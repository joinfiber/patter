"""
Patter — Voice dictation with AI cleanup.

Hold Ctrl+Win (Windows) or Ctrl+Cmd (Mac) to record, release to transcribe
and paste. Add Shift for optional LLM cleanup of filler words and false starts.

Audio is streamed in 15-second chunks for near-instant results on long recordings.
Works with any OpenAI-compatible STT and LLM endpoint.
"""

import io
import json
import os
import platform
import sys
import threading
import time
import traceback
import wave
from datetime import datetime

import numpy as np
import pyperclip
import requests
import sounddevice as sd
from pynput import keyboard

from overlay import DictationOverlay

IS_MAC = platform.system() == "Darwin"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Config ──────────────────────────────────────────────────────────────────

def load_config():
    config_path = os.path.join(SCRIPT_DIR, "config.json")
    if not os.path.exists(config_path):
        print("ERROR: config.json not found.")
        print("  Copy config.example.json to config.json and add your API key.")
        sys.exit(1)
    with open(config_path, "r") as f:
        return json.load(f)

CONFIG = load_config()

API_KEY = os.environ.get("PATTER_API_KEY", CONFIG.get("api_key", ""))
STT_MODE = CONFIG.get("stt_mode", "api")
STT_MODEL = CONFIG.get("stt_model", "openai/whisper-large-v3-turbo")
STT_ENDPOINT = CONFIG.get("stt_endpoint", "https://api.deepinfra.com/v1/audio/transcriptions")
LOCAL_STT_MODEL = CONFIG.get("local_stt_model", "small")
LLM_MODEL = CONFIG.get("llm_model", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
LLM_ENDPOINT = CONFIG.get("llm_endpoint", "https://api.deepinfra.com/v1/openai/chat/completions")
SAMPLE_RATE = CONFIG.get("sample_rate", 16000)
CHUNK_SECONDS = CONFIG.get("chunk_seconds", 15)

LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ── Local Whisper (lazy-loaded) ─────────────────────────────────────────────

_local_whisper = None

def get_local_whisper():
    global _local_whisper
    if _local_whisper is None:
        from faster_whisper import WhisperModel
        print(f"  Loading local Whisper model '{LOCAL_STT_MODEL}'...")
        _local_whisper = WhisperModel(LOCAL_STT_MODEL, device="cpu", compute_type="int8")
        print(f"  Model loaded.")
    return _local_whisper

# ── State ───────────────────────────────────────────────────────────────────

recording = False
use_llm_this_recording = False
audio_frames = []
lock = threading.Lock()
overlay = None

chunk_results = []
chunk_threads = []
chunk_frame_count = 0

session = requests.Session()
if API_KEY:
    session.headers.update({"Authorization": f"Bearer {API_KEY}"})

# ── Audio encoding ──────────────────────────────────────────────────────────

def frames_to_wav(frames):
    """Convert raw PCM frames to WAV bytes in memory."""
    audio = np.concatenate(frames, axis=0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()

def frames_to_ogg(frames):
    """Compress PCM frames to OGG Vorbis. Safe for chunks under 45s."""
    import soundfile as sf  # lazy: not needed in local mode, cached after first call
    audio = np.concatenate(frames, axis=0).astype(np.float32) / 32768.0
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="OGG", subtype="VORBIS")
    return buf.getvalue()

# ── Transcription ───────────────────────────────────────────────────────────

MAX_RETRIES = 3

def transcribe_api(audio_bytes, filename="audio.wav", mime="audio/wav"):
    """Send audio to STT endpoint with retry on transient errors."""
    files = {"file": (filename, audio_bytes, mime)}
    data = {"model": STT_MODEL}

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = session.post(STT_ENDPOINT, files=files, data=data, timeout=120)
            if resp.status_code == 429:
                if attempt < MAX_RETRIES:
                    wait = 3 * (attempt + 1)
                    print(f"  [!] Rate limited, retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
            resp.raise_for_status()
            return resp.json().get("text", "").strip()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"  [!] Timeout, retrying in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise

def transcribe_local(wav_bytes):
    """Transcribe audio locally using faster-whisper."""
    model = get_local_whisper()
    buf = io.BytesIO(wav_bytes)
    segments, _ = model.transcribe(buf, language="en", beam_size=1, vad_filter=False)
    return " ".join(seg.text.strip() for seg in segments).strip()

# ── LLM cleanup ─────────────────────────────────────────────────────────────

CLEANUP_PROMPT = """\
You are a text filter. You receive dictated text and return a cleaned version. \
You are NOT a chatbot. Do NOT converse, reply, acknowledge, or add anything.

Rules: Remove filler words (um, uh, like, you know, okay, so) and \
repeated/false-start phrases. Keep ALL substantive content. Do not summarize. \
Output ONLY the cleaned text and NOTHING else. STOP after the cleaned text.

Input: "So I was like, um, I think we should, we should probably just, you \
know, go with the first option and then and then move on to the next thing."
Output: "I think we should probably just go with the first option and then \
move on to the next thing."

Input: "Okay so this is a test to see what happens when I, uh, when I ramble \
for a bit. I wonder what I should have for dinner. What do you think I should \
have for dinner?"
Output: "This is a test to see what happens when I ramble for a bit. I wonder \
what I should have for dinner. What do you think I should have for dinner?"
"""

def llm_cleanup(raw_text):
    """Run the raw transcription through an LLM for cleanup."""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": CLEANUP_PROMPT},
            {"role": "user", "content": raw_text},
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    resp = session.post(LLM_ENDPOINT, json=payload, timeout=60)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    cleaned = resp.json()["choices"][0]["message"]["content"].strip()

    # Guard against common LLM output quirks: wrapping in quotes,
    # or appending chatbot commentary after a double newline.
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
    if "\n\n" in cleaned:
        cleaned = cleaned.split("\n\n")[0].strip()

    print(f"  [LLM] {elapsed:.1f}s", flush=True)
    return cleaned

# ── History log ─────────────────────────────────────────────────────────────

def log_entry(duration_s, raw_text, final_text, total_time):
    """Append a timestamped entry to today's log file."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"{today}.md")
    now = datetime.now().strftime("%H:%M:%S")
    word_count = len(final_text.split())
    wpm = int(word_count / (duration_s / 60)) if duration_s > 0 else 0

    entry = (
        f"## {now}\n"
        f"- **Duration:** {duration_s:.1f}s | **Words:** {word_count} "
        f"| **WPM:** {wpm} | **Latency:** {total_time:.1f}s\n"
        f"- **Raw:** {raw_text}\n"
        f"- **Clean:** {final_text}\n\n"
    )

    is_new = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if is_new:
            f.write(f"# Dictation Log \u2014 {today}\n\n")
        f.write(entry)

# ── Paste to active window ──────────────────────────────────────────────────

def paste_text(text):
    """Copy text to clipboard and simulate paste."""
    pyperclip.copy(text)
    time.sleep(0.15)
    kb = keyboard.Controller()
    for mod in [keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
                keyboard.Key.cmd, keyboard.Key.cmd_r, keyboard.Key.shift,
                keyboard.Key.shift_r]:
        try:
            kb.release(mod)
        except Exception:
            pass
    time.sleep(0.05)
    paste_mod = keyboard.Key.cmd if IS_MAC else keyboard.Key.ctrl
    kb.press(paste_mod)
    kb.tap("v")
    kb.release(paste_mod)

# ── Chunked transcription ──────────────────────────────────────────────────

def dispatch_chunk(chunk_frames, chunk_index):
    """Encode and transcribe a chunk in the background."""
    try:
        if STT_MODE == "local":
            audio_bytes, fname, mime = frames_to_wav(chunk_frames), "audio.wav", "audio/wav"
        else:
            audio_bytes, fname, mime = frames_to_ogg(chunk_frames), "audio.ogg", "audio/ogg"
        raw_kb = sum(f.shape[0] for f in chunk_frames) * 2 // 1024
        print(f"    [Chunk {chunk_index}] {len(audio_bytes)//1024}KB "
              f"(from {raw_kb}KB) sending...", flush=True)
        t0 = time.perf_counter()
        if STT_MODE == "local":
            text = transcribe_local(audio_bytes)
        else:
            text = transcribe_api(audio_bytes, fname, mime)
        elapsed = time.perf_counter() - t0
        print(f'    [Chunk {chunk_index}] {elapsed:.1f}s', flush=True)
        with lock:
            while len(chunk_results) <= chunk_index:
                chunk_results.append(None)
            chunk_results[chunk_index] = text
    except Exception as e:
        print(f"    [Chunk {chunk_index}] Error: {e}", flush=True)
        with lock:
            while len(chunk_results) <= chunk_index:
                chunk_results.append(None)
            chunk_results[chunk_index] = ""

def audio_callback(indata, frames, time_info, status):
    """Capture audio and dispatch chunks every CHUNK_SECONDS."""
    global chunk_frame_count
    if not recording:
        return
    audio_frames.append(indata.copy())
    if overlay:
        rms = float(np.sqrt(np.mean(indata.astype(np.float64) ** 2)))
        overlay.push_audio_level(rms)
    chunk_frame_count += frames
    if chunk_frame_count >= CHUNK_SECONDS * SAMPLE_RATE:
        with lock:
            chunk_frames = list(audio_frames)
            audio_frames.clear()
            chunk_index = len(chunk_threads)
        chunk_frame_count = 0
        t = threading.Thread(target=dispatch_chunk, args=(chunk_frames, chunk_index), daemon=True)
        t.start()
        chunk_threads.append(t)

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16",
    callback=audio_callback,
    blocksize=1024,
)

# ── Pipeline ────────────────────────────────────────────────────────────────

def process_audio():
    """Transcribe remaining audio, wait for chunks, assemble, and paste."""
    global chunk_frame_count
    try:
        with lock:
            remaining_frames = list(audio_frames)
            audio_frames.clear()
        chunk_frame_count = 0

        all_threads = list(chunk_threads)
        remainder_duration = sum(f.shape[0] for f in remaining_frames) / SAMPLE_RATE if remaining_frames else 0
        total_chunks = len(all_threads) + (1 if remaining_frames and remainder_duration >= 0.3 else 0)

        if total_chunks == 0 and remainder_duration < 0.3:
            print("  [!] Too short, skipping")
            return

        t_total = time.perf_counter()

        if remaining_frames and remainder_duration >= 0.3:
            remainder_index = len(all_threads)
            print(f"  [Remainder] {remainder_duration:.1f}s left to process", flush=True)
            dispatch_chunk(remaining_frames, remainder_index)

        for i, t in enumerate(all_threads):
            t.join(timeout=120)
            if overlay:
                frac = (i + 1) / total_chunks * 0.7
                overlay.schedule(overlay.set_progress, frac)

        with lock:
            raw_text = " ".join(r for r in chunk_results if r).strip()

        if not raw_text:
            print("  [!] Empty transcription")
            return

        print(f"  [STT] assembled {total_chunks} chunk(s)", flush=True)

        if use_llm_this_recording:
            if overlay:
                overlay.schedule(overlay.set_progress, 0.75, "cleaning")
            final_text = llm_cleanup(raw_text)
        else:
            final_text = raw_text

        elapsed_total = time.perf_counter() - t_total
        total_duration = len(all_threads) * CHUNK_SECONDS + remainder_duration
        print(f"  [Total] {elapsed_total:.1f}s to process {total_duration:.0f}s of audio")

        log_entry(total_duration, raw_text, final_text, elapsed_total)

        if overlay:
            overlay.schedule(overlay.set_progress, 1.0, "done")
        paste_text(final_text)
        print(f"  [Pasted]")
        time.sleep(0.5)

    except Exception as e:
        print(f"  [!] Error: {e}")
        traceback.print_exc()
    finally:
        if overlay:
            overlay.schedule(overlay.hide)

# ── Hotkey handling ─────────────────────────────────────────────────────────

ctrl_pressed = False
win_pressed = False
shift_pressed = False

def on_press(key):
    global recording, ctrl_pressed, win_pressed, shift_pressed
    global use_llm_this_recording, chunk_frame_count

    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        ctrl_pressed = True
    elif key in (keyboard.Key.cmd, keyboard.Key.cmd_r):
        win_pressed = True
    elif key in (keyboard.Key.shift, keyboard.Key.shift_r):
        shift_pressed = True

    if ctrl_pressed and win_pressed and not recording:
        recording = True
        use_llm_this_recording = shift_pressed
        audio_frames.clear()
        chunk_results.clear()
        chunk_threads.clear()
        chunk_frame_count = 0
        mode = "LLM" if use_llm_this_recording else "raw"
        print(f"\n  Recording ({mode})...", flush=True)
        if overlay:
            overlay.schedule(overlay.show_recording, shift_pressed)

def on_release(key):
    global recording, ctrl_pressed, win_pressed, shift_pressed

    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        ctrl_pressed = False
    elif key in (keyboard.Key.cmd, keyboard.Key.cmd_r):
        win_pressed = False
    elif key in (keyboard.Key.shift, keyboard.Key.shift_r):
        shift_pressed = False

    if recording and (not ctrl_pressed or not win_pressed):
        recording = False
        print("  Stopped. Processing...", flush=True)
        if overlay:
            overlay.schedule(overlay.show_processing)
        threading.Thread(target=process_audio, daemon=True).start()

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    global overlay

    needs_api_key = STT_MODE == "api"
    if needs_api_key and (not API_KEY or API_KEY == "YOUR_API_KEY_HERE"):
        print("ERROR: No API key configured.")
        print("  Set PATTER_API_KEY env var, or add 'api_key' to config.json.")
        print()
        print("  For local-only mode (no API key needed), set:")
        print('    "stt_mode": "local"  in config.json')
        sys.exit(1)

    hotkey = "Ctrl+Cmd" if IS_MAC else "Ctrl+Win"
    stt_display = f"local ({LOCAL_STT_MODEL})" if STT_MODE == "local" else f"api ({STT_MODEL})"
    config_path = os.path.join(SCRIPT_DIR, "config.json")

    print("=" * 50)
    print("  Patter")
    print("=" * 50)
    print(f"  STT:      {stt_display}")
    if needs_api_key:
        print(f"  LLM:      {LLM_MODEL}")
    print(f"  Chunks:   {CHUNK_SECONDS}s")
    print(f"  Hotkey:   {hotkey} (raw)")
    if needs_api_key:
        print(f"            Shift+{hotkey} (LLM cleanup)")
    print(f"  Config:   {config_path}")
    print(f"  Logs:     {LOG_DIR}")
    print("-" * 50)
    print("  Settings (edit config.json):")
    print('    "stt_mode"       "api" or "local"')
    print('    "stt_model"      model name for STT')
    print('    "stt_endpoint"   any OpenAI-compatible URL')
    print('    "llm_model"      model name for cleanup')
    print('    "chunk_seconds"  audio chunk size (default 15)')
    print("=" * 50)

    stream.start()

    if STT_MODE == "local":
        get_local_whisper()
    elif needs_api_key:
        try:
            session.head(STT_ENDPOINT, timeout=5)
        except Exception:
            pass

    print("Ready.\n")

    overlay = DictationOverlay()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    try:
        overlay.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down.")
        stream.stop()
        listener.stop()

if __name__ == "__main__":
    main()
