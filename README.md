# Patter

Voice dictation that runs locally on your CPU for free, or through a cloud API for pennies. Hold a hotkey, speak, release — your words appear wherever your cursor is.

The key idea: audio is streamed in 15-second chunks that are transcribed in parallel while you speak. By the time you stop talking, only the last chunk remains. Result: **~1.3 seconds of latency regardless of how long you spoke** — even on a laptop CPU with no GPU.

## Why

Commercial dictation tools charge $10-15/month. The underlying technology — OpenAI's Whisper model — is open source and runs well on modern hardware. Patter assembles the pieces: local or cloud transcription, optional LLM cleanup, a floating overlay, and the chunked streaming architecture that makes it fast.

## Quick Start

```bash
git clone https://github.com/joinfiber/patter.git
cd patter
pip install -r requirements.txt
cp config.example.json config.json
```

**Local mode (free, no account needed):**
```bash
pip install faster-whisper
```
Set `"stt_mode": "local"` in config.json. First run downloads the model (~500MB one-time for `small`). That's it.

**Cloud mode (faster, needs API key):**
Set `"stt_mode": "api"` and add your API key to config.json. See [Provider Setup](#provider-setup) below.

**Run:**
```bash
python patter.py
```

## Hotkeys

| Action | Windows | macOS |
|---|---|---|
| Dictate | `Ctrl+Win` (hold) | `Ctrl+Cmd` (hold) |
| Dictate + LLM cleanup | `Shift+Ctrl+Win` (hold) | `Shift+Ctrl+Cmd` (hold) |

Release the keys to stop recording. Text is transcribed and pasted at your cursor.

The LLM cleanup mode removes filler words (um, uh, like, you know), false starts, and repeated phrases while preserving your meaning. Requires a cloud API.

## Performance

Tested on an AMD Ryzen 7 7700 (no discrete GPU, 32GB RAM):

| Mode | Latency after release | Cost |
|---|---|---|
| **Local (small model)** | ~1.3s | Free |
| **Local (medium model)** | ~3-4s | Free |
| **Cloud API (DeepInfra)** | ~0.9-1.5s | ~$0.35/month |
| **Cloud + LLM cleanup** | ~2-3s | ~$0.35/month |

Local latency is constant regardless of recording length thanks to chunked streaming. A 10-second clip and a 10-minute ramble both finish in ~1.3 seconds.

With a CUDA-capable GPU, local performance would be substantially faster.

## How It Works

```
[Hold hotkey] → Mic captures audio
                  ↓
          Every 15 seconds, a chunk is:
            1. Encoded to OGG (compressed ~7x)
            2. Transcribed in the background
               (locally or via API)
                  ↓
[Release hotkey] → Final partial chunk sent
                  ↓
          All transcriptions assembled in order
                  ↓
          (Optional) LLM cleanup pass
                  ↓
          Text pasted at cursor via clipboard
```

## Provider Setup

Cloud mode works with any service that exposes an OpenAI-compatible API. Set `stt_endpoint`, `llm_endpoint`, and `api_key` in config.json.

### Example Configurations

**DeepInfra** (cheapest cloud option — ~$0.35/month at 12k words/day):
```json
{
    "api_key": "your-deepinfra-key",
    "stt_mode": "api",
    "stt_endpoint": "https://api.deepinfra.com/v1/audio/transcriptions",
    "stt_model": "openai/whisper-large-v3-turbo",
    "llm_endpoint": "https://api.deepinfra.com/v1/openai/chat/completions",
    "llm_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
}
```

**OpenAI** (~$3/month):
```json
{
    "api_key": "sk-your-openai-key",
    "stt_mode": "api",
    "stt_endpoint": "https://api.openai.com/v1/audio/transcriptions",
    "stt_model": "whisper-1",
    "llm_endpoint": "https://api.openai.com/v1/chat/completions",
    "llm_model": "gpt-4o-mini"
}
```

**Groq** (fast, has free tier):
```json
{
    "api_key": "your-groq-key",
    "stt_mode": "api",
    "stt_endpoint": "https://api.groq.com/openai/v1/audio/transcriptions",
    "stt_model": "whisper-large-v3-turbo",
    "llm_endpoint": "https://api.groq.com/openai/v1/chat/completions",
    "llm_model": "llama-3.3-70b-versatile"
}
```

**Local (fully offline, free):**
```json
{
    "stt_mode": "local",
    "local_stt_model": "small"
}
```

Available local models (one-time download):

| Model | Size | Quality | Speed (CPU) |
|---|---|---|---|
| `tiny` | ~75MB | Basic | Fastest |
| `base` | ~150MB | Decent | Fast |
| `small` | ~500MB | Good | ~1.3s/chunk |
| `medium` | ~1.5GB | Very good | ~3-4s/chunk |
| `large-v3-turbo` | ~3GB | Best (matches cloud) | Needs GPU |

## Configuration Reference

| Field | Default | Description |
|---|---|---|
| `api_key` | — | API key for cloud provider. Also settable via `PATTER_API_KEY` env var. |
| `stt_mode` | `"api"` | `"api"` for cloud, `"local"` for offline |
| `stt_model` | `"openai/whisper-large-v3-turbo"` | Model identifier for cloud STT |
| `stt_endpoint` | DeepInfra | OpenAI-compatible STT endpoint URL |
| `llm_model` | `"meta-llama/Llama-3.3-70B-Instruct-Turbo"` | Model for LLM cleanup |
| `llm_endpoint` | DeepInfra | OpenAI-compatible chat completions endpoint |
| `local_stt_model` | `"small"` | Whisper model size for local mode |
| `sample_rate` | `16000` | Audio sample rate in Hz |
| `chunk_seconds` | `15` | Seconds of audio per streaming chunk |

## Logs

Daily logs are saved to `logs/YYYY-MM-DD.md` with timestamps, word counts, WPM, processing latency, and both raw and cleaned transcriptions.

## macOS Setup

macOS requires Accessibility permissions for global hotkeys and simulated paste. Go to **System Settings > Privacy & Security > Accessibility** and add your terminal app.

## Requirements

- Python 3.10+
- A microphone
- For local mode: `pip install faster-whisper` (~500MB model download)
- For cloud mode: an API key from any supported provider

## License

MIT
