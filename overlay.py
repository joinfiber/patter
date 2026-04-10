"""
Floating overlay widget for voice dictation status.
Live waveform with volume feedback during recording,
progress bar with phase indicator during processing.
"""

import tkinter as tk
import math
import platform
import time
import threading
import collections

# System font per platform
_FONT_FAMILY = {
    "Windows": "Segoe UI",
    "Darwin": "SF Pro",
}.get(platform.system(), "sans-serif")


class DictationOverlay:
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"

    # Volume thresholds (RMS of int16 samples, range 0-32768)
    VOL_QUIET = 600
    VOL_LOUD = 14000

    # Palette
    BG = "#111118"
    BORDER = "#2a2a3a"
    TEXT = "#b0b0c0"
    TEXT_DIM = "#555568"
    GREEN = "#30d080"
    YELLOW = "#e0b040"
    RED = "#e04040"
    BLUE = "#4080ff"
    BAR_BG = "#1c1c28"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.92)
        # Use a color key for transparency so only the pill shape is visible
        self._TRANSPARENT = "#010101"
        self.root.configure(bg=self._TRANSPARENT)
        self.root.wm_attributes("-transparentcolor", self._TRANSPARENT)

        self.state = self.IDLE
        self.anim_tick = 0
        self._visible = False

        # Audio levels fed from audio thread
        self._lock = threading.Lock()
        self._current_rms = 0.0
        self._level_history = collections.deque(maxlen=48)

        # Processing state
        self._proc_start = 0.0
        self._proc_progress = 0.0   # 0..1 target
        self._proc_smooth = 0.0     # smoothed display value
        self._proc_phase = "transcribing"

        # Dimensions
        self.width = 300
        self.height = 52

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - self.width) // 2
        y = screen_h - self.height - 60
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")

        self.canvas = tk.Canvas(
            self.root, width=self.width, height=self.height,
            bg=self._TRANSPARENT, highlightthickness=0
        )
        self.canvas.pack()

        # Drag support
        self._drag_x = 0
        self._drag_y = 0
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)

        # Border pill
        self._draw_pill(1, 1, self.width - 1, self.height - 1, self.BG, self.BORDER)

        # ── Recording elements ──
        self._rec_dot = self.canvas.create_oval(0, 0, 0, 0, fill=self.RED, outline="")
        self._rec_label = self.canvas.create_text(
            0, 0, anchor="w", text="", font=(_FONT_FAMILY, 9, "bold"), fill=self.TEXT
        )
        self._vol_hint = self.canvas.create_text(
            0, 0, anchor="e", text="", font=(_FONT_FAMILY, 8), fill=self.TEXT_DIM
        )

        # Waveform bars
        self._num_bars = 32
        self._bar_x0 = 56          # left edge of bar area
        self._bar_area_w = self.width - self._bar_x0 - 12
        self._bar_spacing = self._bar_area_w / self._num_bars
        self._bar_w = max(3, self._bar_spacing - 2)
        self._bars = []
        for i in range(self._num_bars):
            bx = self._bar_x0 + i * self._bar_spacing
            bar = self.canvas.create_rectangle(
                bx, 26, bx + self._bar_w, 26,
                fill=self.GREEN, outline="", width=0
            )
            self._bars.append(bar)

        # ── Processing elements ──
        self._proc_label = self.canvas.create_text(
            self.width // 2, 16, anchor="center",
            text="", font=(_FONT_FAMILY, 9, "bold"), fill=self.TEXT
        )
        self._proc_time = self.canvas.create_text(
            self.width - 14, 16, anchor="e",
            text="", font=(_FONT_FAMILY, 8), fill=self.TEXT_DIM
        )
        bar_x1, bar_x2 = 14, self.width - 14
        bar_y1, bar_y2 = 33, 39
        self._pbar_bg = self.canvas.create_rectangle(
            bar_x1, bar_y1, bar_x2, bar_y2,
            fill=self.BAR_BG, outline=self.BORDER, width=1
        )
        self._pbar_fg = self.canvas.create_rectangle(
            bar_x1, bar_y1, bar_x1, bar_y2,
            fill=self.BLUE, outline=""
        )

        # Start everything hidden
        self._hide_recording_elements()
        self._hide_processing_elements()
        self.root.withdraw()
        self._animate()

    # ── Drawing helpers ──

    def _draw_pill(self, x1, y1, x2, y2, fill, outline):
        r = (y2 - y1) // 2
        self.canvas.create_arc(x1, y1, x1 + 2 * r, y2, start=90, extent=180, fill=fill, outline=outline)
        self.canvas.create_arc(x2 - 2 * r, y1, x2, y2, start=270, extent=180, fill=fill, outline=outline)
        self.canvas.create_rectangle(x1 + r, y1, x2 - r, y2, fill=fill, outline=outline)

    def _hide_recording_elements(self):
        self.canvas.coords(self._rec_dot, 0, 0, 0, 0)
        self.canvas.itemconfig(self._rec_label, text="")
        self.canvas.itemconfig(self._vol_hint, text="")
        for b in self._bars:
            self.canvas.itemconfig(b, state="hidden")

    def _hide_processing_elements(self):
        self.canvas.itemconfig(self._proc_label, text="")
        self.canvas.itemconfig(self._proc_time, text="")
        self.canvas.coords(self._pbar_bg, 0, 0, 0, 0)
        self.canvas.coords(self._pbar_fg, 0, 0, 0, 0)

    # ── Public API (called via schedule() for thread safety) ──

    def push_audio_level(self, rms):
        """Feed current RMS level from audio thread. rms is raw int16 magnitude."""
        with self._lock:
            self._current_rms = rms
            self._level_history.append(rms)

    def show_recording(self, enhanced=False):
        self.state = self.RECORDING
        self.anim_tick = 0
        with self._lock:
            self._level_history.clear()
            self._current_rms = 0.0

        self._hide_processing_elements()

        label = "REC  LLM" if enhanced else "REC"
        self.canvas.itemconfig(self._rec_label, text=label)
        self.canvas.coords(self._rec_label, 30, self.height // 2)
        self.canvas.itemconfig(self._vol_hint, text="")
        self.canvas.coords(self._vol_hint, self.width - 10, self.height // 2)

        for b in self._bars:
            self.canvas.itemconfig(b, state="normal")

        if not self._visible:
            self.root.deiconify()
            self._visible = True

    def show_processing(self):
        self.state = self.PROCESSING
        self.anim_tick = 0
        self._proc_start = time.perf_counter()
        self._proc_progress = 0.0
        self._proc_smooth = 0.0
        self._proc_phase = "transcribing"

        self._hide_recording_elements()

        bar_x1, bar_x2 = 14, self.width - 14
        bar_y1, bar_y2 = 33, 39
        self.canvas.coords(self._pbar_bg, bar_x1, bar_y1, bar_x2, bar_y2)
        self.canvas.coords(self._pbar_fg, bar_x1, bar_y1, bar_x1, bar_y2)
        self.canvas.itemconfig(self._proc_label, text="Transcribing...")
        self.canvas.coords(self._proc_label, self.width // 2, 18)
        self.canvas.coords(self._proc_time, self.width - 14, 18)

        if not self._visible:
            self.root.deiconify()
            self._visible = True

    def set_progress(self, fraction, phase=None):
        """Update processing progress. fraction: 0..1, phase: transcribing|cleaning|done."""
        self._proc_progress = min(1.0, max(0.0, fraction))
        if phase:
            self._proc_phase = phase
            labels = {
                "transcribing": "Transcribing...",
                "cleaning": "Cleaning up...",
                "done": "Done",
            }
            self.canvas.itemconfig(self._proc_label, text=labels.get(phase, "Processing..."))

    def hide(self):
        self.state = self.IDLE
        self._hide_recording_elements()
        self._hide_processing_elements()
        self.root.withdraw()
        self._visible = False

    # ── Animation ──

    def _volume_color(self, rms):
        if rms < self.VOL_QUIET:
            return self.YELLOW
        if rms > self.VOL_LOUD:
            return self.RED
        return self.GREEN

    def _animate(self):
        self.anim_tick += 1
        if self.state == self.RECORDING:
            self._animate_recording()
        elif self.state == self.PROCESSING:
            self._animate_processing()
        self.root.after(33, self._animate)

    def _animate_recording(self):
        with self._lock:
            rms = self._current_rms
            history = list(self._level_history)

        # Pulsing rec dot
        pulse = 0.5 + 0.5 * math.sin(self.anim_tick * 0.18)
        r_val = int(180 + 75 * pulse)
        self.canvas.itemconfig(self._rec_dot, fill=f"#{r_val:02x}2020")
        sz = 4.5 + 1.5 * pulse
        cx, cy = 17, self.height // 2
        self.canvas.coords(self._rec_dot, cx - sz, cy - sz, cx + sz, cy + sz)

        # Volume hint
        if rms < self.VOL_QUIET:
            self.canvas.itemconfig(self._vol_hint, text="speak up", fill=self.YELLOW)
        elif rms > self.VOL_LOUD:
            self.canvas.itemconfig(self._vol_hint, text="too loud", fill=self.RED)
        else:
            self.canvas.itemconfig(self._vol_hint, text="")

        # Waveform bars
        n_hist = len(history)
        center_y = self.height // 2
        max_h = 18

        for i in range(self._num_bars):
            # Map bar to history sample
            if n_hist > 0:
                idx = max(0, n_hist - self._num_bars + i)
                if idx < n_hist:
                    level = history[idx]
                else:
                    level = 0.0
            else:
                level = 0.0

            # Perceptual scaling: sqrt for better visual spread
            norm = min(1.0, level / 12000.0)
            norm = math.sqrt(norm)
            h = max(1.5, norm * max_h)

            bx = self._bar_x0 + i * self._bar_spacing
            self.canvas.coords(self._bars[i],
                               bx, center_y - h / 2,
                               bx + self._bar_w, center_y + h / 2)
            self.canvas.itemconfig(self._bars[i], fill=self._volume_color(level))

    def _animate_processing(self):
        elapsed = time.perf_counter() - self._proc_start

        # Smooth approach toward target
        self._proc_smooth += (self._proc_progress - self._proc_smooth) * 0.08

        # Slow time-based creep so it never looks frozen
        creep = min(0.12, elapsed * 0.008)
        display = self._proc_smooth + creep
        if self._proc_phase != "done":
            display = min(display, 0.95)
        else:
            display = min(display + 0.3, 1.0)  # snap toward 100% on done

        # Bar geometry
        bar_x1, bar_x2 = 14, self.width - 14
        bar_y1, bar_y2 = 33, 39
        fill_x = bar_x1 + (bar_x2 - bar_x1) * display
        self.canvas.coords(self._pbar_fg, bar_x1, bar_y1, fill_x, bar_y2)

        # Color: shimmer blue while active, solid green on done
        if self._proc_phase == "done":
            self.canvas.itemconfig(self._pbar_fg, fill=self.GREEN)
        else:
            shimmer = 0.5 + 0.5 * math.sin(self.anim_tick * 0.18)
            r = int(50 + 30 * shimmer)
            g = int(110 + 30 * shimmer)
            self.canvas.itemconfig(self._pbar_fg, fill=f"#{r:02x}{g:02x}ff")

        # Elapsed time
        self.canvas.itemconfig(self._proc_time, text=f"{elapsed:.1f}s")

    # ── Drag ──

    def _on_drag_start(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag_move(self, event):
        x = self.root.winfo_x() + event.x - self._drag_x
        y = self.root.winfo_y() + event.y - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    def run(self):
        self.root.mainloop()

    def schedule(self, func, *args):
        """Thread-safe: run func on the tkinter main thread."""
        self.root.after(0, func, *args)
