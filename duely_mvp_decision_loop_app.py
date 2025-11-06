# app.py — fresh, stable single-file MVP (planner-first, OCR as a tool)
# -------------------------------------------------------------------
# This script implements a minimal desktop "perception → decision" loop
# for your Duely assistant. The agent:
#   1) Captures the current screen (DXCAM if available; MSS otherwise)
#   2) Checks visual stability (via perceptual hash) to avoid reprocessing
#   3) Builds a tiny observation and asks a planner (LLM stub) whether to OCR
#   4) If yes, OCRs the active-window region and summarizes to clipboard
#   5) Sleeps according to a target FPS and repeats
#
# Notes on architecture:
# - The StabilityGate detects when the screen has stopped changing enough to act.
# - The planner() is a stub that can be wired to an LLM (e.g., Ollama). For now
#   it can be forced via SETTINGS["planner_force_ocr"].
# - OCR is kept as a tool (ocr_pass), isolated from the main loop.
# - The code is intentionally single-file and dependency-light so we can iterate.

from __future__ import annotations
import sys, time, re
from typing import Optional, Tuple
import numpy as np

import json, os, subprocess
# Imaging / OCR
from PIL import Image, ImageOps, ImageFilter
import imagehash
import pytesseract

# Windows helpers (used to detect the active window bounds for cropping)
import pygetwindow as gw

# Clipboard (used to copy summary after OCR)
import pyperclip

# Capture backends (DXCAM optional, MSS default)
try:
    import dxcam
    _DXCAM_OK = True
except Exception:
    _DXCAM_OK = False  # If DXCAM import fails, we will fall back to MSS

try:
    import mss
except Exception:
    mss = None  # MSS might also be unavailable in some environments

# ---------------- SETTINGS ----------------
# All operational toggles live here to keep the loop readable.
SETTINGS = {
    "capture_fps": 2,            # Target capture rate (frames per second)
    "phash_threshold": 10,       # Max perceptual-hash distance to consider frames "similar"
    "stable_seconds": 0.8,       # How long frames must remain similar to be considered stable
    "ocr_max_chars": 2000,       # OCR output is truncated to this many characters
    "crop_ratio": 0.90,          # Fallback center-crop ratio when no active window
    "tesseract_cmd": None,       # Optional: full path to tesseract executable
    "debug": False,              # Extra prints when True
    "prefer_dxcam": False,       # If True (and DXCAM is ok), try DXCAM first
    "planner_force_ocr": True,   # For testing: force planner to use OCR
}

# Allow explicit Tesseract path if needed (e.g., on Windows when not in PATH)
if SETTINGS.get("tesseract_cmd"):
    pytesseract.pytesseract.tesseract_cmd = SETTINGS["tesseract_cmd"]

# ---------------- Capture ----------------
# Two concrete capture implementations (+ a small orchestrator):
#   _grab_dxcam(): fast on Windows GPUs; returns RGB ndarray
#   _grab_mss(): reliable CPU fallback; returns RGB ndarray
#   grab_frame(): chooses backend, returns (frame, timestamp)

def _grab_dxcam() -> np.ndarray:
    """Capture a single frame using DXCAM and convert BGR→RGB.

    Returns
    -------
    np.ndarray
        H×W×3 RGB image as uint8 array.
    """
    cam = dxcam.create(output_idx=0)
    frame = cam.grab()  # ndarray (H, W, 3) in BGR order
    if frame is None:
        # DXCAM occasionally returns None if no frame is available
        raise RuntimeError("dxcam returned None frame")
    return frame[..., ::-1].copy()  # Convert BGR→RGB in-place-copy for safety


def _grab_mss() -> np.ndarray:
    """Capture a single frame using MSS and convert BGRA→RGB."""
    if mss is None:
        raise RuntimeError("mss not available.")
    with mss.mss() as sct:
        mon = sct.monitors[1]             # Monitor 1 = virtual full desktop
        img = np.array(sct.grab(mon))     # Returns BGRA uint8
        rgb = img[..., :3][..., ::-1]     # Strip alpha; convert BGR→RGB
        return rgb.copy()


def grab_frame() -> Tuple[np.ndarray, float]:
    """Return a frame and a timestamp using the preferred backend."""
    ts = time.time()
    arr = None
    if SETTINGS.get("prefer_dxcam") and _DXCAM_OK:
        try:
            arr = _grab_dxcam()
        except Exception:
            # Fall back to MSS if DXCAM fails mid-run
            arr = _grab_mss()
    else:
        arr = _grab_mss()
    if SETTINGS.get("debug"):
        print("[debug] frame:", arr.shape)
    return arr, ts

# ---------------- Stability Gate ----------------
# The gate uses perceptual hashing (pHash) to measure how much the current
# frame differs from the previous one. If the distance remains below a
# threshold long enough (stable_seconds), we consider the screen "stable".

class StabilityGate:
    def __init__(self, phash_threshold: int = 6, stable_seconds: float = 1.5):
        self.thresh = phash_threshold           # max pHash distance to be "similar"
        self.stable_seconds = stable_seconds    # duration required to call it stable
        self._last_hash: Optional[imagehash.ImageHash] = None
        self._stable_since: Optional[float] = None

    def _hash(self, frame_rgb: np.ndarray) -> imagehash.ImageHash:
        """Compute the pHash of the frame (robust to small visual changes)."""
        return imagehash.phash(Image.fromarray(frame_rgb))

    def update(self, frame_rgb: np.ndarray, ts: float):
        """Update stability state given the new frame and return a tuple:
        (is_stable_now, since_timestamp_or_None, current_hash)
        """
        h = self._hash(frame_rgb)
        if self._last_hash is None:
            # First frame: initialize tracking but not yet stable
            self._last_hash = h
            self._stable_since = None
            return False, None, h

        dist = (h - self._last_hash)  # Hamming distance between pHashes
        if dist <= self.thresh:
            # Similar enough → either start or continue the stability timer
            if self._stable_since is None:
                self._stable_since = ts
            stable_for = ts - self._stable_since
            self._last_hash = h
            return stable_for >= self.stable_seconds, self._stable_since, h
        else:
            # Significant change → reset stability
            self._last_hash = h
            self._stable_since = None
            return False, None, h

# ---------------- OCR helpers (tool) ----------------
# These helpers are isolated so the OCR path can be unit-tested independently.


def _center_crop(rgb: np.ndarray, ratio=0.75) -> np.ndarray:
    """Return a center crop of the frame by the given ratio (0<ratio≤1)."""
    h, w, _ = rgb.shape
    nh, nw = int(h * ratio), int(w * ratio)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    return rgb[y0:y0 + nh, x0:x0 + nw]


def _preprocess_for_ocr(rgb: np.ndarray) -> Image.Image:
    """Lightweight preprocessing to help Tesseract: grayscale → autocontrast →
    upscale slightly → sharpen."""
    img = Image.fromarray(rgb)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.resize((int(img.width * 1.5), int(img.height * 1.5)))
    img = img.filter(ImageFilter.SHARPEN)
    return img


def crop_active_window(frame_rgb: np.ndarray) -> np.ndarray:
    """Crop to the active window bounds when available; otherwise use a
    center crop (per SETTINGS["crop_ratio"]). This reduces OCR noise by
    focusing on the most likely region of interest."""
    try:
        win = gw.getActiveWindow()
        if not win:
            return _center_crop(frame_rgb, SETTINGS.get("crop_ratio", 0.90))
        l, t, r, b = win.left, win.top, win.right, win.bottom
        h, w, _ = frame_rgb.shape
        # Clamp to frame bounds to prevent negative/overflow indices
        l, t = max(0, l), max(0, t)
        r, b = min(w, r), min(h, b)
        # Avoid tiny crops (e.g., when the OS reports a 0×0 window)
        if r - l < 100 or b - t < 100:
            return _center_crop(frame_rgb, SETTINGS.get("crop_ratio", 0.90))
        return frame_rgb[t:b, l:r]
    except Exception:
        # If pygetwindow fails (no permission / platform quirk), fall back
        return _center_crop(frame_rgb, SETTINGS.get("crop_ratio", 0.90))


def is_meaningful(text: str) -> bool:
    """Heuristic to decide whether OCR text looks like real content.
    - Require a minimum length
    - Require a minimum number of word-like tokens
    - Require at least half the characters to be alphabetic
    This avoids copying fragments, artifacts, or UI chrome.
    """
    if not text or len(text) < 60:
        return False
    words = re.findall(r"[A-Za-z]{3,}", text)
    if len(words) < 20:
        return False
    alpha_ratio = sum(ch.isalpha() for ch in text) / max(1, len(text))
    return alpha_ratio >= 0.5


def ocr_text(roi_rgb: np.ndarray, max_chars: int = 2000) -> str:
    """Run Tesseract OCR on the preprocessed region and trim output."""
    cfg = "--oem 1 --psm 3 -l eng --dpi 220"
    text = pytesseract.image_to_string(_preprocess_for_ocr(roi_rgb), config=cfg).strip()
    return (text[:max_chars] + " …") if len(text) > max_chars else text


def quick_summarize(text: str, max_bullets: int = 5) -> str:
    """Very lightweight extractive summarizer:
    - Split lines, normalize whitespace
    - Score lines by uppercase starts, punctuation, and length
    - Keep the top-N as bullet points
    """
    if not text:
        return "(no text detected)"
    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in text.splitlines()]
    scored = []
    for ln in lines:
        if not ln or len(ln) < 5:
            continue
        score = 0
        score += sum(c.isupper() for c in ln[:30])   # Title-ish lines
        score += ln.count(":") + ln.count("-") + ln.count("•")
        score += min(len(ln) // 40, 3)               # Prefer longer lines up to a cap
        scored.append((score, ln))
    scored.sort(reverse=True)
    keep = [ln for _, ln in scored[:max_bullets]]
    return "\n".join(f"• {ln}" for ln in keep) if keep else "(no salient lines found)"

# ---------------- Planner (stub) ----------------
# The planner returns a JSON-like dict indicating whether to use OCR now.
# You can wire it to an LLM by setting the DUELY_LLM env var, e.g.:
#   DUELY_LLM="ollama:llama3"
# If no LLM is configured, it defaults to "skip" unless planner_force_ocr=True.


def planner(observation: dict) -> dict:
    # Hard override for testing: keep the outer loop deterministic while iterating
    if SETTINGS.get("planner_force_ocr") is True:
        return {"use_ocr": True, "notes": "forced by settings"}

    # If no LLM configured, default to not using OCR (safe)
    llm = os.environ.get("DUELY_LLM", "").lower()  # e.g., "ollama:llama3"
    if not llm:
        return {"use_ocr": False, "notes": "no LLM configured"}

    # Build a compact prompt for the LLM. In practice you might include
    # a downscaled thumbnail or features; we keep it text-only here.
    prompt = f"""
You are a tool selector for a desktop agent.
Decide if OCR is appropriate right now. Output ONLY JSON with keys: use_ocr (true/false), notes.
Observation:
- ts: {observation.get('ts')}
- frame_shape: {observation.get('frame_shape')}
- focus_window: {observation.get('focus_window')}
Guidance:
- Use OCR when the active app likely has text (docs, email, browser reading, IDE, PDF, slides).
- Skip when it looks like gaming/video/fullscreen graphics/empty desktop.
JSON only:
"""

    # Example: DUELY_LLM="ollama:llama3"
    if llm.startswith("ollama:"):
        model = llm.split(":", 1)[1]
        try:
            res = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            txt = res.stdout.decode("utf-8").strip()
            # Extract JSON if the model wraps extra text around it
            start = txt.find("{"); end = txt.rfind("}")
            if start != -1 and end != -1 and end > start:
                txt = txt[start:end+1]
            return json.loads(txt)
        except Exception as e:
            # On LLM errors, fail safely (skip OCR) and note the error
            return {"use_ocr": False, "notes": f"LLM error: {e}"}

    # Unknown DUELY_LLM scheme → default safe behavior
    return {"use_ocr": False, "notes": f"unsupported DUELY_LLM={llm}"}


# ---------------- One-call OCR pass (tool) ----------------
# A single function that crops, OCRs, summarizes, and copies to clipboard.


def ocr_pass(frame: np.ndarray) -> str | None:
    # Focus the OCR on the active window (or center crop fallback)
    roi = crop_active_window(frame)

    # Raw OCR text (bounded by SETTINGS["ocr_max_chars"]) for safety
    txt = ocr_text(roi, max_chars=SETTINGS["ocr_max_chars"])

    # Skip if text likely isn't meaningful (UI fragments / noise)
    if not is_meaningful(txt):
        print("[agent] OCR skipped — low meaningful content.")
        return None

    # Extractive bullet summary for quick glance + shareability
    summary = quick_summarize(txt)
    print(f"\n=== SUMMARY ===\n{summary}\n===============\n")

    # Best-effort copy to clipboard to speed up downstream use
    try:
        pyperclip.copy(summary)
        print("[agent] Copied summary to clipboard.")
    except Exception as e:
        print(f"[agent] Clipboard copy failed: {e}")
    return summary

# ---------------- Main loop ----------------
# The main loop coordinates capture → stability → planning → OCR.


def main():
    # Stability gate tuned by top-level SETTINGS for consistency
    gate = StabilityGate(
        phash_threshold=SETTINGS["phash_threshold"],
        stable_seconds=SETTINGS["stable_seconds"],
    )

    cooldown = 6.0          # Minimum seconds between planner triggers
    last_trigger = 0.0       # Timestamp of last planner decision
    last_fire_hash = None    # Hash of last frame when planner ran (avoid repeats)
    rearm_needed = True      # Requires an unstable period before re-trigger

    print("[agent] Armed. Planner-first. (Ctrl+C to exit)")
    while True:
        # 1) Capture the latest frame and timestamp
        frame, ts = grab_frame()

        # 2) Update stability state (returns: is_stable_now, since_ts, curr_hash)
        is_stable, since, curr_hash = gate.update(frame, ts)

        # If the screen is changing again, allow a future trigger after cooldown
        if not is_stable:
            rearm_needed = True

        # 3) If stable and cooled down, consider invoking the planner
        if is_stable:
            # Avoid refiring if the scene hash is near-identical to last fire
            same_as_last = (last_fire_hash is not None and (curr_hash - last_fire_hash) <= 2)

            # Conditions to call planner:
            # - We had instability since the last fire (rearm_needed)
            # - Enough time has passed (cooldown)
            # - The scene is not virtually identical to the last fired state
            if rearm_needed and (ts - last_trigger) > cooldown and not same_as_last:
                # Build a compact observation for the planner
                obs = {
                    "ts": ts,
                    "frame_shape": frame.shape,
                    "focus_window": getattr(gw.getActiveWindow(), "title", None),
                }
                decision = planner(obs)
                if SETTINGS.get("debug"):
                    print(f"[planner] decision: {decision}")

                # 4) Execute OCR tool if planner says so; otherwise skip
                if decision.get("use_ocr"):
                    _ = ocr_pass(frame)
                else:
                    print("[agent] Skipping OCR (planner-first design).")

                # Bookkeeping to throttle/avoid redundant triggers
                last_trigger = ts
                last_fire_hash = curr_hash
                rearm_needed = False
                time.sleep(0.5)  # small post-action pause

        # Optional heartbeat during development
        if SETTINGS.get("debug"):
            print("[tick]", time.strftime("%H:%M:%S"))

        # 5) Sleep to match target FPS; ensure non-negative
        time.sleep(max(0, 1.0 / SETTINGS["capture_fps"]))


if __name__ == "__main__":
    print("STARTED app.py with:", sys.executable)
    time.sleep(0.2)
    try:
        main()
    except Exception:
        # Print the traceback to stderr and wait for Enter so console windows
        # (e.g., on Windows) don’t disappear immediately after a crash.
        import traceback
        traceback.print_exc()
        input("\n[agent] Crashed. Press Enter to close…")
