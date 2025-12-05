# canvas_5.py — multi-turn Duely MVP with planner flags wired to execution

from __future__ import annotations
import sys, time, re
from typing import Optional, Tuple
import numpy as np
import pyautogui
import json, os, subprocess
from google import genai
from google.genai import types
from io import BytesIO

# Imaging / OCR
from PIL import Image, ImageOps, ImageFilter
import imagehash
import pytesseract

# Windows helpers
import pygetwindow as gw

# Clipboard
import pyperclip

# Capture backends (DXCAM optional, MSS fallback)
try:
    import dxcam
    _DXCAM_OK = True
except Exception:
    _DXCAM_OK = False

try:
    import mss
except Exception:
    mss = None

# Optional hotkey stop
try:
    import keyboard
    _KEYBOARD_OK = True
except Exception:
    keyboard = None
    _KEYBOARD_OK = False

STOP_REQUESTED = False


def _install_hotkeys():
    global STOP_REQUESTED
    if not _KEYBOARD_OK:
        return
    try:
        def trigger_stop():
            global STOP_REQUESTED
            STOP_REQUESTED = True

        keyboard.add_hotkey("esc", trigger_stop)
        keyboard.add_hotkey("ctrl+shift+q", trigger_stop)
    except Exception:
        pass


SETTINGS = {
    "capture_fps": 1,
    "phash_threshold": 10,
    "stable_seconds": 0.8,
    "stability_timeout": 4.0,
    "ocr_max_chars": 2000,
    "crop_ratio": 0.90,
    "tesseract_cmd": None,
    "debug": False,
    "prefer_dxcam": False,
    "planner_force_ocr": True,
}

# High-level task description for this run.
# Edit this string to change what Duely is trying to accomplish.
TASK_PROMPT = (
    "Open a new text file on my computer and type 'Hello World!'"
)

pyautogui.FAILSAFE = True

if SETTINGS.get("tesseract_cmd"):
    pytesseract.pytesseract.tesseract_cmd = SETTINGS["tesseract_cmd"]

# ---------------- Capture ----------------

def _grab_dxcam() -> np.ndarray:
    cam = dxcam.create(output_idx=0)
    frame = cam.grab()
    if frame is None:
        raise RuntimeError("dxcam returned None frame")
    return frame[..., ::-1].copy()


def _grab_mss() -> np.ndarray:
    if mss is None:
        raise RuntimeError("mss not available.")
    with mss.mss() as sct:
        mon = sct.monitors[1]
        img = np.array(sct.grab(mon))
        rgb = img[..., :3][..., ::-1]
        return rgb.copy()


def grab_frame() -> Tuple[np.ndarray, float]:
    ts = time.time()
    if SETTINGS.get("prefer_dxcam") and _DXCAM_OK:
        try:
            arr = _grab_dxcam()
        except Exception:
            arr = _grab_mss()
    else:
        arr = _grab_mss()
    if SETTINGS.get("debug"):
        print("[debug] frame:", arr.shape)
    return arr, ts


# ---------------- Stability Gate ----------------

class StabilityGate:
    def __init__(self, phash_threshold: int = 6, stable_seconds: float = 1.5):
        self.thresh = phash_threshold
        self.stable_seconds = stable_seconds
        self._last_hash: Optional[imagehash.ImageHash] = None
        self._stable_since: Optional[float] = None

    def _hash(self, frame_rgb: np.ndarray) -> imagehash.ImageHash:
        return imagehash.phash(Image.fromarray(frame_rgb))

    def update(self, frame_rgb: np.ndarray, ts: float):
        h = self._hash(frame_rgb)
        if self._last_hash is None:
            self._last_hash = h
            self._stable_since = None
            return False, None, h

        dist = (h - self._last_hash)
        if dist <= self.thresh:
            if self._stable_since is None:
                self._stable_since = ts
            stable_for = ts - self._stable_since
            self._last_hash = h
            return stable_for >= self.stable_seconds, self._stable_since, h
        else:
            self._last_hash = h
            self._stable_since = None
            return False, None, h


# ---------------- OCR helpers ----------------

def _center_crop(rgb: np.ndarray, ratio=0.75) -> np.ndarray:
    h, w, _ = rgb.shape
    nh, nw = int(h * ratio), int(w * ratio)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    return rgb[y0:y0 + nh, x0:x0 + nw]


def _preprocess_for_ocr(rgb: np.ndarray) -> Image.Image:
    img = Image.fromarray(rgb)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.resize((int(img.width * 1.5), int(img.height * 1.5)))
    img = img.filter(ImageFilter.SHARPEN)
    return img


def crop_active_window(frame_rgb: np.ndarray) -> np.ndarray:
    try:
        win = gw.getActiveWindow()
        if not win:
            return _center_crop(frame_rgb, SETTINGS.get("crop_ratio", 0.90))
        l, t, r, b = win.left, win.top, win.right, win.bottom
        h, w, _ = frame_rgb.shape
        l, t = max(0, l), max(0, t)
        r, b = min(w, r), min(h, b)
        if r - l < 100 or b - t < 100:
            return _center_crop(frame_rgb, SETTINGS.get("crop_ratio", 0.90))
        return frame_rgb[t:b, l:r]
    except Exception:
        return _center_crop(frame_rgb, 0.90)


def is_meaningful(text: str) -> bool:
    if not text or len(text) < 60:
        return False
    words = re.findall(r"[A-Za-z]{3,}", text)
    if len(words) < 20:
        return False
    alpha_ratio = sum(ch.isalpha() for ch in text) / max(1, len(text))
    return alpha_ratio >= 0.5


def ocr_text(roi_rgb: np.ndarray, max_chars: int = 2000) -> str:
    cfg = "--oem 1 --psm 3 -l eng --dpi 220"
    text = pytesseract.image_to_string(_preprocess_for_ocr(roi_rgb), config=cfg).strip()
    return (text[:max_chars] + " …") if len(text) > max_chars else text


def quick_summarize(text: str, max_bullets: int = 5) -> str:
    if not text:
        return "(no text detected)"
    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in text.splitlines()]
    scored = []
    for ln in lines:
        if not ln or len(ln) < 5:
            continue
        score = 0
        score += sum(c.isupper() for c in ln[:30])
        score += ln.count(":") + ln.count("-") + ln.count("•")
        score += min(len(ln) // 40, 3)
        scored.append((score, ln))
    scored.sort(reverse=True)
    keep = [ln for _, ln in scored[:max_bullets]]
    return "\n".join(f"• {ln}" for ln in keep) if keep else "(no salient lines found)"


def ocr_pass(frame: np.ndarray) -> str | None:
    roi = crop_active_window(frame)
    txt = ocr_text(roi, max_chars=SETTINGS["ocr_max_chars"])
    if not is_meaningful(txt):
        print("[agent] OCR skipped — low meaningful content.")
        return None
    summary = quick_summarize(txt)
    print(f"\n=== SUMMARY ===\n{summary}\n===============\n")
    try:
        pyperclip.copy(summary)
        print("[agent] Copied summary to clipboard.")
    except Exception as e:
        print(f"[agent] Clipboard copy failed: {e}")
    return summary


# ---------------- High-res fallback helper ----------------

def request_high_res_plan(prompt: str, image_bytes: bytes) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return ""
    try:
        client = genai.Client(api_key=api_key)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image_part],
        )
        return response.text or ""
    except Exception:
        return ""


# ---------------- Planner parsing + state ----------------

def parse_llm_response(raw: str):  # LLM -> structured plan + notes
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("LLM response could not be parsed: no JSON block found")

    try:
        payload = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM JSON decoding failed: {e}")

    if not isinstance(payload.get("action_summary"), str):
        raise RuntimeError("LLM JSON missing or invalid 'action_summary' string")
    if not isinstance(payload.get("steps"), list):
        raise RuntimeError("LLM JSON missing or invalid 'steps' list")

    todo = payload.get("todo", [])
    if not isinstance(todo, list):
        raise RuntimeError("LLM JSON missing or invalid 'todo' list")

    # Option C: do not enforce notes type here; let the LLM control it.
    # If "notes" is omitted, we treat it as "no change" later.
    if "notes" in payload:
        notes = payload["notes"]
    else:
        notes = None

    action_summary = payload["action_summary"]
    steps = payload["steps"]

    raw_expect_change = payload.get("expect_change", None)
    if raw_expect_change not in (True, False, None):
        raise RuntimeError("LLM JSON invalid 'expect_change' (must be true, false, or null)")
    expect_change = raw_expect_change

    needs_high_res = bool(payload.get("needs_high_res", False))

    skip_turn = bool(payload.get("skip_turn", False))
    sleep_seconds = payload.get("sleep_seconds", 0.0)
    try:
        sleep_seconds = float(sleep_seconds)
    except Exception:
        sleep_seconds = 0.0

    task_done = bool(payload.get("task_done", False))

    return action_summary, steps, todo, notes, expect_change, needs_high_res, skip_turn, sleep_seconds, task_done


def update_memory(memory: dict, action_summary: str, todo: list, notes):
    new_memory = dict(memory) if memory is not None else {}
    turn = int(new_memory.get("turn", 0)) + 1
    new_memory["turn"] = turn
    new_memory["last_action"] = action_summary
    new_memory["todo"] = todo

    # Option C: do not force notes into a dict.
    # - If notes is None (key omitted), preserve existing notes.
    # - If notes is present (any JSON type), overwrite previous notes with it.
    if notes is not None:
        new_memory["notes"] = notes

    return new_memory


def update_status(status: dict, expect_change, needs_high_res: bool,
                  skip_turn: bool, sleep_seconds: float, task_done: bool) -> dict:
    new_status = dict(status) if status is not None else {}
    new_status["expect_change"] = expect_change
    new_status["needs_high_res"] = bool(needs_high_res)
    new_status["skip_turn"] = bool(skip_turn)
    new_status["sleep_seconds"] = float(max(0.0, sleep_seconds))
    new_status["task_done"] = bool(task_done)
    return new_status


def plan_turn(frame_rgb: np.ndarray, memory: dict, status: dict):
    img_full = Image.fromarray(frame_rgb)

    buf_full = BytesIO()
    img_full.save(buf_full, format="PNG")
    image_bytes_full = buf_full.getvalue()

    img_low = img_full.copy()
    img_low.thumbnail((1280, 720))
    buf_low = BytesIO()
    img_low.save(buf_low, format="PNG")
    image_bytes_low = buf_low.getvalue()

    thumb_w, thumb_h = img_low.size

    memory_text = json.dumps(memory or {}, ensure_ascii=False)
    status = dict(status or {})
    status["screen"] = {"width": int(thumb_w), "height": int(thumb_h)}
    status_text = json.dumps(status, ensure_ascii=False)
    prompt = f"""
You are Duely, a desktop control agent. You control the user's real desktop with mouse and keyboard.

You work in discrete TURNS. Each turn you:
- Look at the screenshot image.
- Read MEMORY_JSON (your own past plan state).
- Read STATUS_JSON (environment signals).
- Read TASK_PROMPT (the overall goal).
- Decide what to do THIS TURN only and output JSON.

MEMORY_JSON (read-only):
{memory_text}

STATUS_JSON (read-only):
{status_text}

TASK_PROMPT:
{TASK_PROMPT}

Coordinate system:
- (0,0) is the top-left of the full desktop image.
- All x,y are desktop pixels with 0 <= x < screen.width and 0 <= y < screen.height.

Allowed actions:
- move_mouse: {{"action":"move_mouse","x":int,"y":int,"duration":float?}}
- click:      {{"action":"click","button":"left"|"right"|"middle","clicks":int?,"interval":float?}}
- type:       {{"action":"type","text":string,"interval":float?}}
- key:        {{"action":"key","key":string}}
- key_combo:  {{"action":"key_combo","keys":[string,...]}}
- scroll:     {{"action":"scroll","clicks":int,"x":int?,"y":int?}}
- drag_mouse: {{"action":"drag_mouse","x":int,"y":int,"duration":float?,"button":"left"|"right"|"middle"}}
- sleep:      {{"action":"sleep","seconds":float}}

Todo list (your internal plan):
- Use todo items for meaningful subgoals toward TASK_PROMPT.
- Task status is one of: "todo" → "in_progress" → "done".
- At the start of a turn, read MEMORY_JSON.todo to see which task (if any) is already "in_progress".
- When you decide to work on a todo task this turn, keep it as "todo" in MEMORY_JSON, but in your NEW todo list you return, set that task to "in_progress".
- Only mark a task "done" on a later turn, after it was already "in_progress" in MEMORY_JSON and you can now see that your previous actions visibly completed it.
- Never change a task directly from "todo" to "done" in a single turn.
- If the screen already looks complete at the start of a run, treat that as stale state for this run, not work you just performed.
- Always return the full todo list each turn.

Notes (scratchpad memory):
- Use the "notes" field as a JSON object for any additional state that helps you reason across turns.
- Examples: retry counters, flags like {{"saw_old_text": true}}, cached text, last window title, or whether you are waiting for a specific dialog.
- Prefer small, factual keys and values over long prose.
- You may add or update keys in "notes" as your understanding of the situation evolves.
- Avoid deleting useful keys unless you intentionally want to clear them.
- Send nothing in notes if no new notes are to be written
- Try to reformat any mistake in notes to dictionary format

Pacing and stability:
- expect_change=true if your actions should visibly change the screen by next turn.
- expect_change=false if you expect the screen to stay mostly the same.
- expect_change=null if unsure.
- If STATUS_JSON.expected_change_but_no_change is true, assume the UI did not respond; adjust your plan.
- If STATUS_JSON.unexpected_change is true, re-evaluate where you are before acting.
- Use sleep_seconds to request extra cooldown time between turns when needed.
- Use skip_turn=true when you want the next turn to observe only.

Return ONLY valid JSON like this:
{{
  "action_summary": "short description of what you will do this turn",
  "expect_change": true | false | null,
  "needs_high_res": true | false,
  "skip_turn": true | false,
  "sleep_seconds": number,
  "task_done": true | false,
  "todo": [
    {{"task": "string", "status": "todo" | "in_progress" | "done"}}
  ],
  "steps": [ {{"action": "..."}}, {{"action": "..."}} ]
}}

No explanations. JSON only.
"""

    llm = os.environ.get("DUELY_LLM", "").lower()

    if llm.startswith("gemini"):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set; cannot call Gemini")
        client = genai.Client(api_key=api_key)
        image_part_low = types.Part.from_bytes(data=image_bytes_low, mime_type="image/png")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image_part_low],
        )
        raw = response.text or ""
    else:
        raise RuntimeError("No supported LLM configured (expected DUELY_LLM to start with 'gemini')")

    (action_summary, steps, todo, notes, expect_change, needs_high_res,
     skip_turn, sleep_seconds, task_done) = parse_llm_response(raw)

    if needs_high_res:
        raw_hi = request_high_res_plan(prompt, image_bytes_full)
        if raw_hi:
            (action_summary, steps, todo, notes, expect_change, needs_high_res2,
             skip_turn, sleep_seconds, task_done) = parse_llm_response(raw_hi)
            needs_high_res = needs_high_res2

    new_memory = update_memory(memory, action_summary, todo, notes)
    new_status = update_status(status, expect_change, needs_high_res,
                               skip_turn, sleep_seconds, task_done)

    return action_summary, steps, new_memory, new_status, raw


# ---------------- Stability-based capture between turns ----------------

def wait_for_stable_frame(gate: StabilityGate,
                          last_turn_hash: Optional[imagehash.ImageHash],
                          expect_change_hint):
    timeout = float(SETTINGS.get("stability_timeout", 4.0))
    phash_threshold = int(SETTINGS.get("phash_threshold", 10))
    target_dt = 1.0 / max(1, int(SETTINGS.get("capture_fps", 2)))

    start = time.time()
    chosen_frame: Optional[np.ndarray] = None
    chosen_ts: float = 0.0
    chosen_hash: Optional[imagehash.ImageHash] = None

    while True:
        frame, ts = grab_frame()
        is_stable, _, current_hash = gate.update(frame, ts)

        chosen_frame, chosen_ts, chosen_hash = frame, ts, current_hash

        now = time.time()
        if is_stable or (now - start) >= timeout:
            break
        time.sleep(target_dt)

    changed_vs_last_turn = False
    if chosen_hash is not None and last_turn_hash is not None:
        try:
            changed_vs_last_turn = (chosen_hash - last_turn_hash) > phash_threshold
        except Exception:
            changed_vs_last_turn = False

    expected_change_but_no_change = False
    unexpected_change = False

    if expect_change_hint is True:
        if not changed_vs_last_turn:
            expected_change_but_no_change = True
    elif expect_change_hint is False:
        if changed_vs_last_turn:
            unexpected_change = True

    return chosen_frame, chosen_ts, chosen_hash, expected_change_but_no_change, unexpected_change


# ---------------- Action execution ----------------

def clamp_point(x, y):
    """Clamp a point to the current primary screen bounds.

    This prevents the agent from sending the mouse off-screen, regardless of
    the user's monitor or laptop resolution.
    """
    screen_w, screen_h = pyautogui.size()
    x = max(0, min(screen_w - 1, int(x)))
    y = max(0, min(screen_h - 1, int(y)))
    return x, y


def move_mouse(x, y, duration=0.25):  # smoother glide default
    x, y = clamp_point(x, y)
    pyautogui.moveTo(x, y, duration=duration)
def click(button="left", clicks=1, interval=0.05):
    pyautogui.click(button=button, clicks=clicks, interval=interval)


def type_text(text, interval=0.02):
    pyautogui.write(text, interval=interval)


def press_key(key):
    pyautogui.press(key)


def key_combo(keys):
    pyautogui.hotkey(*keys)


def scroll(clicks, x=None, y=None):
    if x is not None and y is not None:
        x, y = clamp_point(x, y)
        pyautogui.scroll(clicks, x=x, y=y)
    else:
        pyautogui.scroll(clicks)


def drag_mouse(x, y, duration=0.2, button="left"):
    x, y = clamp_point(x, y)
    pyautogui.dragTo(x, y, duration=duration, button=button)


def user_requested_stop() -> bool:
    return STOP_REQUESTED


def execute_step(step):
    if user_requested_stop():
        raise KeyboardInterrupt("Emergency stop requested")

    action = step.get("action")

    if action == "move_mouse":
        move_mouse(step["x"], step["y"], step.get("duration", 0.25))
        return

    if action == "click":
        click(
            button=step.get("button", "left"),
            clicks=step.get("clicks", 1),
            interval=step.get("interval", 0.05)
        )
        return

    if action == "type":
        type_text(step.get("text", ""), step.get("interval", 0.02))
        return

    if action == "key":
        press_key(step["key"])
        return

    if action == "key_combo":
        key_combo(step["keys"])
        return

    if action == "scroll":
        scroll(
            step.get("clicks", -500),
            step.get("x"),
            step.get("y"),
        )
        return

    if action == "drag_mouse":
        drag_mouse(
            step["x"],
            step["y"],
            step.get("duration", 0.2),
            step.get("button", "left"),
        )
        return

    if action == "sleep":
        time.sleep(step.get("seconds", 0.2))
        return

    print("Unknown action:", action)


# ---------------- Main loop ----------------


def _suspicious_todo_jump(prev_memory: dict, new_memory: dict) -> bool:
    try:
        old_todo = (prev_memory or {}).get("todo") or []
        new_todo = (new_memory or {}).get("todo") or []
        old_map = {}
        for item in old_todo:
            if isinstance(item, dict):
                task = item.get("task")
                status = item.get("status")
                if isinstance(task, str) and isinstance(status, str):
                    old_map[task] = status
        for item in new_todo:
            if not isinstance(item, dict):
                continue
            task = item.get("task")
            status = item.get("status")
            if not isinstance(task, str) or not isinstance(status, str):
                continue
            if status == "done" and old_map.get(task) == "todo":
                return True
    except Exception:
        pass
    return False


def main():
    _install_hotkeys()

    memory = {
        "turn": 0,
        "last_action": None,
        "todo": [],
        "notes": {},
    }
    status: dict = {}

    gate = StabilityGate(
        phash_threshold=SETTINGS.get("phash_threshold", 10),
        stable_seconds=SETTINGS.get("stable_seconds", 0.8),
    )

    last_turn_hash: Optional[imagehash.ImageHash] = None
    screen_ref_hash: Optional[imagehash.ImageHash] = None
    screen_ref_ts: Optional[float] = None
    screen_age = 0.0

    print("[agent] Duely multi-turn planner MVP.")

    turn_idx = 0
    while True:

        turn_idx += 1
        if user_requested_stop():
            print("[agent] Stop requested, exiting loop.")
            break

        expect_hint = status.get("expect_change", None)

        if expect_hint is None:

            frame, ts = grab_frame()
            try:
                current_hash = imagehash.phash(Image.fromarray(frame))
            except Exception:
                current_hash = None
            expected_change_but_no_change = False
            unexpected_change = False
        else:

            frame, ts, current_hash, expected_change_but_no_change, unexpected_change = (
                wait_for_stable_frame(gate, last_turn_hash, expect_hint)
            )

        status["expected_change_but_no_change"] = bool(expected_change_but_no_change)
        status["unexpected_change"] = bool(unexpected_change)

        if current_hash is not None:
            print("hello this is loop", turn_idx)
            last_turn_hash = current_hash
            try:
                if screen_ref_hash is None:
                    screen_ref_hash = current_hash
                    screen_ref_ts = ts
                    screen_age = 0.0
                else:
                    dist = current_hash - screen_ref_hash
                    if dist <= SETTINGS.get("phash_threshold", 10):
                        if screen_ref_ts is not None:
                            screen_age = ts - screen_ref_ts
                    else:
                        screen_ref_hash = current_hash
                        screen_ref_ts = ts
                        screen_age = 0.0
            except Exception:
                pass


        status["screen_age"] = float(screen_age)

        if SETTINGS.get("debug"):
            print(f"[agent] Turn {turn_idx}: ts={ts}, shape={frame.shape}, screen_age={screen_age:.2f}s")

        action_summary, steps, new_memory, new_status, raw = plan_turn(frame, memory, status)

        # Guard against illegal todo transitions (todo -> done in one step).
        if _suspicious_todo_jump(memory, new_memory):
            print("[agent] Suspicious todo->done jump detected; skipping this turn's plan.")
            # Do not update memory or status, do not execute steps; just sleep and retry.
            base_sleep = 1.0 / max(1, SETTINGS.get("capture_fps", 2))
            extra_sleep = float(status.get("sleep_seconds", 0.0) or 0.0)
            time.sleep(base_sleep + extra_sleep)
            continue

        # (removed RAW LLM RESPONSE printing)
        # print("===== RAW LLM RESPONSE") =====")
        # print(raw)
        print("===== PARSED PLAN =====")
        print("action_summary:", action_summary)
        print("steps:", steps)
        print("new_memory:", new_memory)
        print("new_status:", new_status)
        print("task_done:", new_status.get("task_done"))
        print("==============================")

        memory, status = new_memory, new_status

        # If the planner declares the high-level task finished, exit cleanly.
        if bool(status.get("task_done")):
            print("[agent] task_done=True from planner — exiting main loop.")
            break

        skip_turn = bool(status.get("skip_turn", False))
        extra_sleep = float(status.get("sleep_seconds", 0.0) or 0.0)
        base_sleep = 1.0 / max(1, SETTINGS.get("capture_fps", 2))

        if skip_turn:
            if SETTINGS.get("debug"):
                print(f"[agent] Turn {turn_idx}: skip_turn=True, sleeping {base_sleep + extra_sleep:.2f}s")
            time.sleep(base_sleep + extra_sleep)
            continue

        # Scale coordinates from model thumbnail space to full screen space
        scale_x = scale_y = 1.0
        screen_info = status.get("screen")
        if isinstance(screen_info, dict):
            try:
                thumb_w = int(screen_info.get("width") or 0)
                thumb_h = int(screen_info.get("height") or 0)
                if thumb_w > 0 and thumb_h > 0:
                    full_w, full_h = pyautogui.size()
                    scale_x = full_w / thumb_w
                    scale_y = full_h / thumb_h
            except Exception:
                pass

        for step in steps:
            if "x" in step and "y" in step:
                try:
                    step["x"] = int(step["x"] * scale_x)
                    step["y"] = int(step["y"] * scale_y)
                except Exception:
                    pass
            execute_step(step)

        if extra_sleep > 0:
            time.sleep(extra_sleep)
        time.sleep(base_sleep)


if __name__ == "__main__":
    print("STARTED canvas_5.py with:", sys.executable)
    time.sleep(0.2)
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        input("[agent] Crashed. Press Enter to close…")
