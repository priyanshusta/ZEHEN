r"""ZEHEN: control your PC with hand gestures. Camera runs hidden; a see-through HUD shows feedback.

Install:  python -m pip install -r requirements.txt
Run:      python zehen.py            (finds your real camera; python zehen.py 1 forces camera #1)
Test:     python zehen.py --test
Build:    double-click build.bat  ->  dist\ZEHEN.exe  (needs ~3 GB free disk space while building)
Quit:     the X on the guide panel. Touch the real mouse/touchpad and ZEHEN lets go of the cursor.
Settings: %USERPROFILE%\.zehen.json  ("main_hand": "Left" for left-handers, "apps": [names for the launcher])
Voice:    say "Zehen, open Spotify". Free cloud key (faster): panel > Get key > Paste key, or env ZEHEN_API_KEY.
          No key: a small offline model downloads on first use (needs ~1.6 GB free disk). "voice": false turns it off
"""
import ctypes
import difflib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
import webbrowser
import zipfile
from pathlib import Path
from tkinter import messagebox
from types import SimpleNamespace

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")  # else camera open takes ~20 s on Windows
import cv2
import numpy as np
import mediapipe as mp
import pyautogui  # import before tk: it makes the process DPI-aware so tk and pyautogui agree on pixels
from mediapipe.tasks.python import BaseOptions, vision

# --- tuning knobs: every camera, hand and screen is different ---
MARGIN = 0.18                      # frame edge ignored when mapping to screen (smaller = more hand travel)
MIN_CUTOFF, BETA = 1.0, 0.007      # One Euro filter: lower MIN_CUTOFF = less jitter, higher BETA = less lag
PINCH_ON, PINCH_OFF = 0.33, 0.48   # thumb-finger gap / hand size; the gap between them stops flicker
FIST_GUARD = 0.5                   # index tip must be this far from its knuckle to start a pinch (fists don't click)
STABLE_FRAMES = 3                  # a gesture must hold this many frames before it counts (~100 ms)
CLICK_LOCK_S = 0.15                # cursor freezes this long when a pinch starts, so clicks land where aimed
HOLD_S = 0.8                       # seconds to hold palm / three fingers before they fire
TUTORIAL_SPEECH = [
    "Let's begin. Raise your index finger and keep the rest of your hand relaxed. Move slowly, and the cursor follows your knuckle.",
    "Now pinch your thumb and index finger together, quickly, as if picking up a grain of rice. That's a click.",
    "Pinch twice, fast. That's a double click. It opens apps, files and folders.",
    "Pinch, hold, and move your hand. Whatever you grabbed comes along. That's a drag.",
    "Two fingers up, like a peace sign. Raise your hand to scroll up, lower it to scroll down.",
    "Index finger up, then touch your thumb to your middle finger. That's a right click.",
    "Use both hands. Pinch with each, then spread them apart to zoom in, or bring them together to zoom out.",
    "With your other hand, wave sideways, open palm, quickly. That jumps to your last app.",
    "Open your palm and hold it still for a moment. The launcher appears.",
    "Show three fingers and hold. The writing pad opens. Pinch and write in the air, and I'll type it for you.",
    "Now let me learn your voice, so I wake only for you. Say Zehen, three times, with a short pause between each.",
    "Now the fun part. Say Zehen, then ask me something. Try: open Notepad.",
]
TUTORIAL_END = "That's everything. Point at the little pill to open this panel again. Say Zehen whenever you need me."
PRAISE = ["Nice.", "Perfect.", "Smooth.", "There you go.", "Well done."]
DWELL_S, CLOSE_S = 0.6, 5.0        # side panel: point at the pill this long to open it; folds after this long unused
SCROLL_SPEED = 400 if sys.platform == "win32" else 10  # Windows wheel units are 120 per notch
SCROLL_DEADZONE = 0.03
LIVE_MIN_CHANGE = 0.3              # avg frame-to-frame pixel change for a "live" camera (real ~2, placeholder 0)
ZOOM_STEP = 0.12                   # two-hand pinch: hands must spread/close this fraction for one zoom notch
SWIPE_DIST, SWIPE_S, SWIPE_COOLDOWN = 0.18, 0.3, 0.9  # other-hand swipe: frame fraction travelled within SWIPE_S
INK_PAUSE_S = 0.9                  # writing pad: pause this long after a stroke and the ink turns into text
INK_MIN_CUTOFF, INK_BETA = 0.5, 0.004  # pen smoothing on the pad: steadier than the cursor (lower = smoother)
INK_HOOK = 0.15                    # look this far into each end of a stroke for the hook a pinch leaves
INK_STRAIGHT = 0.08                # strays less than this x its length from a line -> drawn perfectly straight
INK_ROUND = 0.16                   # closed loop whose radius varies less than this -> drawn as a clean oval
TAP_S, DOUBLE_TAP_S = 0.4, 0.7      # pinch shorter than TAP_S is a click; 2nd pinch within DOUBLE_TAP_S = open
MOUSE_TAKEOVER_PX, MOUSE_TAKEOVER_S = 25, 1.5  # real mouse moved this far -> hands off the cursor this long

BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))  # PyInstaller unpacks bundled files here
MODEL = BASE / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
CONFIG = Path.home() / ".zehen.json"
CAMERA_HELP = ("Can't reach the camera.\n\n"
               "•  Close other apps using it (Zoom, Teams, Camera app).\n"
               "•  Windows Settings › Privacy & security › Camera: turn on 'Camera access' "
               "and 'Let desktop apps access your camera'.\n"
               "•  Some laptops have a camera shutter or an F-key that turns the camera off.\n"
               "•  Virtual cameras (PRISM, OBS) are skipped; ZEHEN needs a real one plugged in.")
CONSENT = ("ZEHEN uses your camera to see your hand, so you can control your computer with gestures.\n\n"
           "•  The video never appears on screen.\n"
           "•  The video is never recorded, saved or sent anywhere: it stays on this computer.\n"
           "•  It also listens for the word \"Zehen\" through the microphone, on this computer. Only after you say "
           "it, what you ask is sent to the free AI service you chose (or answered offline if you chose none).\n"
           "•  Pause or quit any time from the ZEHEN panel.")

POSES = {(1, 0, 0, 0): "point", (1, 1, 0, 0): "scroll", (1, 1, 1, 0): "three",
         (1, 1, 1, 1): "palm", (0, 0, 0, 0): "fist"}
GUIDE = [("point", "☝", "Index finger up", "Move cursor"),
         ("pinch", "🤏", "Pinch thumb + index", "Click · hold to drag · tap tiles"),
         ("double", "🤏", "Pinch twice quickly", "Open apps, files, folders"),
         ("rpinch", "👌", "Index up, thumb→middle", "Right-click"),
         ("scroll", "✌", "Two fingers up", "Scroll ↕"),
         ("palm", "🖐", "Open palm, hold", "Launcher: apps, windows · closes panels"),
         ("three", "✍", "Three fingers, hold", "Writing pad"),
         ("zoom", "🙌", "Both hands pinch, spread", "Zoom in / out"),
         ("swipe", "👋", "Swipe your other hand", "Switch app"),
         ("fist", "✊", "Fist", "Pause")]
LABELS = {"point": "MOVE", "pinch": "GRAB", "rpinch": "RIGHT CLICK", "scroll": "SCROLL", "three": "WRITING PAD",
          "palm": "LAUNCHER", "fist": "PAUSED", "zoom": "ZOOM", "swipe": "SWITCH", "double": "OPEN"}
# (active gesture, seconds held, instruction, hint)
TUTORIAL = [("point", 1.0, "Raise your index finger ☝", "Move your hand: the cursor follows your knuckle."),
            ("pinch", 0.05, "Pinch thumb + index 🤏", "A quick pinch is a click."),
            ("double", 0.0, "Pinch twice quickly 🤏🤏", "That's a double-click: opens apps, files and folders."),
            ("pinch", 0.8, "Pinch, hold and move", "That's drag: grab a file or window and carry it."),
            ("scroll", 1.0, "Two fingers up ✌", "Raise your hand to scroll up, lower it to scroll down."),
            ("rpinch", 0.05, "Index up, touch thumb to middle", "That's a right-click."),
            ("zoom", 0.5, "Both hands: pinch, then spread 🙌", "Spread to zoom in, bring together to zoom out."),
            ("swipe", 0.0, "Swipe your other hand sideways 👋", "Open palm, quick wave: switches to your last app."),
            ("palm", HOLD_S, "Open palm 🖐, hold", "The launcher: your apps, open windows and controls."),
            ("three", HOLD_S, "Three fingers up, hold ✍", "The writing pad: pinch and write, it types for you."),
            ("enroll", 0.0, "Teach me your voice 🎙", "Say “Zehen” three times, with a short pause between."),
            ("voice", 0.0, "Say “Zehen” 🎙", "Then ask: “open Notepad”. It does it, then answers out loud.")]
KEY_ROWS = ["1234567890", "QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM",
            ["ESC", "TAB", "CAPS", "SPACE", "BACK", "ENTER", "WIN", "✕"]]
SPECIAL = {"SPACE": "space", "BACK": "backspace", "ENTER": "enter", "ESC": "esc", "TAB": "tab",
           "CAPS": "capslock", "WIN": "win" if sys.platform == "win32" else "command"}
DEFAULT_APPS = ["Google Chrome", "Microsoft Edge", "File Explorer", "WhatsApp", "Spotify", "Settings",
                "Calculator", "Notepad", "Visual Studio Code", "Microsoft Store", "Paint", "Photos"]
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
KEY = "#000102"   # overlay colour that Windows makes see-through; near-black so text edges look like shadow
CYAN, DIM, DARK, GREEN, AMBER, BG, TILE, TEXT = ("#00e5ff", "#007a8a", "#003a44", "#3dff9a", "#ffb340",
                                                 "#08121a", "#0d1f2a", "#cfeff4")
CHIP, HAIR, SUB = "#0b222c", "#1d6674", "#7cc4cf"  # side panel: solid chips on the glass, hairlines, secondary text
COLORS = {"cyan": CYAN, "dim": DIM, "green": GREEN, "amber": AMBER, "white": TEXT}

pyautogui.PAUSE = 0  # default 0.1 s sleep per call would cap tracking at ~10 fps
pyautogui.FAILSAFE = False  # its corner stop fired whenever you used the touchpad near a corner;
                            # ZEHEN yields to the real mouse instead (see _mouse_taken)


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)


def fingers_up(lm):
    # index, middle, ring, pinky. Tip further from the wrist than its middle joint = extended.
    # Works with a tilted hand, unlike comparing y. Thumb is too unreliable to count.
    return tuple(int(dist(lm[0], lm[t]) > dist(lm[0], lm[t - 2]) * 1.1) for t in (8, 12, 16, 20))


def classify(lm, was_pinched):
    """Gesture name for one frame. Distances are divided by hand size so it works near or far."""
    size = dist(lm[0], lm[9]) or 1e-6
    d = dist(lm[4], lm[8]) / size
    if d < PINCH_OFF if was_pinched else (d < PINCH_ON and dist(lm[8], lm[5]) / size > FIST_GUARD):
        return "pinch"
    up = fingers_up(lm)
    if up[0] and dist(lm[4], lm[12]) / size < PINCH_ON:
        return "rpinch"
    return POSES.get(up, "?")


class OneEuro:
    """One Euro filter (Casiez 2012): heavy smoothing when the hand is still, little lag when it moves fast."""

    def __init__(self, min_cutoff=MIN_CUTOFF, beta=BETA):
        self.t = self.x = None
        self.dx = 0.0
        self.min_cutoff, self.beta = min_cutoff, beta

    def __call__(self, x, t):
        if self.t is None:
            self.t, self.x = t, x
            return x
        dt, self.t = max(t - self.t, 1e-6), t
        alpha = lambda cutoff: 1 / (1 + 1 / (2 * math.pi * cutoff * dt))
        self.dx += alpha(1.0) * ((x - self.x) / dt - self.dx)
        self.x += alpha(self.min_cutoff + self.beta * abs(self.dx)) * (x - self.x)
        return self.x


def unhook(P, window=3):
    """Cut the hook off the start of a stroke: closing a pinch jerks the pen, leaving a short tail that
    meets the real stroke at a sharp angle. Finds the sharpest turn in the first INK_HOOK of the path
    and starts there; a stroke that starts smoothly is left alone."""
    total = sum(math.dist(a, b) for a, b in zip(P, P[1:]))
    best, best_turn, run = 0, math.radians(80), 0.0  # a pinch hook turns hard: sideways or back
    for k in range(1, len(P) - 1):  # from the very first point: a quick flick may be just 1-2 points long
        run += math.dist(P[k - 1], P[k])
        if run > INK_HOOK * total:
            break
        # direction in over the hook, direction out over the next few points (one point alone is noise)
        (ax, ay), (bx, by), (cx, cy) = P[max(0, k - window)], P[k], P[min(len(P) - 1, k + window)]
        u, v = (bx - ax, by - ay), (cx - bx, cy - by)
        if any(u) and any(v):
            turn = math.acos(max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (math.hypot(*u) * math.hypot(*v)))))
            if turn > best_turn:
                best, best_turn = k, turn
    return P[best:]


def beautify(points, aspect=16 / 9):
    """Clean one air-written stroke: trim the hooks left by closing/opening the pinch, smooth out hand
    tremor, and snap near-straight lines perfectly straight and near-round loops into clean ovals.
    Points are screen fractions; aspect (width / height) keeps the geometry in true proportions."""
    if len(points) < 5:
        return points  # a tap: the dot on an i
    P = [(x * aspect, y) for x, y in points]
    seg = [math.dist(a, b) for a, b in zip(P, P[1:])]
    total = sum(seg)
    if not total:
        return points[:1]
    P = unhook(unhook(P)[::-1])[::-1]  # both ends
    mean = lambda pts: (sum(q[0] for q in pts) / len(pts), sum(q[1] for q in pts) / len(pts))
    P = [P[0]] + [mean(P[max(0, n - 2):n + 3]) for n in range(1, len(P) - 1)] + [P[-1]]  # tremor
    for _ in range(2):  # Chaikin corner cutting: rounds what's left of the jitter into a curve
        P = [P[0]] + [q for a, b in zip(P, P[1:]) for q in ((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]),
                                                           (0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))] + [P[-1]]
    a, b = P[0], P[-1]
    chord, length = math.dist(a, b), sum(math.dist(p, q) for p, q in zip(P, P[1:]))
    back = lambda pts: [(x / aspect, y) for x, y in pts]
    if chord and max(abs((b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1])) for p in P) / chord \
            < INK_STRAIGHT * chord:
        return back([(a[0] + (b[0] - a[0]) * t / 24, a[1] + (b[1] - a[1]) * t / 24) for t in range(25)])
    if chord < 0.2 * length:  # it closes on itself: is it an oval?
        xs, ys = [p[0] for p in P], [p[1] for p in P]
        cx, cy, rx, ry = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2
        if rx and ry:
            r = [math.hypot((p[0] - cx) / rx, (p[1] - cy) / ry) for p in P]
            avg = sum(r) / len(r)
            if (sum((v - avg) ** 2 for v in r) / len(r)) ** 0.5 / avg < INK_ROUND:
                start = math.atan2((P[0][1] - cy) / ry, (P[0][0] - cx) / rx)
                turn = 1 if sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(P, P[1:] + P[:1])) > 0 else -1
                return back([(cx + rx * math.cos(start + turn * math.tau * t / 48),
                              cy + ry * math.sin(start + turn * math.tau * t / 48)) for t in range(49)])
    return back(P)


def key_rects(x0=0.16, y0=0.56, x1=0.84, y1=0.94):
    """Keyboard layout in screen fractions: [(label, left, top, right, bottom)]."""
    rh, rects = (y1 - y0) / len(KEY_ROWS), []
    for r, row in enumerate(KEY_ROWS):
        kw = (x1 - x0) / len(row)
        rects += [(k, x0 + i * kw, y0 + r * rh, x0 + (i + 1) * kw, y0 + (r + 1) * rh) for i, k in enumerate(row)]
    return rects


KEYS = key_rects()


def inside(r, x, y):
    return r[0] <= x < r[2] and r[1] <= y < r[3]


def btn(id, label, x0, y0, x1, y1, action, size=14):
    return SimpleNamespace(id=id, label=label, r=(x0, y0, x1, y1), action=action, size=size)


def panel(boxes=(), texts=(), buttons=(), pad=None):
    """What the HUD draws for the open panel. All positions are screen fractions."""
    return SimpleNamespace(boxes=list(boxes), texts=list(texts), buttons=list(buttons), pad=pad)


def open_folder(path=Path.home()):
    if sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


# --- Windows: apps, windows, handwriting ---

if sys.platform == "win32":
    from ctypes import wintypes
    u32, dwm = ctypes.WinDLL("user32"), ctypes.WinDLL("dwmapi")  # own handles: argtypes here don't leak elsewhere
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    for name, args in (("EnumWindows", [WNDENUMPROC, wintypes.LPARAM]), ("IsWindowVisible", [wintypes.HWND]),
                       ("GetWindow", [wintypes.HWND, wintypes.UINT]), ("GetWindowTextLengthW", [wintypes.HWND]),
                       ("GetWindowTextW", [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]),
                       ("GetClassNameW", [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]),
                       ("GetWindowLongW", [wintypes.HWND, ctypes.c_int]), ("IsIconic", [wintypes.HWND]),
                       ("ShowWindow", [wintypes.HWND, ctypes.c_int]),
                       ("SwitchToThisWindow", [wintypes.HWND, wintypes.BOOL]),
                       ("PostMessageW", [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM])):
        getattr(u32, name).argtypes = args
    u32.GetWindow.restype = u32.GetForegroundWindow.restype = wintypes.HWND
    dwm.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]


def window_text(h, fn="GetWindowTextW"):
    buf = ctypes.create_unicode_buffer(256)
    getattr(u32, fn)(h, buf, 256)
    return buf.value


def list_windows():
    """App windows the taskbar would show: [(hwnd, title)], most recently used first."""
    if sys.platform != "win32":
        return []
    out = []

    def check(h, _):
        if (u32.IsWindowVisible(h) and not u32.GetWindow(h, 4) and u32.GetWindowTextLengthW(h)  # 4 = GW_OWNER
                and not u32.GetWindowLongW(h, -20) & 0x80):                                     # TOOLWINDOW
            cloaked = ctypes.c_int()  # hidden UWP / other-desktop windows are "cloaked"
            dwm.DwmGetWindowAttribute(h, 14, ctypes.byref(cloaked), 4)
            title = window_text(h)
            if not cloaked.value and title not in ("Program Manager", "ZEHEN"):
                out.append((h, title))
        return True

    u32.EnumWindows(WNDENUMPROC(check), 0)
    return out


def window_label(title):
    for sep in (" - ", " — ", " – "):  # Chrome/Edge use " - "; Firefox/Notion use " — "
        if sep in title:
            doc, _, app = title.rpartition(sep)
            return f"{app[:22]}\n{doc[:26]}".strip()
    return title[:22].strip()


def focus_window(h):
    if u32.IsIconic(h):
        u32.ShowWindow(h, 9)  # SW_RESTORE
    u32.SwitchToThisWindow(h, True)  # unlike SetForegroundWindow, works from a background app


def window_cmd(cmd):
    """Maximize / minimize / close the window you were using (the HUD never takes focus, so it's foreground)."""
    h = u32.GetForegroundWindow()
    if not h or window_text(h, "GetClassNameW") in ("Progman", "WorkerW", "Shell_TrayWnd"):
        return  # desktop or taskbar: closing it would open the shutdown dialog
    if cmd == "close":
        u32.PostMessageW(h, 0x0010, 0, 0)  # WM_CLOSE: the app can still ask to save
    else:
        u32.ShowWindow(h, 3 if cmd == "max" else 6)


def installed_apps():
    """Everything in the Start menu (desktop + Store apps): [(name, app_id)]."""
    if sys.platform != "win32":
        return []
    ps = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-StartApps | ConvertTo-Json -Compress"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, encoding="utf-8",
                         timeout=60, creationflags=0x08000000).stdout  # CREATE_NO_WINDOW: no console flash
    apps = json.loads(out or "[]")
    apps = [apps] if isinstance(apps, dict) else apps
    return [(a["Name"].strip(), a["AppID"]) for a in apps if not a["AppID"].startswith("http")]  # skip web links


def launch(app_id):
    os.startfile("shell:AppsFolder\\" + app_id)


def fold(s):
    """Lowercase with accents stripped ('Paramètres' -> 'parametres', 'Caméra' -> 'camera')."""
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower().strip()) if not unicodedata.combining(c))


def search_apps(apps, query, n=5):
    """Best app matches for handwritten text. Fuzzy and accent-insensitive: 'spotlfy' -> Spotify,
    'parametres' -> Paramètres."""
    q = fold(query)
    if not q:
        return []
    score = lambda name: (2 if (f := fold(name)).startswith(q) else 1.5 if q in f
                          else difflib.SequenceMatcher(None, q, f[:len(q) + 2]).ratio())
    ranked = sorted(apps, key=lambda a: -score(a[0]))
    return [a for a in ranked if score(a[0]) >= 0.75][:n]


def recognize(strokes, w, h, timeout=5.0):
    """Windows' built-in handwriting recognition (offline). strokes: lists of (x, y) screen fractions.
    Returns candidates, best first: the whole text, then alternatives when it's a single word.
    Polls the result instead of `await`ing it: asyncio's WinRT callback randomly never fired (about half
    the time), which froze the pad. Call it on a worker thread; it sleeps while Windows works."""
    from winrt.windows.foundation import AsyncStatus, Point
    from winrt.windows.ui.input.inking import (InkRecognitionTarget, InkRecognizerContainer,
                                               InkStrokeBuilder, InkStrokeContainer)
    builder, box = InkStrokeBuilder(), InkStrokeContainer()
    for s in strokes:
        s = s if len(s) > 1 else s + [(s[0][0] + 0.002, s[0][1])]  # a tap is a dot (the i in "hi")
        box.add_stroke(builder.create_stroke([Point(x * w, y * h) for x, y in s]))
    recognizer = InkRecognizerContainer()  # keep a reference until the result is in
    op = recognizer.recognize_async(box, InkRecognitionTarget.ALL)
    give_up = time.monotonic() + timeout
    while op.status == AsyncStatus.STARTED:
        if time.monotonic() > give_up:
            op.cancel()
            raise RuntimeError("handwriting took too long; try again")
        time.sleep(0.01)
    words = [c for c in (list(r.get_text_candidates()) for r in op.get_results()) if c]
    if not words:
        return []
    return [" ".join(c[0] for c in words)] + (words[0][1:4] if len(words) == 1 else [])


# --- Voice: say "Zehen, ...", a free LLM turns it into actions on your PC ---

DATA = Path.home() / ".zehen"  # models downloaded on first use live here
VOSK_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"  # 40 MB, Apache-2.0
VOSK_DIR = DATA / "vosk-model-small-en-us-0.15"
LOCAL_URL = "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf"  # 1.1 GB, Apache-2.0
LOCAL_FILE = DATA / "Qwen3-1.7B-Q4_K_M.gguf"
LLAMA_WHEELS = "https://abetlen.github.io/llama-cpp-python/whl/cpu"  # prebuilt, so no compiler is needed
# her voice: Kokoro-82M (Apache-2.0), a natural neural voice that runs offline; 120 MB, downloaded on first use
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_FILES = ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")
VOICE_NAME = "af_heart"  # other female voices in the pack: af_bella, af_nicole, af_sarah, af_sky, bf_emma, bf_isabella
# Vosk's small model has no word "zehen" (it hears "the him"), but it hears "Zehen" as "zane" every time and almost
# nothing else as "zane", so that is the wake grammar. Say it differently? Set "wake": ["your", "spellings"] in the config
WAKE_WORDS = ["zane"]
GREETINGS = ["Good {part}. Noor online, running on electricity and mild sarcasm. Say Zehen, and I'll handle the rest.",
             "Good {part}. Systems nominal, ego slightly inflated. Say Zehen whenever you need me.",
             "Good {part}. I'm Noor. I do the clicking, you take the credit. Just say Zehen.",
             "Good {part}. Noor reporting for duty. No coffee break required. Say Zehen.",
             "Online and unreasonably cheerful. I'm Noor. Say Zehen, and I'll make your computer behave.",
             "Good {part}. Noor here. Fully charged, vaguely overqualified. Say Zehen."]
WAKE_SPELLINGS = ["zehen", "zen", "zane", "zayn", "zain", "zahen", "sehen", "zayan", "sen", "zayed", "zhen"]
WAKE_STABLE = 5  # the wake word must be the phrase's first word for this many 0.1 s steps ("scroll" briefly sounds like it)
WAKE_LIKE = {"zane", "zen", "zain", "zayn", "the", "then", "this", "he", "hes", "him", "head", "hen", "sane", "said",
             "saint", "zayed", "hey", "a", "is"}  # what Vosk's full vocabulary writes for "Zehen": the second opinion
# a key's prefix says which free service it's from; any OpenAI-compatible endpoint also works via "llm" in the config.
# Free tiers change (Groq dropped Llama 3.3 from its free plan in 2026): override "model" in the config if one goes
PROVIDER_DEFAULTS = {"Groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
                     "Gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-flash-latest"),
                     "OpenRouter": ("https://openrouter.ai/api/v1", "openrouter/free"),
                     "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini")}
KEY_PREFIXES = [("gsk_", "Groq"), ("AIza", "Gemini"), ("sk-or-", "OpenRouter"), ("sk-", "OpenAI")]  # longest first
PROVIDER_NAMES = ["Auto-detect", *PROVIDER_DEFAULTS, "Custom"]  # Custom: any OpenAI-compatible service
KEY_PAGE = "https://console.groq.com/keys"
ASK_SYSTEM = """You are NOOR, the voice assistant that runs this Windows PC for its owner, and you also just talk with them.
Speak the way a warm, quick, dry-witted person does: natural sentences, contractions, no lists, no markdown, no emoji.
Do the job first, then (only sometimes, and only if it is natural) add a short light remark. Never use "sir" or "madam".
The request was transcribed from speech and may contain mishearings, wrong spellings or missing words: work out what
the user most likely meant, using the installed apps and open windows. If it is truly unclear, ask one short question.
Reply with ONE JSON object and nothing else:
{"say": "<what you say out loud: a 2 to 6 word confirmation, or a real answer of 1 or 2 sentences when asked a question>", "do": [ACTION, ...], "done": true}
Never reply with just "Done": say what you did ("Opened Spotify."). Never add filler like "let me know if you need anything else". Ask a question back only when you truly need the answer.
ACTION is an object with exactly one key:
{"open_app": "<app name>"}  {"focus_window": "<part of an open window's title>"}  {"open_url": "https://..."}
{"search_web": "<query>"}  {"type": "<text to type>"}  {"press": "<key or combo like ctrl+t or win+d>"}
{"window": "max"|"min"|"close"}  {"volume": "up"|"down"}  {"media": "playpause"|"nexttrack"|"prevtrack"}
{"open_folder": "desktop"|"downloads"|"documents"|"pictures"|"music"|"videos"}
{"play": "first"|"2"|"<words from a video's title>"} plays a video on the page, or finds it on YouTube
{"focus_input": "search"|"message"|"address"} puts the cursor in that box; {"dictate": "on"} then types what the user says
{"tab": "<words from a browser tab's title>"}  {"switch": "<words from a window or tab title>"}  switch tabs and windows
{"powershell": "<command>"}  runs hidden as the user and returns its output: use it for anything else on Windows
(files, folders, settings, processes, network, installs).
{"wait": "<seconds, max 5>"}  {"ui_list": ""} returns the names of what you can click in the window in front
{"click": "<name from ui_list>"} clicks it.  {"scroll": "down 6"|"up 6"|"top"|"bottom"}  {"zoom": "in"|"out"}
{"mouse": "click"|"double"|"right"} clicks wherever the pointer is now (the user points with their hand).
To work inside an app: open_app, wait, ui_list, click, type, press enter.
Set "done": false when you must read an output (powershell, ui_list) before the next step. At most 5 actions per step.
Chat, questions, advice, jokes and facts: answer in "say" and leave "do" empty. Act, don't ask permission."""
PLEASANT = {"thank you", "thanks", "thank you so much", "thanks a lot", "ok", "okay", "cool", "nice", "great", "good", "awesome",
            "perfect", "bye", "goodbye", "that's all", "no thanks", "fine", "alright", "got it", "hmm", "yes", "no",
            "yeah", "yep", "nope", "sure", "right", "oh", "wow", "hello", "hi", "hey"}
ACTIONS = {"open_app", "focus_window", "open_url", "search_web", "type", "press", "window", "volume", "media",
           "powershell", "wait", "ui_list", "click", "scroll", "mouse", "zoom", "open_folder", "tab", "switch", "play",
           "focus_input", "dictate"}
FOLDERS = {"desktop": "shell:Desktop", "downloads": "shell:Downloads", "documents": "shell:Personal",
           "pictures": "shell:My Pictures", "music": "shell:My Music", "videos": "shell:My Video"}
MAX_STEPS = 6
_local, _local_lock, _tts, _local_used = None, threading.Lock(), None, 0.0
_stt_lock, _tts_lock = threading.Lock(), threading.Lock()


def fetch(url, dest, unzip=False, progress=lambda done, size: None):
    """Download once into ~/.zehen. Refuses up front when the disk can't hold it."""
    DATA.mkdir(exist_ok=True)
    part = dest.with_suffix(f".{threading.get_ident()}.part")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ZEHEN"}), timeout=30) as r:
            size, free = int(r.headers.get("Content-Length", 0)), shutil.disk_usage(DATA).free
            if free < size + 5e8:
                raise RuntimeError(f"Needs {(size + 5e8) / 1e9:.1f} GB free on this drive, you have {free / 1e9:.1f} GB")
            with open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    progress(f.tell(), size)
        if unzip:
            with zipfile.ZipFile(part) as z:
                z.extractall(DATA)
        else:
            part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


def _unload_when_idle():
    """The offline brain is about 2 GB in memory: let it go after three quiet minutes, reload on the next request."""
    global _local
    while _local is not None:
        time.sleep(20)
        if time.monotonic() - _local_used > 60 and _local_lock.acquire(blocking=False):
            try:
                brain, _local = _local, None
                try:
                    brain.close()
                except Exception:
                    pass
                del brain
                import gc
                gc.collect()
                trim_memory()
            finally:
                _local_lock.release()


def local_llm(note=lambda s, frac=None: None, load=True):
    """The offline brain: Qwen3 1.7B through llama.cpp. Installed and downloaded on first use, free forever.
    load=False only makes sure it is installed and downloaded (first-run setup), without using the memory."""
    global _local, _local_used
    _local_used = time.monotonic()
    if _local is None:
        try:
            from llama_cpp import Llama
        except ImportError:
            if getattr(sys, "frozen", False):
                raise RuntimeError("The offline brain needs the Python install, not the .exe: paste a free key")
            note("Installing the offline brain…")
            if subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--only-binary=:all:",
                               "--extra-index-url", LLAMA_WHEELS], capture_output=True,
                              creationflags=0x08000000).returncode:
                raise RuntimeError("Couldn't install the offline brain: paste a free key instead")
            from llama_cpp import Llama
        if not LOCAL_FILE.exists():
            fetch(LOCAL_URL, LOCAL_FILE, progress=lambda d, n: note(
                f"Downloading brain {d / 1e6:.0f}/{n / 1e6:.0f} MB", d / n if n else None))
        if not load:
            return None
        note("Waking the offline brain…")
        _local = Llama(model_path=str(LOCAL_FILE), n_ctx=2048, n_batch=256, n_threads=max(2, (os.cpu_count() or 4) // 2),
                       verbose=False)  # 2048 tokens is plenty for one request, and half the memory of 4096
        threading.Thread(target=_unload_when_idle, daemon=True).start()
    return _local


_stt = None
WHISPER_URL = "https://huggingface.co/Systran/faster-whisper-%s/resolve/main/"
WHISPER_FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")


def stt_engine(cfg, note=lambda s, frac=None: None):
    """Whisper through faster-whisper, int8 on the CPU: about 0.4 s for a spoken command, 145 MB, downloaded once.
    It copes with accents, spells names properly and is told your app and window names, so "Jupyter" stays Jupyter."""
    global _stt, _stt_used
    with _stt_lock:
        _stt_used = time.monotonic()
        if _stt is None:
            name = cfg.get("stt_model", "base.en")  # tiny.en is quicker, small.en is sharper (480 MB)
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                if getattr(sys, "frozen", False):
                    raise RuntimeError("Speech recognition needs the Python install, not the .exe")
                note("Installing speech recognition…")
                if subprocess.run([sys.executable, "-m", "pip", "install", "faster-whisper"], capture_output=True,
                                  creationflags=0x08000000).returncode:
                    raise RuntimeError("Couldn't install speech recognition")
                from faster_whisper import WhisperModel
            folder = DATA / ("whisper-" + name)
            folder.mkdir(parents=True, exist_ok=True)
            for f in WHISPER_FILES:
                if not (folder / f).exists():
                    fetch(WHISPER_URL % name + f, folder / f, progress=lambda d, n: note(
                        f"Downloading hearing {d / 1e6:.0f}/{n / 1e6:.0f} MB", d / n if n else None))
            _stt = WhisperModel(str(folder), device="cpu", compute_type="int8", cpu_threads=min(4, os.cpu_count() or 2))
        return _stt


def sane_transcript(text, hints=(), other=""):
    """False for what Whisper writes when there is nothing to hear: its hints read back, one phrase on repeat, or a
    long sentence where the other ear heard a word or two."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    if not words:
        return False
    if len(words) >= 6 and len(set(words)) / len(words) < 0.5:
        return False
    hint_words = {w for h in hints for w in re.findall(r"[a-z0-9']+", h.lower())}
    if len(words) >= 3 and sum(w in hint_words for w in words) / len(words) > 0.6:
        return False
    return not (other and len(other.split()) <= 2 and len(words) > 6)


def hear_local(cfg, pcm, hints=()):
    """What was said, from 16 kHz int16 audio. Segments Whisper itself doubts are dropped."""
    x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    if len(x) < 4800:
        return ""
    prompt = "Zehen, voice commands for a PC assistant. " + ", ".join(h for h in list(hints)[:14] if h)[:260]
    segs, _ = stt_engine(cfg).transcribe(x, language="en", beam_size=3, temperature=0, initial_prompt=prompt,
                                         condition_on_previous_text=False)
    return " ".join(g.text.strip() for g in segs if g.no_speech_prob < 0.6 and g.avg_logprob > -1.3).strip()


_tts_used = _stt_used = 0.0


def trim_memory():
    """Hand idle pages back to Windows so Task Manager shows what ZEHEN really needs right now."""
    if sys.platform == "win32":
        k = ctypes.windll.kernel32
        k.SetProcessWorkingSetSize(k.GetCurrentProcess(), ctypes.c_size_t(-1), ctypes.c_size_t(-1))


def memory_keeper():
    """Runs for the life of the app: Noor's voice and ear (about 300 MB between them) are dropped after two idle minutes
    and reload in 1 to 2 s, which is hidden behind you speaking; after a quiet minute the working set is trimmed."""
    global _tts, _stt
    import gc
    while True:
        time.sleep(20)
        now = time.monotonic()
        if _tts is not None and now - _tts_used > 120 and _tts_lock.acquire(blocking=False):
            try:
                _tts = None
                gc.collect()
            finally:
                _tts_lock.release()
        if _stt is not None and now - _stt_used > 120 and _stt_lock.acquire(blocking=False):
            try:
                _stt = None
                gc.collect()
            finally:
                _stt_lock.release()
        if now - max(_tts_used, _stt_used, _local_used) > 60:
            trim_memory()


def tts_engine():
    global _tts, _tts_used
    with _tts_lock:
        _tts_used = time.monotonic()
        if _tts is None:
            from kokoro_onnx import Kokoro
            for name in KOKORO_FILES:
                if not (DATA / name).exists():
                    fetch(KOKORO_URL + name, DATA / name)
            _tts = Kokoro(str(DATA / KOKORO_FILES[0]), str(DATA / KOKORO_FILES[1]))
        return _tts


NAMES = {"noor": "नूर", "zehen": "ज़ेहेन"}  # an English voice reads "Noor" as "nor": say them from Hindi spelling instead


def tts_phonemes(tts, text):
    """Phonemes for the voice: Devanagari runs (and our names, swapped to Devanagari) are read as Hindi, the rest
    as English, so "I'm Noor" comes out "nuur", not "nor"."""
    for en, hi in NAMES.items():
        text = re.sub(rf"\b{en}\b", hi, text, flags=re.I)
    return " ".join(tts.tokenizer.phonemize(p, "hi" if "\u0900" <= p.strip()[0] <= "\u097f" else "en-us")
                    for p in re.findall(r"[\u0900-\u097f]+|[^\u0900-\u097f]+", text) if p.strip())


def llm_endpoint(cfg):
    """(base_url, model, key) from the config or env, or None: then the offline model answers."""
    c = cfg.get("llm", {})
    key = c.get("api_key") or os.environ.get("ZEHEN_API_KEY", "")
    if not key:
        return None
    url, model = PROVIDER_DEFAULTS.get(provider_of(key, c), PROVIDER_DEFAULTS["Groq"])
    return c.get("base_url") or url, c.get("model") or model, key


def provider_of(key, llm=None):
    """Which service a key belongs to: the one you picked, else its prefix, else None."""
    chosen = (llm or {}).get("provider", "")
    return chosen if chosen in PROVIDER_DEFAULTS else next((n for p, n in KEY_PREFIXES if key.startswith(p)), None)


def verify_key(key, provider="Auto-detect", base_url="", model=""):
    """Ask the service one tiny question with this key. {ok, message, provider, base_url, model}: never raises."""
    name = provider if provider in PROVIDER_DEFAULTS else (provider_of(key) if provider == "Auto-detect" else None)
    url, mdl = PROVIDER_DEFAULTS.get(name, ("", ""))
    url, mdl = (base_url or url).strip(), (model or mdl).strip()
    label = name or "this service"
    out = {"ok": False, "provider": name or "Custom", "base_url": url, "model": mdl}
    if not url or not mdl:
        out["message"] = ("Couldn't tell which service that key is for. Pick one from the list, or choose Custom and "
                          "enter its address and model.")
        return out
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(
        {"model": mdl, "max_tokens": 8, "messages": [{"role": "user", "content": "Reply with the single word OK"}]}
    ).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + key, "User-Agent": "ZEHEN"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            json.load(r)["choices"][0]
        out.update(ok=True, message=f"Verified: {label} answered with {mdl}.")
    except urllib.error.HTTPError as e:
        out["message"] = {401: f"{label} rejected this key.", 403: f"{label} rejected this key.",
                          404: f"The key may be fine, but {label} has no model called '{mdl}'. Set another.",
                          429: ""}.get(e.code, f"{label} replied with error {e.code}.")
        if e.code == 429:  # a valid key that is out of free requests for now
            out.update(ok=True, message=f"Verified: {label} knows this key (free limit reached for the moment).")
    except (urllib.error.URLError, TimeoutError, OSError):
        out["message"] = "Couldn't reach the service. Check your connection and the address."
    except (KeyError, ValueError, IndexError):
        out["message"] = f"{label} answered, but not in the OpenAI format. Is the address right?"
    return out


def parse_plan(text):
    """Model reply -> (say, actions, done). Only known actions with plain string values survive. Small models
    wrap JSON in <think> tags or write {"open_app", "x"} for {"open_app": "x"}: both are repaired, and raw model
    text is never shown or spoken."""
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.S).strip()
    if "{" not in text:
        return text[:200], [], True  # it just talked: that is the answer
    try:
        plan = json.loads(text[text.index("{"):text.rindex("}") + 1])
        do = plan.get("do", [])
        do = [do] if isinstance(do, dict) else do if isinstance(do, list) else []
        acts = [a for a in do if isinstance(a, dict) and len(a) == 1]
        say, done = str(plan.get("say", "")), plan.get("done", True) is not False
    except (ValueError, AttributeError, TypeError):
        m = re.search(r'"say"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        say = m[1].replace('\\"', '"').replace("\\n", " ") if m else ""
        acts = [{k: v} for k, v in re.findall(r'\{\s*"(\w+)"\s*[:,]\s*"((?:[^"\\]|\\.)*)"\s*\}', text) if k != "say"]
        done = not re.search(r'"done"\s*:\s*false', text)
        if not say and not acts:
            return "Sorry, I couldn't work that one out.", [], True
    acts = [a for a in acts if next(iter(a)) in ACTIONS and isinstance(next(iter(a.values())), str)]
    return say[:200], acts[:5], done


# everyday commands that need no AI, so they are instant, work with no key and no internet, and never misfire
QUICK_KEYS = {"go back": "alt+left", "back": "alt+left", "go forward": "alt+right", "forward": "alt+right",
              "refresh": "f5", "reload": "f5", "new tab": "ctrl+t", "open new tab": "ctrl+t", "close tab": "ctrl+w",
              "next tab": "ctrl+tab", "previous tab": "ctrl+shift+tab", "copy": "ctrl+c", "paste": "ctrl+v",
              "cut": "ctrl+x", "undo": "ctrl+z", "redo": "ctrl+y", "select all": "ctrl+a", "save": "ctrl+s",
              "save it": "ctrl+s", "find": "ctrl+f", "enter": "enter", "press enter": "enter", "escape": "esc",
              "cancel": "esc", "press escape": "esc", "tab": "tab", "delete": "delete", "backspace": "backspace",
              "page down": "pagedown", "page up": "pageup", "full screen": "f11", "fullscreen": "f11",
              "switch window": "alt+tab", "switch app": "alt+tab", "show desktop": "win+d", "task view": "win+tab",
              "screenshot": "win+shift+s", "take a screenshot": "win+shift+s", "mute": "volumemute",
              "unmute": "volumemute", "mute it": "volumemute"}
QUICK_WORDS = {"play": "playpause", "pause": "playpause", "resume": "playpause", "play pause": "playpause",
               "pause music": "playpause", "play music": "playpause", "stop music": "playpause",
               "pause it": "playpause", "play it": "playpause", "next song": "nexttrack", "next track": "nexttrack",
               "skip": "nexttrack", "skip song": "nexttrack", "next": "nexttrack", "previous song": "prevtrack",
               "previous track": "prevtrack", "last song": "prevtrack", "previous": "prevtrack"}


QUICK_HEARD = {"scrawl": "scroll", "scrolling": "scroll", "scrolled": "scroll", "scrolls": "scroll", "clicks": "click",
               "clicked": "click", "clique": "click", "klick": "click", "clic": "click", "quick": "click", "opened": "open",
               "opening": "open", "opens": "open"}
SPLIT_WORDS = [("you tube", "youtube"), ("g mail", "gmail"), ("face book", "facebook"), ("insta gram", "instagram"),
               ("git hub", "github"), ("chat gpt", "chatgpt"), ("net flix", "netflix"), ("link din", "linkedin"),
               ("what's app", "whatsapp"), ("whats app", "whatsapp"), ("note pad", "notepad")]
QUICK_PHRASES = [*QUICK_KEYS, *QUICK_WORDS, "scroll down", "scroll up", "scroll down a lot", "scroll up a lot", "click",
                 "double click", "right click", "zoom in", "zoom out", "volume up", "volume down", "close window",
                 "minimize", "maximize"]


SITES = {"youtube": "https://www.youtube.com", "gmail": "https://mail.google.com", "google": "https://www.google.com",
         "facebook": "https://www.facebook.com", "instagram": "https://www.instagram.com", "twitter": "https://x.com",
         "github": "https://github.com", "chatgpt": "https://chatgpt.com", "claude": "https://claude.ai",
         "netflix": "https://www.netflix.com", "amazon": "https://www.amazon.com", "wikipedia": "https://www.wikipedia.org",
         "reddit": "https://www.reddit.com", "linkedin": "https://www.linkedin.com", "maps": "https://maps.google.com",
         "google maps": "https://maps.google.com", "google drive": "https://drive.google.com",
         "whatsapp web": "https://web.whatsapp.com", "stack overflow": "https://stackoverflow.com",
         "twitch": "https://www.twitch.tv", "pinterest": "https://www.pinterest.com", "outlook": "https://outlook.live.com"}


def _quick_one(t, text, apps):
    """One everyday command, already lower-cased and cleaned: (what to show, [actions]) or None."""
    if t in QUICK_KEYS:
        return t.capitalize(), [{"press": QUICK_KEYS[t]}]
    if t in QUICK_WORDS:
        return "Media: " + t, [{"media": QUICK_WORDS[t]}]
    if m := re.fullmatch(r"(?:scroll|go|move) (down|up)(?: (a lot|a bit|a little|more|fast|slightly|slowly))?", t):
        amount = {"a lot": 15, "more": 10, "fast": 15, "a bit": 3, "a little": 3, "slightly": 2, "slowly": 2}.get(m[2], 6)
        return "Scrolling " + m[1], [{"scroll": f"{m[1]} {amount}"}]
    if m := re.fullmatch(r"(?:scroll |go )?(?:to )?(?:the )?(top|bottom)(?: of the page)?", t):
        return "Scrolling to the " + m[1], [{"scroll": m[1]}]
    if re.fullmatch(r"(?:left |single )?(?:click|tap)(?: (?:it|this|that|on it))?|select (?:it|this|that)|press it", t):
        return "Click", [{"mouse": "click"}]
    if re.fullmatch(r"double (?:click|tap)(?: (?:it|this|that))?|open (?:it|this|that)", t):
        return "Double click", [{"mouse": "double"}]
    if re.fullmatch(r"right (?:click|tap)(?: (?:it|this|that))?", t):
        return "Right click", [{"mouse": "right"}]
    if m := re.fullmatch(r"zoom (in|out)", t):
        return "Zoom " + m[1], [{"zoom": m[1]}]
    if re.fullmatch(r"turn (?:the )?(?:volume )?(?:up|down)|(?:turn )?(?:the )?volume (?:up|down)|turn it (?:up|down)|louder|quieter|softer", t):
        d = "up" if "up" in t or t == "louder" else "down"
        return "Volume " + d, [{"volume": d}]
    if re.fullmatch(r"close (?:this |the )?(?:window|app|program)|close it", t):
        return "Closing", [{"window": "close"}]
    if re.fullmatch(r"minimi[sz]e(?: (?:this|it|the window))?", t):
        return "Minimize", [{"window": "min"}]
    if re.fullmatch(r"maximi[sz]e(?: (?:this|it|the window))?", t):
        return "Maximize", [{"window": "max"}]
    if (m := re.fullmatch(r"(?:type|write|dictate) (.+)", text.strip(" .,!?"), re.I)) and not re.match(
            r"(?:a|an|the|my|me|to|about|something|some|note|email|letter|message|story|poem)\b|.*\b(?:in|on|into|using) \w+$",
            m[1], re.I):  # "type hello" is quick; "write a note in notepad" needs the AI
        return "Typing", [{"type": m[1]}]
    if t in ("what time is it", "what's the time", "whats the time", "what is the time", "tell me the time", "the time",
             "current time", "time"):
        return time.strftime("It's %I:%M %p.").replace("It's 0", "It's "), []  # an answer, not an action
    if t in ("what is the date", "what's the date", "whats the date", "what day is it", "what is today's date",
             "today's date", "what is the day today", "what's today", "what day is today"):
        return time.strftime("Today is %A, %B %d.").replace(" 0", " "), []
    if m := re.fullmatch(r"(?:search google for|search for|google search|look up|search|google) (.+)", t):
        return "Searching", [{"search_web": m[1]}]
    if m := re.fullmatch(r"(?:(?:open|start|launch|run) )?(?:jupiter|jupyter)(?: (notebook|lab))?", t):  # no Start menu entry: it's a command
        tool = "lab" if m[1] == "lab" else "notebook"
        return "Starting Jupyter " + tool, [{"powershell": (
            "$j = Get-Command jupyter -ErrorAction SilentlyContinue; "
            f"if ($j) {{ Start-Process -FilePath $j.Source -ArgumentList '{tool}'; 'started' }} "
            f"else {{ python -c 'import {tool}' 2>$null; if ($LASTEXITCODE -eq 0) {{ Start-Process python "
            f"-ArgumentList '-m','{tool}'; 'started' }} else {{ 'NOTFOUND' }} }}")}]
    if t in ("install jupyter", "install jupiter", "install jupyter notebook"):
        return "Installing Jupyter", [{"powershell": "python -m pip install notebook 2>&1 | Select-Object -Last 1"}]
    if m := re.fullmatch(r"(?:play|open|watch|click|start)(?: on)?(?: the)? (first|second|third|fourth|fifth|top|1st|2nd|3rd|4th|5th|\d)"
                         r"(?: video| one| result| link| song)?", t):
        return "Playing video " + m[1], [{"play": m[1]}]
    if m := re.fullmatch(r"(?:play|watch|open)(?: the)?(?: video| song)? (.+?) (?:on youtube|video|video on youtube)", t):
        return "Playing " + m[1], [{"open_url": YT_SEARCH + urllib.parse.quote(m[1])}, {"wait": "4.5"}, {"play": "1"}]
    if m := re.fullmatch(r"(?:search|find|look for)(?: for)? (.+?) on youtube", t):
        return "Searching YouTube", [{"open_url": YT_SEARCH + urllib.parse.quote(m[1])}]
    if m := re.fullmatch(r"(?:click|select|focus|go to|open|use|put the cursor in)(?: on| in)?(?: the)? "
                         r"(search|type|text|message|chat|address|url|comment|input|reply|prompt)(?: bar| box| field| area| input)?", t):
        return "Text box", [{"focus_input": m[1]}]
    if re.fullmatch(r"(?:start )?(?:dictate|dictation|dictation mode|typing mode|start typing|type for me|take dictation)", t):
        return "Ready to type", [{"dictate": "on"}]
    if (m := re.fullmatch(r"(?:play|watch) (.+)", t)) and not re.search(r" on (?!youtube)\w+$", t):
        return "Playing " + m[1], [{"play": m[1]}]  # on the page, else found on YouTube
    if m := re.fullmatch(r"(?:switch|go|jump|change|move) (?:back )?to (?:the )?tab (\d)", t):
        return "Tab " + m[1], [{"press": "ctrl+" + m[1]}]
    if m := re.fullmatch(r"(?:(?:switch|go|jump|change|move)(?: back)? to|(?:open|show|select|find|focus))(?: the)? (.+?) tab", t):
        return "Switching to the " + m[1] + " tab", [{"tab": m[1]}]
    if m := re.fullmatch(r"(?:switch|jump|change|move|alt tab)(?: back)? to (?:the )?(.+?)(?: window| app)?", t):
        return "Switching to " + m[1], [{"switch": m[1]}]
    if m := re.fullmatch(r"(?:open|show|go to)(?: me)?(?: my| the)? (desktop|downloads|documents|pictures|music|videos)(?: folder)?", t):
        return "Opening " + m[1], [{"open_folder": m[1]}]
    if m := re.fullmatch(r"(?:open|launch|start|run|go to|visit)(?: the)?(?: website| site)? (.+?)(?: website| site)?", t):
        name = m[1].replace(" dot ", ".")
        exact = next((a for a in apps if fold(a[0]) == fold(name)), None)
        if exact:
            return "Opening " + exact[0], [{"open_app": exact[0]}]
        if name in SITES:
            return "Opening " + name, [{"open_url": SITES[name]}]
        if re.fullmatch(r"[\w-]+(?:\.[\w-]+)+", name):
            return "Opening " + name, [{"open_url": "https://" + name}]
        if search_apps(apps, name, 1):
            return "Opening " + name, [{"open_app": name}]
        return None
    if (m := re.fullmatch(r"(?:click|press|tap|select|choose)(?: on)?(?: the)? (.+)", t)) and m[1] not in (
            "it", "this", "that", "here", "there", "enter", "escape"):
        return "Clicking " + m[1], [{"click": m[1]}]
    return None


def quick_command(text, apps):
    """(what to show, [actions]) for "scroll down", "click", "open youtube", "click on videos", "go back",
    "type hello", and chains like "open youtube and click on trending"; None when it needs the AI."""
    def one(part):
        t = " ".join(re.sub(r"[^\w\s']", " ", re.sub(r"(?<=\w)\.(?=\w)", " dot ", part.lower())).split())
        t = re.sub(r"^((please|hey|ok|okay|can you|could you|would you|just|now) )+", "", t)
        t = re.sub(r"( (please|now|for me|here|there|again))+$", "", t)
        t = " ".join(QUICK_HEARD.get(w, w) for w in t.split())
        if t in ("like", "lick"):
            t = "click"  # Vosk's small model writes a short "click" as "like"
        for heard, word in SPLIT_WORDS:  # "you tube" -> "youtube"
            t = re.sub(rf"\b{heard}\b", word, t)
        got = _quick_one(t, part, apps)
        if not got:  # a small mishearing: "scroll dawn", "double clic"
            near = difflib.get_close_matches(t, QUICK_PHRASES, 1, 0.86)
            got = _quick_one(near[0], near[0], apps) if near and near[0] != t else None
        return got
    if got := one(text):
        return got
    parts = [p for p in re.split(r"\s*,\s*|\s+and then\s+|\s+then\s+|\s+and\s+", text.strip(" .")) if p.strip()]
    steps = [one(p) for p in parts] if len(parts) > 1 else []
    if steps and all(steps):
        acts = []
        for i, (_, a) in enumerate(steps):
            acts += a
            if i < len(steps) - 1 and any(k in ("open_url", "open_app", "search_web") for x in a for k in x):
                acts.append({"wait": "3.5"})  # let the page or app appear before the next step
        return steps[0][0] + ", then " + steps[-1][0].lower(), acts
    return None


def _chat_local(messages, note):
    last = {**messages[-1], "content": messages[-1]["content"] + " /no_think"}  # Qwen3: skip the long thinking
    with _local_lock:
        out = local_llm(note).create_chat_completion(  # json_object: the decoder can only write valid JSON
            messages=messages[:-1] + [last], max_tokens=300, temperature=0.1, response_format={"type": "json_object"})
    global _local_used
    _local_used = time.monotonic()  # the quiet minute starts when she finishes, not when she started
    return out["choices"][0]["message"]["content"]


def chat(cfg, messages, note=lambda s, frac=None: None):
    """One model call: the free cloud key if there is one, else the offline model. A key that is missing, rejected,
    out of free requests or offline never leaves her speechless: she falls back to the offline brain."""
    ep = llm_endpoint(cfg)
    if not ep:
        return _chat_local(messages, note)
    url, model, key = ep
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(
        {"model": model, "temperature": 0.1, "max_tokens": 600, "messages": messages}).encode(), {
        "Content-Type": "application/json", "Authorization": "Bearer " + key,
        "User-Agent": "ZEHEN"})  # Cloudflare in front of some providers rejects Python's default agent
    try:
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            raise RuntimeError({401: "key rejected: paste a new one", 403: "key rejected: paste a new one",
                                404: "that model is gone: set another in .zehen.json",
                                429: "free limit reached"}.get(e.code, f"AI error {e.code}"))
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError):
            raise RuntimeError("couldn't reach the AI")
    except RuntimeError as cloud_error:
        note(f"{cloud_error}. Using my offline brain.")
        try:
            return _chat_local(messages, note)
        except RuntimeError:
            raise cloud_error  # no offline brain either: say what went wrong with the key


def run_powershell(cmd, timeout=30):
    ps = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + cmd
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                           encoding="utf-8", errors="replace", timeout=timeout, creationflags=0x08000000)
    except subprocess.TimeoutExpired:
        return "(timed out after %ds)" % timeout
    return ((r.stdout + r.stderr).strip() or "(no output)")[:1500]


UI_KINDS = {"ButtonControl": "button", "EditControl": "text box", "ListItemControl": "item", "TabItemControl": "tab",
            "MenuItemControl": "menu", "HyperlinkControl": "link", "CheckBoxControl": "checkbox",
            "ComboBoxControl": "dropdown", "TreeItemControl": "item", "RadioButtonControl": "option"}


def _uia_top(auto, top=None):
    if top is not None:
        return top
    try:
        c = auto.GetForegroundControl()
        return c.GetTopLevelControl() if c else None
    except Exception:
        return None


def ui_controls(limit=400, top=None):
    """Clickable things in the window in front, read through Windows UI Automation (no screenshots, no vision
    model, almost no RAM): [(kind, name, control)]. Works with any app that exposes accessibility, which is
    nearly all of them, Electron and browsers included."""
    import uiautomation as auto
    out, t0 = [], time.monotonic()
    try:
        with auto.UIAutomationInitializerInThread():
            top = _uia_top(auto, top)
            for c, _ in auto.WalkControl(top, maxDepth=45) if top else []:  # web pages sit 25+ levels deep
                if time.monotonic() - t0 > 4 or len(out) >= limit:
                    break  # a huge page: work with what we have
                try:
                    if c.ControlTypeName in UI_KINDS and (name := c.Name.strip()) and not c.IsOffscreen:
                        out.append((UI_KINDS[c.ControlTypeName], name.replace("\n", " "), c))
                except Exception:
                    continue
    except Exception:
        pass
    return out


NUMWORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
            "eight": "8", "nine": "9", "ten": "10"}


def _toks(text):
    """'SEM7_ML_PB_2026.pdf' -> ['sem', '7', 'ml', 'pb', '2026', 'pdf']; 'sam seven' -> ['sam', '7']."""
    return [NUMWORDS.get(w, w) for w in re.findall(r"[a-z]+|\d+", fold(text))]


def best_match(query, names, need=0.6):
    """Index of the name that best matches what you said, or None. Forgiving of how speech writes things:
    "sam seven tfpb" finds "SEM7_ML_PB_2026_TEST-2.pdf" (digits, spelling, word order)."""
    q = _toks(query)
    best, pick = need - 1e-9, None
    for i, name in enumerate(names):
        n = _toks(name)
        hit = sum(any(w == x or (len(w) >= 3 and len(x) >= 3 and (w in x or x in w
                      or difflib.SequenceMatcher(None, w, x).ratio() >= 0.66)) for x in n) for w in q)
        score = hit / len(q) if q else 0
        if score > best:
            best, pick = score, i
    return pick


def ui_tabs(top=None):
    """The tabs of the browser (or any tabbed app) in front: [(title, control)]."""
    import uiautomation as auto
    out = []
    try:
        with auto.UIAutomationInitializerInThread():
            top = _uia_top(auto, top)
            for c, _ in (auto.WalkControl(top, maxDepth=12) if top else []):
                try:
                    if c.ControlTypeName == "TabItemControl" and (name := c.Name.strip()) and not c.IsOffscreen:
                        out.append((name, c))
                except Exception:
                    continue
    except Exception:
        pass
    return out


def switch_tab(query, top=None):
    tabs = ui_tabs(top)
    i = best_match(query, [n for n, _ in tabs])
    if i is None:
        seen = ", ".join(n[:25] for n, _ in tabs[:4])
        return f"no tab like '{query}'." + (f" I can see: {seen}" if seen else " I don't see any tabs here.")
    tabs[i][1].Click()
    return "switched to " + re.sub(r" - (?:Memory usage|Audio playing|Camera|Microphone).*$", "", tabs[i][0])[:40]


def switch_to(query):
    """Windows first (apps), then the tabs of the window in front."""
    wins = list_windows()
    i = best_match(query, [t for _, t in wins])
    if i is not None:
        focus_window(wins[i][0])
        return "switched to " + wins[i][1][:40]
    out = switch_tab(query)
    return out if out.startswith("switched") else f"no window or tab like '{query}'."


def ui_list():
    return "\n".join(f"{k}: {n[:50]}" for k, n, _ in ui_controls(60)) or "(nothing clickable found)"


def ui_click(name):
    found = ui_controls()
    want = fold(name)
    for test in (lambda n: fold(n) == want, lambda n: want in fold(n),
                 lambda n: difflib.SequenceMatcher(None, want, fold(n)).ratio() > 0.75):  # exact, contains, close
        for k, n, c in found:
            if test(n):
                c.Click()
                return f"clicked {k}: {n[:50]}"
    return f"no '{name}' here: use ui_list"


ORDINALS = {"first": 1, "1st": 1, "top": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
            "fifth": 5, "5th": 5}
YT_SEARCH = "https://www.youtube.com/results?search_query="


def video_links(top=None):
    """Videos on the page in front, in reading order: YouTube and most video sites name a video's link with its title and
    length ("... 7 minutes, 6 seconds"). Ads are skipped."""
    import uiautomation as auto
    found, t0 = [], time.monotonic()
    try:
        with auto.UIAutomationInitializerInThread():
            top = _uia_top(auto, top)
            for c, _ in auto.WalkControl(top, maxDepth=60) if top else []:
                if time.monotonic() - t0 > 5:
                    break
                try:
                    if c.ControlTypeName == "HyperlinkControl":
                        n = c.Name.strip()
                        if re.search(r"\b\d+\s+(?:second|minute|hour)s?\b", n) and not re.search(r"sponsored|\bad\b", n, re.I) \
                                and not c.IsOffscreen and c.BoundingRectangle.width() >= 120:
                            r = c.BoundingRectangle
                            found.append((r.top // 40, r.left, n, c))  # rows of 40 px, then left to right
                except Exception:
                    continue
    except Exception:
        pass
    found.sort(key=lambda f: (f[0], f[1]))
    return [(n, c) for _, _, n, c in found]


def play_video(query, top=None):
    """"1" (the first), "3", or words from a title. Words that match nothing on the page search YouTube and play the
    top result."""
    q = query.strip().lower()
    vids = video_links(top)
    if q.isdigit() or q in ORDINALS:
        i = (int(q) if q.isdigit() else ORDINALS[q]) - 1
        if i < len(vids):
            vids[i][1].Click()
            return "playing " + re.sub(r" \d+ (?:second|minute|hour).*$", "", vids[i][0])[:60]
        return "no video here: " + (f"only {len(vids)} on the page" if vids else "none on this page")
    i = best_match(query, [n for n, _ in vids])
    if i is not None:
        vids[i][1].Click()
        return "playing " + re.sub(r" \d+ (?:second|minute|hour).*$", "", vids[i][0])[:60]
    webbrowser.open(YT_SEARCH + urllib.parse.quote(query))  # not on this page: find it on YouTube
    for _ in range(4):
        time.sleep(2.5)
        top_now = None
        if (vids := video_links(top_now)):
            vids[0][1].Click()
            return "playing " + re.sub(r" \d+ (?:second|minute|hour).*$", "", vids[0][0])[:60]
    return f"no video like '{query}' found"


def focus_input(hint="", top=None):
    """Put the cursor in a text box of the app in front, ready for you to speak: "search" finds the search bar,
    "message" the chat box, otherwise the biggest box on screen."""
    boxes = [(n, c) for k, n, c in ui_controls(400, top) if k == "text box"]
    words = _toks(hint)
    key = [w for w in words if w in ("search", "address", "url", "message", "chat", "comment", "reply", "prompt", "type")]
    pick = next((c for n, c in boxes if any(w in _toks(n) or (w == "url" and "address" in _toks(n)) for w in key)), None)
    if pick is None and boxes:
        pick = max(boxes, key=lambda b: b[1].BoundingRectangle.width() * b[1].BoundingRectangle.height())[1]
    if pick is None:
        return "no text box here"
    pick.Click()
    try:
        pick.SetFocus()
    except Exception:
        pass
    return "ready to type"


def whisper(cfg, pcm):
    """Accurate speech-to-text through Groq's free Whisper, when the key is a Groq one. '' means: use Vosk's."""
    ep = llm_endpoint(cfg)
    if not ep or not ep[2].startswith("gsk_"):
        return ""
    wav = io.BytesIO()
    with wave.open(wav, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(pcm)
    b = uuid.uuid4().hex
    field = lambda k, v: f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    body = (field("model", "whisper-large-v3-turbo") + field("response_format", "text") + field("temperature", "0")
            + f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
              'Content-Type: audio/wav\r\n\r\n'.encode() + wav.getvalue() + f"\r\n--{b}--\r\n".encode())
    req = urllib.request.Request(ep[0].rstrip("/") + "/audio/transcriptions", body, {
        "Content-Type": "multipart/form-data; boundary=" + b, "Authorization": "Bearer " + ep[2],
        "User-Agent": "ZEHEN"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read().decode().strip()
    except Exception:
        return ""


_MEL = None


def mfcc(pcm):
    """12 cepstral coefficients per 10 ms of 16 kHz audio (bytes of int16), loudness and mic tone removed.
    The voice's fingerprint for one short word."""
    import numpy as np
    global _MEL
    x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    if len(x) < 1200:
        return None
    x = np.append(x[0], x[1:] - np.float32(0.97) * x[:-1])  # pre-emphasis
    frames = 1 + (len(x) - 400) // 160
    if _MEL is None:  # 26 mel filters from 80 Hz to 7.6 kHz, and the DCT that turns them into cepstra
        mel = lambda f: 2595 * np.log10(1 + f / 700)
        pts = 700 * (10 ** (np.linspace(mel(80), mel(7600), 28) / 2595) - 1)
        bins = np.floor((512 + 1) * pts / 16000).astype(int)
        fb = np.zeros((26, 257), dtype=np.float32)
        for m in range(26):
            l, c, r = bins[m], bins[m + 1], bins[m + 2]
            fb[m, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
            fb[m, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
        dct = np.cos(np.pi / 26 * (np.arange(26) + .5)[None, :] * np.arange(1, 13)[:, None]).astype(np.float32)
        _MEL = (fb, dct, np.hamming(400).astype(np.float32))
    fr = x[np.arange(400)[None, :] + 160 * np.arange(frames)[:, None]] * _MEL[2]
    power = np.abs(np.fft.rfft(fr, 512)) ** 2
    c = np.log(power @ _MEL[0].T + np.float32(1e-10)) @ _MEL[1].T
    energy = power.sum(1)
    voiced = energy > np.percentile(energy, 25)
    return c - (c[voiced].mean(0) if voiced.any() else c.mean(0))


def dtw(template, audio):
    """How far `audio` is from saying `template`: dynamic time warping, free start within the first 0.3 s and free
    end, so a word spoken fast or slow still lines up. Returns (score, end frame); lower is closer."""
    import numpy as np
    T, N = len(template), len(audio)
    if not T or not N:
        return float("inf"), 0
    cost = np.linalg.norm(template[:, None, :] - audio[None, :, :], axis=-1).astype(np.float32)
    D = np.full((T, N), np.inf, dtype=np.float32)
    free = min(N, 30)
    D[0, :free] = cost[0, :free]
    for j in range(free, N):
        D[0, j] = cost[0, j] + D[0, j - 1]
    for i in range(1, T):
        prev, row = D[i - 1], cost[i]
        cur = np.empty(N, dtype=np.float32)
        cur[0] = row[0] + prev[0]
        cur[1:] = row[1:] + np.minimum(prev[:-1], prev[1:])
        for j in range(1, N):
            if cur[j - 1] + row[j] < cur[j]:
                cur[j] = cur[j - 1] + row[j]
        D[i] = cur
    start = min(int(T * .6), N - 1)
    j = int(D[T - 1, start:].argmin())
    return float(D[T - 1, start + j] / T), start + j


class Voice:
    """Hands-free: say "Zehen" (wake word via Vosk, offline, always listening on this PC) -> your request (Whisper if you
    have a Groq key, else Vosk) -> on_text(text) returns what to say back. state drives the HUD."""

    def __init__(self, cfg, on_text, paused, note=lambda s, frac=None: None, chatty=lambda: False, hints=lambda: [],
                 keep=lambda: False, setup=None):
        self.setup = setup
        self.cfg, self.on_text, self.paused, self.note, self.chatty = cfg, on_text, paused, note, chatty
        self.keep = keep  # dictating: stay in listening mode, no wake word needed
        self.hints, self.alts = hints, []  # names to listen for; the other ear's version of the last request
        self.state, self.error, self.muted = "loading", "", False  # loading|idle|listening|thinking|speaking|off
        self.level, self.floor, self.spoke = 0.0, 300.0, False  # mic loudness 0..1 for the HUD; room noise; did we talk
        self.ready, self.talk = threading.Event(), threading.Lock()  # greeting done; one voice at a time
        self.follow, self.chain, self.chain_on = False, 0, False  # keep listening only after a question of hers
        self.partial, self.heard, self.heard_at = "", "", -99.0  # your words as you speak them, and the last request
        self.wake_score = 0.0  # how close the last wake word was; a borderline one is double-checked by Whisper
        self.tpl, self.thr = self._load_templates()  # your own "Zehen", learned from three samples (see enroll)
        self.enroll_left, self.samples, self.enrolled_now = 0, [], False
        self.go = threading.Event()  # set by "listen now": skips the wake word
        self.words, self.tts_ok = cfg.get("wake", WAKE_WORDS), True

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _load_templates():
        try:
            import numpy as np
            z = np.load(DATA / "wake.npz")
            return [z[k] for k in sorted(z.files) if k.startswith("t")], float(z["thr"])
        except (OSError, ValueError, ImportError):
            return None, 0.0

    def enroll(self):
        """Learn how you say "Zehen": three samples. After this it wakes only for that sound, in your voice,
        whatever your accent, instead of leaning on a speech model that has never heard the word."""
        self.samples, self.enroll_left, self.enrolled_now = [], 3, False

    def _take_sample(self, hist, onset):
        a = max(onset - 1, 0)
        self.samples.append(b"".join(hist[a:len(hist) - 4]))  # from just before the word to the end of the sound
        self.enroll_left -= 1
        if self.enroll_left:
            self.say(random.choice(["Again.", "Once more.", "One more time."]))
            return
        import numpy as np
        tpl = [f for f in (mfcc(x) for x in self.samples) if f is not None]
        if len(tpl) < 3:
            self.enroll_left, self.samples = 3, []
            self.say("That was too short. Let's try again. Say Zehen.")
            return
        intra = [dtw(tpl[i], tpl[j])[0] for i in range(3) for j in range(3) if i != j]
        self.tpl, self.thr = tpl, min(max(intra) * 2.6, 19.0)  # you score 1.2 to 2 times the spread, other words 2.6 times and up; Whisper settles the edge
        DATA.mkdir(exist_ok=True)
        np.savez(DATA / "wake.npz", thr=self.thr, **{f"t{i}": t for i, t in enumerate(tpl)})
        self.enrolled_now = True
        self.say("Got it. I'll know you by your voice now.")

    def _dtw_wake(self, hist, onset):
        """(is it your wake word?, where it ends in seconds from the phrase start) for the first sound of a phrase."""
        a = max(onset - 1, 0)
        f = mfcc(b"".join(hist[a:a + 16]))  # about 1.5 s from just before the first sound
        if f is None:
            return False, None
        score, end = min(dtw(t, f) for t in self.tpl)
        self.wake_score = score
        if os.environ.get("ZEHEN_DEBUG"):
            print(f"[wake] score {score:.1f} / limit {self.thr:.1f} -> {'WAKE' if score < self.thr else 'no'}", flush=True)
        return score < self.thr, a * 0.1 + end * 0.01

    def listen_now(self):
        self.go.set()

    def shush(self):
        """"Stop" / "be quiet": cut her off mid-sentence."""
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    def say(self, text):
        """Speak without blocking the caller (the tutorial's voice)."""
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()

    def _prewarm(self):
        for load in (lambda: stt_engine(self.cfg), tts_engine):
            try:
                load()
            except Exception:
                pass

    def _warm_ears(self):
        try:
            stt_engine(self.cfg, self.note)
            self.note("")
        except Exception:
            pass  # the small model still hears; she tries again next launch

    def _warm_voice(self):
        try:
            tts_engine()
        except Exception:
            pass  # speak() falls back to Windows' voice
        if self.cfg.get("greeting", True):
            h = time.localtime().tm_hour
            part = "morning" if h < 12 else "afternoon" if h < 18 else "evening"
            self.speak(random.choice(GREETINGS).format(part=part))
        brain = self.setup.item("brain") if self.setup is not None and self.cfg.get("brain_setup", True) else None
        need_brain = brain is not None and brain["state"] in ("pending", "working")
        if need_brain:
            self.speak("Before we start, I need to set up my brain. It's a one-time download of about a gigabyte. "
                       "Carry on with the tutorial. I'll tell you when I'm ready.")
        self.ready.set()
        if need_brain:
            brain["finished"].wait()  # the checklist does the downloading; the panel shows it
            self.note("")
            if brain["state"] == "done":
                self.say("My brain is installed. Ready when you are.")
            elif brain["state"] == "failed":
                self.say("I couldn't finish setting up my brain. You can add a free key from the panel instead.")

    def _run(self):
        try:
            self._loop()
        except ImportError:
            self.state, self.error = "off", "Voice needs: pip install vosk sounddevice"
        except Exception as e:
            self.state, self.error = "off", str(e) if isinstance(e, RuntimeError) else f"Voice: {e}"[:90]

    def _loop(self):
        import queue
        import sounddevice as sd
        import vosk
        vosk.SetLogLevel(-1)
        if self.setup is not None:
            self.setup.core.wait()  # the checklist is fetching the essentials: packages, wake word, voice, hearing
        if not VOSK_DIR.exists():
            fetch(VOSK_URL, VOSK_DIR, unzip=True)
        model = vosk.Model(str(VOSK_DIR))
        threading.Thread(target=self._warm_voice, daemon=True).start()  # her first words shouldn't wait on a download
        threading.Thread(target=self._warm_ears, daemon=True).start()
        q = queue.Queue()
        wake = lambda: vosk.KaldiRecognizer(model, 16000, json.dumps(self.words + ["[unk]"]))
        rec, quiet, hist, stable = wake(), 0, [], 0  # hist: this phrase's audio so far
        onset, loud_n, checked = None, 0, False  # where the phrase's first sound is; how much sound; already judged
        with sd.RawInputStream(samplerate=16000, blocksize=1600, dtype="int16", channels=1,
                               callback=lambda d, *_: q.put(bytes(d))):
            self.state = "idle"
            while True:
                try:
                    block = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if self.muted or self.paused():
                    continue
                loud = self._loud(block)
                quiet = 0 if loud else quiet + 1
                woke, pre, cut = False, b"", None
                if quiet == 5 or len(hist) > 100:  # half a second of quiet (or 10 s of talk): the phrase is over
                    if self.enroll_left and onset is not None and 3 <= loud_n <= 25:
                        self._take_sample(hist, onset)
                    elif self.tpl and not self.enroll_left and not checked and onset is not None and loud_n >= 3:
                        woke, cut = self._dtw_wake(hist, onset)  # a short "Zehen" on its own ends here
                        pre = b"".join(hist)
                    rec.Reset()
                    hist.clear()
                    stable, onset, loud_n, checked = 0, None, 0, False
                hist.append(block)
                if loud:
                    onset = len(hist) - 1 if onset is None else onset
                    loud_n += 1
                pushed = self.go.is_set()
                if self.enroll_left:
                    pass
                elif self.tpl:  # your own wake word: judged 0.7 s into the phrase, while you may still be talking
                    if not woke and not checked and loud_n == 7 and onset is not None:
                        checked = True
                        woke, cut = self._dtw_wake(hist, onset)
                        pre = b"".join(hist)
                elif self._heard_wake(rec, block):  # no voice learned yet: the speech model's best guess
                    stable += 1 if stable >= 0 else 0
                    if stable == WAKE_STABLE:
                        woke = self._wake_like(model, b"".join(hist))
                        pre = b"".join(hist)
                        stable = stable if woke else -10 ** 6
                else:
                    stable = 0 if stable >= 0 else stable
                if pushed or woke:
                    self.go.clear()
                    self.spoke, self.follow, self.chain_on = False, False, bool(pushed and self.chain)
                    if not pushed:
                        self.chain = 0
                    self._command(model, q, b"" if pushed else pre, None if pushed else cut,
                                  confirm=bool(self.tpl) and self.wake_score >= 0.9 * self.thr)
                    rec, hist, stable, onset, loud_n, checked = wake(), [], 0, None, 0, False
                    while self.spoke and not q.empty():
                        q.get_nowait()  # her own voice, echoing in the mic. Anything you said meanwhile is kept
                    if self.follow or self.keep():
                        time.sleep(0.5)  # let the room's echo of her own voice die away first
                        while not q.empty():
                            q.get_nowait()
                        self.go.set()  # she asked a question: listen for the answer, no wake word needed
                    self.state = "idle"

    def _loud(self, block):
        """True when this 0.1 s of audio is speech. Also feeds the HUD's level meter and tracks the room's noise."""
        import numpy as np
        level = float(np.sqrt(np.mean(np.frombuffer(block, np.int16).astype(np.float32) ** 2)))
        self.floor = min(level, self.floor * 1.002 + 0.5)  # the room's quietest level: drops at once, creeps up slowly
        self.level = self.level * 0.6 + 0.4 * min(1.0, level / 3500)
        return level > max(120.0, self.floor * 2.5)

    def _heard_wake(self, rec, block):
        """True only when a phrase *begins* with the wake word: the same sound in the middle of a sentence
        ("that's insane") is ignored, so ordinary talk never wakes it. (Vosk merges unknown words into one
        [unk], so "hey zehen" can't be told from "the plane is insane": the wake word has to come first.)"""
        done = rec.AcceptWaveform(block)
        words = json.loads(rec.Result() if done else rec.PartialResult()).get("text" if done else "partial", "").split()
        return bool(words) and words[0] in self.words

    @staticmethod
    def _wake_like(model, pcm):
        import vosk
        r = vosk.KaldiRecognizer(model, 16000)
        r.AcceptWaveform(pcm)
        first = json.loads(r.FinalResult())["text"].split(" ")[0]
        return first in WAKE_LIKE

    def _command(self, model, q, phrase=b"", cut=None, confirm=False):
        """Hear the request. phrase = the audio so far, wake word included: it's transcribed again from the start
        and everything up to the end of the wake word is cut by its timestamps, so nothing is lost or left over."""
        import queue
        import vosk
        self.state = "listening"
        threading.Thread(target=self._prewarm, daemon=True).start()  # her ear and voice load while you are still talking
        rec = vosk.KaldiRecognizer(model, 16000)
        rec.SetWords(True)
        audio, words, spoke, t0 = [], [], False, time.monotonic()
        feed = [phrase[i:i + 3200] for i in range(0, len(phrase), 3200)]
        quiet, live_loud, phrase_loud, phrase_cmd, n = 0, 0, 0, 0, 0
        while time.monotonic() - t0 < 15:
            live = not feed
            n += 0 if live else 1  # which 0.1 s of the pre-recorded phrase this is
            if feed:
                block = feed.pop(0)
            else:
                try:
                    block = q.get(timeout=0.5)
                except queue.Empty:
                    continue
            audio.append(block)
            if self._loud(block):
                quiet = 0
                live_loud += live
                phrase_loud += not live
                phrase_cmd += (not live) and cut is not None and n * 0.1 >= cut  # sound after the wake word: a request
            else:
                quiet += 1
            if rec.AcceptWaveform(block):
                words += json.loads(rec.Result()).get("result", [])
                self.partial = " ".join(w["word"] for w in words)
            else:
                part = json.loads(rec.PartialResult())["partial"]
                if part:
                    spoke = True
                    self.partial = (" ".join(w["word"] for w in words) + " " + part).strip()
            if phrase and self.partial:  # the tail of "Zehen" shouldn't show up as your first word
                toks = self.partial.split()
                while len(toks) > 1 and toks[0] in {"the", "then", "this", "him", "he", "hen", "head", "hank", "like", "zane"}:
                    toks.pop(0)
                self.partial = " ".join(toks)
            if live and quiet >= 7 and (live_loud or phrase_cmd >= 2 or (cut is None and phrase_loud >= 8)
                                        or time.monotonic() - t0 > 2.5):
                break  # a real pause after you stopped talking: that was the whole request
            if not spoke and not words and live and time.monotonic() - t0 > (10 if self.keep() else 6):
                return  # woke, then nobody spoke
        words += json.loads(rec.FinalResult()).get("result", [])
        pcm = b"".join(audio)
        if not (live_loud or phrase_cmd >= 2 or (cut is None and phrase_loud >= 8)):
            self.state = "idle"
            return  # nothing was said after the wake word: don't let anything invent a request out of silence
        if cut is None:
            cut = 0.0
        if phrase and words and cut == 0.0:
            g = vosk.KaldiRecognizer(model, 16000, json.dumps(self.words + ["[unk]"]))
            g.SetWords(True)
            g.AcceptWaveform(pcm)
            # where the wake word ends: the grammar's say-so, else a typical "Zehen" length after the first word
            cut = next((w["end"] for w in json.loads(g.FinalResult()).get("result", []) if w["word"] in self.words),
                       words[0]["start"] + 0.4)
        text = " ".join(w["word"] for w in words if w["start"] >= cut)
        self.partial = ""
        if os.environ.get("ZEHEN_DEBUG"):
            print(f"[heard] cut {cut:.2f}s words {[(w['word'], round(w['start'], 2)) for w in words]}", flush=True)
        if phrase:  # the tail of "Zehen" can still decode as a word or two of its own ("the head", "like")
            junk, toks = {"the", "then", "this", "him", "he", "hen", "head", "hank", "like", "zane", "zen", "zain", "zehen", "he's"}, text.split()
            for _ in range(2):
                if toks and len(toks) > 1 and toks[0] in junk:
                    toks.pop(0)
            text = " ".join(toks)
        self.state = "thinking"
        # the best ear available: Groq's Whisper (with a Groq key), else Whisper on this PC, else the small model's guess
        heard, small = whisper(self.cfg, pcm), text
        if not heard:
            try:
                hints = self.hints()
                heard = hear_local(self.cfg, pcm, hints)  # the whole phrase, wake word included
                if heard and not sane_transcript(heard, hints, small):
                    heard = ""  # it was making things up: trust the small model's few words instead
            except Exception:
                heard = ""
        if os.environ.get("ZEHEN_DEBUG"):
            print(f"[ear] whisper {heard!r} | vosk {small!r} | confirm {confirm}", flush=True)
        if heard:
            heard = re.sub(r"^(?:zehen|zehn|zahen|sehen)[\s,.:!-]*", "", heard, flags=re.I) or heard  # fused: "Zehenclick"
            first = heard.split(" ")[0]
            wakeish = bool(difflib.get_close_matches(first.strip(",.!?").lower(), WAKE_SPELLINGS, 1, 0.6))
            if phrase and confirm and not wakeish:  # a borderline wake that Whisper didn't hear as "Zehen": a false alarm
                self.state = "idle"
                return
            for _ in range(2):  # drop a leading "Zehen" however it was written
                first, _, rest = heard.partition(" ")
                if rest and difflib.get_close_matches(first.strip(",.!?").lower(), WAKE_SPELLINGS, 1, 0.6):
                    heard = rest.lstrip(" ,.")
            text = heard
        self.alts = [small] if small and small.lower().strip(" .") != text.lower().strip(" .") else []
        if len(text.strip()) > 1:
            self.heard, self.heard_at = text.strip(), time.monotonic()  # stays on screen a few seconds, under the pill
            reply = self.on_text(text.strip())
            self.speak(reply)
            # she keeps listening only when she asked you a question, so a plain answer or "you're welcome" can't
            # start a conversation with the room; and only for a couple of turns
            self.chain = self.chain + 1 if self.chain_on else 0
            self.follow = (bool(reply) and self.chatty() and reply.rstrip().endswith("?") and self.chain < 3
                           and not self.cfg.get("no_follow_up"))
        else:
            self.state = "idle"

    def speak(self, text):
        """Say it out loud: Kokoro's neural female voice, or Windows' own voice if that isn't available.
        The mic is muted meanwhile so it can't hear itself."""
        if not text:
            return
        self.talk.acquire()  # one voice at a time: the tutorial, the greeting and replies queue up politely
        self.state, self.muted, self.spoke = "speaking", True, True
        try:
            if self.tts_ok:
                try:
                    import sounddevice as sd
                    tts = tts_engine()
                    samples, rate = tts.create(tts_phonemes(tts, text), voice=self.cfg.get("voice_name", VOICE_NAME),
                                               speed=1.05, is_phonemes=True)
                    sd.play(samples, rate)
                    sd.wait()
                    return
                except Exception:
                    self.tts_ok = False  # no model yet and no internet, or no kokoro-onnx: don't retry every sentence
            ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  "$s.SelectVoiceByHints('Female'); $s.Speak($env:ZEHEN_SAY)")
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], env={**os.environ, "ZEHEN_SAY": text},
                           timeout=30, creationflags=0x08000000)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            self.muted = False
            if self.state == "speaking":
                self.state = "idle"
            self.talk.release()


def clipboard_text():
    ps = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Clipboard -Raw"
    return subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, encoding="utf-8",
                          timeout=10, creationflags=0x08000000).stdout.strip()


def ctrl_scroll(notches):
    pyautogui.keyDown("ctrl")
    try:
        pyautogui.scroll(notches * (120 if sys.platform == "win32" else 1))
    finally:
        pyautogui.keyUp("ctrl")


def make_detector():
    if not MODEL.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.6, min_tracking_confidence=0.6))


# --- First run: one checklist for everything ZEHEN needs. What is already on this PC is skipped. ---

SETUP_MODULES = {"vosk": "vosk", "sounddevice": "sounddevice", "faster_whisper": "faster-whisper",
                 "kokoro_onnx": "kokoro-onnx", "uiautomation": "uiautomation"}


def _missing_packages():
    import importlib.util
    return [pip for mod, pip in SETUP_MODULES.items() if importlib.util.find_spec(mod) is None]


def _install_packages(note):
    if getattr(sys, "frozen", False):
        raise RuntimeError("these are bundled in the .exe")
    miss = _missing_packages()
    note("Installing " + ", ".join(miss))
    if subprocess.run([sys.executable, "-m", "pip", "install", *miss], capture_output=True,
                      creationflags=0x08000000).returncode:
        raise RuntimeError("pip couldn't install " + ", ".join(miss))


def _fetch_all(base, folder, files, label):
    """Run-function for a model made of several files: only the missing ones are downloaded."""
    def run(note):
        folder.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files):
            if not (folder / f).exists():
                fetch(base + f, folder / f, progress=lambda d, n, i=i: note(
                    f"{label} {d / 1e6:.0f}/{n / 1e6:.0f} MB", (i + (d / n if n else 0)) / len(files)))
    return run


def setup_items(cfg):
    """The checklist, in order. check(): already here? skip(): not needed right now. run(note): get it."""
    name = cfg.get("stt_model", "base.en")
    whisper_dir = DATA / ("whisper-" + name)
    items = [
        dict(key="packages", name="Python packages", core=True, check=lambda: not _missing_packages(),
             run=_install_packages),
        dict(key="hands", name="Hand tracking model", core=False, check=MODEL.exists,
             run=lambda note: fetch(MODEL_URL, MODEL, progress=lambda d, n: note(
                 f"Hand model {d / 1e6:.0f}/{n / 1e6:.0f} MB", d / n if n else None)))]
    if cfg.get("voice", True):
        items += [
            dict(key="wake", name="Wake word (Vosk)", core=True, check=VOSK_DIR.exists,
                 run=lambda note: fetch(VOSK_URL, VOSK_DIR, unzip=True, progress=lambda d, n: note(
                     f"Wake word {d / 1e6:.0f}/{n / 1e6:.0f} MB", d / n if n else None))),
            dict(key="voice", name="Her voice (Kokoro)", core=True,
                 check=lambda: all((DATA / f).exists() for f in KOKORO_FILES),
                 run=_fetch_all(KOKORO_URL, DATA, KOKORO_FILES, "Voice")),
            dict(key="hearing", name="Hearing (Whisper)", core=True,
                 check=lambda: all((whisper_dir / f).exists() for f in WHISPER_FILES),
                 run=_fetch_all(WHISPER_URL % name, whisper_dir, WHISPER_FILES, "Hearing")),
            dict(key="brain", name="Offline brain (Qwen3)", core=False,
                 skip=lambda: "using your cloud key" if llm_endpoint(cfg) else "",
                 check=lambda: LOCAL_FILE.exists() and _llama_ready(),
                 run=lambda note: local_llm(note, load=False))]
    return items


def _llama_ready():
    import importlib.util
    return importlib.util.find_spec("llama_cpp") is not None or bool(getattr(sys, "frozen", False))


class Setup:
    """Runs the checklist on a thread. State per item: pending, working, found (already here), skipped, done, failed.
    Anything missing is fetched; anything present is left alone, so a fork's first run gets everything and every later
    run takes a blink. A failure (offline, disk full) is shown and retried next launch."""

    def __init__(self, cfg, items=None):
        self.items = items if items is not None else setup_items(cfg)
        for it in self.items:
            it.setdefault("skip", lambda: "")
            it.update(state="pending", frac=None, msg="", finished=threading.Event())
        self.core, self.done, self.running = threading.Event(), threading.Event(), False

    def start(self):
        if self.running:
            return
        self.running = True
        self.done.clear()
        for it in self.items:
            it["finished"].clear()
            it["state"], it["msg"] = "pending", ""
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        for it in self.items:
            it["state"] = "working"
            try:
                if why := it["skip"]():
                    it["state"], it["msg"] = "skipped", why
                elif it["check"]():
                    it["state"], it["msg"] = "found", "already here"
                else:
                    it["run"](lambda msg, frac=None, it=it: it.update(msg=msg, frac=frac))
                    it["state"], it["msg"] = "done", ""
            except Exception as e:
                it["state"], it["msg"] = "failed", (str(e) or type(e).__name__)[:70]
            it["frac"] = None
            it["finished"].set()
            if all(i["finished"].is_set() for i in self.items if i.get("core")):
                self.core.set()
        self.core.set()
        self.running = False
        self.done.set()

    def item(self, key):
        return next((i for i in self.items if i["key"] == key), None)

    def progress(self):
        """(finished, total, the item being worked on or None)."""
        total = len(self.items)
        return (sum(i["state"] in ("found", "skipped", "done", "failed") for i in self.items), total,
                next((i for i in self.items if i["state"] == "working"), None))


class Tracker:
    """Camera + gestures on a background thread. The UI only reads self.state, a fresh snapshot per frame.
    Main hand (right, or whichever is alone) drives the cursor; the other hand zooms and switches apps."""

    def __init__(self, cam, cfg, start=True):
        self.cam, self.cfg = cam, cfg
        self.running, self.paused, self.error, self.mouse_down = True, False, None, False
        self.lock = threading.Lock()  # guards self.ink: the recognizer thread trims it while we draw into it
        self.mode, self.search = None, False  # mode: None | "launcher" | "pad" | "keyboard"
        self.reply, self.reply_at, self.dl = "", -99.0, None  # voice: what it last said, or what went wrong, and when
        self.ink, self.stroke, self.ink_at, self.recognizing = [], None, 0.0, False
        self.chunks, self.query, self.alts = [], "", []  # chunks: text the pad typed, so "⌫ word" can undo it
        self.windows, self.apps, self.search_cache = [], [], ("", [])
        self.hover, self.pressed, self.pressed_until = None, None, 0.0
        self.flash, self.flash_until = "", 0.0
        self.pinch2, self.zoom_d0, self.zoom_since = False, None, 0.0
        self.swipe_hist, self.swipe_at, self.swipe_cool = [], -9.0, 0.0
        self.last_set, self.mouse_free_at = None, 0.0  # where we last put the cursor; when we may again
        self.tap_at, self.tap_pos, self.double_at = -9.0, None, -9.0  # last quick click: double-pinch detection
        self.ui, self.aspect = panel(), 16 / 9
        self._publish(0.0, [], None, None, 0.0, None, 0.0, False)
        self.setup = Setup(cfg) if start else None  # first run: fetch whatever is missing, skip what is here
        if start:
            threading.Thread(target=memory_keeper, daemon=True).start()
        self.memory, self.camera_on = [], cfg.get("camera_on", True)  # memory: the last few things you asked
        self.chatty = False  # the last request was a conversation (no actions): she keeps listening for your reply
        self.dict_until, self.last_typed = 0.0, ""  # dictation: what you say is typed, until you say stop
        self.voice = (Voice(cfg, self._agent, lambda: self.paused, self._say, lambda: self.chatty, self._hints,
                      self.dictating, self.setup)
                      if start and cfg.get("voice", True) else None)
        if start:
            self.setup.start()
            if self.voice and sys.platform == "win32":
                self.voice.start()
            threading.Thread(target=self._load_apps, daemon=True).start()
            threading.Thread(target=self._run, daemon=True).start()

    def _load_apps(self):
        cache = DATA / "apps.json"  # last time's list is there at once; today's replaces it a few seconds later
        try:
            self.apps = [tuple(a) for a in json.loads(cache.read_text(encoding="utf-8"))]
        except (OSError, ValueError):
            pass
        try:
            self.apps = installed_apps()
            DATA.mkdir(exist_ok=True)
            cache.write_text(json.dumps(self.apps), encoding="utf-8")
        except Exception:
            pass  # launcher still has windows, controls and the writing pad

    def _run(self):
        try:
            self._loop()
        except Exception as e:
            self.error = str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}"
        finally:
            self.running = False
            if self.mouse_down:
                pyautogui.mouseUp()  # never leave the button stuck

    def _release(self):
        if self.mouse_down:
            pyautogui.mouseUp()
            self.mouse_down = False

    def _flash(self, text, secs=1.2):
        self.flash, self.flash_until = text, time.monotonic() + secs

    def _loop(self):
        if self.setup is not None:
            self.setup.item("hands")["finished"].wait(180)  # its file may still be arriving on a first run
        det = make_detector()  # released while the camera is off, loaded again when it is back
        sw, sh = pyautogui.size()
        fx, fy = OneEuro(), OneEuro()
        pfx, pfy = OneEuro(INK_MIN_CUTOFF, INK_BETA), OneEuro(INK_MIN_CUTOFF, INK_BETA)
        self.aspect = sw / sh
        last_timestamp_ms = -1
        raw, count, gesture, since, fired = None, 0, None, 0.0, False
        lock_until, scroll_y0, main_label = 0.0, None, None
        m = lambda v: min(max((v - MARGIN) / (1 - 2 * MARGIN), 0.0), 1.0)
        try:
            while self.running:
                now = time.monotonic()
                if not self.camera_on:  # voice only: release the camera so its light goes out
                    if self.cam is not None:
                        self.cam.release()
                        self.cam = None
                        if det is not None:
                            det.close()
                            det = None
                    self._release()
                    self._end_stroke(now)
                    raw, count, gesture, scroll_y0, self.zoom_d0, self.hover = None, 0, None, None, None, None
                    self.ui = self._layout()
                    self._publish(now, [], None, None, 0.0, None, now, False)
                    time.sleep(0.1)
                    continue
                if det is None:
                    det = make_detector()
                if self.cam is None:
                    self.cam = open_camera(self.cfg.get("camera", 0))
                    if self.cam is None:
                        self.camera_on = False  # couldn't reopen it: stay voice only rather than crash
                        self._say("Camera won't open. Staying voice only.")
                        continue
                ok, frame = self.cam.read()
                if not ok:
                    raise RuntimeError(CAMERA_HELP)
                rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)  # mirror: move right = cursor right
                timestamp_ms = max(int(now * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                # MediaPipe labels handedness assuming a mirrored image, which we give it, so labels are real
                main = self.cfg.get("main_hand", "Right")
                hands = [] if self.paused else sorted(
                    zip((h[0].category_name for h in res.handedness), res.hand_landmarks), key=lambda h: h[0] != main)
                self.ui = self._layout()
                if not hands:
                    self._release()  # hand left mid-drag: never leave the button stuck
                    self._end_stroke(now)
                    raw, count, gesture, scroll_y0, self.zoom_d0, self.hover = None, 0, None, None, None, None
                    self._publish(now, [], None, None, 0.0, None, now, False)
                    continue
                (label, lm), lm2 = hands[0], (hands[1][1] if len(hands) > 1 else None)
                if label != main_label:  # the other hand took over the cursor: don't smooth across hands
                    main_label, fx, fy = label, OneEuro(), OneEuro()
                    pfx, pfy = OneEuro(INK_MIN_CUTOFF, INK_BETA), OneEuro(INK_MIN_CUTOFF, INK_BETA)

                # pointer = index knuckle: it stays put while pinching, unlike the fingertip
                x, y = fx(m(lm[5].x) * sw, now), fy(m(lm[5].y) * sh, now)
                # the pen is where a pen would be: between the pinched thumb and index tips
                pen = pfx(m((lm[4].x + lm[8].x) / 2) * sw, now), pfy(m((lm[4].y + lm[8].y) / 2) * sh, now)
                if self.mode == "pad":
                    x, y = pen
                nx, ny = x / sw, y / sh
                self.hover = next((b.id for b in self.ui.buttons if inside(b.r, nx, ny)), None)
                g2 = classify(lm2, self.pinch2) if lm2 is not None else None
                self.pinch2 = g2 == "pinch"

                g = classify(lm, raw == "pinch")
                count, raw = (count + 1 if g == raw else 1), g
                if count >= STABLE_FRAMES and g != gesture:  # gesture changed (debounced)
                    prev, held = gesture, now - since
                    gesture, since, fired = g, now, False
                    if prev == "pinch":
                        if self.mouse_down and held < TAP_S:  # a quick click: a second one soon means "open"
                            self.tap_at = now
                        self._release()
                        self._end_stroke(now)
                    if g == "pinch":
                        self._pinch_start(nx, ny, both_hands=self.pinch2)
                        lock_until = now + CLICK_LOCK_S
                    elif g == "rpinch" and not self.mode:
                        pyautogui.rightClick()

                progress = 0.0
                if gesture in ("palm", "three") and not fired:
                    progress = min((now - since) / HOLD_S, 1.0)
                    if progress >= 1:
                        fired = True
                        if gesture == "palm":
                            self.set_mode(None if self.mode else "launcher")
                        else:
                            self.set_mode(None if self.mode == "pad" else "pad")

                if self.stroke is not None and gesture == "pinch":  # writing: extend the ink line
                    lx, ly = self.stroke[-1]
                    if abs(nx - lx) + abs(ny - ly) > 0.002:
                        self.stroke.append((nx, ny))

                zoom = not self.mode and gesture == "pinch" and self.pinch2
                if zoom:
                    self._release()  # a two-hand pinch is a zoom, never a drag
                    d = math.hypot(lm[5].x - lm2[5].x, lm[5].y - lm2[5].y)
                    if self.zoom_d0 is None:
                        self.zoom_d0, self.zoom_since = d, now
                    elif abs(d / self.zoom_d0 - 1) > ZOOM_STEP:
                        ctrl_scroll(1 if d > self.zoom_d0 else -1)
                        self.zoom_d0 = d
                else:
                    self.zoom_d0 = None

                if lm2 is not None and g2 == "palm" and not zoom:  # other hand waves sideways -> switch app
                    self.swipe_hist = [(t, sx) for t, sx in self.swipe_hist if now - t < SWIPE_S] + [(now, lm2[9].x)]
                    if now > self.swipe_cool and abs(self.swipe_hist[-1][1] - self.swipe_hist[0][1]) > SWIPE_DIST:
                        pyautogui.hotkey("alt", "tab")
                        self.swipe_at, self.swipe_cool, self.swipe_hist = now, now + SWIPE_COOLDOWN, []
                        self._flash("⇄  SWITCH APP", 0.8)
                else:
                    self.swipe_hist = []

                if (not self._mouse_taken(now) and not self.mode and not zoom and gesture in ("point", "pinch")
                        and now >= lock_until and now >= self.tap_at + DOUBLE_TAP_S):
                    # (after a quick click the cursor holds still, so a second pinch opens the same thing)
                    self.last_set = (int(min(max(x, 0), sw - 1)), int(min(max(y, 0), sh - 1)))
                    pyautogui.moveTo(*self.last_set)
                else:
                    self.last_set = None
                if gesture == "scroll" and not self.mode:
                    scroll_y0 = lm[5].y if scroll_y0 is None else scroll_y0
                    off = scroll_y0 - lm[5].y  # joystick: distance from where the V started
                    if abs(off) > SCROLL_DEADZONE:
                        pyautogui.scroll(int((off - math.copysign(SCROLL_DEADZONE, off)) * SCROLL_SPEED))
                else:
                    scroll_y0 = None

                if (self.mode == "pad" and self.ink and self.stroke is None and not self.recognizing
                        and now - self.ink_at > INK_PAUSE_S):
                    self.recognizing = True
                    threading.Thread(target=self._recognize, args=([list(s) for s in self.ink],), daemon=True).start()

                ptr2 = (m(lm2[5].x), m(lm2[5].y)) if lm2 is not None else None
                self._publish(now, [lm] + ([lm2] if lm2 is not None else []), (nx, ny), ptr2, progress,
                              gesture, since, zoom)
        finally:
            if self.cam is not None:
                self.cam.release()

    def _publish(self, now, hands, ptr, ptr2, progress, gesture, since, zoom):
        active, active_since = gesture, since
        if zoom:
            active, active_since = "zoom", self.zoom_since
        if now - self.swipe_at < 0.6:
            active, active_since = "swipe", self.swipe_at
        if now - self.double_at < 0.6:
            active, active_since = "double", self.double_at
        with self.lock:
            ink = [list(s) for s in self.ink]
        self.state = SimpleNamespace(
            hands=[[(p.x, p.y) for p in lm] for lm in hands], gesture=gesture, active=active, since=active_since,
            ptr=ptr, ptr2=ptr2, progress=progress, zoom=zoom, mode=self.mode, ui=self.ui, hover=self.hover,
            pressed=self.pressed if now < self.pressed_until else None, ink=ink,
            flash=self.flash if now < self.flash_until else "")

    # --- panels: launcher, writing pad, keyboard ---

    def set_mode(self, mode, search=False):
        self._release()
        with self.lock:
            self.ink, self.stroke = [], None
        self.mode, self.search, self.query, self.alts, self.chunks = mode, search, "", [], []
        if mode == "launcher":
            self.windows = list_windows()

    def _pinch_start(self, nx, ny, both_hands):
        if self.mode:
            b = next((b for b in self.ui.buttons if b.id == self.hover), None)
            if b:
                self.pressed, self.pressed_until = b.id, time.monotonic() + 0.3
                b.action()
            elif self.ui.pad and inside(self.ui.pad, nx, ny):
                with self.lock:
                    self.stroke = [(nx, ny)]
                    self.ink.append(self.stroke)
        elif not both_hands and time.monotonic() >= self.mouse_free_at:
            now = time.monotonic()
            if now - self.tap_at < DOUBLE_TAP_S:
                # Windows only counts a double-click if both land within ~4 px and 0.5 s; hand clicks miss that,
                # so send a real one at exactly where the first click went
                pyautogui.doubleClick(*self.tap_pos)
                self.tap_at, self.double_at = -9.0, now
                self._flash("↗  OPEN", 0.6)
                return
            self.tap_pos = pyautogui.position()
            pyautogui.mouseDown()
            self.mouse_down = True

    def _mouse_taken(self, now):
        """True while you're using the real mouse or touchpad: the cursor isn't where we left it last frame,
        so ZEHEN keeps its hands off it for a moment instead of fighting you."""
        if self.last_set is not None:
            cx, cy = pyautogui.position()
            if abs(cx - self.last_set[0]) + abs(cy - self.last_set[1]) > MOUSE_TAKEOVER_PX:
                self._release()
                self.last_set, self.mouse_free_at = None, now + MOUSE_TAKEOVER_S
                self._flash("🖱  MOUSE IN USE", MOUSE_TAKEOVER_S)
        return now < self.mouse_free_at

    def _end_stroke(self, now):
        if self.stroke is not None:
            with self.lock:  # swap in the cleaned-up stroke: what you see is what gets read
                if self.ink and self.ink[-1] is self.stroke:
                    self.ink[-1] = beautify(self.stroke, self.aspect)
            self.stroke, self.ink_at = None, now

    def _recognize(self, strokes):
        try:
            cands = recognize(strokes, *pyautogui.size())
            if cands:
                self._got_text(cands)
        except Exception as e:
            self._flash(f"Handwriting failed: {e}", 3)
        finally:
            with self.lock:
                self.ink = self.ink[len(strokes):]  # keep anything written while we were reading
            self.recognizing = False

    def _got_text(self, cands):
        best, self.alts = cands[0], cands[1:4]
        if self.search:
            self.query = f"{self.query} {best}".strip()
        else:
            gap = " " if self.chunks and self.chunks[-1][-1:].isalnum() and best[:1].isalnum() else ""
            self._type(gap + best)

    def _type(self, text):
        if text.isascii():
            # a small gap per key: at full speed Notepad dropped letters ("look ma" arrived as "look a")
            pyautogui.write(text, interval=0.015)
        else:  # Hindi, accents, emoji: paste it, since keys can't type them
            subprocess.run(["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value $env:ZEHEN_SAY"],
                           env={**os.environ, "ZEHEN_SAY": text}, creationflags=0x08000000)
            pyautogui.hotkey("ctrl", "v")
        self.chunks.append(text)

    def _use_alt(self, alt):
        if self.search:
            self.query = " ".join(self.query.split(" ")[:-1] + [alt])
        elif self.chunks:
            last = self.chunks.pop()
            pyautogui.press("backspace", presses=len(last))
            self._type((" " if last.startswith(" ") else "") + alt)
        self.alts = []

    def _backspace(self):
        if self.search:
            self.query = self.query[:-1]
            return
        pyautogui.press("backspace")
        if self.chunks:
            self.chunks[-1] = self.chunks[-1][:-1]
            if not self.chunks[-1]:
                self.chunks.pop()

    def _delete_word(self):
        if self.search:
            self.query = self.query.rpartition(" ")[0]
        elif self.chunks:
            pyautogui.press("backspace", presses=len(self.chunks.pop()))
        else:
            pyautogui.hotkey("ctrl", "backspace")  # nothing written here yet: delete the app's previous word
        self.alts = []

    def _space(self):
        if self.search:
            self.query += " "
        else:
            self._type(" ")

    def dictating(self):
        return time.monotonic() < self.dict_until

    def _dictate(self, text):
        """While dictating, what you say is typed as you say it (Whisper's punctuation and capitals included). A few
        phrases control it. Returns what to say back, or None when this isn't a dictation matter."""
        t = " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())
        if t in ("stop typing", "stop dictation", "stop dictating", "finish typing", "done typing", "end dictation",
                 "i'm done", "that's all", "stop", "cancel"):
            self.dict_until = 0.0
            return "Done typing."
        if t in ("enter", "press enter", "send", "send it", "send message", "submit", "search", "go", "search it"):
            pyautogui.press("enter")
            self.dict_until = 0.0  # you sent it: the text box is done
            return ""
        if t in ("new line", "next line", "new paragraph"):
            pyautogui.press("enter")
        elif t in ("delete that", "scratch that", "undo that", "remove that"):
            pyautogui.press("backspace", presses=max(len(self.last_typed), 1))
            self.last_typed = ""
        elif t in ("clear", "clear all", "clear everything"):
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("backspace")
        else:
            self._type(text.strip() + " ")
            self.last_typed = text.strip() + " "
        self.dict_until = time.monotonic() + 25
        self._say("")
        return ""

    def _hints(self):
        """Words worth listening for: your apps, the sites she knows, the windows open right now."""
        return [n for n, _ in self.apps[:40]] + [k.title() for k in list(SITES)[:12]] + [
            t[:30] for _, t in list_windows()[:5]]

    def _say(self, text, frac=None):
        self.reply, self.reply_at, self.dl = text, time.monotonic(), frac  # frac: download progress 0..1, else None

    def _agent(self, text):
        """Voice: ask the model, do what it says, show it the output, repeat until done. Runs on the voice thread.
        Pausing (fist or ⏸) stops it between steps. Returns what to say back."""
        if m := re.search(r"\b(?:(?:turn|switch) (on|off) (?:the )?camera|camera (on|off)|(voice only|no camera))\b",
                          fold(text)):
            on = (m[1] or m[2]) == "on"
            self.set_camera(on)
            return "Camera on. Hands are back." if on else "Camera off. Voice only, as requested."
        if self.voice and re.search(r"\b(?:learn|train|teach|retrain|calibrate|recalibrate)\b.*\bvoice\b", fold(text)):
            self.voice.enroll()
            return "Say Zehen, three times, with a short pause between each."
        fresh = load_config()  # the file is the truth: delete the key there and the next request is offline
        if "llm" in fresh:
            self.cfg["llm"] = fresh["llm"]
        else:
            self.cfg.pop("llm", None)
        t = " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())
        if re.fullmatch(r"(?:remove|delete|forget|clear|drop) (?:my |the |your )?(?:api )?key|(?:use|go) (?:the )?(?:offline|local)(?: brain| mode)?", t):
            self.cfg.pop("llm", None)
            save_config(self.cfg)
            return "Key removed. I'm using my offline brain now."
        if t in ("stop", "shut up", "be quiet", "quiet", "silence", "stop talking", "enough", "that's enough", "hush",
                 "never mind", "nevermind", "cancel that", "cancel"):
            if self.voice:
                self.voice.shush()
            return ""
        if t in PLEASANT:  # "thanks", "okay", "cool": no reply, and certainly no chat
            return ""
        if self.dictating() and (r := self._dictate(text)) is not None:
            return r
        if self.setup and re.search(r"\b(?:run|do|check|redo|repeat|finish|start)\b.*\bset ?up\b|\binstall everything\b|"
                                    r"\bdownload (?:everything|what you need)\b", fold(text)):
            self.setup.start()
            return "Checking what I need. I'll only fetch what's missing."
        alts = self.voice.alts if self.voice else []
        for heard in [text, *alts]:  # the other ear may have caught it ("click" for "like"): try every version
            quick = quick_command(heard, self.apps)
            if quick:
                text = heard
                break
        if not quick and len(text.split()) < 2:
            return ""  # one stray word ("hand") is noise, not a request: she does not make something up
        if quick and not quick[1]:  # a fact she knows for certain (the time): say it, no model to make one up
            self._say(quick[0])
            return quick[0]
        if quick:  # no model, no wait: scroll, click, go back... happen at once and quietly
            self._say(quick[0])
            out = [o for o in (self._do(*next(iter(a.items()))) for a in quick[1]) if o]
            self.memory = (self.memory + [f"{text} => {quick[0]}"])[-5:]
            lost = next((o for o in out if o.startswith(("no tab", "no window"))), "")
            if lost:  # tell her what is there, so you can say it again properly
                return lost.replace("no tab like", "I can't find a tab like").replace("no window or tab like", "I can't find")
            if any(o.startswith(("ready", "clicked text box")) for o in out):  # a text box has the cursor: speak now
                self.dict_until = time.monotonic() + 25
                return "Ready to type."
            playing = next((o for o in out if o.startswith("playing")), "")
            if playing:
                self._say(playing.capitalize() + ".")
                return ""
            if any(o.startswith("no video") for o in out):
                return "I can't find that video here."
            if any(o.startswith("no text box") for o in out):
                return "I can't find a text box here."
            bad = next((o for o in out if o.startswith("no '")), "")
            if any("NOTFOUND" in o for o in out):
                return "Jupyter isn't installed on this PC. Say: install Jupyter, and I'll add it."
            return ("I can't find that here." if bad else "")
        windows = "; ".join(t[:40] for _, t in list_windows()[:10]) or "none"
        earlier = "".join(f"Earlier: {e}\n" for e in self.memory)  # so "now click the first one" makes sense
        msgs = [{"role": "system", "content": ASK_SYSTEM},
                {"role": "user", "content": f"Installed apps: {', '.join(n for n, _ in self.apps[:80])}\n"
                                            f"Open windows: {windows}\n{earlier}"
                                            + (f"Another transcript of the same audio: {alts[0]}\n" if alts else "")
                                            + f"Request (from speech recognition): {text}"}]
        self._say("")
        say, acts, failed, worked = "", [], [], []
        self.chatty = False
        try:
            for _ in range(MAX_STEPS):
                reply = chat(self.cfg, msgs, self._say)
                say, acts, done = parse_plan(reply)
                out = []
                for a in acts:
                    kind, v = next(iter(a.items()))
                    self._say(self._describe(kind, v))  # shows in the pill while it works: what she is doing right now
                    if (o := self._do(kind, v)):
                        out.append(o)
                        if o.startswith("no "):
                            failed.append(o)
                        elif o.startswith(("opened", "switched", "playing", "clicked")):
                            worked.append(o)
                if done or not out or self.paused:
                    break
                msgs += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": "Output:\n" + "\n".join(out)}]
        except RuntimeError as e:
            say = str(e)
        except Exception as e:
            say = f"{type(e).__name__}: {e}"
        say = " ".join(re.split(r"(?<=[.!?]) ", say)[:2])  # two sentences at most: she talks to be useful, not to talk
        say = re.sub(r"\s+", " ", re.sub(r"[^\w\s.,!?'\-:;()%/&+@#$]", "", say)).strip()  # no emoji or markdown to read aloud
        if failed:  # it didn't work: say so, whatever the model claimed
            what = failed[0].split("'")[1] if "'" in failed[0] else "that"
            say = f"I couldn't find {what}."
        elif acts and say.lower().strip(" .!") in ("", "done", "ok", "okay"):
            say = (worked[-1][:1].upper() + worked[-1][1:] if worked else  # what really happened, not "Done"
                   self._describe(*next(iter(acts[-1].items())), past=True)) + "."
        elif not acts and say.lower().strip(" .!") in ("", "done", "ok", "okay", "sure", "alright"):
            say = ""  # nothing to do and nothing to say: stay quiet rather than make something up
        self.chatty = not acts and bool(say)
        self._say(say)  # also clears any download bar
        self.memory = (self.memory + [f"{text} => {self.reply}"])[-5:]
        return self.reply

    @staticmethod
    def _describe(kind, v, past=False):
        """One honest phrase for an action: "Opening Spotify" while it runs, "Opened Spotify" when it has."""
        host = urllib.parse.urlparse(v).netloc.removeprefix("www.") if kind == "open_url" else v
        now, then = {"open_app": ("Opening {}", "Opened {}"), "focus_window": ("Switching to {}", "Switched to {}"),
                     "open_url": ("Opening {}", "Opened {}"), "search_web": ("Searching for {}", "Searched for {}"),
                     "type": ("Typing", "Typed it"), "press": ("Pressing {}", "Pressed {}"),
                     "powershell": ("Running a command", "Ran the command"), "click": ("Clicking {}", "Clicked {}"),
                     "play": ("Finding the video", "Playing it"), "tab": ("Switching tabs", "Switched tabs"),
                     "switch": ("Switching to {}", "Switched to {}"), "scroll": ("Scrolling", "Scrolled"),
                     "open_folder": ("Opening {}", "Opened {}"), "ui_list": ("Looking at the screen", "Looked"),
                     "wait": ("Waiting", "Waited"), "volume": ("Volume {}", "Volume {}"), "window": ("Window {}", "Window {}"),
                     "media": ("Media", "Media"), "mouse": ("Clicking", "Clicked"), "zoom": ("Zooming", "Zoomed"),
                     "focus_input": ("Finding the text box", "Ready to type"), "dictate": ("Ready to type", "Ready to type")
                     }.get(kind, ("Working", "Done"))
        return (then if past else now).format(host[:40])

    def set_camera(self, on):
        """Camera off = voice only: the sensor really stops (its light goes out) and hands are ignored."""
        self.camera_on = on
        self.cfg["camera_on"] = on
        save_config(self.cfg)

    def _do(self, kind, v):
        """Run one model-chosen action; powershell returns its output so the model can see it."""
        if kind == "open_app":
            if m := search_apps(self.apps, v, 1):
                launch(m[0][1])
                return "opened " + m[0][0]
            return f"no app like '{v}'"
        elif kind == "focus_window":
            if m := next((h for h, t in list_windows() if fold(v) in fold(t)), None):
                focus_window(m)
                return "switched to " + v
            return f"no window like '{v}'"
        elif kind == "open_url":
            if v.startswith(("http://", "https://")):
                webbrowser.open(v)
                return "opened " + urllib.parse.urlparse(v).netloc
            return f"no link like '{v}'"
        elif kind == "search_web":
            webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote(v))
        elif kind == "type":
            self._type(v)
        elif kind == "press":
            keys = [k.strip().lower() for k in v.split("+")]
            if all(k in pyautogui.KEYBOARD_KEYS for k in keys):
                pyautogui.hotkey(*keys)
        elif kind == "window" and v in ("max", "min", "close"):
            window_cmd(v)
        elif kind == "volume" and v in ("up", "down"):
            pyautogui.press("volume" + v, presses=5)
        elif kind == "media" and v in ("playpause", "nexttrack", "prevtrack"):
            pyautogui.press(v)
        elif kind == "powershell":
            return run_powershell(v)
        elif kind == "scroll":
            d, _, n = v.partition(" ")
            if d in ("top", "bottom"):
                pyautogui.press("home" if d == "top" else "end")
            elif d in ("up", "down"):
                pyautogui.scroll((1 if d == "up" else -1) * int(n or 6) * (120 if sys.platform == "win32" else 1))
        elif kind == "mouse":
            {"click": pyautogui.click, "double": pyautogui.doubleClick, "right": pyautogui.rightClick}.get(
                v, lambda: None)()
        elif kind == "zoom":
            ctrl_scroll(3 if v == "in" else -3)
        elif kind == "play":
            return play_video(v)
        elif kind == "focus_input":
            return focus_input(v)
        elif kind == "dictate":
            self.dict_until = time.monotonic() + 25 if v == "on" else 0.0
            return "ready to type"
        elif kind == "tab":
            return switch_tab(v)
        elif kind == "switch":
            return switch_to(v)
        elif kind == "open_folder" and v in FOLDERS:
            os.startfile(FOLDERS[v])
        elif kind == "wait":
            time.sleep(min(float(v) if v.replace(".", "", 1).isdigit() else 1.0, 5.0))
        elif kind == "ui_list":
            return ui_list()
        elif kind == "click":
            return ui_click(v)

    def _paste_key(self):
        key = clipboard_text()
        if len(key) < 20 or " " in key or "\n" in key:
            self._say("Copy your key first, then say: paste my key")
            return
        self._say("Checking the key…")
        res = verify_key(key)
        if res["ok"]:
            self.cfg["llm"] = {"api_key": key, "provider": res["provider"], "verified": True}
            save_config(self.cfg)
        self._say(res["message"])

    def _enter(self):
        if self.search:
            if results := self._results():
                self._launch(*results[0])
        else:
            pyautogui.press("enter")
            self.chunks = []

    def _clear_ink(self):
        with self.lock:
            self.ink, self.stroke = [], None

    def _results(self):
        if self.search_cache[0] != self.query:
            self.search_cache = (self.query, search_apps(self.apps, self.query))
        return self.search_cache[1]

    def _launch(self, name, app_id):
        launch(app_id)
        self._flash(f"▶  {name.upper()}")
        self.set_mode(None)

    def _then_close(self, fn):
        return lambda: (fn(), self.set_mode(None))

    def _key(self, k):
        if k == "✕":
            self.set_mode(None)
        else:
            pyautogui.press(SPECIAL.get(k, k.lower()))

    def _layout(self):
        if self.mode == "launcher":
            return self._launcher()
        if self.mode == "pad":
            return self._pad()
        if self.mode == "keyboard":
            return panel(buttons=[btn(k, k, *r, lambda k=k: self._key(k), 18 if len(k) == 1 else 13)
                                  for k, *r in KEYS])
        return panel()

    def _launcher(self):
        X0, X1, COLS, TH, GAP = 0.12, 0.88, 6, 0.1, 0.008
        tw = (X1 - X0) / COLS
        texts = [("◉ ZEHEN", X0, 0.105, 22, "cyan", "w"),
                 ("point and pinch to tap · hold open palm to close", X1, 0.105, 12, "dim", "e")]
        buttons, y = [], 0.145

        def section(title, items, empty):
            nonlocal y
            texts.append((title, X0, y, 12, "dim", "w"))
            y += 0.02
            if not items:
                texts.append((empty, X0, y + 0.03, 13, "white", "w"))
                y += 0.07
                return
            for i, (label, action) in enumerate(items):
                r, c = divmod(i, COLS)
                top = y + r * (TH + GAP)
                buttons.append(btn(f"{title}{i}", label, X0 + c * tw + 0.003, top, X0 + (c + 1) * tw - 0.003,
                                   top + TH, action, 13))
            y += math.ceil(len(items) / COLS) * (TH + GAP) + 0.025

        section("OPEN WINDOWS", [(window_label(t), self._then_close(lambda h=h: focus_window(h)))
                                 for h, t in self.windows[:COLS]], "No open windows")
        names = {fold(n): (n, a) for n, a in self.apps}
        favs = [names[fold(n)] for n in self.cfg.get("apps", DEFAULT_APPS) if fold(n) in names][:COLS - 2]
        section("APPS", [(f"▶\n{n}", lambda n=n, a=a: self._launch(n, a)) for n, a in favs]
                + [("🔍\nSearch all apps", lambda: self.set_mode("pad", search=True)),
                   ("🎙\nVoice", self._then_close(lambda: self.voice and self.voice.listen_now()))], "")
        section("CONTROLS", [
            ("⌨\nKeyboard", lambda: self.set_mode("keyboard")),
            ("✍\nWriting pad", lambda: self.set_mode("pad")),
            ("📂\nFiles", self._then_close(open_folder)),
            ("🖥\nDesktop", self._then_close(lambda: pyautogui.hotkey("win", "d"))),
            ("🗂\nTask view", self._then_close(lambda: pyautogui.hotkey("win", "tab"))),
            ("📸\nScreenshot", self._then_close(lambda: pyautogui.hotkey("win", "shift", "s"))),
            ("⬜\nMaximize", self._then_close(lambda: window_cmd("max"))),
            ("➖\nMinimize", self._then_close(lambda: window_cmd("min"))),
            ("✖\nClose window", self._then_close(lambda: window_cmd("close"))),
            ("🔉\nVolume −", lambda: pyautogui.press("volumedown", presses=5)),  # stays open: tap again
            ("🔊\nVolume +", lambda: pyautogui.press("volumeup", presses=5)),
            ("⏯\nPlay / Pause", lambda: pyautogui.press("playpause"))], "")
        return panel(boxes=[(0.1, 0.07, 0.9, y)], texts=texts, buttons=buttons)

    def _pad(self):
        L, R, T, B = 0.2, 0.8, 0.21, 0.62
        texts = [("🔍 SEARCH APPS" if self.search else "✍ WRITING PAD", L, 0.105, 20, "cyan", "w"),
                 ("pinch and write in the box · pause and it turns into text", R, 0.105, 12, "dim", "e"),
                 (f"🔍  {self.query}▏" if self.search else "typed:  " + "".join(self.chunks)[-44:],
                  L, 0.165, 16, "white", "w")]
        buttons = []
        if self.recognizing:
            texts.append(("reading your handwriting…", R, 0.165, 13, "green", "e"))
        elif self.alts:  # "did you mean" chips: tap one to swap the last word
            for i, a in enumerate(reversed(self.alts)):
                buttons.append(btn(f"alt{i}", a, R - (i + 1) * 0.095 + 0.003, 0.145, R - i * 0.095 - 0.003,
                                   0.188, lambda a=a: self._use_alt(a), 13))
        keys = [("⌫", self._backspace), ("⌫ word", self._delete_word), ("␣ space", self._space),
                ("↵ enter", self._enter), ("clear ink", self._clear_ink), ("✕ close", lambda: self.set_mode(None))]
        kw = (R - L) / len(keys)
        buttons += [btn(f"pad{i}", label, L + i * kw + 0.003, B + 0.015, L + (i + 1) * kw - 0.003, B + 0.085, fn, 16)
                    for i, (label, fn) in enumerate(keys)]
        bottom = B + 0.1
        if self.search:
            results = self._results()
            texts.append(("matches · tap to open" if results else "write an app name in the box",
                          L, bottom + 0.012, 12, "dim", "w"))
            rw = (R - L) / 5
            buttons += [btn(f"app{i}", f"▶\n{n}", L + i * rw + 0.003, bottom + 0.03, L + (i + 1) * rw - 0.003,
                            bottom + 0.13, lambda n=n, a=a: self._launch(n, a), 13)
                        for i, (n, a) in enumerate(results)]
            bottom += 0.145
        return panel(boxes=[(L - 0.02, 0.07, R + 0.02, bottom)], texts=texts, buttons=buttons, pad=(L, T, R, B))


def click_through(win, clicks=False):
    """Windows: keep the window out of the taskbar and from stealing focus (typing must reach your app).
    clicks=False also lets mouse clicks fall through to whatever is underneath."""
    if sys.platform != "win32":
        return  # ponytail: overlay blocks clicks on macOS/Linux; needs native window flags there
    u = ctypes.windll.user32
    win.update_idletasks()
    hwnd = u.GetParent(win.winfo_id())
    style = u.GetWindowLongW(hwnd, -20) | 0x80000 | 0x80 | 0x08000000  # LAYERED | TOOLWINDOW | NOACTIVATE
    u.SetWindowLongW(hwnd, -20, style if clicks else style | 0x20)     # | TRANSPARENT (click-through)


def glass(win, tint):
    """Windows: frosted-acrylic blur behind the window (tint is 0xAABBGGRR) and rounded corners.
    Pixels drawn in the window's transparent colour show the blur. Returns False where Windows can't."""
    if sys.platform != "win32":
        return False
    win.update_idletasks()
    hwnd = u32.GetParent(win.winfo_id())

    class Accent(ctypes.Structure):
        _fields_ = [("state", ctypes.c_int), ("flags", ctypes.c_int), ("tint", ctypes.c_uint), ("anim", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [("attr", ctypes.c_int), ("data", ctypes.c_void_p), ("size", ctypes.c_size_t)]
    u32.SetWindowCompositionAttribute.argtypes = [wintypes.HWND, ctypes.c_void_p]
    dwm.DwmSetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    dwm.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(ctypes.c_int(2)), 4)  # DWMWA_WINDOW_CORNER_PREFERENCE: round (Win 11)
    accent = Accent(4, 2, tint, 0)  # 4 = ACCENT_ENABLE_ACRYLICBLURBEHIND
    data = Data(19, ctypes.cast(ctypes.byref(accent), ctypes.c_void_p), ctypes.sizeof(accent))  # 19 = WCA_ACCENT_POLICY
    return bool(u32.SetWindowCompositionAttribute(hwnd, ctypes.byref(data)))


class HUD:
    def __init__(self, root, tracker, cfg):
        self.root, self.t, self.cfg = root, tracker, cfg
        self.s = root.winfo_fpixels("1i") / 96  # Windows display scaling (125% -> 1.25)
        self.W, self.H = root.winfo_screenwidth(), root.winfo_screenheight()
        self.tut = None if cfg.get("tutorial_done") else 0
        self.great_until = self.step_start = 0.0
        self.open, self.armed, self.over_since, self.left_at = False, True, None, None  # side panel: pill or full
        self.cw = self.ch = None  # the panel's current (eased) size
        self.bars, self.pill_bottom = [3.0] * 5, 0  # the wave's bar heights; where the pill ends (captions go below)
        self.spoken, self.praise = None, False  # which tutorial step she last spoke

        root.overrideredirect(True)
        root.geometry(f"{self.W}x{self.H}+0+0")
        root.attributes("-topmost", True)
        root.attributes("-transparentcolor", KEY)
        root.config(bg=KEY)
        self.c = tk.Canvas(root, bg=KEY, highlightthickness=0)
        self.c.pack(fill="both", expand=True)
        click_through(root)

        self.panel = tk.Toplevel(root)
        self.panel.overrideredirect(True)
        self.panel.attributes("-topmost", True)
        self.pc = tk.Canvas(self.panel, bg=KEY, highlightthickness=0)
        self.pc.pack(fill="both", expand=True)
        for tag, fn in (("btn_pause", self.toggle_pause), ("btn_min", self.toggle_open),
                        ("btn_quit", self.quit), ("btn_tut", self.restart_tutorial), ("btn_skip", self.skip_tutorial),
                        ("btn_voice", lambda: self.t.voice and self.t.voice.listen_now()),
                        ("btn_cam", lambda: self.t.set_camera(not self.t.camera_on)),
                        ("btn_addkey", lambda: self.open_key_dialog()),
                        ("btn_getkey", lambda: webbrowser.open(KEY_PAGE)),
                        ("btn_pastekey", lambda: threading.Thread(target=self.t._paste_key, daemon=True).start())):
            self.pc.tag_bind(tag, "<Button-1>", lambda e, fn=fn: fn())
        click_through(self.panel, clicks=True)
        self.frosted = glass(self.panel, 0x661A1208)  # tint = BG at ~40%: the desktop shows through, blurred
        if self.frosted:
            self.panel.attributes("-transparentcolor", KEY)  # undrawn pixels become the blurred glass
        else:
            self.pc.config(bg=BG, highlightthickness=1, highlightbackground=DIM)  # older Windows: solid panel
        self.tick()

    def px(self, v):
        return int(v * self.s)

    def font(self, size, bold=False):
        return ("Segoe UI", self.px(size) * -1, "bold" if bold else "normal")  # negative = pixels

    # --- controls ---
    def toggle_pause(self):
        self.t.paused = not self.t.paused

    def toggle_open(self):
        self.open = not self.open
        self.armed = False  # pointing at the pill that just closed must not reopen it: leave first

    def open_key_dialog(self):
        """A small window with a box to paste any AI key into, a Verify button that really asks the service, and
        a way to remove the key again. Any OpenAI-compatible service works: pick Custom and give its address."""
        from tkinter import ttk
        if getattr(self, "keywin", None) is not None and self.keywin.winfo_exists():
            return self.keywin.lift()
        win = self.keywin = tk.Toplevel(self.root)
        win.title("ZEHEN: AI key")
        win.configure(bg=BG, padx=22, pady=18)
        win.attributes("-topmost", True)
        win.resizable(False, False)
        lab = lambda text, **kw: tk.Label(win, text=text, bg=BG, fg=kw.pop("fg", SUB), font=("Segoe UI", 10), anchor="w", **kw)
        lab("Paste an API key from Groq, Gemini, OpenRouter, OpenAI or any OpenAI-compatible service.",
            fg=TEXT).grid(row=0, column=0, columnspan=3, sticky="w")
        lab("Free keys: console.groq.com/keys  or  aistudio.google.com/apikey").grid(row=1, column=0, columnspan=3,
                                                                                    sticky="w", pady=(0, 10))
        key = tk.StringVar()
        box = tk.Entry(win, textvariable=key, width=52, show="\u2022", bg=TILE, fg=TEXT, insertbackground=CYAN,
                       relief="flat", font=("Consolas", 11), highlightthickness=1, highlightbackground=DIM,
                       highlightcolor=CYAN)
        box.grid(row=2, column=0, columnspan=2, sticky="we", ipady=6)
        btn = lambda text, cmd, **kw: tk.Button(win, text=text, command=cmd, bg=CHIP, fg=kw.pop("fg", CYAN), relief="flat",
                                                activebackground=DARK, activeforeground=CYAN, padx=12, pady=4,
                                                font=("Segoe UI", 10, "bold"), cursor="hand2", **kw)
        def paste():
            try:
                key.set(win.clipboard_get().strip())
            except tk.TclError:
                status.config(text="The clipboard is empty: copy your key first.", fg=AMBER)
        btn("Paste", paste).grid(row=2, column=2, padx=(8, 0))
        show = tk.IntVar()
        tk.Checkbutton(win, text="show key", variable=show, bg=BG, fg=SUB, selectcolor=BG, activebackground=BG,
                       activeforeground=SUB, command=lambda: box.config(show="" if show.get() else "\u2022")
                       ).grid(row=3, column=0, sticky="w", pady=(4, 0))
        lab("Service").grid(row=4, column=0, sticky="w", pady=(8, 0))
        who = tk.StringVar(value="Auto-detect")
        pick = ttk.Combobox(win, textvariable=who, values=PROVIDER_NAMES, state="readonly", width=18)
        pick.grid(row=5, column=0, sticky="w")
        url, mdl = tk.StringVar(), tk.StringVar()
        custom = tk.Frame(win, bg=BG)
        for r, (name, var) in enumerate((("Address (ends in /v1)", url), ("Model", mdl))):
            tk.Label(custom, text=name, bg=BG, fg=SUB, font=("Segoe UI", 9)).grid(row=r, column=0, sticky="w")
            tk.Entry(custom, textvariable=var, width=44, bg=TILE, fg=TEXT, insertbackground=CYAN, relief="flat",
                     font=("Consolas", 10)).grid(row=r, column=1, padx=8, pady=2, ipady=3)
        pick.bind("<<ComboboxSelected>>", lambda e: custom.grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))
                  if who.get() == "Custom" else custom.grid_remove())
        saved = self.cfg.get("llm", {})
        status = tk.Label(win, bg=BG, fg=GREEN if saved.get("api_key") else SUB, font=("Segoe UI", 10), anchor="w",
                          wraplength=470, justify="left",
                          text="A key is saved and verified." if saved.get("verified") else
                          "A key is saved." if saved.get("api_key") else "No key yet: ZEHEN uses its offline brain.")
        status.grid(row=7, column=0, columnspan=3, sticky="w", pady=(12, 8))
        bar = tk.Frame(win, bg=BG)
        bar.grid(row=8, column=0, columnspan=3, sticky="we")

        def verify():
            k = key.get().strip()
            if not k:
                return status.config(text="Paste a key first.", fg=AMBER)
            status.config(text="Asking the service…", fg=AMBER)
            go.config(state="disabled")
            args = (k, who.get(), url.get(), mdl.get())

            def work():
                res = verify_key(*args)
                win.after(0, lambda: done(k, res))
            threading.Thread(target=work, daemon=True).start()

        def done(k, res):
            go.config(state="normal")
            status.config(text=res["message"], fg=GREEN if res["ok"] else "#ff6b6b")
            if res["ok"]:  # verified: keep it, and the next request already uses it
                llm = {"api_key": k, "provider": res["provider"], "verified": True}
                if who.get() == "Custom" or url.get().strip() or mdl.get().strip():
                    llm.update(base_url=res["base_url"], model=res["model"])
                self.cfg["llm"] = llm
                save_config(self.cfg)
                self.t._say(f"{res['provider']} key verified.")
                win.after(1600, win.destroy)

        def remove():
            self.cfg.pop("llm", None)
            save_config(self.cfg)
            status.config(text="Key removed. ZEHEN is using its offline brain.", fg=GREEN)
            self.t._say("Key removed. Using my offline brain.")
        go = btn("Verify and save", verify)
        go.pack(in_=bar, side="left")
        btn("Remove key", remove, fg=AMBER).pack(in_=bar, side="left", padx=8)
        btn("Close", win.destroy, fg=SUB).pack(in_=bar, side="right")
        box.focus_set()
        win.bind("<Return>", lambda e: verify())
        win.update_idletasks()
        win.geometry(f"+{(self.W - win.winfo_width()) // 2}+{(self.H - win.winfo_height()) // 3}")

    def restart_tutorial(self):
        self.tut, self.step_start, self.spoken = 0, time.monotonic(), None

    def narrate(self, now):
        """The tutorial out loud, step by step, so you can watch your hand instead of reading."""
        v = self.t.voice
        if not v or not self.cfg.get("tutorial_voice", True) or not v.ready.is_set() or now < self.great_until:
            return
        if self.spoken != self.tut:
            self.spoken = self.tut
            text = TUTORIAL_SPEECH[self.tut] if self.tut < len(TUTORIAL) else TUTORIAL_END
            v.say((random.choice(PRAISE) + " " if self.praise else "") + text)
            self.praise = False
            if self.tut < len(TUTORIAL) and TUTORIAL[self.tut][0] == "enroll":
                v.enroll()  # listens as soon as she has finished speaking

    def skip_tutorial(self):
        self.tut, self.great_until = len(TUTORIAL), 0.0

    def quit(self):
        self.t.running = False
        self.root.destroy()

    # --- frame loop ---
    def tick(self):
        if self.t.error:
            messagebox.showerror("ZEHEN", self.t.error)
            return self.quit()
        now, s = time.monotonic(), self.t.state
        if self.tut is not None and now > self.great_until:
            if self.tut == len(TUTORIAL):
                self.tut = None
                self.cfg["tutorial_done"] = True
                save_config(self.cfg)
            else:
                g, hold = TUTORIAL[self.tut][:2]
                # the gesture must start after the step appeared, so one long pinch can't pass two steps
                voice = self.t.voice
                if g in ("voice", "enroll") and not voice:
                    self.tut += 1  # no voice on this PC: nothing to try
                elif (voice and voice.enrolled_now if g == "enroll" else
                      voice and voice.state in ("listening", "thinking", "speaking") if g == "voice" else
                        s.active == g and s.since >= self.step_start and now - s.since >= hold):
                    self.tut += 1
                    self.praise = True
                    self.great_until = now + (3.0 if self.tut == len(TUTORIAL) else 1.0)
                    self.step_start = self.great_until
        if self.tut is not None:
            self.narrate(now)
        self.draw_overlay(s, now)
        self.draw_panel(s, now)
        self.root.after(33, self.tick)  # ~30 fps, matches the camera

    def draw_overlay(self, s, now):
        c, W, H, px = self.c, self.W, self.H, self.px
        c.delete("all")
        ui = s.ui
        for x0, y0, x1, y1 in ui.boxes:
            c.create_rectangle(x0 * W, y0 * H, x1 * W, y1 * H, fill=BG, outline=DIM, width=2)
        if ui.pad:
            x0, y0, x1, y1 = ui.pad
            c.create_rectangle(x0 * W, y0 * H, x1 * W, y1 * H, fill=TILE, outline=CYAN, width=2)
            c.create_line(x0 * W + px(20), y0 * H + (y1 - y0) * H * 0.72, x1 * W - px(20),
                          y0 * H + (y1 - y0) * H * 0.72, fill=DARK, width=2)  # baseline, like ruled paper
            if not s.ink:
                c.create_text((x0 + x1) / 2 * W, (y0 + y1) / 2 * H, text="write here", fill=DARK,
                              font=self.font(40, True))
        for b in ui.buttons:
            x0, y0, x1, y1 = b.r
            on, hit = b.id == s.hover, b.id == s.pressed
            c.create_rectangle(x0 * W, y0 * H, x1 * W, y1 * H, fill=GREEN if hit else CYAN if on else TILE,
                               outline=CYAN if on else DIM, width=2)
            c.create_text((x0 + x1) / 2 * W, (y0 + y1) / 2 * H, text=b.label, justify="center",
                          fill=BG if on or hit else CYAN, font=self.font(b.size, True), width=(x1 - x0) * W - 8)
        for text, x, y, size, col, anchor in ui.texts:
            c.create_text(x * W, y * H, text=text, anchor=anchor, fill=COLORS[col], font=self.font(size, True))
        for stroke in s.ink:
            if len(stroke) > 1:
                c.create_line(*[v for x, y in stroke for v in (x * W, y * H)], fill=CYAN, width=px(5),
                              capstyle="round", joinstyle="round", smooth=True)
            else:
                (x, y), r = stroke[0], px(3)
                c.create_oval(x * W - r, y * H - r, x * W + r, y * H + r, fill=CYAN, outline="")
        if s.ptr2:
            x2, y2, r2 = s.ptr2[0] * W, s.ptr2[1] * H, px(14)
            c.create_oval(x2 - r2, y2 - r2, x2 + r2, y2 + r2, outline=GREEN, width=px(2))
            if s.zoom:
                c.create_line(x2, y2, s.ptr[0] * W, s.ptr[1] * H, fill=GREEN, width=px(2), dash=(6, 4))
        if s.ptr:
            self.draw_pointer(c, s, now)
        if s.flash:
            c.create_text(W / 2 + 2, px(40) + 2, text=s.flash, fill=DARK, font=self.font(28, True))
            c.create_text(W / 2, px(40), text=s.flash, fill=CYAN, font=self.font(28, True))
        v, y = self.t.voice, self.pill_bottom + px(12)
        you = (v.partial or (v.heard if now - v.heard_at < 8 else "")) if v else ""
        if you:  # what you said, small and quiet, as you say it
            for dx, dy, fg in ((1, 1, "#02060a"), (0, 0, SUB)):
                c.create_text(W - px(18) + dx, y + dy, text="\u201c" + you + "\u201d", anchor="ne", justify="right",
                              fill=fg, font=("Segoe UI", -px(11), "italic"), width=px(300))
            y += px(20) * (1 + len(you) // 48)
        age = now - self.t.reply_at
        if self.t.reply and age < 7 and self.voice_mode(now) != "working":  # what she said: small, below the pill
            col = TEXT if age < 5.5 else SUB if age < 6.3 else DIM  # fades out
            for dx, dy, fg in ((1, 1, "#02060a"), (0, 0, col)):
                c.create_text(W - px(18) + dx, y + dy, text=self.t.reply, anchor="ne",
                              justify="right", fill=fg, font=self.font(13), width=px(300))

    def draw_pointer(self, c, s, now):
        px = self.px
        x, y = s.ptr[0] * self.W, s.ptr[1] * self.H
        grab = s.gesture == "pinch"
        r = px(11 if grab else 18)
        for i, col in enumerate((DARK, DIM)):  # fake glow: tk has no alpha
            rr = r + px(6 - 3 * i)
            c.create_oval(x - rr, y - rr, x + rr, y + rr, outline=col, width=px(3))
        c.create_oval(x - r, y - r, x + r, y + r, outline=CYAN, width=px(2), fill=CYAN if grab else "")
        c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=CYAN, outline="")
        R, spin = px(30), now * 160 % 360
        for k in range(3):  # spinning HUD ticks
            c.create_arc(x - R, y - R, x + R, y + R, start=spin + k * 120, extent=40,
                         style="arc", outline=DIM, width=px(2))
        if s.progress:
            R2 = px(38)
            c.create_arc(x - R2, y - R2, x + R2, y + R2, start=90, extent=-359.9 * s.progress,
                         style="arc", outline=CYAN, width=px(4))
        label = "" if s.mode and s.active in ("point", "pinch") else LABELS.get(s.active, "")
        if label:
            c.create_text(x + px(44), y + px(4), text=label, anchor="w", fill=CYAN, font=self.font(13, True))

    def mono(self, size, bold=False):
        return ("Consolas", self.px(size) * -1, "bold" if bold else "normal")

    def chip(self, x0, y0, x1, y1, r=10, fill=CHIP, outline="", tags=(), width=1, state="normal"):
        """Rounded rectangle on the side panel (a smoothed polygon: tk has no rounded rectangles)."""
        pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1,
               x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
        return self.pc.create_polygon(pts, smooth=True, fill=fill, outline=outline, width=width, tags=tags,
                                      state=state)  # "disabled": for rims and decoration, so clicks reach what's under

    def voice_mode(self, now):
        """What the voice is doing: loading | idle | listening | speaking | working | paused | off."""
        v = self.t.voice
        if self.t.paused:
            return "paused"
        if v and v.enroll_left:
            return "enrolling"
        state = v.state if v else "off"
        if state == "thinking" or (self.t.dl is not None and now - self.t.reply_at < 30):
            return "working"
        if self.t.setup is not None and self.t.setup.running and state in ("loading", "idle", "off"):
            return "working"  # fetching what's missing: the wave ripples
        return state

    def status(self, s, now):
        """What the pill says: (label, detail, colour). Voice first, then what your hand is doing."""
        v, mode = self.t.voice, self.voice_mode(now)
        if mode == "paused":
            return "PAUSED", "fist or the pause button to resume", AMBER
        st = self.t.setup
        if mode == "working" and st is not None and st.running and not (v and v.state == "thinking"):
            done, total, cur = st.progress()
            return "SETTING UP", f"{done}/{total}  {cur['name'] if cur else ''}"[:30], AMBER
        if mode == "working":  # never the word "thinking": say what is actually happening
            return (self.t.reply or "").strip()[:30], "", AMBER
        if mode == "enrolling":
            return "SAY ZEHEN", f"{4 - v.enroll_left} of 3", CYAN
        if self.t.dictating() and mode != "speaking":
            return "TYPING", 'say "stop typing" to end', GREEN
        if mode in ("listening", "speaking"):
            return mode.upper(), "go ahead" if mode == "listening" else "", GREEN
        if s.mode:
            return {"launcher": "LAUNCHER", "pad": "WRITING PAD", "keyboard": "KEYBOARD"}[s.mode], "open palm closes it", CYAN
        gesture = next((g for g in GUIDE if g[0] == s.active), None)
        if gesture and s.hands:
            return LABELS[s.active], gesture[3], CYAN
        if not self.t.camera_on:
            return "VOICE ONLY", 'say "zehen" to talk', CYAN
        idle = v.error if v and mode == "off" else 'say "zehen" to talk' if mode == "idle" else "waking up"
        return ("TRACKING", idle, GREEN) if s.hands else ("SHOW YOUR HAND", idle, AMBER)

    def draw_wave(self, c, x, cy, mode, now, col, hands=False, tags=()):
        """Five bars that move like the voice: they follow the mic when listening, ripple when working, swell
        when speaking, and just breathe when idle. Each eases toward its target, so nothing ever jumps."""
        px, v = self.px, self.t.voice
        level = v.level if v else 0.0
        for i in range(5):
            ph = now * 9 + i * 1.1
            target = {"listening": 4 + 26 * level * (0.55 + 0.45 * math.sin(ph) ** 2),
                      "enrolling": 4 + 26 * level * (0.55 + 0.45 * math.sin(ph) ** 2),
                      "speaking": 7 + 12 * (0.5 + 0.5 * math.sin(ph * .8)) * (0.6 + 0.4 * math.sin(now * 3 + i)),
                      "working": 6 + 10 * (0.5 + 0.5 * math.sin(now * 5.5 - i * 0.9))}.get(
                mode, (5, 9, 13, 9, 5)[i] * (0.8 + 0.2 * math.sin(now * 2 + i * .6)) if hands else
                3.5 + 1.8 * math.sin(now * 1.7 + i * .7))
            self.bars[i] += (target - self.bars[i]) * 0.35
            h, bx = self.bars[i] * self.s, x + i * px(7)
            self.chip(bx, cy - h / 2, bx + px(3), cy + h / 2, px(1.5), fill=col, tags=tags, state="disabled")

    def draw_panel(self, s, now):
        c, px = self.pc, self.px
        full = self.tut is not None or (self.open and not s.mode)
        tw = px(300 if full else 236)
        c.delete("all")
        mode = self.voice_mode(now)
        label, detail, col = self.status(s, now)
        # fluid size: the window eases toward its target instead of snapping, so opening and folding flow
        if self.cw is None:
            self.cw, self.ch = tw, px(46)
        self.cw += (tw - self.cw) * 0.34
        w = int(self.cw)
        if full:
            target_h = self.draw_full(c, s, now, w, label, col, mode)
        else:  # the pill: one glass capsule, always there, says what you're showing it
            target_h = px(46)
            self.draw_wave(c, px(16), target_h / 2, mode, now, col, hands=bool(s.hands) or not self.t.camera_on)
            c.create_text(px(58), target_h / 2 - (px(8) if detail else 0), text=label, anchor="w", fill=col,
                          font=self.mono(12, True))
            if detail:
                c.create_text(px(58), target_h / 2 + px(9), text=detail[:30], anchor="w", fill=SUB, font=self.mono(9))
            frac = self.t.dl
            if frac is None and label == "SETTING UP":
                cur = self.t.setup.progress()[2]
                frac = cur["frac"] if cur else None
            if mode == "working" and frac is not None:
                c.create_rectangle(px(58), target_h - px(8), w - px(16), target_h - px(6), fill=DARK, outline="")
                c.create_rectangle(px(58), target_h - px(8), px(58) + (w - px(74)) * frac, target_h - px(6),
                                   fill=AMBER, outline="")
        self.ch += (target_h - self.ch) * 0.34
        h = int(self.ch)
        if self.frosted:  # hairline rim, so the glass has an edge
            self.chip(1, 1, w - 2, h - 2, px(23) if not full else px(16), fill="", outline=HAIR, state="disabled")
        x0, y0 = self.W - w - px(16), px(16)
        self.panel.geometry(f"{w}x{h}+{x0}+{y0}")
        self.pill_bottom = y0 + h
        self.dwell(now, w, h, x0, y0)

    def dwell(self, now, w, h, x0, y0):
        """Hands-free open and close: point at the pill for a moment and it opens; leave the open panel
        alone for a few seconds and it folds back. (Glass pixels aren't clickable, so a real click can't open it.)"""
        cx, cy = pyautogui.position()
        over = x0 <= cx <= x0 + w and y0 <= cy <= y0 + h
        if self.tut is not None or self.t.state.mode:
            self.over_since = self.left_at = None
            return
        if not self.open:
            self.armed = self.armed or not over
            self.over_since = (self.over_since or now) if over and self.armed else None
            if self.over_since and now - self.over_since > DWELL_S:
                self.open, self.over_since, self.left_at = True, None, None
        else:
            self.left_at = None if over else (self.left_at or now)
            if self.left_at and now - self.left_at > CLOSE_S:
                self.open, self.left_at, self.armed = False, None, False

    def draw_full(self, c, s, now, w, label, col, mode):
        px, m = self.px, self.px(14)
        # header: wordmark + live status, then camera, pause, fold and quit
        c.create_oval(m, px(20), m + px(10), px(30), fill=col, outline="")
        c.create_text(m + px(20), px(25), text="ZEHEN", anchor="w", fill=TEXT, font=self.mono(15, True))
        c.create_text(m + px(76), px(26), text="// " + {"show your hand": "no hand", "voice only": "voice"}.get(label.lower(), label.lower())[:9],
                      anchor="w", fill=col, font=self.mono(10, True))
        cam = self.t.camera_on
        x1 = w - m
        for tag, glyph, ww, tint in (("btn_quit", "✕", 24, CYAN), ("btn_min", "–", 24, CYAN),
                                     ("btn_pause", "▶" if self.t.paused else "❚❚", 24, CYAN),
                                     ("btn_cam", "CAM" if cam else "CAM", 40, GREEN if cam else AMBER)):
            self.chip(x1 - px(ww), px(13), x1, px(37), px(12), fill=CHIP, outline=tint if tag == "btn_cam" else HAIR, tags=tag)
            c.create_text(x1 - px(ww) / 2, px(25), text=glyph, fill=tint, font=self.font(9 if tag == "btn_cam" else 11, True),
                          tags=tag)
            if tag == "btn_cam" and not cam:
                c.create_line(x1 - px(ww) + px(6), px(31), x1 - px(6), px(19), fill=AMBER, width=px(2), tags=tag)
            x1 -= px(ww + 6)
        # wireframe hands: proof it's tracking, without any video. Main hand cyan, other green.
        bx, by, bw, bh = m, px(48), w - 2 * m, px(84)
        self.chip(bx, by, bx + bw, by + bh, px(12), fill=BG, outline=HAIR)
        fw = bh * 4 / 3  # camera is 4:3
        X = lambda p: bx + (bw - fw) / 2 + p[0] * fw
        Y = lambda p: by + p[1] * bh
        for hand, line, dot in zip(s.hands, (DIM, "#1f8f5a"), (CYAN, GREEN)):
            for a, b in HAND_LINKS:
                c.create_line(X(hand[a]), Y(hand[a]), X(hand[b]), Y(hand[b]), fill=line, width=px(2))
            for p in hand:
                c.create_oval(X(p) - 2, Y(p) - 2, X(p) + 2, Y(p) + 2, fill=dot, outline="")
        if not s.hands:
            c.create_text(bx + bw / 2, by + bh / 2, fill=DIM, font=self.mono(9),
                          text="camera off: voice only" if not cam else "raise your hand in front of the camera")
        y = by + bh + px(10)
        y += self.draw_voice(c, y, w, now, mode)
        y += self.draw_setup(c, y, w)
        return self.draw_tutorial(c, y, w, now) if self.tut is not None else self.draw_guide(c, s, y, w)

    def draw_setup(self, c, y, w):
        """The first-run checklist: each thing ZEHEN needs, whether it was already here, and how far along it is."""
        st = self.t.setup
        if st is None or not (st.running or any(i["state"] == "failed" for i in st.items)):
            return 0
        px, m = self.px, self.px(14)
        rows = st.items
        h = px(32) + len(rows) * px(20)
        self.chip(m, y, w - m, y + h, px(12), fill=CHIP, outline=AMBER if st.running else "#ff6b6b")
        done, total, _ = st.progress()
        c.create_text(m + px(12), y + px(15), text=f"SETUP  {done}/{total}", anchor="w", fill=AMBER, font=self.mono(9, True))
        c.create_text(w - m - px(12), y + px(15), text="what's here is skipped" if st.running else "retries next launch",
                      anchor="e", fill=SUB, font=self.mono(8))
        for i, it in enumerate(rows):
            ry = y + px(32) + i * px(20) + px(9)
            text, col = {"found": ("ready", GREEN), "done": ("ready", GREEN), "skipped": ("skipped", SUB),
                         "pending": ("waiting", SUB), "failed": ((it["msg"] or "failed")[:24], "#ff6b6b"),
                         "working": (f"{int(it['frac'] * 100)}%" if it["frac"] is not None else "...", AMBER)}[it["state"]]
            c.create_text(m + px(12), ry, text=it["name"], anchor="w", fill=TEXT if it["state"] == "working" else SUB,
                          font=self.mono(9))
            c.create_text(w - m - px(12), ry, text=text, anchor="e", fill=col, font=self.mono(9, True))
            if it["state"] == "working" and it["frac"] is not None:
                c.create_rectangle(m + px(12), ry + px(8), m + px(12) + (w - 2 * m - px(24)) * it["frac"], ry + px(10),
                                   fill=AMBER, outline="")
        return h + px(8)

    def draw_voice(self, c, y, w, now, mode):
        """Voice chip: the live wave, what it is doing and a hint. Returns the height used."""
        px, m, v = self.px, self.px(14), self.t.voice
        label, col = {"loading": ("Starting", AMBER), "idle": ('Say "Zehen"', CYAN), "listening": ("Listening", GREEN),
                      "working": ((self.t.reply or "Working").strip()[:26], AMBER), "speaking": ("Speaking", GREEN),
                      "enrolling": ('Say "Zehen"', CYAN),
                      "paused": ("Paused", AMBER), "off": ("Voice off", DIM)}[mode]
        line = ("go ahead" if mode == "listening" else f"{4 - v.enroll_left} of 3" if mode == "enrolling" else
                (v.error if v and mode == "off" else "")
                or ("or point here and tap" if mode == "idle" else ""))
        h = px(50)
        self.chip(m, y, w - m, y + h, px(12), fill="#0a2e38", outline=col, tags="btn_voice")
        self.draw_wave(c, m + px(16), y + h / 2, mode, now, col, tags="btn_voice")
        c.create_text(m + px(58), y + (px(18) if line else h / 2), text=label, anchor="w", fill=col,
                      font=self.font(13, True), tags="btn_voice")
        if line:
            c.create_text(m + px(58), y + px(36), text=line[:40], anchor="w", fill=SUB, font=self.mono(9), tags="btn_voice")
        if mode == "working" and self.t.dl is not None:  # download bar along the bottom of the chip
            c.create_rectangle(m + px(14), y + h - px(7), w - m - px(14), y + h - px(4), fill=DARK, outline="",
                               state="disabled")
            c.create_rectangle(m + px(14), y + h - px(7), m + px(14) + (w - 2 * m - px(28)) * self.t.dl, y + h - px(4),
                               fill=AMBER, outline="", state="disabled")
        used = h + px(8)
        ep, saved = llm_endpoint(self.t.cfg), self.t.cfg.get("llm", {})
        brain = ("Offline brain" if not ep else
                 f"{provider_of(ep[2], saved) or 'Custom'} key" + (" verified" if saved.get("verified") else ""))
        c.create_text(m, y + used + px(11), text=brain[:22], anchor="w", fill=GREEN if ep else SUB, font=self.mono(9))
        x1 = w - m
        for tag, text in ((("btn_addkey", "Change key"),) if ep else (("btn_addkey", "Add key"), ("btn_getkey", "Get key"))):
            ww = px(78 if text == "Change key" else 66)
            self.chip(x1 - ww, y + used, x1, y + used + px(22), px(11), fill=CHIP, outline=HAIR, tags=tag)
            c.create_text(x1 - ww / 2, y + used + px(11), text=text, fill=CYAN, font=self.font(9, True), tags=tag)
            x1 -= ww + px(6)
        used += px(30)
        return used

    def draw_tutorial(self, c, y, w, now):
        px, great, m = self.px, now < self.great_until, self.px(14)
        if self.tut == len(TUTORIAL):
            title, hint = "You're all set 🚀", "Point at the pill to open this panel again."
        else:
            _, _, title, hint = TUTORIAL[self.tut]
        self.chip(m, y, w - m, y + px(124), px(12), fill=CHIP, outline=GREEN if great else HAIR)
        c.create_text(m + px(12), y + px(14), text=f"TUTORIAL {min(self.tut + 1, len(TUTORIAL))}/{len(TUTORIAL)}",
                      anchor="w", fill=SUB, font=self.mono(9, True))
        c.create_text(w - m - px(12), y + px(14), text="skip ›", anchor="e", fill=CYAN, font=self.mono(10, True),
                      tags="btn_skip")
        c.create_text(m + px(12), y + px(30), text="✓ GREAT!" if great and self.tut < len(TUTORIAL) else title,
                      anchor="nw", fill=GREEN if great else CYAN, font=self.font(15, True), width=w - 2 * m - px(24))
        c.create_text(m + px(12), y + px(58), text=hint, anchor="nw", fill="#9fdce6", font=self.font(11),
                      width=w - 2 * m - px(24))
        for i in range(len(TUTORIAL)):  # progress dots
            cx = m + px(16) + i * px(15)
            c.create_oval(cx - 4, y + px(108) - 4, cx + 4, y + px(108) + 4, outline=CYAN,
                          fill=CYAN if i < self.tut else "")
        return y + px(138)

    def draw_guide(self, c, s, y, w):
        px, m = self.px, self.px(14)
        c.create_text(m, y + px(6), text="GESTURES", anchor="w", fill=SUB, font=self.mono(9, True))
        c.create_text(w - m, y + px(6), text="tutorial ↻", anchor="e", fill=CYAN, font=self.mono(9, True),
                      tags="btn_tut")
        y += px(18)
        for g, icon, how, what in GUIDE:
            on = g == s.active
            if on:  # the row you're doing lights up and says what it does
                self.chip(px(8), y, w - px(8), y + px(40), px(10), fill=DARK, outline=CYAN)
            c.create_text(px(22), y + px(12), text=icon, anchor="w", fill=CYAN, font=self.font(13))
            c.create_text(px(48), y + px(12), text=how, anchor="w", fill=CYAN if on else TEXT,
                          font=self.font(11, True))
            if on:
                c.create_text(px(48), y + px(29), text=what, anchor="w", fill="#9fdce6", font=self.mono(9))
            y += px(40 if on else 25)
        return y + px(8)


def load_config():
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")


def open_camera(index):
    """Open one camera and prove it's live. Virtual cameras (PRISM Lens, OBS, Snap...) and privacy blocks
    open fine but send a black or frozen placeholder; a real sensor always has frame-to-frame noise."""
    cam = cv2.VideoCapture(index)
    if cam.isOpened():
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        frames = [f for ok, f in (cam.read() for _ in range(15)) if ok][-8:]  # skip warm-up frames
        changes = [cv2.absdiff(a, b).mean() for a, b in zip(frames, frames[1:])]
        if len(changes) >= 3 and sum(changes) / len(changes) > LIVE_MIN_CHANGE:
            return cam
    cam.release()
    return None


def find_camera(cfg, forced=None):
    """First live camera, trying the forced / last-working one first. Remembers the winner in cfg."""
    order = [forced] if forced is not None else [cfg.get("camera"), *range(5)]
    for i in dict.fromkeys(i for i in order if i is not None):  # dedupe, keep order
        if cam := open_camera(i):
            cfg["camera"] = i
            return cam
    return None


def camera_setup(root, cfg, forced=None):
    """Permission screen on first run, or when no live camera is found. Returns an open camera or None."""
    if cfg.get("camera_ok") and (cam := find_camera(cfg, forced)):
        save_config(cfg)
        return cam  # already allowed and working: straight in, no screen
    got = {}
    win = tk.Toplevel(root, bg=BG, padx=28, pady=24)
    win.title("ZEHEN")
    win.resizable(False, False)
    win.attributes("-topmost", True)
    if (BASE / "zehen.ico").exists():
        win.iconbitmap(BASE / "zehen.ico")
    tk.Label(win, text="◉ ZEHEN", fg=CYAN, bg=BG, font=("Segoe UI", 24, "bold")).pack(anchor="w")
    tk.Label(win, text="Camera permission", fg=DIM, bg=BG, font=("Segoe UI", 11, "bold")).pack(anchor="w")
    body = tk.Label(win, text=CONSENT, fg=TEXT, bg=BG, font=("Segoe UI", 11), wraplength=460, justify="left")
    body.pack(anchor="w", pady=(16, 22))
    row = tk.Frame(win, bg=BG)
    row.pack(fill="x")
    button = lambda text, cmd, primary=False: tk.Button(
        row, text=text, command=cmd, relief="flat", bd=0, cursor="hand2", padx=18, pady=8,
        font=("Segoe UI", 11, "bold"), bg=CYAN if primary else DARK, fg=BG if primary else CYAN,
        activebackground="#7ff2ff" if primary else DIM, activeforeground=BG)

    def blocked():
        body.config(text=CAMERA_HELP, fg=AMBER)
        allow.config(text="Try again")
        if sys.platform == "win32":
            settings.pack(side="left", padx=(10, 0))

    def try_camera():
        body.config(text="Looking for your camera…", fg=CYAN)
        win.update()
        if cam := find_camera(cfg, forced):
            cfg["camera_ok"] = True
            save_config(cfg)
            got["cam"] = cam
            win.destroy()
        else:
            blocked()

    allow = button("Allow camera", try_camera, primary=True)
    allow.pack(side="left")
    settings = button("Open camera settings", lambda: os.startfile("ms-settings:privacy-webcam"))
    quit_btn = button("Quit", win.destroy)
    quit_btn.config(bg=BG, fg=DIM)  # plain text link, so it doesn't merge with the settings button
    quit_btn.pack(side="right", padx=(12, 0))
    if cfg.get("camera_ok"):
        blocked()  # they allowed before, so the camera itself is the problem: say what to fix
    win.update_idletasks()
    win.geometry(f"+{(win.winfo_screenwidth() - win.winfo_width()) // 2}"
                 f"+{(win.winfo_screenheight() - win.winfo_height()) // 3}")
    win.focus_force()
    win.wait_window()
    return got.get("cam")


def main(forced_camera=None):
    root = tk.Tk()
    root.withdraw()
    cfg = load_config()
    cam = camera_setup(root, cfg, forced_camera)
    if not cam:
        return root.destroy()
    root.deiconify()
    HUD(root, Tracker(cam, cfg), cfg)
    root.mainloop()


def run_setup_cli():
    """python zehen.py --setup: fetch everything ZEHEN needs, skipping what is already here, and print the checklist."""
    st = Setup(load_config())
    st.start()
    shown = {}
    while not st.done.is_set() or any(i["state"] != shown.get(i["key"]) for i in st.items):
        for i in st.items:
            line = f"  {i['name']:26} {i['state']:8} {i['msg']}"
            if i["state"] == "working" and i["msg"]:
                print("\r" + line, end="", flush=True)
            elif shown.get(i["key"]) != i["state"] and i["state"] != "pending":
                print("\r" + line + " " * 10, flush=True)
            shown[i["key"]] = i["state"]
        time.sleep(0.3)
    print("Setup finished." if all(i["state"] != "failed" for i in st.items) else "Setup finished with problems (see above).")


def selftest():
    def hand(up, thumb_on_index=False):
        P = lambda x, y: SimpleNamespace(x=x, y=y, z=0.0)
        lm = [P(0.5, 0.5) for _ in range(21)]
        lm[0], lm[4] = P(0.5, 0.9), P(0.15, 0.62)
        for i, (t, u) in enumerate(zip((8, 12, 16, 20), up)):
            fx = 0.4 + i * 0.07
            lm[t - 3], lm[t - 2] = P(fx, 0.6), P(fx, 0.5)        # knuckle, middle joint
            lm[t] = P(fx, 0.3) if u else P(fx, 0.62)             # tip: up, or curled into the palm
        if thumb_on_index:
            lm[4] = P(lm[8].x, lm[8].y)
        return lm

    assert classify(hand((1, 0, 0, 0)), False) == "point"
    assert classify(hand((1, 1, 0, 0)), False) == "scroll"
    assert classify(hand((1, 1, 1, 0)), False) == "three"
    assert classify(hand((1, 1, 1, 1)), False) == "palm"
    assert classify(hand((0, 0, 0, 0)), False) == "fist"
    assert classify(hand((1, 0, 0, 0), thumb_on_index=True), False) == "pinch"
    assert classify(hand((0, 0, 0, 0), thumb_on_index=True), False) == "fist"  # fist must not click
    f = OneEuro()
    assert [round(f(100.0, t / 30)) for t in range(5)] == [100] * 5  # still hand -> no drift
    assert len(KEYS) == sum(len(r) for r in KEY_ROWS)

    import random
    rnd = random.Random(3)
    shake = lambda pts, amount=0.004: [(x + rnd.uniform(-amount, amount), y + rnd.uniform(-amount, amount))
                                       for x, y in pts]
    def hooked(pts):  # the tails a real pinch leaves: ~12% of the stroke's size, at a right angle
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        h = 0.12 * max(max(xs) - min(xs), max(ys) - min(ys))

        def tail(end, toward):  # a short spur sticking out sideways from the stroke's end
            dx, dy = end[0] - toward[0], end[1] - toward[1]
            n = math.hypot(dx, dy) or 1
            px, py = -dy / n * h, dx / n * h
            return [(end[0] + px * t / 3, end[1] + py * t / 3) for t in range(1, 4)]
        return tail(pts[0], pts[3])[::-1] + pts + tail(pts[-1], pts[-4])
    messy_line = hooked(shake([(0.3 + 0.004 * k, 0.4 + 0.002 * k) for k in range(60)]))
    clean = beautify(messy_line)
    (ax, ay), (bx, by) = clean[0], clean[-1]
    assert all(abs((bx - ax) * (ay - y) - (ax - x) * (by - ay)) < 1e-9 for x, y in clean)  # perfectly straight
    messy_o = shake([(0.5 + 0.05 * math.cos(k / 20 * math.tau), 0.5 + 0.09 * math.sin(k / 20 * math.tau))
                     for k in range(21)] * 1, 0.006)
    ring = beautify(messy_o)
    assert max(abs(math.hypot((x - 0.5) / 0.05, (y - 0.5) / 0.09) - 1) for x, y in ring) < 0.25  # a clean oval
    zigzag = [(0.3 + 0.02 * k, 0.4 + (0.05 if k % 2 else 0)) for k in range(8)]
    z = beautify(zigzag)
    assert max(y for _, y in z) - min(y for _, y in z) > 0.02  # a w stays a w, not a line
    assert beautify([(0.1, 0.1), (0.1, 0.1)]) == [(0.1, 0.1), (0.1, 0.1)]  # taps (dots) untouched

    apps = [("Spotify", "a"), ("Settings", "b"), ("Google Chrome", "c"), ("Calculator", "d"),
            ("PostgreSQL documentation", "e"), ("Photos", "f"), ("Paramètres", "g"), ("Caméra", "h")]
    assert search_apps(apps, "spotlfy")[0][0] == "Spotify"  # handwriting typo still finds it
    assert search_apps(apps, "chrome")[0][0] == "Google Chrome"
    assert search_apps(apps, "parametres")[0][0] == "Paramètres"  # unaccented handwriting finds accented app
    assert search_apps(apps, "camera")[0][0] == "Caméra"
    assert search_apps(apps, "zzzz") == []
    assert [n for n, _ in search_apps(apps, "spot")] == ["Spotify"]  # no loose junk like PostgreSQL
    assert window_label("Instagram - Google Chrome") == "Google Chrome\nInstagram"
    assert window_label("Wikipédia — Mozilla Firefox") == "Mozilla Firefox\nWikipédia"

    apps_q = [("Spotify", "a"), ("Notepad", "b")]
    qc = lambda x: quick_command(x, apps_q)
    assert qc("Scroll down") == ("Scrolling down", [{"scroll": "down 6"}])
    assert qc("please scroll up a lot.") == ("Scrolling up", [{"scroll": "up 15"}])
    assert qc("scrawl down")[1] == [{"scroll": "down 6"}] and qc("scroll dawn")[1] == [{"scroll": "down 6"}]
    assert qc("Click.")[1] == [{"mouse": "click"}] and qc("quick")[1] == [{"mouse": "click"}]
    assert qc("double click")[1] == [{"mouse": "double"}] and qc("right click")[1] == [{"mouse": "right"}]
    assert qc("go back")[1] == [{"press": "alt+left"}] and qc("Zoom in")[1] == [{"zoom": "in"}]
    assert qc("down") is None and qc("turn it up")[1] == [{"volume": "up"}]  # a stray "down" is not a command
    assert qc("volume down")[1] == [{"volume": "down"}] and qc("pause")[1] == [{"media": "playpause"}]
    assert qc("open spotify")[1] == [{"open_app": "Spotify"}] and qc("open the unknown thing") is None
    assert qc("open YouTube")[1] == [{"open_url": "https://www.youtube.com"}]  # a site, not an app
    assert qc("open google.com")[1] == [{"open_url": "https://google.com"}] and qc("go to github")[1][0]["open_url"]
    assert qc("click on trending")[1] == [{"click": "trending"}]
    assert qc("open youtube and click on trending")[1] == [{"open_url": "https://www.youtube.com"}, {"wait": "3.5"},
                                                         {"click": "trending"}]
    assert qc("type cats and dogs")[1] == [{"type": "cats and dogs"}]  # "and" inside a request stays inside
    assert qc("Type Hello World.")[1] == [{"type": "Hello World"}]
    assert qc("write a note saying buy milk in notepad") is None  # too much for a shortcut: the AI plans it
    assert "NOTFOUND" in qc("open jupyter notebook")[1][0]["powershell"] and qc("jupiter notebook")[1] and qc("install jupyter")[1]
    class Fake429(urllib.error.HTTPError):
        def __init__(self): super().__init__("u", 429, "limit", {}, None)
    real_open, real_local = urllib.request.urlopen, globals()["local_llm"]
    reply_obj = SimpleNamespace(create_chat_completion=lambda **kw: {"choices": [{"message": {"content": "offline answer"}}]})
    def dead(*a, **k): raise Fake429()
    urllib.request.urlopen = dead
    globals()["local_llm"] = lambda note=None, load=True: reply_obj
    try:
        keyed = {"llm": {"api_key": "gsk_" + "x" * 30}}
        assert chat(keyed, [{"role": "user", "content": "hi"}]) == "offline answer"  # rate-limited key: offline takes over
        assert chat({}, [{"role": "user", "content": "hi"}]) == "offline answer"      # no key at all: offline
    finally:
        urllib.request.urlopen, globals()["local_llm"] = real_open, real_local
    import io as _io
    class Resp(_io.BytesIO):
        __enter__ = lambda self: self
        __exit__ = lambda self, *a: False
    seen_urls = []
    def fake_open(req, timeout=0):
        seen_urls.append(req.full_url)
        if "bad" in req.headers.get("Authorization", ""):
            raise urllib.error.HTTPError(req.full_url, 401, "no", {}, None)
        return Resp(b'{"choices": [{"message": {"content": "OK"}}]}')
    urllib.request.urlopen = fake_open
    try:
        good = verify_key("AIza" + "x" * 30)
        assert good["ok"] and good["provider"] == "Gemini" and "generativelanguage" in seen_urls[-1], good
        assert verify_key("sk-" + "x" * 30)["provider"] == "OpenAI" and verify_key("sk-or-" + "x" * 20)["provider"] == "OpenRouter"
        assert not verify_key("gsk_bad")["ok"] and "rejected" in verify_key("gsk_bad")["message"]
        unk = verify_key("mystery-key-123")
        assert not unk["ok"] and "Pick one" in unk["message"]  # can't guess: asks, doesn't send the key anywhere
        cust = verify_key("anything", "Custom", "https://api.together.xyz/v1", "meta-llama/Llama-3-8b")
        assert cust["ok"] and seen_urls[-1].startswith("https://api.together.xyz/v1/chat/completions")
    finally:
        urllib.request.urlopen = real_open
    dt = Tracker(None, {}, start=False)
    typed = []
    real_w, real_p = pyautogui.write, pyautogui.press
    pyautogui.write = lambda text, interval=0: typed.append(text)
    pyautogui.press = lambda k, **kw: typed.append(("press", k))
    try:
        dt.dict_until = time.monotonic() + 25
        assert dt._agent("Hello, how are you today?") == "" and typed == ["Hello, how are you today? "]  # typed as spoken
        assert dt._agent("new line") == "" and typed[-1] == ("press", "enter") and dt.dictating()
        assert dt._agent("send") == "" and not dt.dictating()  # sending ends it
        dt.dict_until = time.monotonic() + 25
        assert dt._agent("stop typing") == "Done typing." and not dt.dictating()
    finally:
        pyautogui.write, pyautogui.press = real_w, real_p
    calls = []
    def item(key, here=False, skip="", boom=False, core=False):
        def run(note):
            calls.append(key)
            note("half way", 0.5)
            if boom:
                raise RuntimeError("no internet")
        return dict(key=key, name=key, core=core, check=lambda: here, run=run, skip=lambda: skip)
    st = Setup({}, [item("a", here=True, core=True), item("b", core=True), item("c", skip="cloud key"), item("d", boom=True)])
    st.start()
    assert st.done.wait(10) and st.core.is_set()
    got = {i["key"]: i["state"] for i in st.items}
    assert got == {"a": "found", "b": "done", "c": "skipped", "d": "failed"}, got  # present: skipped; missing: fetched
    assert calls == ["b", "d"] and st.item("d")["msg"] == "no internet" and st.progress()[:2] == (4, 4)
    hints = ["Alacritty", "Auto-Tune Pro", "Spotify", "YouTube"]
    assert not sane_transcript("Alacritty, Auto-Tune Pro, Auto-Tune Pro, Auto-Tune Pro, Auto-Tune Pro", hints)  # the leak
    assert not sane_transcript("Thank you. Thank you. Thank you. Thank you. Thank you. Thank you.", hints)
    assert not sane_transcript("this is a very long made up sentence about nothing at all", hints, "him")
    assert sane_transcript("Open Spotify and YouTube.", hints) and sane_transcript("What is the capital of France?", hints, "what is")
    ht = Tracker(None, {}, start=False)
    ht.apps = [("Spotify", "s")]
    ht._do = lambda kind, v: ("opened Spotify" if kind == "open_app" and v == "spotify" else f"no app like '{v}'")
    script2 = iter(['{"say": "Done.", "do": [{"open_app": "spotify"}]}', '{"say": "Done.", "do": [{"open_app": "zzz"}]}',
                    '{"say": "Done.", "do": []}'])
    chat_before = globals()["chat"]
    globals()["chat"] = lambda cfg, msgs, note=None: next(script2)
    try:
        assert ht._agent("get my music app going please") == "Opened Spotify."   # not a blind "Done."
        assert ht._agent("get the zzz thing going for me") == "I couldn't find zzz."  # honest when it fails
        assert ht._agent("blah blah more words here") == ""                            # nothing to do: silence
    finally:
        globals()["chat"] = chat_before
    quiet_t = Tracker(None, {}, start=False)
    assert quiet_t._agent("thank you") == "" and quiet_t._agent("okay") == ""  # pleasantries: silence, not a chat
    names = ["SEM7_ML_PB_2026_TEST-2.pdf", "API Keys - GroqCloud", "(908) YouTube"]
    assert [best_match(q, names) for q in ("SAM 7 TFPB", "sam seven", "youtube", "groq", "api keys", "banana")] == [
        0, 0, 2, 1, 1, None], [best_match(q, names) for q in ("SAM 7 TFPB", "sam seven", "youtube", "groq", "banana")]
    assert qc("switch to the SAM 7 TFPB tab")[1] == [{"tab": "sam 7 tfpb"}]
    assert qc("go to youtube tab")[1] == [{"tab": "youtube"}] and qc("switch to tab 2")[1] == [{"press": "ctrl+2"}]
    assert qc("switch to chrome")[1] == [{"switch": "chrome"}] and qc("new tab")[1] == [{"press": "ctrl+t"}]
    assert qc("go to youtube")[1] == [{"open_url": "https://www.youtube.com"}]  # "go to" a site still opens the site
    assert qc("opened Jupyter Notebook")[1] and qc("search for flights to goa")[1] == [{"search_web": "flights to goa"}]
    assert qc("play the first video")[1] == [{"play": "first"}] and qc("play second video")[1] == [{"play": "second"}]
    assert qc("play arijit singh on youtube")[1][0]["open_url"].endswith("arijit%20singh") and qc("play arijit singh on youtube")[1][-1] == {"play": "1"}
    assert qc("play jaan e jaan")[1] == [{"play": "jaan e jaan"}] and qc("play some relaxing music on spotify") is None
    assert qc("click on the search bar")[1] == [{"focus_input": "search"}] and qc("click in the message box")[1] == [{"focus_input": "message"}]
    assert qc("start typing")[1] == [{"dictate": "on"}] and qc("click on trending")[1] == [{"click": "trending"}]
    assert qc("what time is it")[1] == [] and qc("what time is it")[0].startswith("It's ")
    assert qc("what is the weather") is None and qc("move my downloads to d drive") is None  # these need the AI
    say, acts, done = parse_plan('Sure!\n```json\n{"say": "Opening Spotify", "do": [{"open_app": "Spotify"}, {"rm": "x"}, '
                                 '{"type": 5}, {"press": "ctrl+t", "x": "y"}, "junk"]}\n```')
    assert (say, acts, done) == ("Opening Spotify", [{"open_app": "Spotify"}], True), (say, acts)  # bad actions dropped
    assert parse_plan("just chatting") == ("just chatting", [], True)
    bad = '<think>\n\n</think>\n{"say": "Opening the desktop folder...","do":[{"open_app", "File Explorer"}], "done": true}'
    assert parse_plan(bad) == ("Opening the desktop folder...", [{"open_app": "File Explorer"}], True), parse_plan(bad)
    assert parse_plan("{ nonsense }")[0].startswith("Sorry") and "{" not in parse_plan('{"x": [1,')[0]  # never raw JSON
    assert qc("open my desktop folder")[1] == [{"open_folder": "desktop"}] and qc("show downloads")[1][0]["open_folder"]
    assert parse_plan('{"say": "x", "do": [{"powershell": "dir"}], "done": false}')[2] is False
    assert len(parse_plan('{"do": [' + ", ".join(['{"type": "a"}'] * 8) + "]}")[1]) == 5
    assert run_powershell("Write-Output hello") == "hello"
    fake = SimpleNamespace(tokenizer=SimpleNamespace(phonemize=lambda t, lang: f"<{lang}:{t.strip()}>"))
    assert tts_phonemes(fake, "I'm Noor. Say Zehen!") == "<en-us:I'm> <hi:नूर> <en-us:. Say> <hi:ज़ेहेन> <en-us:!>"
    assert run_powershell("Start-Sleep 5", timeout=1).startswith("(timed out")
    assert llm_endpoint({}) is None or os.environ.get("ZEHEN_API_KEY")
    assert llm_endpoint({"llm": {"api_key": "AIza" + "x" * 30}})[0].startswith("https://generativelanguage")
    assert llm_endpoint({"llm": {"api_key": "gsk_" + "x" * 30, "model": "m"}})[1] == "m"  # config overrides defaults

    t = Tracker(None, {}, start=False)  # every panel lays out, every tile fits on screen
    t.apps = apps
    sent, real_hot, real_open = [], pyautogui.hotkey, webbrowser.open
    pyautogui.hotkey, webbrowser.open = lambda *k: sent.append(k), lambda u: sent.append(u)
    for kind, v in (("press", "ctrl+t"), ("press", "win+r"), ("press", "rm -rf"), ("open_url", "file:///c:/x"),
                    ("open_url", "https://example.com"), ("window", "format")):
        t._do(kind, v)
    pyautogui.hotkey, webbrowser.open = real_hot, real_open
    assert sent == [("ctrl", "t"), ("win", "r"), "https://example.com"], sent  # no file:// links, no invented keys
    script = iter(['{"say": "Checking", "do": [{"powershell": "Write-Output 42"}], "done": false}',
                   '{"say": "It is 42", "do": [], "done": true}'])
    seen, real_chat = [], chat
    globals()["chat"] = lambda cfg, msgs, note=None: (seen.append(msgs[-1]["content"]), next(script))[1]
    try:
        assert t._agent("what is the answer") == "It is 42"
    finally:
        globals()["chat"] = real_chat
    assert seen[1] == "Output:\n42", seen  # the model got to see what its command printed
    real_save = save_config
    globals()["save_config"] = lambda c: None  # the test must not touch your real settings
    try:
        assert t._agent("turn off the camera").startswith("Camera off") and not t.camera_on
        assert t._agent("camera on").startswith("Camera on") and t.camera_on
    finally:
        globals()["save_config"] = real_save
    tone = lambda f0, f1: (np.sin(np.cumsum(np.linspace(f0, f1, 9600)) * 2 * np.pi / 16000) * 12000).astype(np.int16).tobytes()
    a1, a2, b1 = mfcc(tone(300, 900)), mfcc(tone(310, 880)), mfcc(tone(1500, 400))
    assert dtw(a1, a2)[0] < dtw(a1, b1)[0] * 0.6  # the same sweep matches itself far better than a different one
    assert dtw(a1, a2[:10])[1] < 10  # short audio buffer (< 0.6 * T) must not raise ValueError on argmin
    assert parse_plan('<think>unclosed reasoning {"open_app": "Spotify"}')[1] == []
    assert parse_plan('{"say": "Opened Spotify.", "do": {"open_app": "Spotify"}}')[1] == [{"open_app": "Spotify"}]
    sent2 = []
    real_io = pyautogui.scroll, pyautogui.click, pyautogui.press
    pyautogui.scroll, pyautogui.click, pyautogui.press = (lambda n: sent2.append(("scroll", n)),
                                                            lambda: sent2.append("click"),
                                                            lambda k, **kw: sent2.append(("press", k)))
    for kind, v in (("scroll", "down 6"), ("scroll", "up"), ("scroll", "bottom"), ("mouse", "click"), ("mouse", "bogus")):
        t._do(kind, v)
    pyautogui.scroll, pyautogui.click, pyautogui.press = real_io
    k = 120 if sys.platform == "win32" else 1
    assert sent2 == [("scroll", -6 * k), ("scroll", 6 * k), ("press", "end"), "click"], sent2
    for mode, search in (("launcher", False), ("pad", False), ("pad", True), ("keyboard", False)):
        t.set_mode(mode, search)
        t.query = "spot"
        ui = t._layout()
        assert ui.buttons and all(0 <= v <= 1 for b in ui.buttons for v in b.r), mode
    real = pyautogui.position, pyautogui.mouseDown, pyautogui.doubleClick
    calls = []
    pyautogui.position = lambda: (500, 500)
    pyautogui.mouseDown = lambda *a: calls.append(("down",))
    pyautogui.doubleClick = lambda *a: calls.append(("double", *a))
    t.set_mode(None)
    t._pinch_start(0, 0, both_hands=False)                  # first pinch: a normal press
    t.mouse_down, t.tap_at = False, time.monotonic()        # ...released quickly = a click
    t._pinch_start(0, 0, both_hands=False)                  # second pinch soon after: open it
    assert calls == [("down",), ("double", 500, 500)], calls
    t._pinch_start(0, 0, both_hands=False)                  # a third pinch is a fresh click again
    assert calls[-1] == ("down",), calls
    t.mouse_down = False
    pyautogui.mouseDown, pyautogui.doubleClick = real[1], real[2]
    t.last_set = (500, 503)
    assert not t._mouse_taken(10.0)                         # where we left it (rounding): still ours
    t.last_set = (900, 500)
    assert t._mouse_taken(10.0) and t._mouse_taken(11.0)   # you moved it: hands off...
    assert not t._mouse_taken(10.0 + MOUSE_TAKEOVER_S)      # ...for MOUSE_TAKEOVER_S only
    pyautogui.position = real[0]
    t.set_mode("pad")
    t.alts = ["x"]
    assert any(b.id.startswith("alt") for b in t._layout().buttons)

    if sys.platform == "win32":  # real handwriting recognition, from a worker thread like the pad
        circle = lambda cx, cy, r: [(cx - r * math.sin(a / 10 * math.pi), cy - r * math.cos(a / 10 * math.pi))
                                    for a in range(21)]
        line = lambda x, y0, y1: [(x, y0 + (y1 - y0) * i / 10) for i in range(11)]
        oil = [[(x / 1000, y / 1000) for x, y in s] for s in
               (circle(60, 130, 28), line(130, 100, 160), [(130, 75)], line(180, 40, 160))]
        oil_messy = [beautify(hooked(shake(s, 0.0025)) if len(s) > 1 else s, 1.0) for s in oil]  # shaky + hooks
        got = []
        th = threading.Thread(target=lambda: got.extend(recognize(o, 1000, 1000) for o in (oil, oil, oil_messy)))
        th.start()
        th.join(20)
        assert len(got) == 3 and all("oil" in [c.lower() for c in cands] for cands in got), got
        assert list_windows() is not None
        assert isinstance(ui_list(), str)  # UI Automation answers (whatever window is in front)
    make_detector()  # model + mediapipe binaries load (catches broken .exe bundles)
    print("selftest ok")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "--test":
        selftest()
    elif arg == "--setup":
        run_setup_cli()
    elif arg is None or arg.isdigit():
        main(int(arg) if arg else None)
    else:
        sys.exit(__doc__)  # --help or a typo: show usage instead of a traceback
