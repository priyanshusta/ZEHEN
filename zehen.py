r"""ZEHEN: control your PC with hand gestures. Camera runs hidden; a see-through HUD shows feedback.

Install:  python -m pip install -r requirements.txt
Run:      python zehen.py            (finds your real camera; python zehen.py 1 forces camera #1)
Test:     python zehen.py --test
Build:    double-click build.bat  ->  dist\ZEHEN.exe  (needs ~1 GB free disk space while building)
Quit:     the X on the guide panel. Touch the real mouse/touchpad and ZEHEN lets go of the cursor.
Settings: %USERPROFILE%\.zehen.json  ("main_hand": "Left" for left-handers, "apps": [names for the launcher])
"""
import ctypes
import difflib
import json
import math
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
import unicodedata
import urllib.request
from pathlib import Path
from tkinter import messagebox
from types import SimpleNamespace

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")  # else camera open takes ~20 s on Windows
import cv2
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
           "•  Nothing is recorded, saved or sent anywhere: it stays on this computer.\n"
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
            ("three", HOLD_S, "Three fingers up, hold ✍", "The writing pad: pinch and write, it types for you.")]
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

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t), ("_pad", ctypes.c_ubyte * 8)]

    class _INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("ki", _KEYBDINPUT)]

    u32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]


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


def unicode_events(text):
    """UTF-16LE (scan, flags) pairs for Win32 SendInput(KEYEVENTF_UNICODE): layout-independent, no clipboard."""
    raw = text.encode("utf-16-le")
    units = [int.from_bytes(raw[i:i + 2], "little") for i in range(0, len(raw), 2)]
    return [(u, f) for u in units for f in (0x0004, 0x0006)]  # KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP


def ctrl_scroll(notches):
    pyautogui.keyDown("ctrl")
    try:
        pyautogui.scroll(notches * (120 if sys.platform == "win32" else 1))
    finally:
        pyautogui.keyUp("ctrl")


def make_detector():
    if not MODEL.exists():
        tmp = MODEL.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(MODEL)
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.6, min_tracking_confidence=0.6))


class Tracker:
    """Camera + gestures on a background thread. The UI only reads self.state, a fresh snapshot per frame.
    Main hand (right, or whichever is alone) drives the cursor; the other hand zooms and switches apps."""

    def __init__(self, cam, cfg, start=True):
        self.cam, self.cfg = cam, cfg
        self.running, self.paused, self.error, self.mouse_down = True, False, None, False
        self.lock = threading.Lock()  # guards self.ink: the recognizer thread trims it while we draw into it
        self.mode, self.search = None, False  # mode: None | "launcher" | "pad" | "keyboard"
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
        if start:
            threading.Thread(target=self._load_apps, daemon=True).start()
            threading.Thread(target=self._run, daemon=True).start()

    def _load_apps(self):
        try:
            self.apps = installed_apps()
        except Exception:
            self.apps = []  # launcher still has windows, controls and the writing pad

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
        det, cam = make_detector(), self.cam
        sw, sh = pyautogui.size()
        fx, fy = OneEuro(), OneEuro()
        pfx, pfy = OneEuro(INK_MIN_CUTOFF, INK_BETA), OneEuro(INK_MIN_CUTOFF, INK_BETA)
        self.aspect = sw / sh
        raw, count, gesture, since, fired = None, 0, None, 0.0, False
        lock_until, scroll_y0, main_label = 0.0, None, None
        m = lambda v: min(max((v - MARGIN) / (1 - 2 * MARGIN), 0.0), 1.0)
        try:
            while self.running:
                ok, frame = cam.read()
                if not ok:
                    raise RuntimeError(CAMERA_HELP)
                now = time.monotonic()
                rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)  # mirror: move right = cursor right
                res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(now * 1000))
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
            cam.release()

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
        # a small gap per key: at full speed Notepad dropped letters ("look ma" arrived as "look a")
        if text.isascii() or sys.platform != "win32":
            pyautogui.write(text, interval=0.015)
        else:
            for ch in text:
                if ch.isascii():
                    pyautogui.write(ch, interval=0.015)
                else:
                    ev = (_INPUT * len(pairs := unicode_events(ch)))(*(
                        _INPUT(1, _KEYBDINPUT(0, u, f, 0, 0)) for u, f in pairs))
                    u32.SendInput(len(ev), ev, ctypes.sizeof(_INPUT))
                    time.sleep(0.015)
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
        favs = [names[fold(n)] for n in self.cfg.get("apps", DEFAULT_APPS) if fold(n) in names][:COLS - 1]
        section("APPS", [(f"▶\n{n}", lambda n=n, a=a: self._launch(n, a)) for n, a in favs]
                + [("🔍\nSearch all apps", lambda: self.set_mode("pad", search=True))], "")
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


class HUD:
    def __init__(self, root, tracker, cfg):
        self.root, self.t, self.cfg = root, tracker, cfg
        self.s = root.winfo_fpixels("1i") / 96  # Windows display scaling (125% -> 1.25)
        self.W, self.H = root.winfo_screenwidth(), root.winfo_screenheight()
        self.tut = None if cfg.get("tutorial_done") else 0
        self.great_until = self.step_start = 0.0
        self.collapsed = False

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
        self.pc = tk.Canvas(self.panel, bg=BG, highlightthickness=1, highlightbackground=DIM)
        self.pc.pack(fill="both", expand=True)
        for tag, fn in (("btn_pause", self.toggle_pause), ("btn_min", self.toggle_collapse),
                        ("btn_quit", self.quit), ("btn_tut", self.restart_tutorial), ("btn_skip", self.skip_tutorial)):
            self.pc.tag_bind(tag, "<Button-1>", lambda e, fn=fn: fn())
        click_through(self.panel, clicks=True)
        self.tick()

    def px(self, v):
        return int(v * self.s)

    def font(self, size, bold=False):
        return ("Segoe UI", self.px(size) * -1, "bold" if bold else "normal")  # negative = pixels

    # --- controls ---
    def toggle_pause(self):
        self.t.paused = not self.t.paused

    def toggle_collapse(self):
        self.collapsed = not self.collapsed

    def restart_tutorial(self):
        self.tut, self.step_start = 0, time.monotonic()

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
                if s.active == g and s.since >= self.step_start and now - s.since >= hold:
                    self.tut += 1
                    self.great_until = now + (3.0 if self.tut == len(TUTORIAL) else 1.0)
                    self.step_start = self.great_until
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

    def draw_panel(self, s, now):
        c, px, w = self.pc, self.px, self.px(330)
        c.delete("all")
        # header
        status, col = (("PAUSED", AMBER) if self.t.paused else ("TRACKING", GREEN) if s.hands
                       else ("SHOW YOUR HAND", AMBER))
        c.create_text(px(14), px(20), text="◉ ZEHEN", anchor="w", fill=CYAN, font=self.font(14, True))
        c.create_text(px(108), px(20), text=status, anchor="w", fill=col, font=self.font(10, True))
        for i, (tag, label) in enumerate((("btn_pause", "▶" if self.t.paused else "⏸"),
                                          ("btn_min", "+" if self.collapsed else "–"), ("btn_quit", "✕"))):
            c.create_text(w - px(84 - i * 30), px(20), text=label, fill=CYAN, font=self.font(15, True), tags=tag)
        h = px(40)
        # an open launcher/pad needs the space: shrink to the header (unless the tutorial is talking)
        if not (self.collapsed or (s.mode and self.tut is None)):
            # wireframe hands: proof it's tracking, without showing any video. Main hand cyan, other green.
            bx, by, bw, bh = px(14), px(44), w - px(28), px(130)
            c.create_rectangle(bx, by, bx + bw, by + bh, outline=DARK)
            fw = bh * 4 / 3  # camera is 4:3
            X = lambda p: bx + (bw - fw) / 2 + p[0] * fw
            Y = lambda p: by + p[1] * bh
            for hand, line, dot in zip(s.hands, (DIM, "#1f8f5a"), (CYAN, GREEN)):
                for a, b in HAND_LINKS:
                    c.create_line(X(hand[a]), Y(hand[a]), X(hand[b]), Y(hand[b]), fill=line, width=px(2))
                for p in hand:
                    c.create_oval(X(p) - 3, Y(p) - 3, X(p) + 3, Y(p) + 3, fill=dot, outline="")
            if not s.hands:
                c.create_text(bx + bw / 2, by + bh / 2, text="raise your hand in front of the camera",
                              fill=DIM, font=self.font(11))
            y = by + bh + px(14)
            h = self.draw_tutorial(c, y, w, now) if self.tut is not None else self.draw_guide(c, s, y, w)
        self.panel.geometry(f"{w}x{h}+{self.W - w - px(16)}+{px(16)}")

    def draw_tutorial(self, c, y, w, now):
        px, great = self.px, now < self.great_until
        if self.tut == len(TUTORIAL):
            title, hint = "You're all set 🚀", "Open palm closes any panel. ⏸ pauses, ✕ quits."
        else:
            _, _, title, hint = TUTORIAL[self.tut]
        c.create_text(px(14), y, text=f"TUTORIAL  {min(self.tut + 1, len(TUTORIAL))} / {len(TUTORIAL)}",
                      anchor="nw", fill=DIM, font=self.font(10, True))
        c.create_text(w - px(14), y, text="skip ›", anchor="ne", fill=DIM, font=self.font(10, True), tags="btn_skip")
        c.create_text(px(14), y + px(22), text="✓ GREAT!" if great and self.tut < len(TUTORIAL) else title,
                      anchor="nw", fill=GREEN if great else CYAN, font=self.font(17, True), width=w - px(28))
        c.create_text(px(14), y + px(54), text=hint, anchor="nw", fill="#9fdce6", font=self.font(12),
                      width=w - px(28))
        for i in range(len(TUTORIAL)):  # progress dots
            cx = px(18) + i * px(16)
            c.create_oval(cx - 4, y + px(104) - 4, cx + 4, y + px(104) + 4, outline=CYAN,
                          fill=CYAN if i < self.tut else "")
        return y + px(120)

    def draw_guide(self, c, s, y, w):
        px = self.px
        c.create_text(w - px(14), y - px(4), text="tutorial ↻", anchor="ne", fill=DIM,
                      font=self.font(10, True), tags="btn_tut")
        y += px(16)
        for g, icon, how, what in GUIDE:
            on = g == s.active
            if on:
                c.create_rectangle(px(8), y - px(2), w - px(8), y + px(30), fill=DARK, outline=CYAN)
            c.create_text(px(18), y + px(14), text=icon, anchor="w", fill=CYAN, font=self.font(15))
            c.create_text(px(50), y + px(7), text=how, anchor="w", fill=CYAN if on else TEXT,
                          font=self.font(12, True))
            c.create_text(px(50), y + px(22), text=what, anchor="w", fill=DIM if not on else CYAN,
                          font=self.font(10))
            y += px(36)
        return y + px(6)


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
    assert unicode_events("é") == [(0x00E9, 0x0004), (0x00E9, 0x0006)]
    if sys.platform == "win32":
        assert ctypes.sizeof(_INPUT) == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)

    t = Tracker(None, {}, start=False)  # every panel lays out, every tile fits on screen
    t.apps = apps
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
    make_detector()  # model + mediapipe binaries load (catches broken .exe bundles)
    print("selftest ok")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "--test":
        selftest()
    elif arg is None or arg.isdigit():
        main(int(arg) if arg else None)
    else:
        sys.exit(__doc__)  # --help or a typo: show usage instead of a traceback
