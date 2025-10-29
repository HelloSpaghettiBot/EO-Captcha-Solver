# ──────────────────────────────────────────────────────────────────────────────
# EO: Triggered OCR (5 spots) → System Keystrokes via REAL SendInput (scan codes)
# CONTINUOUS MODE: Loops forever, solving captchas as they appear
# ──────────────────────────────────────────────────────────────────────────────
# Flow:
#   1) Select TRIGGER area (where "reward" appears) - ONCE at startup
#   2) Select 5 LETTER spots (SPACE to confirm each, ENTER when exactly 5) - ONCE at startup
#   3) Loop forever:
#      - Wait for "reward" trigger (2 consecutive hits)
#      - OCR all 5 spots
#      - Majority-vote letters, focus EndlessOnline, type via SendInput
#      - Return to waiting for next trigger
# ──────────────────────────────────────────────────────────────────────────────

import sys, time, re, ctypes, msvcrt
import cv2
import numpy as np
import pytesseract
import mss
from collections import Counter

import pyautogui
try:
    import win32gui, win32con, win32process
except ImportError:
    win32gui = win32con = win32process = None

from ctypes import wintypes

# ─────────── USER CONFIG ───────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

WINDOW_TITLE_SUBSTR = "EndlessOnline"   # window title substring to focus

TRIGGER_PHRASE = "reward"
TRIGGER_CHECK_DELAY = 0.5
TRIGGER_HITS_REQUIRED = 2

SPOTS_REQUIRED = 5
FRAMES = 50
DELAY = 0.20

CLICK_CENTER_BEFORE_TYPING = True
PRESS_ENTER_AT_END = True
TYPE_INTERVAL = 0.15  # seconds between letters (longer = safer)

# OCR tuning for letters on black with confetti
SAT_THRESH = 30
V_MIN, V_MAX = 110, 230
MIN_CHAR_AREA_RATIO = 0.002
UPSCALE = 2
DILATE_KERNEL = 1

DEBUG_PREVIEW = False

# Cooldown after solving to avoid re-triggering on same captcha
POST_SOLVE_COOLDOWN = 3.0  # seconds to wait after typing solution
# ───────────────────────────────────


# =================== SENDINPUT KEYBOARD (LIKE YOUR EXAMPLE) ===================
user32 = ctypes.WinDLL('user32', use_last_error=True)
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP       = 0x0002
KEYEVENTF_UNICODE     = 0x0004
MAPVK_VK_TO_VSC = 0

wintypes.ULONG_PTR = wintypes.WPARAM

class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx",          wintypes.LONG),
                ("dy",          wintypes.LONG),
                ("mouseData",   wintypes.DWORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk",         wintypes.WORD),
                ("wScan",       wintypes.WORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))
    def __init__(self, *args, **kwds):
        super(KEYBDINPUT, self).__init__(*args, **kwds)
        if not self.dwFlags & KEYEVENTF_UNICODE:
            self.wScan = user32.MapVirtualKeyExW(self.wVk,
                                                 MAPVK_VK_TO_VSC, 0)

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg",    wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                    ("mi", MOUSEINPUT),
                    ("hi", HARDWAREINPUT))
    _anonymous_ = ("_input",)
    _fields_ = (("type",   wintypes.DWORD),
                ("_input", _INPUT))

LPINPUT = ctypes.POINTER(INPUT)

def PressKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode,
                            dwFlags=KEYEVENTF_KEYUP))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

keyCodeMap = {
    'shift'             : "0x10",
    '0'                 : "0x30",
    '1'                 : "0x31",
    '2'                 : "0x32",
    '3'                 : "0x33",
    '4'                 : "0x34",
    '5'                 : "0x35",
    '6'                 : "0x36",
    '7'                 : "0x37",
    '8'                 : "0x38",
    '9'                 : "0x39",
    'a'                 : "0x41",
    'b'                 : "0x42",
    'c'                 : "0x43",
    'd'                 : "0x44",
    'e'                 : "0x45",
    'f'                 : "0x46",
    'g'                 : "0x47",
    'h'                 : "0x48",
    'i'                 : "0x49",
    'j'                 : "0x4A",
    'k'                 : "0x4B",
    'l'                 : "0x4C",
    'm'                 : "0x4D",
    'n'                 : "0x4E",
    'o'                 : "0x4F",
    'p'                 : "0x50",
    'q'                 : "0x51",
    'r'                 : "0x52",
    's'                 : "0x53",
    't'                 : "0x54",
    'u'                 : "0x55",
    'v'                 : "0x56",
    'w'                 : "0x57",
    'x'                 : "0x58",
    'y'                 : "0x59",
    'z'                 : "0x5A",
    'enter'             : "0x0D",
}

def toKeyCode(c):
    keyCode = keyCodeMap.get(c.lower())
    if keyCode:
        return int(keyCode, base=16)
    return None

def type_text_sendinput(text, interval=TYPE_INTERVAL):
    """Types text using SendInput like your F12 example."""
    print(f"[TYPING] Starting SendInput typing: {text}")
    
    for i, char in enumerate(text, 1):
        keycode = toKeyCode(char)
        if keycode:
            print(f"  [{i}/{len(text)}] Pressing key: {char} (VK: {hex(keycode)})")
            PressKey(keycode)
            time.sleep(0.05)  # Hold key briefly
            ReleaseKey(keycode)
            time.sleep(interval)
        else:
            print(f"  [{i}/{len(text)}] Skipping unknown char: {char}")
    
    if PRESS_ENTER_AT_END:
        print("  [ENTER] Pressing enter key")
        enter_code = toKeyCode('enter')
        PressKey(enter_code)
        time.sleep(0.05)
        ReleaseKey(enter_code)
    
    print("[TYPING] SendInput typing complete!")


# =================== OCR / PREPROCESS ===================
def _gray_only_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = (s < SAT_THRESH) & (v >= V_MIN) & (v <= V_MAX)
    return (mask.astype(np.uint8) * 255)

def _area_filter(bin_img, min_area_px):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    out = np.zeros_like(bin_img)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            out[labels == i] = 255
    return out

def preprocess_for_ocr(bgr):
    # 1. HSV gray-only mask
    mask = _gray_only_mask(bgr)
    masked = cv2.bitwise_and(bgr, bgr, mask=mask)

    # 2. Grayscale + CLAHE
    g = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    g = clahe.apply(g)

    # 3. Otsu binarization
    _, bin_img = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Area filter (confetti removal)
    h, w = bin_img.shape[:2]
    min_area = max(20, int(MIN_CHAR_AREA_RATIO * h * w))
    clean = _area_filter(bin_img, min_area_px=min_area)

    # 5. Inverted close to preserve “H” shape
    closed = cv2.bitwise_not(clean)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    clean = cv2.bitwise_not(closed)

    # 6. Stroke strengthening (MORPH_RECT instead of ellipse)
    if DILATE_KERNEL > 0:
        size = max(1, int(DILATE_KERNEL))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=1)

    # 7. Upscale image
    if UPSCALE > 6:
        clean = cv2.resize(clean, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_LINEAR)

    # 8. Optional visual debug
    if DEBUG_PREVIEW:
        cv2.imshow("OCR Preprocess (letter)", clean)
        cv2.waitKey(1)

    return clean

    # 1. HSV gray-only mask
    mask = _gray_only_mask(bgr)
    masked = cv2.bitwise_and(bgr, bgr, mask=mask)

    # 2. Grayscale + Otsu binarization
    g = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Area filter to remove small blobs (confetti)
    h, w = bin_img.shape[:2]
    min_area = max(20, int(MIN_CHAR_AREA_RATIO * h * w))
    clean = _area_filter(bin_img, min_area_px=min_area)

    # 4. FIX for "H" – inverted close to seal inner gaps
    closed = cv2.bitwise_not(clean)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    clean = cv2.bitwise_not(closed)

    # 5. Optional dilation to strengthen strokes
    if DILATE_KERNEL > 0:
        size = max(1, int(DILATE_KERNEL))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=1)

    # 6. Upscale image to help Tesseract
    if UPSCALE > 5:
        clean = cv2.resize(clean, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_LINEAR)

    # 7. Optional preview window
    if DEBUG_PREVIEW:
        cv2.imshow("OCR Preprocess (letter)", clean)
        cv2.waitKey(1)

    return clean

    mask = _gray_only_mask(bgr)
    masked = cv2.bitwise_and(bgr, bgr, mask=mask)
    g = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = bin_img.shape[:2]
    min_area = max(20, int(MIN_CHAR_AREA_RATIO * h * w))
    clean = _area_filter(bin_img, min_area_px=min_area)
    if DILATE_KERNEL > 0:
        size = max(1, int(DILATE_KERNEL))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=1)
    if UPSCALE > 1:
        clean = cv2.resize(clean, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_LINEAR)
    if DEBUG_PREVIEW:
        cv2.imshow("OCR Preprocess (letter)", clean); cv2.waitKey(1)
    return clean

def preprocess_for_line(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = (s < 60) & (v > 100)
    mask = (mask.astype(np.uint8) * 255)
    masked = cv2.bitwise_and(bgr, bgr, mask=mask)
    g = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    _, bin_img = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=1)
    bin_img = cv2.resize(bin_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return bin_img

def ocr_text_line(bgr):
    proc = preprocess_for_line(bgr)
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(proc, config=config)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def ocr_letter_from_bgr(img_bgr):
    proc = preprocess_for_ocr(img_bgr)
    config = "--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    text = pytesseract.image_to_string(proc, config=config)
    letters = [c for c in text if c.isalpha()]
    return letters[0].upper() if letters else ""


# =================== REGION SELECTION ===================
def select_single_region(screen, title="Select Region"):
    saved = None
    selecting = False
    x1 = y1 = x2 = y2 = 0

    def mouse_cb(event, x, y, flags, param):
        nonlocal selecting, x1, y1, x2, y2
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1 = x, y; x2, y2 = x, y; selecting = True
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            x2, y2 = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            x2, y2 = x, y; selecting = False

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse_cb)

    while True:
        display = screen.copy()
        if saved is not None:
            pt1 = (saved["left"], saved["top"])
            pt2 = (saved["left"]+saved["width"], saved["top"]+saved["height"])
            cv2.rectangle(display, pt1, pt2, (0,200,0), 2)
            cv2.putText(display, "Saved", (saved["left"]+5, saved["top"]+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2, cv2.LINE_AA)
        if selecting or (x1!=x2 and y1!=y2):
            cv2.rectangle(display, (x1,y1), (x2,y2), (0,200,255), 1)

        cv2.putText(display, "[SPACE]=confirm  [ENTER]=done  [BACKSPACE]=clear  [ESC/q]=cancel",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        cv2.imshow(title, display)
        key = cv2.waitKey(20)
        if key == -1: continue
        if key in (27, ord('q')): cv2.destroyWindow(title); print("Selection cancelled."); return None
        if key in (8, 127): saved=None; x1=y1=x2=y2=0; print("Cleared saved region.")
        if key == ord(' '):
            xa, ya = min(x1,x2), min(y1,y2); xb, yb = max(x1,x2), max(y1,y2)
            w, h = xb-xa, yb-ya
            if w>5 and h>5:
                saved = {"left": xa, "top": ya, "width": w, "height": h}
                print(f"Saved region: left={xa}, top={ya}, width={w}, height={h}")
                x1=y1=x2=y2=0
            else:
                print(f"Box too small — draw bigger. (w={w}, h={h})")
        if key in (13,10):
            if saved is not None:
                cv2.destroyWindow(title); return saved
            else:
                print("No region saved. Press SPACE to confirm a box first.")

def select_multiple_regions(screen):
    spots = []
    selecting = False
    x1 = y1 = x2 = y2 = 0

    def mouse_cb(event, x, y, flags, param):
        nonlocal selecting, x1, y1, x2, y2
        if event == cv2.EVENT_LBUTTONDOWN:
            x1,y1=x,y; x2,y2=x,y; selecting=True
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            x2,y2=x,y
        elif event == cv2.EVENT_LBUTTONUP:
            x2,y2=x,y; selecting=False

    win = f"Select {SPOTS_REQUIRED} Letter Spots"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        display = screen.copy()
        for idx, r in enumerate(spots, start=1):
            pt1=(r["left"],r["top"]); pt2=(r["left"]+r["width"], r["top"]+r["height"])
            cv2.rectangle(display, pt1, pt2, (0,200,0), 2)
            cv2.putText(display, f"{idx}", (r["left"]+5, r["top"]+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2, cv2.LINE_AA)
        if selecting or (x1!=x2 and y1!=y2):
            cv2.rectangle(display, (x1,y1), (x2,y2), (0,200,255), 1)

        cv2.putText(display,
                    f"Draw {SPOTS_REQUIRED} spots  [SPACE=confirm]  [ENTER=done]  [BACKSPACE=undo]  [ESC/q=cancel]  Saved:{len(spots)}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

        cv2.imshow(win, display)
        key = cv2.waitKey(20)
        if key == -1: continue
        if key in (27, ord('q')): cv2.destroyWindow(win); print("Selection cancelled."); return []
        if key in (8, 127):
            if spots:
                rem=spots.pop(); print(f"Removed: left={rem['left']} top={rem['top']} w={rem['width']} h={rem['height']}")
            else:
                print("No spots to remove.")
        if key == ord(' '):
            xa,ya=min(x1,x2),min(y1,y2); xb,yb=max(x1,x2),max(y1,y2); w,h=xb-xa,yb-ya
            if w>5 and h>5:
                r={"left":xa,"top":ya,"width":w,"height":h}; spots.append(r)
                print(f"Confirmed spot #{len(spots)}: left={xa}, top={ya}, width={w}, height={h}")
                x1=y1=x2=y2=0
            else:
                print(f"Box too small — draw bigger. (w={w}, h={h})")
        if key in (13,10):
            if len(spots)==SPOTS_REQUIRED:
                cv2.destroyWindow(win); return spots
            else:
                print(f"Have {len(spots)}; need {SPOTS_REQUIRED}.")


# =================== TRIGGER OCR ===================
def phrase_is_present_in_region(sct, region, phrase=TRIGGER_PHRASE):
    img = np.array(sct.grab(region))[:, :, :3]
    text = ocr_text_line(img)
    norm = re.sub(r"\s+", " ", text).strip()
    return phrase.lower() in norm

def wait_for_trigger_in_region(trigger_region, captcha_num):
    print(f"⏳ [Captcha #{captcha_num}] Waiting for trigger '{TRIGGER_PHRASE}' in trigger area...")
    hits = 0
    with mss.mss() as sct:
        while True:
            if phrase_is_present_in_region(sct, trigger_region, TRIGGER_PHRASE):
                hits += 1
                if hits >= TRIGGER_HITS_REQUIRED:
                    print(f"✅ [Captcha #{captcha_num}] Trigger detected! Starting OCR.")
                    return
            else:
                hits = 0
            time.sleep(TRIGGER_CHECK_DELAY)


# =================== LETTER SCAN ===================
def scan_region(region, frames=FRAMES, delay=DELAY):
    found = []
    with mss.mss() as sct:
        for _ in range(frames):
            img = np.array(sct.grab(region))[:, :, :3]
            letter = ocr_letter_from_bgr(img)
            if letter:
                print(f"find {letter}")
                found.append(letter)
            time.sleep(delay)
    if not found:
        print("Answer is ?"); return "?"
    ans = Counter(found).most_common(1)[0][0]
    print(f"Answer is {ans}")
    return ans


# =================== FOCUS WINDOW ===================
def _enum_windows_by_title(substr):
    matches=[]
    if not win32gui: return matches
    def _cb(hwnd,_):
        if win32gui.IsWindowVisible(hwnd):
            t=win32gui.GetWindowText(hwnd)
            if substr.lower() in t.lower(): matches.append((hwnd,t))
    win32gui.EnumWindows(_cb,None)
    return matches

def _bring_to_foreground(hwnd):
    if not win32gui: return False
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE); time.sleep(0.05)
    try: win32gui.SetForegroundWindow(hwnd)
    except Exception: pass
    fg=win32gui.GetForegroundWindow()
    if fg!=hwnd:
        try:
            user32_win=ctypes.windll.user32
            cur=win32process.GetWindowThreadProcessId(fg)[0]
            tgt=win32process.GetWindowThreadProcessId(hwnd)[0]
            user32_win.AttachThreadInput(cur, tgt, True)
            win32gui.SetForegroundWindow(hwnd)
            user32_win.AttachThreadInput(cur, tgt, False)
        except Exception: pass
    return win32gui.GetForegroundWindow()==hwnd

def focus_endless_online_and_click_center():
    hwnd=None; title=None; L=T=R=B=0
    if win32gui:
        m=_enum_windows_by_title(WINDOW_TITLE_SUBSTR)
        if m:
            hwnd,title=m[0]
            if not _bring_to_foreground(hwnd): hwnd=None
        if hwnd and CLICK_CENTER_BEFORE_TYPING:
            L,T,R,B=win32gui.GetWindowRect(hwnd)
    if hwnd is None:
        try:
            wins=pyautogui.getWindowsWithTitle(WINDOW_TITLE_SUBSTR)
            if wins:
                w=wins[0]
                if w.isMinimized: w.restore(); time.sleep(0.1)
                w.activate(); time.sleep(0.1)
                title=w.title; L,T,R,B=w.left,w.top,w.right,w.bottom
            else:
                print(f"⚠️ Could not find window containing: '{WINDOW_TITLE_SUBSTR}'"); return False
        except Exception:
            print(f"⚠️ Could not find window containing: '{WINDOW_TITLE_SUBSTR}'"); return False
    print(f"🎯 Focused window: '{title or WINDOW_TITLE_SUBSTR}'")
    if CLICK_CENTER_BEFORE_TYPING:
        cx=int((L+R)/2); cy=int((T+B)/2); pyautogui.click(cx,cy); time.sleep(0.08)
    return True


# =================== INPUT PROMPT ===================
def _wait_for_input(prompt, required_text="#badcaptcha"):
    print(prompt)
    print(f"→ Type '{required_text}' and press SPACE to continue (ESC to cancel)")
    buffer = ""
    while True:
        ch = msvcrt.getch()
        if ch == b'\x1b':  # ESC
            return False
        elif ch == b' ':  # SPACE
            if buffer.strip() == required_text:
                print()
                return True
            else:
                print(f"\n⚠️ Please type '{required_text}' exactly, then press SPACE")
                print(f"  Current input: '{buffer}'")
        elif ch == b'\r':  # ENTER - ignore
            continue
        elif ch == b'\x08':  # BACKSPACE
            if buffer:
                buffer = buffer[:-1]
                print('\r' + ' ' * 80 + '\r' + buffer, end='', flush=True)
        else:
            try:
                char = ch.decode('ascii')
                if char.isprintable():
                    buffer += char
                    print(char, end='', flush=True)
            except:
                pass


# =================== CAPTCHA SOLVING LOOP ===================
def solve_one_captcha(trigger_region, regions, captcha_num):
    """Solves a single captcha instance."""
    # Wait for trigger
    wait_for_trigger_in_region(trigger_region, captcha_num)

    print(f"\n📸 [Captcha #{captcha_num}] Beginning OCR on letter spots...\n")
    answers = []
    for i, region in enumerate(regions, start=1):
        print(f"\n🔍 Spot {i}")
        answers.append(scan_region(region))

    solution = "".join([a for a in answers if a and a.isalpha()]).upper()
    print(f"\n✅ [Captcha #{captcha_num}] Final Answers: {', '.join(f'Spot {i+1}: {a}' for i,a in enumerate(answers))}")
    print(f"📨 [Captcha #{captcha_num}] Solution to type: {solution}")

    if focus_endless_online_and_click_center():
        time.sleep(0.5)
        
        if win32gui:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            if WINDOW_TITLE_SUBSTR.lower() not in (title or "").lower():
                print(f"⚠️ Warning: Window '{title}' is focused instead of EndlessOnline")
                time.sleep(0.2)

        print(f"⌨️  [Captcha #{captcha_num}] Typing solution into the focused window...")
        type_text_sendinput(solution, interval=TYPE_INTERVAL)
        print(f"⌨️  [Captcha #{captcha_num}] Done typing!")
    else:
        print(f"❌ [Captcha #{captcha_num}] Could not focus EndlessOnline to type the solution.")
    
    # Cooldown to avoid re-triggering on same captcha
    print(f"⏱️  [Captcha #{captcha_num}] Cooldown for {POST_SOLVE_COOLDOWN}s to avoid re-trigger...")
    time.sleep(POST_SOLVE_COOLDOWN)


# =================== MAIN ===================
def main():
    print("🧠 EO: OCR-Based Captcha Solver with SendInput (CONTINUOUS MODE)")
    print("="*70)
    input("Press ENTER to start the captcha solver...")

    # Wait for #badcaptcha + SPACE before proceeding
    if not _wait_for_input("\n🔒 Type '#badcaptcha' to confirm you're ready to proceed."):
        print("❌ Operation cancelled by user.")
        return

    # ONE-TIME SETUP: Prepare selection canvases
    print("\n🖼️  Capturing screen for region selection...")
    with mss.mss() as sct:
        screenshot = np.array(sct.grab(sct.monitors[1]))
        screen = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # ONE-TIME: Select trigger area (around "reward")
    print("\n📍 STEP 1: Select the TRIGGER area (where 'reward' text appears)")
    trigger_region = select_single_region(screen, title="Select TRIGGER Area (around 'reward')")
    if not trigger_region:
        print("❌ No trigger area selected. Exiting.")
        return

    # ONE-TIME: Select 5 letter spots
    print("\n📍 STEP 2: Select the 5 letter spots")
    regions = select_multiple_regions(screen)
    if len(regions) != SPOTS_REQUIRED:
        print(f"❌ Not enough letter spots selected (got {len(regions)}, need {SPOTS_REQUIRED}). Exiting.")
        return

    print("\n" + "="*70)
    print("✅ Setup complete! Starting continuous captcha solving loop...")
    print("   The bot will now run forever, solving captchas as they appear.")
    print("   Press Ctrl+C to stop the bot.")
    print("="*70 + "\n")

    # CONTINUOUS LOOP
    captcha_count = 0
    try:
        while True:
            captcha_count += 1
            print(f"\n{'='*70}")
            print(f"🔄 CAPTCHA #{captcha_count}")
            print(f"{'='*70}\n")
            
            solve_one_captcha(trigger_region, regions, captcha_count)
            
            print(f"\n✅ [Captcha #{captcha_count}] Completed! Returning to waiting state...\n")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Bot stopped by user (Ctrl+C)")
        print(f"📊 Total captchas solved: {captcha_count}")
    
    finally:
        if DEBUG_PREVIEW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    if sys.platform.startswith("win") and win32gui is None:
        print("Note: pywin32 not found. Focusing uses pyautogui fallback. For best results: pip install pywin32")
    main()