
<video src="https://raw.githubusercontent.com/HelloSpaghettiBot/EO-Captcha-Solver/main/screen-20251029-073456~2.mp4"
       controls
       loop
       muted
       playsinline
       style="max-width:100%;height:auto;">
</video>


# 🧠 EO Captcha Solver Setup & Usage Guide

## 📦 Installation

### Step 1: Install Python
- Download Python 3.8+ from https://www.python.org/downloads/
- ✅ Check "Add Python to PATH" during installation

### Step 2: Install Tesseract OCR
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install to default location: `C:\Program Files\Tesseract-OCR\`
- ⚠️ If you install elsewhere, update line 25 in the script with your path

### Step 3: Install Python Packages
Open Command Prompt or PowerShell and run:
```bash
pip install opencv-python numpy pytesseract mss pyautogui pywin32
```

---

## 🚀 Running the Solver

### Step 1: Start the Script
```bash
python captcha2.py
```

### Step 2: Safety Confirmation
- Type `#badcaptcha` in the console
- Press **SPACE** to continue

---

## 🎯 Setup Phase (One-Time Configuration)

### 1️⃣ Select Trigger Area
- A screenshot window will appear
- **Draw a box** around the word "reward" (the captcha trigger text)
- Press **SPACE** to confirm the box
- Press **ENTER** when done

### 2️⃣ Select 5 Letter Spots
- **Draw boxes** around each of the 5 captcha letters (in order: left to right)
- Press **SPACE** after each box to confirm
- Press **ENTER** when all 5 are selected

**Controls:**
- `SPACE` = Confirm current selection
- `ENTER` = Finish selection
- `BACKSPACE` = Undo last selection
- `ESC` or `Q` = Cancel

---

## ⚡ Solving Phase (Automatic)

### What Happens Next:
1. ⏳ Script waits for "reward" to appear in trigger area
2. 📸 OCR scans each letter spot 50 times
3. 🗳️ Majority vote determines each letter
4. 🎯 Focuses the game window
5. ⌨️ Types the solution automatically
6. ✅ Presses ENTER (if enabled)

### Example Output:
```
🔍 Spot 1
find N
find N
Answer is N

🔍 Spot 2
find W
find W
Answer is W

✅ Final Answers: Spot 1: N, Spot 2: W, Spot 3: K, Spot 4: T, Spot 5: O
📨 Solution to type: NWKTO
🎯 Focused window: 'EndlessOnline'
⌨️  Typing solution into the focused window...
[TYPING] Starting SendInput typing: NWKTO
  [1/5] Pressing key: n (VK: 0x4e)
  [2/5] Pressing key: w (VK: 0x57)
  [3/5] Pressing key: k (VK: 0x4b)
  [4/5] Pressing key: t (VK: 0x54)
  [5/5] Pressing key: o (VK: 0x4f)
  [ENTER] Pressing enter key
[TYPING] SendInput typing complete!
⌨️  Done.
```

---

## ⚙️ Configuration Options

Edit these at the top of the script:
```python
TRIGGER_PHRASE = "reward"          # Text to detect for trigger
TRIGGER_HITS_REQUIRED = 2          # Consecutive detections needed
SPOTS_REQUIRED = 5                 # Number of letter spots
FRAMES = 50                        # OCR attempts per letter
TYPE_INTERVAL = 0.15               # Delay between keypresses (seconds)
PRESS_ENTER_AT_END = True          # Auto-press Enter after typing
CLICK_CENTER_BEFORE_TYPING = True  # Click game window center first
```

---

## 🐛 Troubleshooting

### ❌ "Could not find window containing: 'EndlessOnline'"
- Make sure the game window title contains "EndlessOnline"
- Update `WINDOW_TITLE_SUBSTR` if your window has a different name

### ❌ OCR returns wrong letters
- Increase `FRAMES` (default 50) for more samples
- Adjust `TYPE_INTERVAL` if letters are typing too fast
- Check `DEBUG_PREVIEW = True` to see OCR preprocessing

### ❌ Keys not registering in game
- Ensure the input field is **active** (cursor blinking)
- Try increasing `TYPE_INTERVAL` (e.g., 0.20 or 0.25)
- Make sure game is truly focused

### ❌ Tesseract not found
- Verify installation path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Update line 25 if installed elsewhere

---

## 📝 Notes

- ⚠️ The script uses **real keyboard input** via SendInput
- 🎮 Game must be **focused** and input field **active**
- 🔄 Run the script fresh for each captcha session
- 💾 Selection areas are NOT saved between runs

---

## 🆘 Support

If you encounter issues:
1. Check the console output for error messages
2. Verify all packages installed correctly
3. Ensure Tesseract OCR is installed
4. Test with `DEBUG_PREVIEW = True` to see OCR processing
