# Texas Hold'em Screenshot + Prediction Project for [WSOP online Hold'em game](https://www.playwsop.com/?utm_source=wsopcom&utm_medium=banner&utm_campaign=wsop.com_930x200_bigbanner&pid=wsop.com&c=animatedbanner_930x200)

This project demonstrates an end-to-end workflow for automatically **screenshotting** parts of the screen, **recognizing** playing cards (suits and ranks), and **comparing** two poker hands (left vs. right) given a set of public cards. It consists of:

- A **SwiftUI** app (macOS) that handles:
  - Automated or one-time screenshots of defined screen regions.
  - Calling a local **Python/Flask** backend for card recognition and poker-hand evaluation.
  - Displaying recognized card images & corresponding predictions (plus an optional minimal mode).

- A **Python/Flask** server that:
  - Loads a CNN model (or other card recognition logic).
  - Receives requests from the SwiftUI app for `/predict_all`.
  - Returns JSON with recognized card strings, best five-card combos, and the winning side.

---

## 1. Prerequisites

1. **macOS** environment with Swift (or Xcode) to build and run the SwiftUI app.
   I also provide a compiled version of the app, download via this link:
   [下载链接](https://drive.google.com/file/d/1fU5TC__nVRITE330NMoU1wMOTpNxgfV_/view?usp=sharing)
2. **Python 3** with:
   - `Flask` (for the web server),
   - `torch`, `torchvision` (for the PyTorch model),
   - `Pillow` (for image loading).
3. Optionally, a trained PyTorch model (`cardnet_resnet18.pth` or similar) placed in the same folder as the Python script, so the server can load it.

---

## 2. Project Structure

A simplified layout example:


   ```bash
my_card_app/
├── app.py                # Python Flask server
├── cardnet_resnet18.pth  # (Optional) PyTorch model weights
├── templates/
│    └── index.html       # Simple front-end for testing
├── screen_shot/
│    └── (Xcode project for SwiftUI)
│         ├── ContentView.swift
│         └── ...
└── README.md             # You're reading this file
      ```

Within the SwiftUI app (`ContentView.swift`), you will find:
- **Coordinates** & bounding boxes for 9 positions:
  - 5 public cards (common1..common5),
  - 2 left-hand cards (left6..left7),
  - 2 right-hand cards (right8..right9).
- A **captureAndEvaluate** function that:
  1. Clears old image files (like `1.png`, `2.png`, etc.),
  2. Screenshots each defined region into those image files,
  3. Calls `http://127.0.0.1:5000/predict_all`,
  4. Displays the resulting recognized cards and winner.

The Python server (`app.py`) has routes:
- `/predict_all`: Looks for images in `src/common`, `src/hand_l`, `src/hand_r`, does card recognition, returns JSON with recognized strings & best combos.

---

## 3. How to Run

1. **Start Python server**:
   ```bash
   cd my_card_app
   python app.py
   # Server listens on http://127.0.0.1:5000
   ```
2. **Open the SwiftUI (Xcode) project**:
   ```bash
   open screen_shot/screen_shot.xcodeproj
   ```
3. **Build & Run** the SwiftUI app.  
   - By default, it expects the server on `127.0.0.1:5000`.  
   - The app will create screenshots with `screencapture -R...`, saving them to `1.png, 2.png, ...` in subfolders (like `src/common`, `src/hand_l`, `src/hand_r`).

4. **Press “截取并识别牌面”** in the app UI to do a one-shot process:
   - The Swift app removes old image files, captures new screenshots, calls Python’s `/predict_all`, and shows recognized results.

5. **Toggle “开始自动截屏”** for repeated screenshots every ~0.6 seconds.  
   - The app calls the same flow each time, updating the displayed predictions.

---

## 4. Features

- **Advanced Settings**:  
  In the SwiftUI app, you can expand “Advanced Settings” to configure each card’s bounding box (x,y,w,h), plus the subfolder for storing each card (`src/common`, `src/hand_l`, etc.).  
  - Press “确认并保存(可选)” to update config in real time and optionally persist it in `UserDefaults`.

- **极简模式 (simpleMode)**:  
  - Toggle “切换到极简模式” in the full UI to reduce the window to a small box that only displays text-based outputs (public cards, left/right combos, and final winner).
  - You can still do “截取并识别牌面” or “开始自动截屏” in minimal mode, but all image previews and advanced settings are hidden.

---

## 5. Troubleshooting

- Make sure your **Python environment** has the right packages: `Flask`, `torch`, etc.  
- If the Swift app logs a 400 status from `/predict_all`, check that exactly 5 public-card images, 2 left-hand images, and 2 right-hand images exist.
- On **macOS 13+**, you may need to enable “Screen Recording” permission for the app or the screencapture command.

---

## 6. License

Feel free to adapt and customize this code for your own projects.  

