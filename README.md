# Capstoneptoject2025
# 🎮 Gesture-Based Media Control 🎥  
A Python-based real-time **hand gesture recognition system** that controls **VLC & Windows Media Player** using AI-powered computer vision.

---

## 🚀 Features
✅ **Real-Time Hand Gesture Detection** using OpenCV  
✅ **Trained Deep Learning Models (CNN & MobileNetV2)**  
✅ **Media Control (Play/Pause, Volume, Track Navigation, Brightness, Mute, Stop)**  
✅ **Works with VLC & Windows Media Player**  
✅ **Standalone `.exe` version available**  

---

## 📂 Project Structure


---

## 📸 **Supported Gestures**
| Gesture            | Action         |
|-------------------|---------------|
| ✋ Open Palm      | Play/Pause     |
| 👍 Thumbs Up      | Volume Up      |
| 👎 Thumbs Down    | Volume Down    |
| 👉 Point Right   | Next Track     |
| 👈 Point Left    | Previous Track |
| 🤫 Finger on Lips | Mute          |
| ✊ Fist           | Stop           |
| ☀️ Hand Open Up   | Brightness Up  |
| 🌑 Hand Open Down | Brightness Down |

---

## 🔧 **Installation & Setup**
### **1️⃣ Install Dependencies**
```bash
pip install tensorflow opencv-python pyautogui keyboard numpy

python real_time_detection.py

pyinstaller --onefile --windowed --icon=icon.ico real_time_detection.py

---
