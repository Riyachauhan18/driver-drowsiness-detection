#Driver Drowsiness Detection System
A real-time driver drowsiness detection system using facial landmark analysis (EAR & MAR).
---
##  Overview

This project detects driver drowsiness using a webcam by analyzing eye closure and yawning patterns. It uses Mediapipe for facial landmark detection and applies adaptive thresholding for personalized accuracy.
---

## Features
* Real-time webcam-based detection
* Eye closure detection using EAR (Eye Aspect Ratio)
* Yawning detection using MAR (Mouth Aspect Ratio)
* Adaptive threshold calibration (user-specific)
* Lightweight and fast (no GPU required)
* Real-time performance (~30 FPS, <100ms latency)

---

## Tech Stack
* Python
* OpenCV
* Mediapipe
* NumPy

---

## How It Works
1. Capture video using webcam
2. Detect face and extract landmarks
3. Calculate EAR & MAR
4. Calibrate user-specific threshold
5. Detect drowsiness based on conditions
6. Display real-time alert

---

## How to Run
pip install -r requirements.txt

python main.py

---

##  Key Innovations
* Adaptive learning using real-time calibration
* Hybrid detection (eye + mouth)
* ML-inspired intelligent system
* Real-time performance

---

##  Future Scope
* Add alarm/voice alert
* Mobile app integration
* Deep learning enhancement
---

⭐ If you like this project, give it a star!
