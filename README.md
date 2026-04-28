# 🚗 Driver Drowsiness Detection System

A real-time driver drowsiness detection system using **Computer Vision + Hybrid Machine Learning**.

---

## 📌 Overview
This project detects driver fatigue using:
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Hybrid logic (Adaptive + ML)

The system works in real-time using a webcam and alerts the user when drowsiness is detected.

---

## 🚀 Features
- Real-time detection (25–30 FPS)
- EAR-based eye closure detection
- MAR-based yawning detection
- Adaptive calibration (works for all users)
- Hybrid model (Rule-based + ML)
- Audio alert system

---

## 📂 Dataset
- Custom dataset (`EAR, MAR, Label`)
- Label:
  - 0 → Alert
  - 1 → Drowsy

---

## 🤖 Models Used
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree

---

## 🧠 Hybrid Approach
The system combines:
- Adaptive thresholding (personalized)
- Relative EAR drop detection
- ML model (secondary validation)

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
