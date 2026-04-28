# 🚗 Driver Drowsiness Detection System
A real-time driver drowsiness detection system using **Computer Vision + Hybrid Machine Learning**.

---

## 📌 Overview
This project detects driver fatigue using facial features:
- **EAR (Eye Aspect Ratio)** → detects eye closure  
- **MAR (Mouth Aspect Ratio)** → detects yawning  

The system works in real-time using a webcam and alerts the user when drowsiness is detected.

---

## 🚀 Features
- Real-time detection (**25–30 FPS**)  
- Eye closure detection using EAR  
- Yawning detection using MAR  
- Adaptive calibration (works for all users)  
- Hybrid approach (**Rule-based + ML**)  
- Audio alert system  

---

## 🧠 Hybrid Approach
The system combines:
- Adaptive thresholding (personalized baseline)
- Relative EAR drop detection
- Machine Learning model (secondary validation)

👉 This improves accuracy and reduces false alerts across different users.

---

## 📂 Dataset
- Custom dataset: **EAR, MAR, Label**
- Extracted from real images and webcam

| Label | Meaning |
|------|--------|
| 0 | Alert |
| 1 | Drowsy |

---

## 🤖 Models Used
- Logistic Regression  
- Support Vector Machine (SVM)  
- Decision Tree  

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
