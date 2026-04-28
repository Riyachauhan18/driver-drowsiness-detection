import cv2
import mediapipe as mp
import numpy as np
import threading
import winsound
from collections import deque
import joblib

model = joblib.load("drowsiness_model.pkl")

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

cap = cv2.VideoCapture(0)

# -------- CALIBRATION --------
calibration_frames = 60
ear_values = []
calibrated = False
baseline = 0

# -------- DETECTION --------
closed_frames = 0
FRAME_THRESHOLD = 8   # balanced

# -------- YAWN --------
yawn_frames = 0
YAWN_THRESHOLD = 12

# -------- SMOOTHING --------
ear_buffer = deque(maxlen=5)

alarm_on = False

def play_alarm():
    winsound.Beep(1200, 120)

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_EAR(lm, eye, w, h):
    pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in eye]
    return (euclidean(pts[1], pts[5]) + euclidean(pts[2], pts[4])) / (2*euclidean(pts[0], pts[3]))

def calculate_MAR(lm, w, h):
    top = (int(lm[13].x*w), int(lm[13].y*h))
    bottom = (int(lm[14].x*w), int(lm[14].y*h))
    left = (int(lm[78].x*w), int(lm[78].y*h))
    right = (int(lm[308].x*w), int(lm[308].y*h))
    return euclidean(top,bottom)/euclidean(left,right)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            ear = (calculate_EAR(lm, LEFT_EYE, w, h) +
                   calculate_EAR(lm, RIGHT_EYE, w, h)) / 2

            ear_buffer.append(ear)
            ear = sum(ear_buffer)/len(ear_buffer)

            mar = calculate_MAR(lm, w, h)

            # -------- CALIBRATION --------
            if not calibrated:
                ear_values.append(ear)
                cv2.putText(frame,"Calibrating...",(30,50),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

                if len(ear_values) >= calibration_frames:
                    baseline = np.mean(ear_values)
                    calibrated = True

            else:
                # -------- RELATIVE DROP --------
                drop = baseline - ear

                if drop > 0.05:
                    closed_frames += 1
                else:
                    closed_frames = 0

                eye_drowsy = closed_frames >= FRAME_THRESHOLD

                # -------- YAWN --------
                if mar > 0.6:
                    yawn_frames += 1
                else:
                    yawn_frames = 0

                yawning = yawn_frames >= YAWN_THRESHOLD

                # -------- ML --------
                ml_pred = model.predict([[ear, mar]])[0]

                # -------- FINAL DECISION --------
                if eye_drowsy:
                    text = "DROWSY ALERT!"
                    color = (0,0,255)

                elif yawning:
                    text = "YAWNING"
                    color = (0,165,255)

                elif ml_pred == 1 and drop > 0.03:
                    text = "DROWSY (ML)"
                    color = (0,0,255)

                else:
                    text = "Normal"
                    color = (0,255,0)

                # -------- ALERT --------
                if "DROWSY" in text and not alarm_on:
                    alarm_on = True
                    threading.Thread(target=play_alarm).start()
                elif "DROWSY" not in text:
                    alarm_on = False

                # -------- DISPLAY --------
                cv2.putText(frame,text,(30,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1,color,3)

                cv2.putText(frame,f"EAR:{round(ear,3)}",(30,90),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

                cv2.putText(frame,f"MAR:{round(mar,3)}",(30,120),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        cv2.imshow("Universal Drowsiness System",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows() 
