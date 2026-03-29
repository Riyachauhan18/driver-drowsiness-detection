import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmarks
MOUTH = [13, 14, 78, 308]

cap = cv2.VideoCapture(0)

calibration_frames = 120
ear_values = []
calibrated = False
dynamic_threshold = 0

closed_counter = 0
yawn_counter = 0

ALERT_TIME = 2  # seconds
start_closed_time = None

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_EAR(landmarks, eye_indices, w, h):
    points = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        points.append((x, y))

    vertical1 = euclidean(points[1], points[5])
    vertical2 = euclidean(points[2], points[4])
    horizontal = euclidean(points[0], points[3])

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def calculate_MAR(landmarks, w, h):
    top = (int(landmarks[13].x * w), int(landmarks[13].y * h))
    bottom = (int(landmarks[14].x * w), int(landmarks[14].y * h))
    left = (int(landmarks[78].x * w), int(landmarks[78].y * h))
    right = (int(landmarks[308].x * w), int(landmarks[308].y * h))

    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)

    mar = vertical / horizontal
    return mar

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = calculate_EAR(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_EAR(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            mar = calculate_MAR(landmarks, w, h)

            # ---------------- CALIBRATION PHASE ----------------
            if not calibrated:
                ear_values.append(ear)
                cv2.putText(frame, "Calibrating... Keep eyes open",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 0), 2)

                if len(ear_values) >= calibration_frames:
                    baseline = sum(ear_values) / len(ear_values)
                    dynamic_threshold = baseline * 0.75
                    calibrated = True
                    print("Calibration Done")
                    print("Baseline EAR:", baseline)
                    print("Dynamic Threshold:", dynamic_threshold)

            # ---------------- DETECTION PHASE ----------------
            else:
                # Eye closure detection
                if ear < dynamic_threshold:
                    if start_closed_time is None:
                        start_closed_time = time.time()
                else:
                    start_closed_time = None

                # Yawn detection
                if mar > 0.6:
                    yawn_counter += 1
                else:
                    yawn_counter = 0

                drowsy = False

                # If eyes closed long enough
                if start_closed_time:
                    if time.time() - start_closed_time > ALERT_TIME:
                        drowsy = True

                # If yawning + semi closed
                if mar > 0.6 and ear < baseline * 0.85:
                    drowsy = True

                if drowsy:
                    text = "DROWSY ALERT!"
                    color = (0, 0, 255)
                else:
                    text = "Normal"
                    color = (0, 255, 0)

                cv2.putText(frame, text, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 3)

                cv2.putText(frame, f"EAR: {round(ear,3)}",
                            (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255,255,255), 2)

                cv2.putText(frame, f"MAR: {round(mar,3)}",
                            (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255,255,255), 2)

        cv2.imshow("Driver Drowsiness - Adaptive System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()