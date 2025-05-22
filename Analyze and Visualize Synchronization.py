import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# CONFIGURATION
video_path = "Video.mp4"
output_folder = "synchronization_mechanics"
csv_filename = "synchronized_mech_scaled.csv"
os.makedirs(output_folder, exist_ok=True)

# Known real-world height of the subject (Usain Bolt in meters)
real_height_m = 1.95

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle

# VIDEO PROCESSING WITH SCALING
cap = cv2.VideoCapture(video_path)
start_time = time.time()
frame_count = 0
records = []
scaling_factor = None  # pixels to meters

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        h, w, _ = frame.shape
        lm = results.pose_landmarks.landmark
        def get_lm(idx): return (int(lm[idx].x * w), int(lm[idx].y * h))

        # Estimate scaling factor from approximate body height (nose to ankle)
        if scaling_factor is None:
            nose_y = lm[mp_pose.PoseLandmark.NOSE].y * h
            ankle_y = lm[mp_pose.PoseLandmark.LEFT_ANKLE].y * h
            pixel_height = abs(ankle_y - nose_y)
            if pixel_height > 0:
                scaling_factor = real_height_m / pixel_height

        # Get joint coordinates
        shoulder_r, elbow_r, wrist_r = get_lm(mp_pose.PoseLandmark.RIGHT_SHOULDER), get_lm(mp_pose.PoseLandmark.RIGHT_ELBOW), get_lm(mp_pose.PoseLandmark.RIGHT_WRIST)
        hip_r, knee_r, ankle_r = get_lm(mp_pose.PoseLandmark.RIGHT_HIP), get_lm(mp_pose.PoseLandmark.RIGHT_KNEE), get_lm(mp_pose.PoseLandmark.RIGHT_ANKLE)

        # Compute angles
        right_elbow_angle = calculate_angle(shoulder_r, elbow_r, wrist_r)
        right_knee_angle = calculate_angle(hip_r, knee_r, ankle_r)

        # Scale vertical wrist Y-position
        wrist_y_scaled = (wrist_r[1] * scaling_factor) if scaling_factor else 0

        # Timestamp
        timestamp = time.time() - start_time
        records.append([timestamp, right_elbow_angle, right_knee_angle, wrist_y_scaled])

        # Annotate frame
        annotated = frame.copy()
        font, color, thickness = cv2.FONT_HERSHEY_SIMPLEX, (255, 255, 255), 2
        cv2.putText(annotated, f"Elbow: {right_elbow_angle:.1f}", (10, 30), font, 0.7, color, thickness)
        cv2.putText(annotated, f"Knee: {right_knee_angle:.1f}", (10, 60), font, 0.7, color, thickness)
        cv2.putText(annotated, f"Wrist Y: {wrist_y_scaled:.2f} m", (10, 90), font, 0.7, color, thickness)
        mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count}.jpg"), annotated)
        frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# SAVE TO CSV
df = pd.DataFrame(records, columns=["Timestamp", "Right Elbow Angle", "Right Knee Angle", "Right Wrist Y (m)"])
df.to_csv(csv_filename, index=False)

# VISUALIZATION
time_vals = df["Timestamp"]
elbow_angles = df["Right Elbow Angle"]
knee_angles = df["Right Knee Angle"]

elbow_peaks, _ = find_peaks(-elbow_angles, distance=5)
knee_peaks, _ = find_peaks(-knee_angles, distance=5)
correlation = elbow_angles.corr(knee_angles)

plt.figure(figsize=(12, 6))
plt.plot(time_vals, elbow_angles, label='Right Elbow Angle', color='blue')
plt.plot(time_vals, knee_angles, label='Right Knee Angle', color='green')
plt.scatter(time_vals[elbow_peaks], elbow_angles[elbow_peaks], color='cyan', label='Elbow Flexion Peaks')
plt.scatter(time_vals[knee_peaks], knee_angles[knee_peaks], color='lime', label='Knee Flexion Peaks')
plt.fill_between(time_vals, elbow_angles, where=(elbow_angles < 90), color='lightcoral', alpha=0.3, label='Elbow Flexion (<90°)')
plt.fill_between(time_vals, elbow_angles, where=(elbow_angles > 150), color='lightgreen', alpha=0.3, label='Elbow Extension (>150°)')
plt.fill_between(time_vals, knee_angles, where=(knee_angles < 90), color='mistyrose', alpha=0.2, label='Knee Flexion (<90°)')
plt.fill_between(time_vals, knee_angles, where=(knee_angles > 150), color='honeydew', alpha=0.2, label='Knee Extension (>150°)')

plt.title(f'Usain Bolt Hand-Leg Synchronization (Scaled)\nCorrelation: {correlation:.2f}')
plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
