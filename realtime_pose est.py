import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from collections import deque

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# Initialize attributes and deque for real-time graphing
timestamps = deque(maxlen=100)
knee_angles_left = deque(maxlen=100)
knee_angles_right = deque(maxlen=100)
stride_lengths = deque(maxlen=100)

# Initialize the figure for real-time graphing
plt.style.use('seaborn-darkgrid')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle("Real-Time Biomechanics Analysis", fontsize=16)
lines = {
    "knee_left": ax1.plot([], [], label="Left Knee Angle", color="blue")[0],
    "knee_right": ax2.plot([], [], label="Right Knee Angle", color="green")[0],
    "stride": ax3.plot([], [], label="Stride Length", color="red")[0],
}

# Graph axes setup
for ax, title in zip([ax1, ax2, ax3], ["Knee Angle (Left)", "Knee Angle (Right)", "Stride Length (px)"]):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 180 if "Angle" in title else 300)
    ax.set_ylabel(title)
    ax.legend(loc="upper right")
ax3.set_xlabel("Time (s)")

# Function to calculate angles
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle

# Function to update the real-time graph
def update_graph(frame_time):
    x_vals = np.array(timestamps) - timestamps[0]
    for attr, data, line in zip(["knee_left", "knee_right", "stride"], 
                                [knee_angles_left, knee_angles_right, stride_lengths], 
                                lines.values()):
        line.set_data(x_vals, data)
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, max(10, x_vals[-1] if x_vals.any() else 0))
    return lines.values()

# Video processing and data recording
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    start_time = time.time()
    biomechanics_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        current_time = time.time() - start_time

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Extract landmarks
            hip_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                        int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
            knee_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * w),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * h))
            ankle_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * w),
                          int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h))
            foot_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x * w),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y * h))
            hip_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                         int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
            knee_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * w),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * h))
            ankle_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w),
                           int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h))
            foot_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x * w),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y * h))

            # Calculate metrics
            knee_angle_left = calculate_angle(hip_left, knee_left, ankle_left)
            knee_angle_right = calculate_angle(hip_right, knee_right, ankle_right)
            stride_length = np.linalg.norm(np.array(foot_left) - np.array(foot_right))

            # Update graphs
            timestamps.append(current_time)
            knee_angles_left.append(knee_angle_left)
            knee_angles_right.append(knee_angle_right)
            stride_lengths.append(stride_length)

            # Record data
            biomechanics_data.append([current_time, knee_angle_left, knee_angle_right, stride_length])

            # Annotate frame
            font, color, thickness = cv2.FONT_HERSHEY_SIMPLEX, (255, 255, 255), 2
            cv2.putText(frame, f"Left Knee Angle: {knee_angle_left:.1f}", (10, 30), font, 0.7, color, thickness)
            cv2.putText(frame, f"Right Knee Angle: {knee_angle_right:.1f}", (10, 60), font, 0.7, color, thickness)
            cv2.putText(frame, f"Stride Length: {stride_length:.1f} px", (10, 90), font, 0.7, color, thickness)
            
            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display frame
        cv2.namedWindow('Real-Time Biomechanics', cv2.WINDOW_NORMAL)
        cv2.imshow('Real-Time Biomechanics', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save data to CSV
    df = pd.DataFrame(biomechanics_data, columns=['Timestamp', 'Left Knee Angle', 'Right Knee Angle', 'Stride Length'])
    df.to_csv('biomechanics_data.csv', index=False)
    print("Biomechanics data saved to 'biomechanics_data.csv'.")

# Run the video processing
video_path = "Video.mp4"  # Replace with your video file path
animation = FuncAnimation(fig, update_graph, interval=100)
plt.ion()  # Enable interactive mode
plt.show(block=False)  # Display the graphs while processing video
process_video(video_path)
