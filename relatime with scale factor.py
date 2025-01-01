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

# Known height of Usain Bolt (in meters)
usain_bolt_height_meters = 1.95  # Usain Bolt's height is 1.95 meters

# Initialize real-time attributes
timestamps = deque(maxlen=100)
knee_angles_left = deque(maxlen=100)
knee_angles_right = deque(maxlen=100)
stride_lengths = deque(maxlen=100)

# Function to calculate angles
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle

# Process video and save scaled data
def process_video_with_scaling(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    biomechanics_data = []
    scaling_factor = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Extract landmarks for height calculation
            head_top = (int(landmarks[mp_pose.PoseLandmark.NOSE].x * w),
                        int(landmarks[mp_pose.PoseLandmark.NOSE].y * h))
            feet_bottom = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * w),
                           int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h))

            # Calculate Usain Bolt's height in pixels
            height_in_pixels = np.linalg.norm(np.array(head_top) - np.array(feet_bottom))
            if not scaling_factor and height_in_pixels > 0:  # Calculate scaling factor once
                scaling_factor = usain_bolt_height_meters / height_in_pixels
                print(f"Calculated Scaling Factor: {scaling_factor:.6f} meters per pixel")

            # Extract other body points for calculations
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

            # Calculate biomechanics metrics
            knee_angle_left = calculate_angle(hip_left, knee_left, ankle_left)
            knee_angle_right = calculate_angle(hip_right, knee_right, ankle_right)
            stride_length_pixels = np.linalg.norm(np.array(foot_left) - np.array(foot_right))
            stride_length_meters = stride_length_pixels * scaling_factor if scaling_factor else 0

            # Record data
            current_time = time.time()
            biomechanics_data.append([current_time, knee_angle_left, knee_angle_right, stride_length_meters])

            # Annotate the frame
            font, color, thickness = cv2.FONT_HERSHEY_SIMPLEX, (255, 255, 255), 2
            cv2.putText(frame, f"Left Knee Angle: {knee_angle_left:.1f}", (10, 30), font, 0.7, color, thickness)
            cv2.putText(frame, f"Right Knee Angle: {knee_angle_right:.1f}", (10, 60), font, 0.7, color, thickness)
            cv2.putText(frame, f"Stride Length: {stride_length_meters:.2f} m", (10, 90), font, 0.7, color, thickness)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the video
        cv2.namedWindow('Real-Time Biomechanics', cv2.WINDOW_NORMAL)
        cv2.imshow('Real-Time Biomechanics', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save biomechanics data to CSV
    df = pd.DataFrame(biomechanics_data, columns=['Timestamp', 'Left Knee Angle', 'Right Knee Angle', 'Stride Length (m)'])
    df.to_csv(output_csv, index=False)
    print(f"Biomechanics data saved to '{output_csv}'.")

def plot_scaled_data(csv_file):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(12, 8))

    # Plot Knee Angles
    plt.subplot(3, 1, 1)
    plt.plot(df['Timestamp'], df['Left Knee Angle'], label='Left Knee Angle', color='blue')
    plt.plot(df['Timestamp'], df['Right Knee Angle'], label='Right Knee Angle', color='green')
    plt.ylabel('Knee Angle (degrees)')
    plt.legend()
    plt.title('Knee Angles Over Time')

    # Plot Stride Length
    plt.subplot(3, 1, 2)
    plt.plot(df['Timestamp'], df['Stride Length (m)'], label='Stride Length', color='red')
    plt.ylabel('Stride Length (meters)')
    plt.legend()
    plt.title('Stride Length Over Time')

    # Add a common X label
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


# Process the video
video_path = "Video.mp4"  # Replace with your video file path
output_csv = "mechanics_in_scale.csv"
process_video_with_scaling(video_path, output_csv)

# Plot the scaled data
plot_scaled_data(output_csv)