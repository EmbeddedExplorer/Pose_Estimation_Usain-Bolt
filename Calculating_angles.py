import os
import cv2
import mediapipe as mp
import numpy as np
from absl import logging

# Suppress TensorFlow and MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.set_verbosity(logging.ERROR)

# Initialize directories
frames_directory = 'Edited_Frames'
poses_directory = 'Poses'
mechanics_directory = 'Mechanics'
os.makedirs(mechanics_directory, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

# Process each frame in the 'Poses' directory
frame_files = [f for f in os.listdir(poses_directory) if f.endswith('.jpg')]
frame_files.sort()

if not frame_files:
    print("No frames found in the 'Poses' directory.")
    exit()

print(f"Processing {len(frame_files)} frames for biomechanics analysis...")

# Function to calculate angles between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (joint)
    c = np.array(c)  # Last point
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# Loop through each frame
for frame_file in frame_files:
    try:
        print(f"Analyzing frame: {frame_file}")
        frame_path = os.path.join(poses_directory, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Error: Failed to read {frame_file}")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Extract landmark coordinates
            landmarks = results.pose_landmarks.landmark
            height, width, _ = frame.shape

            # Define key points
            hip_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * width),
                        int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * height))
            hip_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * width),
                         int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * height))
            knee_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * width),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * height))
            knee_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * width),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * height))
            ankle_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * width),
                          int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * height))
            ankle_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * width),
                           int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * height))
            foot_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x * width),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y * height))
            foot_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x * width),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y * height))

            # Calculate biomechanics metrics
            knee_angle_left = calculate_angle(hip_left, knee_left, ankle_left)
            knee_angle_right = calculate_angle(hip_right, knee_right, ankle_right)
            hip_angle_left = calculate_angle(knee_left, hip_left, hip_right)
            hip_angle_right = calculate_angle(knee_right, hip_right, hip_left)

            # Calculate stride length (distance between left and right feet)
            stride_length = np.linalg.norm(np.array(foot_left) - np.array(foot_right))

            # Annotate the frame
            cv2.putText(frame, f"Left Knee Angle: {knee_angle_left:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Right Knee Angle: {knee_angle_right:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Stride Length: {stride_length:.2f} px", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Save the annotated frame
            output_frame_path = os.path.join(mechanics_directory, frame_file)
            cv2.imwrite(output_frame_path, frame)
            print(f"Saved annotated frame to: {output_frame_path}")

        else:
            print(f"No landmarks detected in frame: {frame_file}")

    except Exception as e:
        print(f"Error processing frame {frame_file}: {e}")

print("Biomechanics analysis completed.")
