import os
import cv2
import mediapipe as mp
from absl import logging

# Suppress TensorFlow and MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.set_verbosity(logging.ERROR)

# Directory where edited frames are saved
frames_directory = 'Edited_Frames'

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

# Directory to save pose-estimated images
poses_directory = 'Poses'
os.makedirs(poses_directory, exist_ok=True)  # Create the 'Poses' folder if it doesn't exist

# Process each frame in the 'Frames' directory
frame_files = [f for f in os.listdir(frames_directory) if f.endswith('.jpg')]
frame_files.sort()  # Ensure frames are processed in order

if not frame_files:
    print("No frames found in the 'Frames' directory.")
    exit()

print(f"Processing {len(frame_files)} frames...")

num_landmarks_detected = 0  # Track the number of frames with detected landmarks

# Loop through each frame
for frame_file in frame_files:
    try:
        print(f"Processing frame: {frame_file}")
        frame_path = os.path.join(frames_directory, frame_file)
        if not os.path.exists(frame_path):
            print(f"Error: {frame_path} does not exist.")
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error: Failed to read {frame_file}")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, _ = frame.shape
        side_length = min(height, width)
        start_x = (width - side_length) // 2
        start_y = (height - side_length) // 2
        cropped_frame = frame[start_y:start_y + side_length, start_x:start_x + side_length]

        results = pose.process(cropped_frame)
        if results.pose_landmarks:
            print(f"Landmarks detected for frame: {frame_file}")
            mp.solutions.drawing_utils.draw_landmarks(
                cropped_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            output_frame_path = os.path.join(poses_directory, frame_file)
            cv2.imwrite(output_frame_path, cropped_frame)
            print(f"Saved pose-estimated frame to: {output_frame_path}")
            num_landmarks_detected +=1
        else:
            print(f"No landmarks detected in frame: {frame_file}")

    except Exception as e:
        print(f"Error processing frame {frame_file}: {e}")

print(f"Pose estimation completed. Detected landmarks in {num_landmarks_detected} out of {len(frame_files)} frames.")
