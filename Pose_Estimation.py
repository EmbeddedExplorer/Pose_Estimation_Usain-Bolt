import os
import cv2
import mediapipe as mp

# Directory where frames are saved
frames_directory = 'Edited_Frames'

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Directory to save pose-estimated images
poses_directory = os.path.join('Poses')
os.makedirs(poses_directory, exist_ok=True)  # Create the 'Poses' folder if it doesn't exist

# Process each frame in the 'Frames' directory
frame_files = [f for f in os.listdir(frames_directory) if f.endswith('.jpg')]
frame_files.sort()  # Ensure frames are processed in order

if not frame_files:
    print("No frames found in the 'Frames' directory.")
    exit()

print(f"Processing {len(frame_files)} frames...")

# Loop through each frame
for frame_file in frame_files:
    frame_path = os.path.join(frames_directory, frame_file)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Error: Could not read {frame_file}")
        continue

    # Convert the BGR frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Draw the landmarks and connections on the frame
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        # Save the frame with pose landmarks in the 'Poses' folder
        output_frame_path = os.path.join(poses_directory, frame_file)
        cv2.imwrite(output_frame_path, frame)
        print(f"Saved pose-estimated image: {output_frame_path}")
    else:
        print(f"No landmarks detected in {frame_file}.")

print("Pose estimation completed. All images saved in the 'Poses' folder.")
