import os
import cv2
import mediapipe as mp

# Directory where frames are saved
frames_directory = 'Frames'

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Process each frame in the 'Frames' directory
frame_files = [f for f in os.listdir(frames_directory) if f.endswith('.jpg')]

# Sort the frames by their filename (optional, if you want them in order)
frame_files.sort()

# Loop through each frame
for frame_file in frame_files:
    # Load the frame
    frame_path = os.path.join(frames_directory, frame_file)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Error: Could not read {frame_file}")
        continue

    # Convert the BGR frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(frame_rgb)

    # If landmarks are detected, visualize them
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            # Get the coordinates of each landmark
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            # Draw the landmark on the image (circle)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Optionally, draw the connections between landmarks
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the frame with pose landmarks
    cv2.imshow('Pose Estimation', frame)

    # Save the frame with landmarks in a new folder (optional)
    output_frame_path = os.path.join(frames_directory, 'With_Landmarks', frame_file)
    if not os.path.exists(os.path.join(frames_directory, 'With_Landmarks')):
        os.makedirs(os.path.join(frames_directory, 'With_Landmarks'))
    cv2.imwrite(output_frame_path, frame)

    # Wait for a key press to continue to the next frame (or break if 'q' is pressed)
    key = cv2.waitKey(100)  # 100 ms delay between frames
    if key == ord('q'):  # If 'q' is pressed, exit the loop
        break

# Release the window
cv2.destroyAllWindows()
