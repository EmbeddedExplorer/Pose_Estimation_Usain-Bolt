import os
import cv2

# Directory to save frames
frames_directory = 'Frames'

# Create the 'Frames' folder if it doesn't exist
if not os.path.exists(frames_directory):
    os.makedirs(frames_directory)

# Desired time interval (in seconds)
desired_time_interval = 0.5  # e.g., every 0.5 seconds

# Open the video file
cap = cv2.VideoCapture('Video.mp4')

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Extract frames at desired intervals
frame_interval = int(fps * desired_time_interval)
frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_interval == 0:
        # Construct file path inside the 'Frames' folder
        file_name = os.path.join(frames_directory, f'frame_{saved_count}.jpg')
        cv2.imwrite(file_name, frame)
        print(f"Saved: {file_name}")
        saved_count += 1
    frame_count += 1

cap.release()
print("Done saving frames.")
