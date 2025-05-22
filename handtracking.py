import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# CSV file to save the data
output_csv = "hand_data.csv"

# Function to calculate angles between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (joint)
    c = np.array(c)  # Last point
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle

# Process video and analyze data
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    start_time = time.time()
    hand_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Extract relevant landmarks
            shoulder_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                              int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
            elbow_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w),
                           int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h))
            wrist_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w),
                           int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h))

            # Calculate the right elbow angle
            right_elbow_angle = calculate_angle(shoulder_right, elbow_right, wrist_right)

            # Track vertical swing path (Y-coordinate of the right wrist)
            vertical_swing_path = wrist_right[1]  # Y-coordinate

            # Calculate hand stride rhythm (difference in vertical position across frames)
            if len(hand_data) > 0:
                previous_y = hand_data[-1][2]  # Last recorded vertical swing path
                hand_stride_rhythm = abs(vertical_swing_path - previous_y)
            else:
                hand_stride_rhythm = 0

            # Record data
            current_time = time.time() - start_time
            hand_data.append([current_time, right_elbow_angle, vertical_swing_path, hand_stride_rhythm])

            # Annotate the frame
            font, color, thickness = cv2.FONT_HERSHEY_SIMPLEX, (255, 255, 255), 2
            cv2.putText(frame, f"Right Elbow Angle: {right_elbow_angle:.1f}", (10, 30), font, 0.7, color, thickness)
            cv2.putText(frame, f"Vertical Swing (Y): {vertical_swing_path}", (10, 60), font, 0.7, color, thickness)
            cv2.putText(frame, f"Hand Stride Rhythm: {hand_stride_rhythm:.1f}", (10, 90), font, 0.7, color, thickness)

            # Draw pose landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the video
        cv2.namedWindow('Hand Mechanics Analysis', cv2.WINDOW_NORMAL)
        cv2.imshow('Hand Mechanics Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save the data to a CSV file
    df = pd.DataFrame(hand_data, columns=['Timestamp', 'Right Elbow Angle', 'Vertical Swing Path (Y)', 'Hand Stride Rhythm'])
    df.to_csv(output_csv, index=False)
    print(f"Data saved to '{output_csv}'.")

# Replace 'Video.mp4' with the path to your video file
video_path = "Video.mp4"
process_video(video_path)
