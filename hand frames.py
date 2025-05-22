import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# Create a directory to save captured frames
output_directory = "Hand_mechanics"
os.makedirs(output_directory, exist_ok=True)

# Function to calculate angles between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (joint)
    c = np.array(c)  # Last point
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle

# Process video and capture frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

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

            # Extract landmarks for elbow angle calculation
            shoulder_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
            elbow_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w),
                          int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h))
            wrist_left = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w),
                          int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h))
            shoulder_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                              int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
            elbow_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w),
                           int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h))
            wrist_right = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w),
                           int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h))

            # Calculate elbow angles
            elbow_angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
            elbow_angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)

            # Annotate the frame
            font, color, thickness = cv2.FONT_HERSHEY_SIMPLEX, (255, 255, 255), 2
            cv2.putText(frame, f"Left Elbow Angle: {elbow_angle_left:.1f}", (10, 30), font, 0.6, color, thickness)
            cv2.putText(frame, f"Right Elbow Angle: {elbow_angle_right:.1f}", (10, 60), font, 0.6, color, thickness)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Save the annotated frame
        output_frame_path = os.path.join(output_directory, f"frame_{frame_count}.jpg")
        cv2.imwrite(output_frame_path, frame)
        print(f"Saved: {output_frame_path}")
        frame_count += 1

        # Display the frame (optional)
        cv2.imshow("Hand Mechanics", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total frames saved: {frame_count}")

# Replace 'Video.mp4' with the path to your video file
video_path = "Video.mp4"
process_video(video_path)
