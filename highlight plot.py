import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
csv_file = 'mechanics_in_scale.csv'  # CSV file path
data = pd.read_csv(csv_file)

# Extract data columns
timestamps = data['Timestamp']
left_knee_angles = data['Left Knee Angle']
right_knee_angles = data['Right Knee Angle']
stride_lengths = data['Stride Length (m)']

# Define thresholds for flexion and extension
flexion_threshold = 90  # Below this angle is flexion
extension_threshold = 150  # Above this angle is extension

# Define normal range for knee angles (example values; adjust as needed)
normal_knee_angle_min = 60
normal_knee_angle_max = 160

# Define normal range for stride length (example values; adjust as needed)
normal_stride_length_min = 1.0  # Meters
normal_stride_length_max = 1.5  # Meters

# Determine flexion and extension regions
left_flexion = left_knee_angles < flexion_threshold
left_extension = left_knee_angles > extension_threshold
right_flexion = right_knee_angles < flexion_threshold
right_extension = right_knee_angles > extension_threshold

# Set up the figure and axes for the plots
plt.style.use('seaborn-darkgrid')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
fig.suptitle("Biomechanics Data Analysis", fontsize=16)

# Plot Left Knee Angle with normal range and flexion/extension highlighting
ax1.plot(timestamps, left_knee_angles, label="Left Knee Angle", color="blue")
ax1.fill_between(timestamps, normal_knee_angle_min, normal_knee_angle_max, color="yellow", alpha=0.2, label="Normal Range")
ax1.fill_between(timestamps, 0, left_knee_angles, where=left_flexion, color="cyan", alpha=0.3, label="Flexion")
ax1.fill_between(timestamps, 0, left_knee_angles, where=left_extension, color="magenta", alpha=0.3, label="Extension")
ax1.set_ylabel("Knee Angle (Left)")
ax1.legend(loc="upper right")

# Plot Right Knee Angle with normal range and flexion/extension highlighting
ax2.plot(timestamps, right_knee_angles, label="Right Knee Angle", color="green")
ax2.fill_between(timestamps, normal_knee_angle_min, normal_knee_angle_max, color="yellow", alpha=0.2, label="Normal Range")
ax2.fill_between(timestamps, 0, right_knee_angles, where=right_flexion, color="cyan", alpha=0.3, label="Flexion")
ax2.fill_between(timestamps, 0, right_knee_angles, where=right_extension, color="magenta", alpha=0.3, label="Extension")
ax2.set_ylabel("Knee Angle (Right)")
ax2.legend(loc="upper right")

# Plot Stride Length with normal range highlighting
ax3.plot(timestamps, stride_lengths, label="Stride Length", color="red")
ax3.fill_between(timestamps, normal_stride_length_min, normal_stride_length_max, color="yellow", alpha=0.2, label="Normal Range")
ax3.set_ylabel("Stride Length (m)")
ax3.set_xlabel("Time (s)")
ax3.legend(loc="upper right")

# Display the plots
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
