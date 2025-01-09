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

# Set up the figure and axes for the plots
plt.style.use('seaborn-darkgrid')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle("Biomechanics Data Analysis", fontsize=16)

# Plot Left Knee Angle
ax1.plot(timestamps, left_knee_angles, label="Left Knee Angle", color="blue")
ax1.set_ylabel("Knee Angle (Left)")
ax1.legend(loc="upper right")

# Plot Right Knee Angle
ax2.plot(timestamps, right_knee_angles, label="Right Knee Angle", color="green")
ax2.set_ylabel("Knee Angle (Right)")
ax2.legend(loc="upper right")

# Plot Stride Length
ax3.plot(timestamps, stride_lengths, label="Stride Length", color="red")
ax3.set_ylabel("Stride Length (m)")
ax3.set_xlabel("Time (s)")
ax3.legend(loc="upper right")

# Display the plots
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
