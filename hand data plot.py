import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('hand_data.csv')

# Extract columns
timestamps = df['Timestamp']
elbow_angles = df['Right Elbow Angle']

# Plot setup
plt.figure(figsize=(12, 6))
plt.plot(timestamps, elbow_angles, label='Right Elbow Angle', color='blue')

# Highlight Flexion (angle < 90°)
plt.fill_between(timestamps, elbow_angles, where=(elbow_angles < 90), 
                 color='lightcoral', alpha=0.4, label='Flexion (<90°)')

# Highlight Extension (angle > 150°)
plt.fill_between(timestamps, elbow_angles, where=(elbow_angles > 150), 
                 color='lightgreen', alpha=0.4, label='Extension (>150°)')

# Labels and formatting
plt.title('Right Elbow Angle Over Time with Flexion and Extension Highlighted')
plt.xlabel('Time (s)')
plt.ylabel('Elbow Angle (degrees)')
plt.ylim(0, 180)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
