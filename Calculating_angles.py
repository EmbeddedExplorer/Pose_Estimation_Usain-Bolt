import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    a, b, c: tuples of (x, y) coordinates.
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Example landmarks: hip, knee, ankle
hip = (hx, hy)
knee = (kx, ky)
ankle = (ax, ay)

knee_angle = calculate_angle(hip, knee, ankle)