from ultralytics import YOLO
import numpy as np
model = YOLO('yolo11n-pose.pt')  # Load model
result = model("../vids/and/testfoot2.mp4") 
def calculate_angle(a, b, c):
    """Calculate the angle between three points (a, b, c) representing (hip, knee, ankle)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    bc = b - c
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

takedown_count = 0
previous_takedown_detected = False  # To track the state of the previous frame

for i in result:
    keypoints = i.keypoints.xy.numpy()  # Extract keypoints as NumPy array
    takedown_detected = False  # Reset takedown detection per frame

    for person in keypoints:
        head_y = person[0][1]
        knee = person[13]
        hip = person[11]
        ankle = person[15]

        # Calculate the body orientation angle
        body_angle = calculate_angle(hip, knee, ankle)

        # Takedown detected if the body angle is lower and head is closer to knee
        if body_angle < 100 and (knee[1] - head_y < 50):
            takedown_detected = True
            break  # Stop checking other people in the frame

    # Only count a new takedown if the previous frame didn't detect one
    if takedown_detected and not previous_takedown_detected:
        takedown_count += 1

    # Update the previous frame's detection status
    previous_takedown_detected = takedown_detected

print(f"Total takedowns detected: {takedown_count}")  # Output: Total takedowns detected: 3