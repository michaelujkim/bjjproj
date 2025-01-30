from ultralytics import YOLO
import numpy as np
import cv2

# Load model
model = YOLO('yolo11n-pose.pt')

# Keypoint smoothing using moving average
class KeypointSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.keypoints_history = []

    def smooth_keypoints(self, keypoints):
        if len(self.keypoints_history) >= self.window_size:
            self.keypoints_history.pop(0)
        self.keypoints_history.append(keypoints)
        return np.nanmean(self.keypoints_history, axis=0)

# Initialize smoother
smoother = KeypointSmoother(window_size=5)

def calculate_angle(a, b, c):
    """Calculate the angle between three points (a, b, c) representing (hip, knee, ankle)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan
    ab = a - b
    bc = b - c
    if np.linalg.norm(ab) == 0 or np.linalg.norm(bc) == 0:
        return np.nan
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# Process video
video_path = "../vids/and/testfoot3.mp4"
cap = cv2.VideoCapture(video_path)

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

takedown_count = 0
previous_takedown_detected = False
previous_frame_keypoints_detected = False
takedown_delay_counter = 0
DELAY_FRAMES = 100

# Store previous frame keypoints for interpolation
previous_keypoints = None

frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    print(f"Processing frame: {frame_number}")
    frame_number += 1

    # Run inference on the frame
    results = model(frame, conf=0.3)  # Lower confidence threshold for better detection

    # Extract keypoints
    keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []
    takedown_detected = False

    # If no keypoints are detected, use previous frame keypoints (interpolation)
    if len(keypoints) == 0:
        if previous_keypoints is not None:
            keypoints = [previous_keypoints]  # Use previous frame keypoints
        else:
            out.write(frame)
            continue

    # Update previous keypoints
    previous_keypoints = keypoints[0] if len(keypoints) > 0 else None

    current_frame_keypoints_detected = True

    for person in keypoints:
        if len(person) < 17:  # Ensure there are enough keypoints (YOLO pose has 17 keypoints)
            continue

        # Smooth keypoints
        smoothed_keypoints = smoother.smooth_keypoints(person)

        # Extract keypoints (adjust indices if necessary)
        head = smoothed_keypoints[0]
        left_knee, left_hip, left_ankle = smoothed_keypoints[13], smoothed_keypoints[11], smoothed_keypoints[15]
        right_knee, right_hip, right_ankle = smoothed_keypoints[14], smoothed_keypoints[12], smoothed_keypoints[16]

        # Skip if any keypoints are missing
        if any(np.isnan([*head, *left_knee, *left_hip, *left_ankle, *right_knee, *right_hip, *right_ankle])):
            continue

        # Visualize keypoints
        for i, kp in enumerate(smoothed_keypoints):
            if not np.isnan(kp[0]) and not np.isnan(kp[1]):
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (int(kp[0]), int(kp[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Calculate angles
        left_body_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_body_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # Debug: Print angles and keypoints
        print(f"Left Angle: {left_body_angle}, Right Angle: {right_body_angle}")
        print(f"Head Y: {head[1]}, Knees Y: {min(left_knee[1], right_knee[1])}")

        # Takedown condition
        if ((left_body_angle < 120 or right_body_angle < 120) and  # Lower angle threshold
            (min(left_knee[1], right_knee[1]) - head[1] < 100)):   # Lower vertical distance threshold
            takedown_detected = True
            break

    if takedown_detected and not previous_takedown_detected and previous_frame_keypoints_detected and takedown_delay_counter == 0:
        takedown_count += 1
        takedown_delay_counter = DELAY_FRAMES
        cv2.putText(frame, f"Takedown Count: {takedown_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    previous_takedown_detected = takedown_detected
    previous_frame_keypoints_detected = current_frame_keypoints_detected

    if takedown_delay_counter > 0:
        takedown_delay_counter -= 1

    # Write frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total takedowns detected: {takedown_count}")