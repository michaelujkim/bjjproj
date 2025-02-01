from ultralytics import YOLO
import numpy as np
import cv2
import time

# Load model
model = YOLO('yolo11n-pose.pt')

def calculate_midpoint(a, b):
    """Calculate the midpoint between two points."""
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

def is_guard_position(top_person, bottom_person):
    """
    Check if the guard position is valid:
    - The legs of the bottom person are around the torso or legs of the top person.
    """
    # Extract keypoints for the bottom person
    bottom_left_knee = bottom_person[13]  # Left knee
    bottom_right_knee = bottom_person[14]  # Right knee
    bottom_left_hip = bottom_person[11]  # Left hip
    bottom_right_hip = bottom_person[12]  # Right hip
    bottom_left_ankle = bottom_person[15]  # Left ankle
    bottom_right_ankle = bottom_person[16]  # Right ankle

    # Extract keypoints for the top person
    top_chest = calculate_midpoint(top_person[5], top_person[6])  # Chest (midpoint of shoulders)
    top_hips = calculate_midpoint(top_person[11], top_person[12])  # Hips (midpoint of hips)
    top_left_knee = top_person[13]  # Left knee
    top_right_knee = top_person[14]  # Right knee
    top_left_ankle = top_person[15]  # Left ankle
    top_right_ankle = top_person[16]  # Right ankle

    # Define the area of the bottom person's legs
    leg_area_left = min(bottom_left_knee[0], bottom_left_hip[0], bottom_left_ankle[0])
    leg_area_right = max(bottom_right_knee[0], bottom_right_hip[0], bottom_right_ankle[0])

    # Check if the top person's torso (chest or hips) is within the leg area
    torso_in_guard = (leg_area_left < top_chest[0] < leg_area_right or
                      leg_area_left < top_hips[0] < leg_area_right)

    # Check if the top person's legs (knees or ankles) are within the leg area
    legs_in_guard = (leg_area_left < top_left_knee[0] < leg_area_right or
                     leg_area_left < top_right_knee[0] < leg_area_right or
                     leg_area_left < top_left_ankle[0] < leg_area_right or
                     leg_area_left < top_right_ankle[0] < leg_area_right)

    # Guard position is valid if either the torso or legs are within the leg area
    return torso_in_guard or legs_in_guard

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

# Process video
video_path = "../vids/and/swfoot1n6.mov"
cap = cv2.VideoCapture(video_path)

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

sweep_count = 0
previous_sweep_detected = False
previous_frame_keypoints_detected = False
sweep_delay_counter = 0
DELAY_FRAMES = 100

# Time in guard position tracking
in_guard_position = False
guard_position_start_time = None
total_time_in_guard = 0

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
    sweep_detected = False

    # If no keypoints are detected, use previous frame keypoints (interpolation)
    if len(keypoints) == 0:
        if previous_keypoints is not None:
            keypoints = [previous_keypoints]  # Use previous frame keypoints
        else:
            out.write(frame)
            continue

    # Update previous keypoints
    previous_keypoints = keypoints[0] if len(keypoints) > 0 else None

    # Skip if fewer than 2 people are detected
    if len(keypoints) < 2:
        out.write(frame)
        continue

    # Identify top and bottom person based on hip position
    person1, person2 = keypoints[0], keypoints[1]
    if person1[11][1] < person2[11][1]:  # Compare hip Y positions
        top_person, bottom_person = person1, person2
    else:
        top_person, bottom_person = person2, person1

    # Smooth keypoints
    smoothed_top_person = smoother.smooth_keypoints(top_person)
    smoothed_bottom_person = smoother.smooth_keypoints(bottom_person)

    # Check guard position
    if is_guard_position(smoothed_top_person, smoothed_bottom_person):
        if not in_guard_position:
            in_guard_position = True
            guard_position_start_time = time.time()  # Start timer
        else:
            # Calculate time spent in guard position
            total_time_in_guard = time.time() - guard_position_start_time
    else:
        if in_guard_position:
            in_guard_position = False
            guard_position_start_time = None

    # Check if the bottom person is now on top (sweep detected)
    if in_guard_position and bottom_person[11][1] < top_person[11][1]:
        sweep_detected = True

    # Count sweeps
    if sweep_detected and not previous_sweep_detected and sweep_delay_counter == 0:
        sweep_count += 1
        sweep_delay_counter = DELAY_FRAMES
        cv2.putText(frame, f"Sweep Count: {sweep_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display time in guard position
    if in_guard_position:
        cv2.putText(frame, f"Time in Guard: {total_time_in_guard:.2f}s", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    previous_sweep_detected = sweep_detected

    if sweep_delay_counter > 0:
        sweep_delay_counter -= 1

    # Write frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total sweeps detected: {sweep_count}")
print(f"Total time in guard position: {total_time_in_guard:.2f}s")