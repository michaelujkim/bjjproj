from ultralytics import YOLO
import numpy as np
import cv2

# Load model
model = YOLO('yolo11n-pose.pt')

def calculate_midpoint(a, b):
    """Calculate the midpoint between two points."""
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

def is_guard_position(top_person, bottom_person):
    """
    Check if the initial guard position is valid:
    - The person on the bottom has their legs between themselves and the person on top.
    """
    # Extract keypoints
    bottom_left_knee = bottom_person[13]  # Left knee
    bottom_right_knee = bottom_person[14]  # Right knee
    top_chest = calculate_midpoint(top_person[5], top_person[6])  # Chest (midpoint of shoulders)
    bottom_chest = calculate_midpoint(bottom_person[5], bottom_person[6])  # Chest (midpoint of shoulders)

    # Check if the top person's chest is between the bottom person's knees
    if (bottom_left_knee[0] < top_chest[0] < bottom_right_knee[0] or
        bottom_right_knee[0] < top_chest[0] < bottom_left_knee[0]):
        return True
    return False

def is_guard_passed(top_person, bottom_person):
    """
    Check if the guard pass is completed:
    - The chest of the person on top is aligned with the chest of the person on the bottom.
    - The vertical distance between their chests is small.
    - No legs are between their chests.
    """
    # Extract keypoints
    top_chest = calculate_midpoint(top_person[5], top_person[6])  # Chest (midpoint of shoulders)
    bottom_chest = calculate_midpoint(bottom_person[5], bottom_person[6])  # Chest (midpoint of shoulders)
    bottom_left_knee = bottom_person[13]  # Left knee
    bottom_right_knee = bottom_person[14]  # Right knee

    # Check if the chests are aligned (X-axis alignment)
    chest_alignment_threshold = 50  # Adjust as needed
    if abs(top_chest[0] - bottom_chest[0]) < chest_alignment_threshold:
        # Check if the vertical distance between chests is small
        vertical_distance_threshold = 100  # Adjust as needed
        if abs(top_chest[1] - bottom_chest[1]) < vertical_distance_threshold:
            # Check if no legs are between their chests
            if not (bottom_left_knee[0] < top_chest[0] < bottom_right_knee[0] or
                    bottom_right_knee[0] < top_chest[0] < bottom_left_knee[0]):
                return True
    return False

# Process video
video_path = "../vids/and/gpfoot1n7.mov"
cap = cv2.VideoCapture(video_path)

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

guard_pass_count = 0
previous_guard_pass_detected = False
previous_frame_keypoints_detected = False
guard_pass_delay_counter = 0
DELAY_FRAMES = 100

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
    guard_pass_detected = False

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

    # Check guard position and guard pass
    if is_guard_position(top_person, bottom_person):
        if is_guard_passed(top_person, bottom_person):
            guard_pass_detected = True

    # Count guard passes
    if guard_pass_detected and not previous_guard_pass_detected and guard_pass_delay_counter == 0:
        guard_pass_count += 1
        guard_pass_delay_counter = DELAY_FRAMES
        cv2.putText(frame, f"Guard Pass Count: {guard_pass_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    previous_guard_pass_detected = guard_pass_detected

    if guard_pass_delay_counter > 0:
        guard_pass_delay_counter -= 1

    # Write frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total guard passes detected: {guard_pass_count}")