from ultralytics import YOLO
import numpy as np
import cv2
model = YOLO('yolo11n-pose.pt')  # Load model
results = model("../vids/and/testfoot2.mp4") 
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

takedown_count = 0
previous_takedown_detected = False  # To track the state of the previous frame
previous_frame_keypoints_detected = False
takedown_delay_counter = 0  # Delay counter to prevent multiple counts within 100 frames
DELAY_FRAMES = 100

for frame_number, result in enumerate(results):
    print(f"Processing frame: {frame_number}")
    frame_image = result.orig_img.copy()  # Retrieve the original frame image
    keypoints = result.keypoints.xy.numpy()  # Extract keypoints as NumPy array
    takedown_detected = False  # Reset takedown detection per frame

    # Ignore frames with no keypoints for takedown counter
    if len(keypoints) == 0 or np.isnan(keypoints).any():
        previous_frame_keypoints_detected = False
        takedown_delay_counter = max(takedown_delay_counter - 1, 0)
        continue

    current_frame_keypoints_detected = True

    for person in keypoints:
        head_y = person[0][1]
        left_knee, left_hip, left_ankle = person[10], person[12], person[14]
        right_knee, right_hip, right_ankle = person[9], person[11], person[13]

        # Check for NaN values before proceeding
        if (np.isnan(left_knee).any() or np.isnan(left_hip).any() or np.isnan(left_ankle).any() or
            np.isnan(right_knee).any() or np.isnan(right_hip).any() or np.isnan(right_ankle).any()):
            continue

        # Calculate the body orientation angle for both legs
        left_body_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_body_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # Takedown detected if both body angles are lower and head is closer to both knees
        if ((left_body_angle < 100 and right_body_angle < 100) and
            (min(left_knee[1], right_knee[1]) - head_y < 50)):
            takedown_detected = True
            break  # Stop checking other people in the frame

    # Only count a new takedown if the previous frame didn't detect one and keypoints were detected
    # and the delay has passed
    if takedown_detected and not previous_takedown_detected and previous_frame_keypoints_detected and takedown_delay_counter == 0:
        takedown_count += 1
        takedown_delay_counter = DELAY_FRAMES
        cv2.putText(frame_image, f"Takedown Count: {takedown_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Display the frame only when the takedown counter increments
        cv2.imshow(f"Frame {frame_number}", frame_image)
        cv2.waitKey(0)  # Wait for user input to move to the next frame

    # Update the previous frame's detection status
    previous_takedown_detected = takedown_detected
    previous_frame_keypoints_detected = current_frame_keypoints_detected

    # Decrease delay counter
    if takedown_delay_counter > 0:
        takedown_delay_counter -= 1

cv2.destroyAllWindows()


print(f"Total takedowns detected: {takedown_count}")  # Output: Total takedowns detected: 3