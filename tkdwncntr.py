from ultralytics import YOLO
import numpy as np
import cv2
model = YOLO('yolo11n-pose.pt')  # Load model
result = model("../vids/and/testfoot2.mp4") 
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

for frame_number, result in enumerate(result):
    print(f"Processing frame: {frame_number}")
    frame_image = result.orig_img.copy()  # Retrieve the original frame image
    keypoints = result.keypoints.xy.numpy()  # Extract keypoints as NumPy array
    takedown_detected = False  # Reset takedown detection per frame

    for person in keypoints:
        head_y = person[0][1]
        left_knee, left_hip, left_ankle = person[10], person[12], person[14]
        right_knee, right_hip, right_ankle = person[9], person[11], person[13]

        # Check for NaN values before proceeding
        if (np.isnan(left_knee).any() or np.isnan(left_hip).any() or np.isnan(left_ankle).any() or
            np.isnan(right_knee).any() or np.isnan(right_hip).any() or np.isnan(right_ankle).any()):
            continue

        # Draw keypoints on the frame
        for point in [person[0], left_knee, left_hip, left_ankle, right_knee, right_hip, right_ankle]:
            cv2.circle(frame_image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

        # Calculate the body orientation angle for both legs
        left_body_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_body_angle = calculate_angle(right_hip, right_knee, right_ankle)

        if not np.isnan(left_body_angle):
            cv2.putText(frame_image, f"L Angle: {int(left_body_angle)}", (int(left_hip[0]), int(left_hip[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if not np.isnan(right_body_angle):
            cv2.putText(frame_image, f"R Angle: {int(right_body_angle)}", (int(right_hip[0]), int(right_hip[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Takedown detected if both body angles are lower and head is closer to both knees
        if ((left_body_angle < 100 and right_body_angle < 100) and
            (min(left_knee[1], right_knee[1]) - head_y < 50)):
            takedown_detected = True
            break  # Stop checking other people in the frame

    # Only count a new takedown if the previous frame didn't detect one
    if takedown_detected and not previous_takedown_detected:
        takedown_count += 1
        cv2.putText(frame_image, f"Takedown Count: {takedown_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame and keep it open until user closes it
    cv2.imshow(f"Frame {frame_number}", frame_image)
    cv2.waitKey(0)  # Wait for user input to move to the next frame

    # Update the previous frame's detection status
    previous_takedown_detected = takedown_detected

cv2.destroyAllWindows()


print(f"Total takedowns detected: {takedown_count}")  # Output: Total takedowns detected: 3