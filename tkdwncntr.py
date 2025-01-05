from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')  # Load model
result = model("../vids/and/testfoot.mp4") 
takedown_count = 0
for result in result:
    keypoints = result.keypoints.xy.numpy()  # Keypoints as NumPy array
    for person in keypoints:
        knee_y, head_y = person[13][1], person[0][1]  # Y-coordinate of knee and head
        if knee_y - head_y < 50:  # Threshold for a person on the ground
            takedown_count += 1 # Inference

print(f"Total takedowns detected: {takedown_count}")  # Output: Total takedowns detected: 3