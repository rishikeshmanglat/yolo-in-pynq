# Import required libraries
import cv2
from ultralytics import YOLO
import numpy as np
import yaml

# Load the class names and total number of classes from the YAML file
with open('C://Users//rishi//Desktop//Main project//kitti - Copy.yaml', 'r') as f:
    data = yaml.safe_load(f)
    class_names = data['names']
    num_classes = data['nc']

# Initialize the model
model = YOLO('C://Users//rishi//Desktop//OBJECT_DETECTION_AND_TRACKING//yolov8n.pt')
model.classes = class_names

# Open the video file
def selectVideo(selection):
    if selection == 1:
        cap = cv2.VideoCapture('C://Users//rishi//Desktop//OBJECT_DETECTION_AND_TRACKING//challenge.mp4')
    elif selection == 2:
        cap = cv2.VideoCapture('C://Users//rishi//Desktop//OBJECT_DETECTION_AND_TRACKING//testing_college.mp4')
    elif selection == 4:
        cap = cv2.VideoCapture(0)
    elif selection == 3:
        cap = cv2.VideoCapture('C://Users//rishi//Desktop//OBJECT_DETECTION_AND_TRACKING//testing_college2.mp4')
    return cap

cap = selectVideo(2)

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video is over
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)

    # Draw the detection results on the frame
    for detection in results[0].boxes:
        box = detection.xyxy[0]
        class_id = int(detection.cls)
        score = detection.conf[0]
        x1, y1, x2, y2 = map(int, box)

        # Only draw the rectangle and label if the class ID is valid
        if 0 <= class_id < num_classes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            class_name = class_names[class_id]
            label = f'{class_name}: {score:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the detection results
    cv2.imshow('Object Detection', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Exit if the user presses the X button
    if cv2.getWindowProperty('Object Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the video and destroy all windows
cap.release()
cv2.destroyAllWindows()