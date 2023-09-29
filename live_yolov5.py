import cv2
import numpy as np
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can specify a different YOLOv5 version if needed

# Open a connection to the camera (0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Perform object detection using YOLOv5
    results = model(frame)

    # Get the bounding boxes, labels, and confidences
    bboxes = results.pred[0][:, :4].cpu().numpy()
    labels = results.pred[0][:, -1].cpu().numpy().astype(int)
    confidences = results.pred[0][:, 4].cpu().numpy()

    # Filter out low-confidence detections (adjust this threshold as needed)
    threshold = 0.5
    mask = confidences > threshold
    bboxes = bboxes[mask]
    labels = labels[mask]

    # Draw bounding boxes and labels on the frame
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = map(int, bbox)
        label_str = str(model.names[label])
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame = cv2.putText(frame, label_str, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
