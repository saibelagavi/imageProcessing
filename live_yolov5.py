import cv2
import torch
import numpy as np  # Add this import for NumPy

# Load YOLOv5 model (use 'yolov5s' for the smallest version, or specify a different one)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open a connection to the camera (0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Perform object detection using YOLOv5
    results = model(frame)

    # Extract object information
    pred = results.pred[0]
    bboxes = pred[:, :4].cpu().numpy()  # Bounding boxes
    labels = pred[:, -1].cpu().numpy().astype(int)  # Class labels
    confidences = pred[:, 4].cpu().numpy()  # Confidences

    # Filter out low-confidence detections (adjust this threshold as needed)
    threshold = 0.5
    mask = confidences > threshold
    bboxes = bboxes[mask]
    labels = labels[mask]
    confidences = confidences[mask]

    # Get unique class labels and counts
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Display object information
    info_str = ""
    for label, count in zip(unique_labels, label_counts):
        label_str = str(model.names[label])
        info_str += f"{label_str}: {count}  "

    # Draw bounding boxes and labels on the frame
    for bbox, label, confidence in zip(bboxes, labels, confidences):
        x1, y1, x2, y2 = map(int, bbox)
        label_str = str(model.names[label])
        info_str += f"{label_str} ({confidence:.2f}), "
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame = cv2.putText(frame, f"{label_str} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with object information
    cv2.putText(frame, info_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
