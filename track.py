import cv2
import numpy as np
from sort import Sort  # SORT tracker

# Load YOLO model
net = cv2.dnn.readNet("yolov8.weights", "yolov8.cfg")  # Update with actual model
tracker = Sort()  # SORT multi-object tracker

cap = cv2.VideoCapture(0)
tracked_path = []  # Stores object path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection using YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    detections = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Filtering for a specific object (e.g., "person")
                x, y, w, h = (detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype("int")
                detections.append([x, y, x+w, y+h, confidence])

    # Track objects
    trackers = tracker.update(np.array(detections))

    # Select only one object (e.g., first detected)
    if len(trackers) > 0:
        x1, y1, x2, y2, obj_id = trackers[0]  # Choose first object
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        tracked_path.append((center_x, center_y))

        # Draw path
        for i in range(1, len(tracked_path)):
            cv2.line(frame, tracked_path[i - 1], tracked_path[i], (0, 255, 0), 2)

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(frame, f"ID: {int(obj_id)}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
