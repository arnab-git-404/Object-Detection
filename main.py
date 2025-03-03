import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Video capture
cap = cv2.VideoCapture(0)

# Tracking variables
tracks = {}
next_id = 1
MAX_DISTANCE = 100  # Pixels between frames to consider same object
TRACK_HISTORY = 25  # How many positions to keep in track visualization

# Store complete paths for final plot
full_paths = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    results = model(frame, verbose=False)[0]
    current_detections = []

    # Process detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        centroid = (int((x1+x2)/2), int((y1+y2)/2))
        class_id = int(box.cls)
        class_name = model.names[class_id]
        conf = float(box.conf)
        
        if conf > 0.5:
            current_detections.append((class_name, centroid, (x1, y1, x2, y2)))

    # Update tracks
    active_ids = []
    for class_name, centroid, bbox in current_detections:
        # Find existing tracks for this class
        candidates = [(tid, track) for tid, track in tracks.items() 
                     if track['class'] == class_name]

        # Find closest existing track
        min_dist = MAX_DISTANCE
        best_tid = None
        for tid, track in candidates:
            last_pos = track['positions'][-1]
            distance = np.linalg.norm(np.array(last_pos) - np.array(centroid))
            if distance < min_dist:
                min_dist = distance
                best_tid = tid

        if best_tid:
            # Update existing track
            tracks[best_tid]['positions'].append(centroid)
            tracks[best_tid]['bbox'] = bbox
            active_ids.append(best_tid)
        else:
            # Create new track
            tracks[next_id] = {
                'class': class_name,
                'positions': [centroid],
                'bbox': bbox
            }
            active_ids.append(next_id)
            next_id += 1

    # Remove stale tracks and maintain history
    for tid in list(tracks.keys()):
        if tid not in active_ids:
            # Save completed path
            if tid not in full_paths:
                full_paths[tid] = tracks[tid]
            del tracks[tid]
        else:
            # Keep only last TRACK_HISTORY positions
            tracks[tid]['positions'] = tracks[tid]['positions'][-TRACK_HISTORY:]

    # Draw real-time visualization
    for tid, track in tracks.items():
        # Draw bounding box
        x1, y1, x2, y2 = track['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw path
        path = track['positions']
        for i in range(1, len(path)):
            cv2.line(frame, path[i-1], path[i], (0, 0, 255), 2)
        
        # Display track ID
        cv2.putText(frame, f"ID:{tid} {track['class']}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Generate and save path visualization
plt.figure(figsize=(10, 8))
for tid, track in full_paths.items():
    path = np.array(track['positions'])
    plt.plot(path[:, 0], path[:, 1], 'o-', label=f"{track['class']} (ID:{tid})")

plt.title('Object Movement Paths')
plt.xlabel('X Position in Frame')
plt.ylabel('Y Position in Frame')
plt.gca().invert_yaxis()  # Match image coordinate system
plt.legend()
plt.grid(True)
plt.savefig('object_paths.png')
plt.close()

cap.release()
cv2.destroyAllWindows()

print("Tracking complete! Saved path visualization as 'object_paths.png'")