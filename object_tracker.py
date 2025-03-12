import cv2
import numpy as np
import time
import os
from datetime import datetime
import argparse
from matplotlib import pyplot as plt
import urllib.request
import ssl
import keyboard



class ObjectTracker:
    def __init__(self, object_class=None, confidence_threshold=0.5):
        # Initialize parameters
        self.object_class = object_class  # If None, track any object
        self.confidence_threshold = confidence_threshold
        
        # Download required files if they don't exist
        self.download_model_files()
        
        # Load YOLO model
        print("Loading YOLO model...")
        self.net = cv2.dnn.readNet(
            "yolov3.weights", 
            "yolov3.cfg"
        )
        
        # Load COCO class names
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layers
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] if isinstance(i, np.ndarray) else layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Colors for visualization (random)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Trajectory storage
        self.trajectories = {}  # Dictionary to store trajectories for each tracked object
        self.current_id = 0     # To assign unique IDs to detected objects
        
        # Frame dimensions
        self.frame_width = None
        self.frame_height = None
        
        # Start time
        self.start_time = time.time()
    
    def download_model_files(self):
        """Download the required YOLO files if they don't exist."""
        # Create SSL context to handle certificate issues
        ssl_context = ssl._create_unverified_context()
        
        # URLs for the required files
        files = {
            "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
            "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
            "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights"
        }
        
        for filename, url in files.items():
            if not os.path.exists(filename):
                print(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filename)
                    print(f"Downloaded {filename} successfully")
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
                    if filename == "yolov3.weights":
                        print("The YOLOv3 weights file is large (236MB). If the download fails, please download it manually from:")
                        print(url)
                        print("And place it in the same directory as this script.")
                        raise Exception("Failed to download required files")
        
    def detect_objects(self, frame):
        if self.frame_width is None:
            self.frame_height, self.frame_width = frame.shape[:2]
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        
        # Get detections
        outputs = self.net.forward(self.output_layers)
        
        # Process detections
        class_ids = []
        confidences = []
        boxes = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter based on confidence and class (if specified)
                if confidence > self.confidence_threshold:
                    if self.object_class is None or self.classes[class_id] == self.object_class:
                        # Get bounding box coordinates
                        center_x = int(detection[0] * self.frame_width)
                        center_y = int(detection[1] * self.frame_height)
                        w = int(detection[2] * self.frame_width)
                        h = int(detection[3] * self.frame_height)
                        
                        # Get top-left corner of bounding box
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
        
        # Apply non-maximum suppression to remove redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
        
        detected_objects = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                center_x = x + w // 2
                center_y = y + h // 2
                
                detected_objects.append({
                    'id': None,  # Will be assigned during tracking
                    'class_id': class_ids[i],
                    'class_name': self.classes[class_ids[i]],
                    'confidence': confidences[i],
                    'box': box,
                    'center': (center_x, center_y)
                })
                
        return detected_objects
    
    def update_trajectories(self, detected_objects, max_distance=50):
        # If no trajectories yet, initialize with detected objects
        if not self.trajectories:
            for obj in detected_objects:
                obj['id'] = self.current_id
                self.trajectories[self.current_id] = {
                    'class_id': obj['class_id'],
                    'class_name': obj['class_name'],
                    'points': [obj['center']],
                    'last_seen': 0,  # Frame counter
                    'color': self.colors[obj['class_id']].tolist()
                }
                self.current_id += 1
            return
        
        # Match detected objects with existing trajectories using simple distance matching
        assigned_trajectories = set()
        assigned_detections = set()
        
        # Calculate distances between current detections and existing trajectories
        for i, obj in enumerate(detected_objects):
            min_distance = float('inf')
            matching_id = None
            
            for traj_id, traj in self.trajectories.items():
                if traj['last_seen'] > 5:  # Skip if not seen recently
                    continue
                    
                last_point = traj['points'][-1]
                distance = np.sqrt((obj['center'][0] - last_point[0])**2 + 
                                  (obj['center'][1] - last_point[1])**2)
                
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    matching_id = traj_id
            
            if matching_id is not None:
                # Update existing trajectory
                self.trajectories[matching_id]['points'].append(obj['center'])
                self.trajectories[matching_id]['last_seen'] = 0
                obj['id'] = matching_id
                assigned_trajectories.add(matching_id)
                assigned_detections.add(i)
        
        # Add new trajectories for unassigned detections
        for i, obj in enumerate(detected_objects):
            if i not in assigned_detections:
                obj['id'] = self.current_id
                self.trajectories[self.current_id] = {
                    'class_id': obj['class_id'],
                    'class_name': obj['class_name'],
                    'points': [obj['center']],
                    'last_seen': 0,
                    'color': self.colors[obj['class_id']].tolist()
                }
                self.current_id += 1
        
        # Increment last_seen counter for trajectories not updated
        for traj_id in self.trajectories:
            if traj_id not in assigned_trajectories:
                self.trajectories[traj_id]['last_seen'] += 1
    
    def draw_trajectories(self, frame):
        # Draw the trajectories on the frame
        for obj_id, traj in self.trajectories.items():
            if traj['last_seen'] > 30:  # Skip drawing if not seen recently
                continue
                
            points = traj['points']
            color = traj['color']
            
            # Draw trajectory line
            if len(points) > 1:
                for i in range(1, len(points)):
                    thickness = int(np.sqrt(64 / float(i + 1)) * 2)
                    cv2.line(frame, points[i-1], points[i], color, thickness)
            
            # Draw the last point with the class name
            if points:
                last_point = points[-1]
                cv2.circle(frame, last_point, 5, color, -1)
                cv2.putText(frame, f"{traj['class_name']} #{obj_id}", 
                           (last_point[0] + 10, last_point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw timestamp
        elapsed_time = time.time() - self.start_time
        cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw instructions
        cv2.putText(frame, "Press 'q' to quit and save trajectories", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def save_trajectories(self, output_dir="outputs"):
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save as PNG visualization
        plt.figure(figsize=(10, 8))
        
        for obj_id, traj in self.trajectories.items():
            if len(traj['points']) < 2:
                continue  # Skip if too few points
                
            points = np.array(traj['points'])
            color = np.array(traj['color']) / 255.0  # Normalize color for matplotlib
            
            plt.plot(points[:, 0], self.frame_height - points[:, 1], 
                    color=color, linewidth=2, label=f"{traj['class_name']} #{obj_id}")
            
            # Mark start and end points
            plt.scatter(points[0, 0], self.frame_height - points[0, 1], 
                       color=color, marker='o', s=50)
            plt.scatter(points[-1, 0], self.frame_height - points[-1, 1], 
                       color=color, marker='x', s=50)
        
        plt.title("Object Trajectories")
        plt.xlabel("X Coordinate (pixels)")
        plt.ylabel("Y Coordinate (pixels)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Flip Y-axis to match image coordinates
        plt.xlim(0, self.frame_width)
        plt.ylim(0, self.frame_height)
        
        # Save figure
        png_path = os.path.join(output_dir, f"trajectories_{timestamp}.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        
        # 2. Save as PDF
        pdf_path = os.path.join(output_dir, f"trajectories_{timestamp}.pdf")
        plt.savefig(pdf_path, bbox_inches='tight')
        
        # 3. Save raw data to CSV
        csv_path = os.path.join(output_dir, f"trajectories_{timestamp}.csv")
        with open(csv_path, 'w') as f:
            f.write("object_id,class_name,timestamp,x,y\n")
            
            for obj_id, traj in self.trajectories.items():
                class_name = traj['class_name']
                for i, point in enumerate(traj['points']):
                    # Estimate timestamp (assuming constant frame rate)
                    est_time = i / 30.0  # Assuming 30 fps
                    f.write(f"{obj_id},{class_name},{est_time:.2f},{point[0]},{point[1]}\n")
        
        print(f"Saved trajectories to:")
        print(f"  - PNG: {png_path}")
        print(f"  - PDF: {pdf_path}")
        print(f"  - CSV: {csv_path}")
        
        return png_path, pdf_path, csv_path

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Object Tracking from Webcam")
    parser.add_argument("--class", dest="object_class", default=None,
                        help="Specify class to track (e.g., 'person', 'car', 'cell phone')")
    parser.add_argument("--camera", dest="camera_id", type=int, default=0,
                        help="Camera ID (default: 0)")
    parser.add_argument("--confidence", dest="confidence", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    # Initialize tracker
    try:
        tracker = ObjectTracker(
            object_class=args.object_class,
            confidence_threshold=args.confidence
        )
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(args.camera_id)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print(f"Starting object tracking...")
    print(f"Press 'q' to stop tracking and save the trajectory.")
    
    if args.object_class:
        print(f"Tracking only '{args.object_class}' objects.")
    else:
        print("Tracking all detected objects.")
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Detect objects
        detected_objects = tracker.detect_objects(frame)
        
        # Update trajectories
        tracker.update_trajectories(detected_objects)
        
        # Draw bounding boxes and trajectories
        frame_with_trajectories = frame.copy()
        
        # Draw bounding boxes
        for obj in detected_objects:
            x, y, w, h = obj['box']
            class_id = obj['class_id']
            color = tracker.colors[class_id].tolist()
            
            cv2.rectangle(frame_with_trajectories, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame_with_trajectories, 
                       f"{obj['class_name']} {obj['confidence']:.2f}", 
                       (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw trajectories
        frame_with_trajectories = tracker.draw_trajectories(frame_with_trajectories)
        
        # Show the frame
        cv2.imshow("Object Tracking", frame_with_trajectories)
        
        # Check for key press
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            print("Stopping tracking...")
            break
    
        if keyboard.is_pressed('q'):  # Alternative key detection
            print("Stopping tracking...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Save trajectories
    if hasattr(tracker, 'frame_width') and tracker.frame_width is not None:
        png_path, pdf_path, csv_path = tracker.save_trajectories()
        
        print("\nTrajectory data saved successfully!")
        print(f"You can find the files at:")
        print(f"  - PNG: {png_path}")
        print(f"  - PDF: {pdf_path}")
        print(f"  - CSV: {csv_path}")
    else:
        print("No frames were processed. No trajectories to save.")

if __name__ == "__main__":
    main()