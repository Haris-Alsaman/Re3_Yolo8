import cv2
import glob
import os.path
import sys

import torch
from ultralytics import YOLO

# Load YOLO model
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))

model_path = "best.pt"  # Update this path
print(os.getcwd())

model = YOLO(model_path)

# Open the video
video_path = "one1.mp4"  # Update this path
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file.")

# Prepare to save the output video
output_path = 'sonuc.mp4'
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

# Initialize re3_tracker
from tracker import re3_tracker
if not os.path.exists(os.path.join(basedir, "data")):
    import tarfile
    tar = tarfile.open(os.path.join(basedir, "data.tar.gz"))
    tar.extractall(path=basedir)
tracker = re3_tracker.Re3Tracker()

initial_bbox = None
is_initialized = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLO for the first frame or if tracking was lost
 
    results = model(frame)
    # Assuming we track the first detected object
    if results[0].boxes is not None :
        if len(results[0].boxes.data.tolist()) > 0:
            r = results[0].boxes.data[0]
            x1, y1, x2, y2 ,  score, class_id= r 
            initial_bbox = [x1.cpu().item(), y1.cpu().item(), x2.cpu().item(), y2.cpu().item()]

            if isinstance(initial_bbox, torch.Tensor):
                    initial_bbox = initial_bbox.cpu().numpy()  
            tracker.track("target_object", frame[:, :, ::-1], initial_bbox)  # Initialize tracker
            is_initialized = True
            frameRGB = frame[:, :, ::-1]
            bbox = tracker.track("target_object", frameRGB)
            if bbox is not None and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    # Optional: Print the coordinates to debug
                    print(f"Coordinates: {x1}, {y1}, {x2}, {y2}")

                    # Draw the tracking bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    out.write(frame)
            else:
                    print("Invalid bounding box:", bbox)
    elif is_initialized :
                bbox = tracker.track("target_object", frameRGB)
                if bbox is not None and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                # Optional: Print the coordinates to debug
                    print(f"Coordinates: {x1}, {y1}, {x2}, {y2}")

                # Draw the tracking bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    out.write(frame)
                else:
                    print("Invalid bounding box:", bbox)
    else:
        out.write(frame)
        continue


    # Write the frame with tracking box to the output video
   # `out.write(frame)` is writing the current frame with the tracking bounding box (if applicable) to
   # the output video file. This allows you to save the processed video with the tracking information
   # included.
    # out.write(frame)

    # Optionally, break the loop if a certain condition is met (e.g., a key press)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
