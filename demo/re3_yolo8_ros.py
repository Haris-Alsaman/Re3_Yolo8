import cv2

import os.path
import sys
import torch
from ultralytics import YOLO

# Initialize YOLO model
basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(basedir, os.pardir)))
model_path = "best.pt"  # Path to the YOLO model
model = YOLO(model_path)

# GStreamer pipeline for receiving video over UDP
gst_pipeline = (
    "udpsrc port=5600 "
    "caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264\" "
    "! rtph264depay "
    "! avdec_h264 "
    "! videoconvert "
    "! appsink emit-signals=True sync=False"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Failed to open video stream!")
    sys.exit(1)

# Prepare to save the output video
output_path = 'result.mp4'
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

# Initialize re3 tracker
from tracker import re3_tracker
if not os.path.exists(os.path.join(basedir, "data")):
    import tarfile
    tar = tarfile.open(os.path.join(basedir, "data.tar.gz"))
    tar.extractall(path=basedir)
tracker = re3_tracker.Re3Tracker()

# Tracking variables
is_initialized = False
initial_bbox = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame from the video stream")
            break

        # Use YOLO for the first frame or if tracking was lost
        results = model(frame)
    # Assuming we track the first detected object
        if results[0].boxes is not None :
            if len(results[0].boxes.data.tolist()) > 0:
                r = results[0].boxes.data[0]
                x11, y11, x22, y22 ,  score, class_id= r 
                print("Coordinates all: ", x11, y11.cpu().item(), x22, y22)
                
                initial_bbox = [x11.cpu().item(), y11.cpu().item(), x22.cpu().item(), y22.cpu().item()]
                # convert to int 
                x11, y11, x22, y22 = map(int, initial_bbox)

                if isinstance(initial_bbox, torch.Tensor):
                        initial_bbox = initial_bbox.cpu().numpy()  
                tracker.track("target_object", frame[:, :, ::-1], initial_bbox)  # Initialize tracker
                is_initialized = True
                frameRGB = frame[:, :, ::-1]
                bbox = tracker.track("target_object", frameRGB)
                if bbox is not None and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        # Optional: Print the coordinates to debug
                        print(f"Coordinates first: {x1}, {y1}, {x2}, {y2}")

                        # Draw the tracking bounding box
                        overlay = frame.copy()
                        # cv2.putText(overlay, "counter: " + '10', (50, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                        cv2.rectangle(overlay,(x11, y11), (x22, y22), (0, 255, 0), 2)
                        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        out.write(frame)
                        cv2.imshow('Video Stream with Tracking', frame)
                else:
                        print("Invalid bounding box:", bbox)
        elif is_initialized :
                    bbox = tracker.track("target_object", frameRGB)
                    if bbox is not None and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                    # Optional: Print the coordinates to debug
                        print(f"Coordinates last: {x1}, {y1}, {x2}, {y2}")

                    # Draw the tracking bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        out.write(frame)
                        cv2.imshow('Video Stream with Tracking', frame)
                    else:
                        print("Invalid bounding box:", bbox)
        else:
            out.write(frame) 
            print("No object detected")
            cv2.imshow('Video Stream with Tracking', frame)
            continue
        # Display the frame
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
