import glob
import os.path
import sys

import cv2

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from tracker import re3_tracker

if not os.path.exists(os.path.join(basedir, "data")):
    import tarfile
    tar = tarfile.open(os.path.join(basedir, "data.tar.gz"))
    tar.extractall(path=basedir)

# Initialize the video writer
video_filename = os.path.join(basedir, 'output_video.avi')
frame_size = (640, 480)
fps = 20  # You might need to adjust this depending on your source video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("/content/onez.mp4", fourcc, fps, frame_size)

tracker = re3_tracker.Re3Tracker()
image_paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "data", "*.jpg")))
initial_bbox = [175, 154, 251, 229]
tracker.track("ball", image_paths[0], initial_bbox)
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        continue  # Skip any images that failed to load
    # Tracker expects RGB, but opencv loads BGR.
    imageRGB = image[:, :, ::-1]
    bbox = tracker.track("ball", imageRGB)
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [0, 0, 255], 5)
    
    # Resize the frame to match the video size, if necessary
    resized_frame = cv2.resize(image, frame_size)
    
    # Write the frame to the video file
    out.write(resized_frame)

# Release the video writer to finish writing the video file
out.release()