import os
import sys
import tarfile
import cv2
import torch
import time
from ultralytics import YOLO
from pyzbar.pyzbar import decode


basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from tracker import re3_tracker

if not os.path.exists(os.path.join(basedir, "data")):
    import tarfile
    tar = tarfile.open(os.path.join(basedir, "data.tar.gz"))
    tar.extractall(path=basedir)
tracker = re3_tracker.Re3Tracker()

def initialize_model(model_path):
    return YOLO(model_path)


def initialize_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        sys.exit(1)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    return cap, out

def initialize_video_stream(gst_pipeline , output_path):
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Failed to open video stream!")
        sys.exit(1)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    return cap, out





def get_initial_bbox(results):
    if results[0].boxes is not None and len(results[0].boxes.data.tolist()) > 0:
        r = results[0].boxes.data[0]
        x1, y1, x2, y2, score, class_id = r
        return [x1.cpu().item(), y1.cpu().item(), x2.cpu().item(), y2.cpu().item()]
    return None

def process_frame_tracker(frame, model, tracker, is_initialized, initial_bbox ,left_margin, right_margin, top_margin, bottom_margin ,border_color, box_thickness , is_centered,centered_time ,w,h,centered_count):
    results = model(frame)
    bbox = get_initial_bbox(results)
    
    if bbox:  # Use YOLO for the first frame or if YOLO detects an object
        initial_bbox = bbox
        tracker.track("target_object", frame[:, :, ::-1], initial_bbox)
        is_initialized = True
    else :
        print( " Yolo No object detected")
        # cv2.putText(frame,"we do not use yolo now ",(50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,255,255),1)

    if is_initialized:  # Use RE3 tracker if the object is already being tracked or YOLO is not detecting anything
        frameRGB = frame[:, :, ::-1]
        bbox = tracker.track("target_object", frameRGB)
        # if bbox is > left_margin and bbox is < right_margin and bbox is > top_margin and bbox is < bottom_margin:

        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # add text time like 00:00:00:0000 to the frame in the top left corner video size is w,h
            cv2.putText(frame,time.strftime("%H:%M:%S"),(0,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7,(0,0,0),1)
            # add text mode like "auto" or "manual" to the frame in the top right corner
            cv2.putText(frame,"Auto",(w-50,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7,(0,0,0),1)
            center_bbox = ((x1 + x2) // 2, (y1 + y2) // 2)
            center_frame = (w // 2, h // 2)
            cv2.line(frame, center_frame, center_bbox, (255, 255, 255), 2)
            if x1 > left_margin and x2 < right_margin and y1 > top_margin and y2 < bottom_margin:
                if not is_centered:
                    centered_time = time.time()
                    is_centered = True
                else :
                    if time.time() - centered_time > 4:
                        is_centered = False
                        centered_count += 1
                        
                        
                cv2.putText(frame,"In Hit Area " + f": {int(time.time() - centered_time)}",(left_margin+10,bottom_margin-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7,(0,0,0),1)        
                cv2.rectangle(frame, (left_margin, top_margin), (right_margin, bottom_margin), (0, 255, 0) , box_thickness)
            else:
                is_centered = False
                cv2.rectangle(frame, (left_margin, top_margin), (right_margin, bottom_margin), border_color, box_thickness)
                
            
        else:
            print("Invalid bounding box:", bbox)
    
    return frame, is_initialized, initial_bbox, is_centered, centered_time, centered_count
def process_frame_qrcode(frame,  is_initialized, initial_bbox):
    detectedBarcode = decode(frame)
    if not detectedBarcode:
        print("No any Barcode Detected")
    else:
        for barcode in detectedBarcode:
            if barcode.data != "":
                cv2.putText(frame,str(barcode.data),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,255),2)
                cv2.imwrite("code.png",frame)
    return frame, is_initialized, initial_bbox
def main():
    basedir = os.path.dirname(__file__)
    sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
    
    model_path = "best.pt"
    video_path = "one4.mp4"
    output_path = "sonuc.mp4"
#     gst_pipeline = (
#     "udpsrc port=5600 "
#     "caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264\" "
#     "! rtph264depay "
#     "! avdec_h264 "
#     "! videoconvert "
#     "! appsink emit-signals=True sync=False"
# )

    model = initialize_model(model_path)
    cap, out = initialize_video(video_path, output_path)
    # cap, out = initialize_video_stream(gst_pipeline, output_path)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    left_margin = int(w * 0.25)
    right_margin = int(w * 0.75)
    top_margin = int(h * 0.1)
    bottom_margin = int(h * 0.9)
    box_thickness = 2
    border_color = (255, 255, 255)
    center_point = (w // 2, h // 2)  # Calculate center point
    
    
    is_qrcode = False
    # get the is_qrocode value from api every 5 seconds
        
    initial_bbox = None
    is_initialized = False
    is_centered = False
    centered_time = time.time()
    centered_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if is_qrcode:
            print("QR Code Mode")
            frame, is_initialized, initial_bbox = process_frame_qrcode(frame, is_initialized, initial_bbox )
        else:
            print("Normal Mode")
            frame, is_initialized, initial_bbox ,is_centered, centered_time , centered_count =  process_frame_tracker(frame, model, tracker, is_initialized, initial_bbox,left_margin, right_margin, top_margin, bottom_margin ,border_color, box_thickness ,is_centered, centered_time,w,h ,centered_count)
        out.write(frame)
        # cv2.imshow('scanner', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
