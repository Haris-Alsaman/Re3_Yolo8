

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator
import time

# Upload the yolov8n.pt file and video file to your Colab environment
# Make sure to provide the correct path to the video file
model = YOLO("best.pt")
cap = cv2.VideoCapture("one1.mp4")  # Provide the correct path

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('z.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Calculate coordinates for the empty white box
left_margin = int(w * 0.25)
right_margin = int(w * 0.75)
top_margin = int(h * 0.1)
bottom_margin = int(h * 0.9)
box_thickness = 3
border_color = (255, 255, 255)

center_point = (w // 2, h // 2)  # Calculate center point

start_time_inside = None
counter_inside = 0
counter = 0

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)
    results = model.track(im0, persist=True ,tracker="bytetrack.yaml" )
    boxes = results[0].boxes.xyxy.cpu()

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            annotator.box_label(box, label=str(track_id), color=colors(int(track_id)))
            annotator.visioneye(box, center_point)
            
            
            if left_margin < box[0] < right_margin and top_margin < box[1] < bottom_margin :
              
                border_color = (0, 255, 0)  # Green if inside the box
                if start_time_inside is None:
                    start_time_inside = time.time()
                else:
                    elapsed_time_inside = time.time() - start_time_inside
                    if elapsed_time_inside >= 1:
                        counter_inside += 1
                        start_time_inside = time.time()
                    # Draw counter on side border
                    cv2.putText(im0, "Time: " + str(counter_inside), (left_margin, top_margin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
            else:
                border_color = (255, 255, 255)
                start_time_inside = None
                counter_inside = 0

        # Draw empty white box
        if counter_inside >= 4:
          counter_inside = 0
          counter += 1
        cv2.putText(im0, "counter: " + str(counter), (right_margin - 50, top_margin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        overlay = im0.copy()
        cv2.rectangle(overlay, (left_margin, top_margin), (right_margin, bottom_margin), border_color , box_thickness)
        im0 = cv2.addWeighted(overlay, 0.5, im0, 0.5, 0)

    out.write(im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()

