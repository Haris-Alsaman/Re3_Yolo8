# import cv2

# # GStreamer pipeline for receiving video over UDP
# gst_pipeline = (
#     "udpsrc port=5600 ! "
#     "application/x-rtp, encoding-name=H264 ! "
#     "rtph264depay ! avdec_h264 ! "
#     "videoconvert ! appsink"
# )

# cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

# if not cap.isOpened():
#     print("Failed to open video stream!")
# else:
#     print("Video stream opened successfully!")
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Failed to get frame from the video stream")
#                 break
            
#             cv2.imshow('Video Stream', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     except KeyboardInterrupt:
#         print("Interrupted by user")
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         print("Video stream closed cleanly")

import cv2
import time

gst_pipeline = (
    "udpsrc port=5600 ! "
    "application/x-rtp, encoding-name=H264 ! "
    "rtph264depay ! avdec_h264 ! "
    "videoconvert ! videorate ! video/x-raw, framerate=60/1 ! "  # Increase the FPS here
    "videoconvert ! appsink"
)

def process_frame(frame):
    cv2.imshow('Video Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

if __name__ == '__main__':
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Failed to open video stream!")
    else:
        print("Video stream opened successfully!")

        start_time = time.time()
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to get frame from the video stream")
                    break

                if not process_frame(frame):
                    break

                frame_count += 1

                if frame_count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    print(f"FPS: {fps:.2f}")

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Video stream closed cleanly")

