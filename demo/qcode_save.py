import cv2
from pyzbar.pyzbar import decode
from pydub import AudioSegment
from pydub.playback import play

# Initialize the GStreamer pipeline for video capture
gst_pipeline = (
    "udpsrc port=5600 ! "
    "application/x-rtp, encoding-name=H264 ! "
    "rtph264depay ! avdec_h264 ! "
    "videoconvert ! videorate ! video/x-raw, framerate=30/1 ! "  # Increase the FPS here
    "videoconvert ! appsink"
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

# Check if the video capture is initialized
if not cap.isOpened():
    print("Error: Could not open video stream")
    
    exit()

# Get the width and height of the frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

# Load the beep sound (uncomment if you have the beep-02.wav file)
# song = AudioSegment.from_wav("beep-02.wav")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the image like a mirror image
    frame = cv2.flip(frame, 1)

    # Detect barcodes in the frame
    detectedBarcodes = decode(frame)

    # If any barcodes are detected
    for barcode in detectedBarcodes:
        # Draw a bounding box around the barcode
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the barcode data
        if barcode.data:
            barcodeData = barcode.data.decode('utf-8')
            cv2.putText(frame, barcodeData, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            # Play the beep sound (uncomment if you have the beep-02.wav file)
            # play(song)
            print(barcodeData)
            cv2.imwrite("code10.png", frame)

    # Write the frame into the video file
    out.write(frame)

    # Display the frame
    cv2.imshow('scanner', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
