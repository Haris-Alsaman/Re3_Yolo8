# let's read a barcode like the machine (product scanner in super market )

# using packages 
# pip install opencv-python 
# pip install pydub 
# pip install pyzbar 

import cv2 
from pyzbar.pyzbar import decode
from pydub import AudioSegment
from pydub.playback import play
from pyzbar.pyzbar import ZBarSymbol

# capture webcam 
# `cap = cv2.VideoCapture(0)` is initializing a video capture object to capture video from the default
# camera (index 0) connected to the system. This object allows you to access frames from the camera
# for further processing, such as barcode detection in this case.
# cap = cv2.VideoCapture(0)
# cap = 
gst_pipeline = (
    "udpsrc port=5600 ! "
    "application/x-rtp, encoding-name=H264 ! "
    "rtph264depay ! avdec_h264 ! "
    "videoconvert ! videorate ! video/x-raw, framerate=30/1 ! "  # Increase the FPS here
    "videoconvert ! appsink"
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
# cap.set(cv2.CAP_PROP_FPS,10)
if not cap.isOpened():
        print("Failed to open video stream!")
else:
        print("Video stream opened successfully!")
# The line `song = AudioSegment.from_wav("beep-02.wav")` is loading an audio file named "beep-02.wav"
# and creating an `AudioSegment` object from it. This object can be used to play the audio file later
# in the code.
#song = AudioSegment.from_wav("beep-02.wav")

while cap.isOpened():
    success,frame = cap.read()
    cv2.imshow('scanner' , frame)
    # detect the barcode 
    detectedBarcode = decode(frame, symbols=[ZBarSymbol.QRCODE])
    # if no any barcode detected  
    if  detectedBarcode:
        # codes in barcode 
        for barcode in detectedBarcode:
            # if barcode is not blank 
            if barcode.data != "":
                cv2.putText(frame,str(barcode.data),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,255),2)
                #play(song)
                print("Barcode successfully detected!")
                print(barcode.data)
                barcode_data = barcode.data.decode('utf-8')
                print(barcode_data)
                cv2.imwrite("code.png",frame)


    if cv2.waitKey(1) == ord('q'):
        break