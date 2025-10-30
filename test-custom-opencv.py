# test-custom-opencv.py
import use_custom_opencv

# Now import OpenCV
import cv2
import os

# Print OpenCV details
print(f"OpenCV version: {cv2.__version__}")
print(f"OpenCV path: {os.path.dirname(cv2.__file__)}")

# Check for GStreamer support
build_info = cv2.getBuildInformation()
gstreamer_line = [line for line in build_info.splitlines() 
                 if "GStreamer:" in line]
print(f"GStreamer support: {gstreamer_line[0] if gstreamer_line else 'Not found'}")

# Test a simple GStreamer pipeline
try:
    pipeline = "videotestsrc ! videoconvert ! appsink"
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if cap.isOpened():
        print("GStreamer pipeline working!")
        ret, frame = cap.read()
        if ret:
            print(f"Successfully read frame with shape: {frame.shape}")
        else:
            print("Failed to read frame")
        cap.release()
    else:
        print("Failed to open GStreamer pipeline")
except Exception as e:
    print(f"Error testing GStreamer: {e}")