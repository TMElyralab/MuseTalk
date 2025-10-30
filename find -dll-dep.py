import os
import sys
import ctypes

# Add the OpenCV bin directory to the DLL search path
opencv_bin = r"D:\tencent\devel\cv\opencv-4.5.5\build\install\x64\vc16\bin"
os.add_dll_directory(opencv_bin)

# Add the directory containing the cv2.pyd file to the Python path
opencv_pyd_dir = r"D:\tencent\devel\cv\opencv-4.5.5\build\lib\python3\Release"
sys.path.insert(0, opencv_pyd_dir)

# Pre-load only the essential DLLs (skip highgui)
essential_dlls = [
    "opencv_core455.dll", 
    "opencv_imgproc455.dll",
    "opencv_imgcodecs455.dll",
    "opencv_videoio455.dll",
    "opencv_flann455.dll",
    "opencv_features2d455.dll"
]

for dll in essential_dlls:
    try:
        dll_path = os.path.join(opencv_bin, dll)
        ctypes.CDLL(dll_path)
        print(f"Successfully pre-loaded {dll}")
    except Exception as e:
        print(f"Failed to load {dll}: {e}")

# Now try importing cv2
try:
    import cv2
    print(f"\nSuccess! OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"\nStill failed: {e}")
