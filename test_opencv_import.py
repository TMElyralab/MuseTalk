
import os
import sys

# Add paths
opencv_python_path = r"D:\tencent\devel\cv\opencv-4.5.5\build\lib\python3\Release"
opencv_bin = r"D:\tencent\devel\cv\opencv-4.5.5\build\install\x64\vc16\bin"
gst_bin = r"E:\gstreamer\1.0\msvc_x86_64\bin"

# Add to Python path
if opencv_python_path not in sys.path:
    sys.path.insert(0, opencv_python_path)

# Add to PATH
os.environ["PATH"] = opencv_bin + os.pathsep + os.environ["PATH"]
os.environ["PATH"] = gst_bin + os.pathsep + os.environ["PATH"]

# Add DLL directories
if sys.version_info >= (3, 8):
    os.add_dll_directory(opencv_bin)
    os.add_dll_directory(gst_bin)

# Try importing OpenCV
try:
    print("Attempting to import cv2...")
    import cv2
    print(f"Success! OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"Error importing cv2: {e}")
    import traceback
    traceback.print_exc()
