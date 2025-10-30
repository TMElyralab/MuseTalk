import os
import sys

# Add DLL directory
opencv_bin = r"D:\tencent\devel\cv\opencv-4.5.5\build\install\x64\vc16\bin"
os.add_dll_directory(opencv_bin)

# Find the exact .pyd file
pyd_dir = r"D:\tencent\devel\cv\opencv-4.5.5\build\lib\python3\Release"
print("Looking for cv2.pyd in:", pyd_dir)
for file in os.listdir(pyd_dir):
    if file.endswith(".pyd"):
        print(f"Found: {file}")

# Add to path and try import
sys.path.insert(0, pyd_dir)
try:
    import cv2
    print("Success!")
except ImportError as e:
    print(f"Failed: {e}")
