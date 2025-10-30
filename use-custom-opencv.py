import os
import sys
import site
import platform

def setup_opencv_environment():
    """Configure environment for custom OpenCV with all required DLLs"""
    print(f"Setting up OpenCV environment for Python {sys.version}")
    
    # Paths to OpenCV and its dependencies
    opencv_python_path = r"D:\tencent\devel\cv\opencv-4.5.5\build\lib\python3\Release"
    opencv_bin = r"D:\tencent\devel\cv\opencv-4.5.5\build\install\x64\vc16\bin"
    opencv_lib = r"D:\tencent\devel\cv\opencv-4.5.5\build\install\x64\vc16\lib"
    gst_bin = r"E:\gstreamer\1.0\msvc_x86_64\bin"
    
    # Add OpenCV module path to Python path
    if opencv_python_path not in sys.path:
        sys.path.insert(0, opencv_python_path)
        print(f"Added to Python path: {opencv_python_path}")
    
    # Add DLL directories to PATH
    paths_to_add = [opencv_bin, opencv_lib, gst_bin]
    for path in paths_to_add:
        if os.path.exists(path):
            if path not in os.environ["PATH"]:
                os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
                print(f"Added to PATH: {path}")
        else:
            print(f"WARNING: Path does not exist: {path}")
    
    # Use add_dll_directory for Python 3.8+
    if sys.version_info >= (3, 8):
        for path in paths_to_add:
            if os.path.exists(path):
                try:
                    os.add_dll_directory(path)
                    print(f"Added DLL directory: {path}")
                except Exception as e:
                    print(f"ERROR adding DLL directory {path}: {e}")
    
    # Print architecture information
    print(f"Python architecture: {platform.architecture()[0]}")
    
    return True

# Execute if run directly
if __name__ == "__main__":
    success = setup_opencv_environment()
    
    if success:
        try:
            import cv2
            print(f"\nOpenCV successfully loaded!")
            print(f"OpenCV version: {cv2.__version__}")
            print(f"OpenCV path: {os.path.dirname(cv2.__file__)}")
            
            # Check for GStreamer support
            build_info = cv2.getBuildInformation()
            gstreamer_line = [line for line in build_info.splitlines() 
                            if "GStreamer:" in line]
            print(f"GStreamer support: {gstreamer_line[0] if gstreamer_line else 'Not found'}")
        except Exception as e:
            print(f"\nERROR loading OpenCV: {e}")
            
            # Provide detailed error information
            import traceback
            traceback.print_exc()