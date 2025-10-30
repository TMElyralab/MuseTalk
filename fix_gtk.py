import os
import sys
import ctypes
from ctypes import windll
import glob
import platform

def inspect_gtk_installation():
    print(f"\n#### GTK Installation Info ####")
    gtk_bin = r'C:\gtk\bin'
    gtk_lib = r'C:\gtk\lib'
    
    # Add both directories to PATH
    os.environ['PATH'] = f"{gtk_bin};{gtk_lib};{os.environ['PATH']}"
    print(f"Python version: {platform.python_version()}")
    
    # Set additional environment variables
    os.environ['GI_TYPELIB_PATH'] = r'C:\gtk\lib\girepository-1.0'
    print(f"GI_TYPELIB_PATH: {os.environ.get('GI_TYPELIB_PATH', 'Not set')}")
    
    # Find all DLLs in GTK directories
    bin_dlls = glob.glob(os.path.join(gtk_bin, "*.dll"))
    lib_dlls = glob.glob(os.path.join(gtk_lib, "*.dll"))
    print(f"Found {len(bin_dlls)} DLLs in {gtk_bin}")
    print(f"Found {len(lib_dlls)} DLLs in {gtk_lib}")
    
    # Critical DLLs that need to be loaded in the right order
    critical_dlls = [
        os.path.join(gtk_bin, "glib-2.0-0.dll"),
        os.path.join(gtk_bin, "gobject-2.0-0.dll"),
        os.path.join(gtk_bin, "gmodule-2.0-0.dll"),
        os.path.join(gtk_bin, "girepository-1.0-1.dll"),
        os.path.join(gtk_bin, "gio-2.0-0.dll"),
        os.path.join(gtk_bin, "ffi-8.dll"),
        os.path.join(gtk_bin, "z.dll"),
        os.path.join(gtk_bin, "libintl-8.dll")
    ]
    
    # Try to load critical DLLs first
    print("\n#### Loading Critical DLLs ####")
    for dll in critical_dlls:
        if os.path.exists(dll):
            try:
                windll.LoadLibrary(dll)
                print(f"✓ Loaded {os.path.basename(dll)}")
            except Exception as e:
                print(f"✗ Failed loading {os.path.basename(dll)}: {e}")
        else:
            print(f"! Missing {os.path.basename(dll)}")
    
    # Try importing gi
    print("\n#### Testing PyGObject Import ####")
    try:
        import gi
        print(f"✓ Successfully imported gi ({gi.__file__})")
        return True
    except ImportError as e:
        print(f"✗ Failed to import gi: {e}")
        return False

if __name__ == "__main__":
    success = inspect_gtk_installation()
    
    if not success:
        print("\n#### Troubleshooting Tips ####")
        print("1. Your PyGObject installation might not be compatible with your GTK installation.")
        print("2. Try reinstalling PyGObject with:")
        print("   pip uninstall pygobject")
        print("   pip install pygobject")
        print("3. If that doesn't work, try installing from an alternate source:")
        print("   pip install --no-binary :all: pygobject")
