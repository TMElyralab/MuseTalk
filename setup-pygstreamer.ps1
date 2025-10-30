# Create this script as setup-pygstreamer.ps1

# Clear any existing paths that might conflict
$env:PATH = ($env:PATH -split ';' | Where-Object { -not $_.Contains('msys64') }) -join ';'

# Add GStreamer paths
$GST_ROOT = "E:\gstreamer\1.0\msvc_x86_64"
$env:PATH = "$GST_ROOT\bin;$env:PATH"
$env:GI_TYPELIB_PATH = "$GST_ROOT\lib\girepository-1.0"
$env:PKG_CONFIG_PATH = "$GST_ROOT\lib\pkgconfig"
$env:GST_PLUGIN_PATH = "$GST_ROOT\lib\gstreamer-1.0"

# Install PyGObjects' Windows binaries
pip install --no-cache-dir PyGObject-stubs
pip install --no-cache-dir pycairo

# Don't forget to install torch/CUDA if needed for MuseTalk
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Run a test script to verify
$TEST_SCRIPT = @'
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)
print(f"GStreamer version: {Gst.version_string()}")
print("GStreamer successfully initialized!")
'@

Write-Host "Creating test script..."
$TEST_SCRIPT | Out-File -FilePath test_gst.py -Encoding utf8
Write-Host "Running test script..."
python test_gst.py
