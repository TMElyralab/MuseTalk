# 1. Create a bridge script named 'setup-gstreamer-env.ps1'
@"
# Path to GStreamer installation
`$GST_ROOT = "E:\\gstreamer\\1.0\\msvc_x86_64"

# Add GStreamer's Python module directory to PYTHONPATH
`$env:PYTHONPATH = "`$GST_ROOT\\lib\\python3\\site-packages;`$env:PYTHONPATH"

# Add GStreamer binary directory to PATH
`$env:PATH = "`$GST_ROOT\\bin;`$env:PATH"

# Set GI_TYPELIB_PATH to find GObject Introspection typelibs
`$env:GI_TYPELIB_PATH = "`$GST_ROOT\\lib\\girepository-1.0"

# Set GST_PLUGIN_PATH to find GStreamer plugins
`$env:GST_PLUGIN_PATH = "`$GST_ROOT\\lib\\gstreamer-1.0"

# Run a simple test to verify it works
python -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; Gst.init(None); print(f'GStreamer version: {Gst.version_string()}'); print('GStreamer successfully initialized!')"
"@ | Out-File -FilePath setup-gstreamer-env.ps1 -Encoding utf8

# 2. Run the script to configure environment
./setup-gstreamer-env.ps1
