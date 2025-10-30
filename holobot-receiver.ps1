<#
.SYNOPSIS
    Launches the GStreamer pipeline to decode and play the MuseTalk real-time A/V stream.

.DESCRIPTION
    This script uses a "zero-copy" video pipeline. The video frame is decoded on the GPU
    and rendered directly to the screen using d3d11videosink without ever being copied
    to system RAM, providing the lowest possible latency and highest performance.

.NOTES
    - Requires GStreamer 1.0 (with msvc_x86_64 and nvcodec packages) to be installed and in the system's PATH.
    - Run this script from a PowerShell terminal.
    - To stop the stream, press Ctrl+C.
#>

# --- Configuration ---
$videoPort = 5000
$audioPort = 5001

# --- User Feedback ---
Write-Host "🚀 Launching GStreamer Decoder (Zero-Copy Video Pipeline)..." -ForegroundColor Green
Write-Host "   - Listening for VIDEO on UDP port: $videoPort"
Write-Host "   - Listening for AUDIO on UDP port: $audioPort"
Write-Host "   (Press Ctrl+C to stop the stream)"
Write-Host ""


# --- Launch GStreamer Pipeline ---
gst-launch-1.0 -v `
    udpsrc port=$videoPort caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" `
        ! rtpjitterbuffer latency=200 `
        ! queue `
        ! rtph264depay `
        ! h264parse `
        ! nvh264dec `
        ! d3d11videosink sync=true qos=true max-lateness=200000000 `
`
    udpsrc port=$audioPort caps="application/x-rtp, media=audio, clock-rate=48000, encoding-name=OPUS, payload=97" `
        ! rtpjitterbuffer latency=375 drop-on-latency=true `
        ! rtpopusdepay `
        ! opusdec plc=true `
        ! audioconvert `
        ! audioresample `
        ! wasapisink sync=true