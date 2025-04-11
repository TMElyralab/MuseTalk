#!/bin/bash
. $(pwd)/musetalk_env/bin/activate
export FFMPEG_PATH=$(pwd)/ffmpeg-4.4-amd64-static
echo "MuseTalk environment activated. FFMPEG_PATH set to $FFMPEG_PATH"
