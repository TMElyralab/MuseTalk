#!/bin/bash

# Set the checkpoints directory
CheckpointsDir="models"

# Create necessary directories
mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper

# Install required packages
echo "Installing dependencies..."
pip install -U "huggingface_hub[cli]<1.0,>=0.19.3"
pip install gdown
sudo apt-get install -y aria2  # or: brew install aria2 on Mac

# Function for fast download with aria2c
fast_download() {
    local url=$1
    local output=$2
    aria2c -x 16 -s 16 -k 1M -c --auto-file-renaming=false --allow-overwrite=true -d "$(dirname "$output")" -o "$(basename "$output")" "$url"
}

echo "Using aria2c for faster downloads..."

# Download MuseTalk V1.0 weights
echo "Downloading MuseTalk V1.0 weights..."
mkdir -p $CheckpointsDir/musetalk
fast_download \
  "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin" \
  "$CheckpointsDir/musetalk/pytorch_model.bin"
fast_download \
  "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json" \
  "$CheckpointsDir/musetalk/musetalk.json"

# Download MuseTalk V1.5 weights
echo "Downloading MuseTalk V1.5 weights..."
mkdir -p $CheckpointsDir/musetalkV15
fast_download \
  "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth" \
  "$CheckpointsDir/musetalkV15/unet.pth"
fast_download \
  "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json" \
  "$CheckpointsDir/musetalkV15/musetalk.json"

# Download SD VAE weights
echo "Downloading SD VAE weights..."
mkdir -p $CheckpointsDir/sd-vae
fast_download \
  "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json" \
  "$CheckpointsDir/sd-vae/config.json"
fast_download \
  "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin" \
  "$CheckpointsDir/sd-vae/diffusion_pytorch_model.bin"

# Download Whisper weights
echo "Downloading Whisper weights..."
mkdir -p $CheckpointsDir/whisper
fast_download \
  "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json" \
  "$CheckpointsDir/whisper/config.json"
fast_download \
  "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin" \
  "$CheckpointsDir/whisper/pytorch_model.bin"
fast_download \
  "https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json" \
  "$CheckpointsDir/whisper/preprocessor_config.json"

# Download DWPose weights
echo "Downloading DWPose weights..."
mkdir -p $CheckpointsDir/dwpose
fast_download \
  "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth" \
  "$CheckpointsDir/dwpose/dw-ll_ucoco_384.pth"

# Download SyncNet weights
echo "Downloading SyncNet weights..."
mkdir -p $CheckpointsDir/syncnet
fast_download \
  "https://huggingface.co/ByteDance/LatentSync/resolve/main/latentsync_syncnet.pt" \
  "$CheckpointsDir/syncnet/latentsync_syncnet.pt"

# Download Face Parse Bisent weights
echo "Downloading Face Parse Bisent weights..."
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $CheckpointsDir/face-parse-bisent/79999_iter.pth

echo "Downloading ResNet18 weights..."
fast_download \
  "https://download.pytorch.org/models/resnet18-5c106cde.pth" \
  "$CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth"

echo "✅ All weights have been downloaded successfully!"