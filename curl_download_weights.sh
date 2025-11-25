#!/bin/bash

# Set the checkpoints directory
CheckpointsDir="models"

# Create necessary directories
mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper

# Install required packages with correct version to avoid conflicts
echo "Installing dependencies..."
pip install -U "huggingface_hub[cli]<1.0,>=0.19.3"
pip install gdown aria2

# Set HuggingFace mirror endpoint for faster downloads (optional)
# Uncomment one of these if downloads are slow:
# export HF_ENDPOINT=https://hf-mirror.com  # China mirror
# export HF_ENDPOINT=https://hf.co  # Alternative

# Use 'hf' command with resume capability
HF_CMD="hf"

echo "Using command: $HF_CMD"

# Download MuseTalk V1.0 weights (using direct URL for faster download)
echo "Downloading MuseTalk V1.0 weights..."
mkdir -p $CheckpointsDir/musetalk
curl -L --progress-bar --create-dirs -C - \
  -o $CheckpointsDir/musetalk/pytorch_model.bin \
  "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin"
curl -L --progress-bar \
  -o $CheckpointsDir/musetalk/musetalk.json \
  "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json"

# Download MuseTalk V1.5 weights (using direct URL)
echo "Downloading MuseTalk V1.5 weights..."
mkdir -p $CheckpointsDir/musetalkV15
curl -L --progress-bar --create-dirs -C - \
  -o $CheckpointsDir/musetalkV15/unet.pth \
  "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth"
curl -L --progress-bar \
  -o $CheckpointsDir/musetalkV15/musetalk.json \
  "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json"

# Download SD VAE weights
echo "Downloading SD VAE weights..."
$HF_CMD download stabilityai/sd-vae-ft-mse \
  --local-dir $CheckpointsDir/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"

# Download Whisper weights
echo "Downloading Whisper weights..."
$HF_CMD download openai/whisper-tiny \
  --local-dir $CheckpointsDir/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# Download DWPose weights
echo "Downloading DWPose weights..."
$HF_CMD download yzd-v/DWPose \
  --local-dir $CheckpointsDir/dwpose \
  --include "dw-ll_ucoco_384.pth"

# Download SyncNet weights
echo "Downloading SyncNet weights..."
$HF_CMD download ByteDance/LatentSync \
  --local-dir $CheckpointsDir/syncnet \
  --include "latentsync_syncnet.pt"

# Download Face Parse Bisent weights
echo "Downloading Face Parse Bisent weights..."
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $CheckpointsDir/face-parse-bisent/79999_iter.pth

echo "Downloading ResNet18 weights..."
curl -L --progress-bar \
  -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth \
  https://download.pytorch.org/models/resnet18-5c106cde.pth

echo "✅ All weights have been downloaded successfully!"
