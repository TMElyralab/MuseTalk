#!/bin/bash

# Set the checkpoints directory
CheckpointsDir="models"

# Create necessary directories
mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper

# Install required packages with correct version to avoid conflicts
echo "Installing dependencies..."
pip install -U "huggingface_hub[cli]<1.0,>=0.19.3"
pip install gdown
export HF_HUB_ENABLE_HF_TRANSFER=1
# Set HuggingFace mirror endpoint (optional, comment out if you want to use default)
# export HF_ENDPOINT=https://hf-mirror.com

# Use 'hf' command directly (modern HuggingFace CLI)
HF_CMD="hf"

echo "Using command: $HF_CMD"

# Download MuseTalk V1.0 weights
echo "Downloading MuseTalk V1.0 weights..."
$HF_CMD download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

# Download MuseTalk V1.5 weights (unet.pth)
echo "Downloading MuseTalk V1.5 weights..."
$HF_CMD download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

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
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth

echo "✅ All weights have been downloaded successfully!"