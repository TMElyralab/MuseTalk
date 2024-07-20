#!/bin/bash

# Define the checkpoints directory
CheckpointsDir="./models"

# Function to download and verify files
download_file() {
  local url=$1
  local output=$2

  wget -O $output $url

  # Verify file size is greater than 0
  if [ ! -s $output ]; then
    echo "Error: File $output is empty or download failed."
    exit 1
  fi
}

# Create the models directory if it does not exist
if [ ! -d "$CheckpointsDir" ]; then
  mkdir -p $CheckpointsDir
  echo "Checkpoint Not Downloaded, start downloading..."
  tic=$(date +%s)

  # Download MuseTalk weights
  mkdir -p $CheckpointsDir
  git clone https://huggingface.co/TMElyralab/MuseTalk $CheckpointsDir

  # Download SD VAE weights
  mkdir -p $CheckpointsDir/sd-vae-ft-mse
  git clone https://huggingface.co/stabilityai/sd-vae-ft-mse $CheckpointsDir/sd-vae-ft-mse

  # Download DWPose weights
  mkdir -p $CheckpointsDir/dwpose
  git clone https://huggingface.co/yzd-v/DWPose $CheckpointsDir/dwpose

  # Download Whisper weights
  mkdir -p $CheckpointsDir/whisper
  download_file "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt" "$CheckpointsDir/whisper/tiny.pt"

  # Download Face Parse Bisent weights
  mkdir -p $CheckpointsDir/face-parse-bisent
  gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $CheckpointsDir/face-parse-bisent/79999_iter.pth

  # Download ResNet weights
  download_file "https://download.pytorch.org/models/resnet18-5c106cde.pth" "$CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth"

  toc=$(date +%s)
  echo "Download completed in $(($toc-$tic)) seconds"

else
  echo "Model already downloaded."
fi
