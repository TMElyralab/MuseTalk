@echo off
setlocal

:: Set the checkpoints directory
set CheckpointsDir=models

:: Create necessary directories
mkdir %CheckpointsDir%\musetalk
mkdir %CheckpointsDir%\musetalkV15
mkdir %CheckpointsDir%\syncnet
mkdir %CheckpointsDir%\dwpose
mkdir %CheckpointsDir%\face-parse-bisent
mkdir %CheckpointsDir%\sd-vae-ft-mse
mkdir %CheckpointsDir%\whisper

:: Install required packages
pip install -U "huggingface_hub[hf_xet]"

:: Set HuggingFace endpoint
set HF_ENDPOINT=https://hf-mirror.com

:: Download MuseTalk weights
hf download TMElyralab/MuseTalk --local-dir %CheckpointsDir%

:: Download SD VAE weights
hf download stabilityai/sd-vae-ft-mse --local-dir %CheckpointsDir%\sd-vae --include "config.json" "diffusion_pytorch_model.bin"

:: Download Whisper weights
hf download openai/whisper-tiny --local-dir %CheckpointsDir%\whisper --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

:: Download DWPose weights
hf download yzd-v/DWPose --local-dir %CheckpointsDir%\dwpose --include "dw-ll_ucoco_384.pth"

:: Download SyncNet weights
hf download ByteDance/LatentSync --local-dir %CheckpointsDir%\syncnet --include "latentsync_syncnet.pt"

:: Download face-parse-bisent weights
hf download ManyOtherFunctions/face-parse-bisent --local-dir %CheckpointsDir%\face-parse-bisent --include "79999_iter.pth" "resnet18-5c106cde.pth"

echo All weights have been downloaded successfully!
endlocal 
