# conda deactivate
# conda env remove -n musetalk -y
ml cuDNN/8.7.0.84-CUDA-11.8.0
conda create -n musetalk python==3.10 -y
conda activate musetalk
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install huggingface_hub==0.25.2
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 
# conda install -c conda-forge ffmpeg
huggingface-cli download TMElyralab/MuseTalk --local-dir models --exclude "*.git*" "README.md" "docs"
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir models/sd-vae-ft-mse --exclude "*.git*" "README.md" "docs"
huggingface-cli download yzd-v/DWPose --local-dir models/dwpose --exclude "*.git*" "README.md" "docs"

mkdir -p models/whisper
wget -O models/whisper/tiny.pt https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt

# pip install gdown
mkdir -p models/face-parse-bisent
cd models/face-parse-bisent
gdown https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812
wget https://download.pytorch.org/models/resnet18-5c106cde.pth
cd -