#!/bin/bash
# filepath: /home/ken/MuseTalk/setup_environment.sh

set -e  # Exit on error

# Check if running under bash
if [ -z "$BASH_VERSION" ]; then
    echo "Error: This script must be run with bash, not sh."
    echo "Please run it as: bash setup_environment.sh"
    exit 1
fi

# Create logs directory
mkdir -p logs

echo "==============================================="
echo "MuseTalk Environment Setup Script"
echo "==============================================="

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv musetalk_env

# Activate virtual environment (use . instead of source for better compatibility)
echo "Activating virtual environment..."
. musetalk_env/bin/activate

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || [ "$python_major" -eq 3 -a "$python_minor" -lt 10 ]; then
    echo "Error: Python version should be >= 3.10. Found $python_version"
    deactivate
    echo "Please upgrade your Python installation."
    exit 1
fi

echo "Python version check passed: $python_version"

# Upgrade pip in the virtual environment
echo "Upgrading pip in virtual environment..."
pip install --upgrade pip

# Check CUDA version if possible
echo "Checking CUDA version..."
if command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    echo "Found CUDA version: $cuda_version (from driver)"
    echo "Note: Recommended CUDA version is 11.7"
else
    echo "Warning: nvidia-smi not found. Cannot verify CUDA version."
    echo "Note: Recommended CUDA version is 11.7"
fi

# Install Python requirements
echo "Installing Python requirements..."
pip install -r requirements.txt 2>&1 | tee logs/pip_install.log
echo "Python requirements installed successfully."

# Install mmlab packages
echo "Installing mmlab packages..."
pip install --no-cache-dir -U openmim 2>&1 | tee logs/mim_install.log
mim install mmengine 2>&1 | tee -a logs/mim_install.log
mim install "mmcv>=2.0.1" 2>&1 | tee -a logs/mim_install.log
mim install "mmdet>=3.1.0" 2>&1 | tee -a logs/mim_install.log
mim install "mmpose>=1.1.0" 2>&1 | tee -a logs/mim_install.log
echo "mmlab packages installed successfully."

# Install huggingface-cli if needed
echo "Installing huggingface_hub CLI..."
pip install -U "huggingface_hub[cli]" 2>&1 | tee logs/hf_install.log

# Check and setup ffmpeg only if needed
echo "Checking for ffmpeg..."
ffmpeg_installed=false

# Check if ffmpeg is in PATH
if command -v ffmpeg &> /dev/null; then
    echo "ffmpeg is already installed in system PATH."
    ffmpeg_installed=true
    system_ffmpeg_path=$(which ffmpeg | xargs dirname)
    echo "System ffmpeg found at: $system_ffmpeg_path"
    echo "export FFMPEG_PATH=$system_ffmpeg_path" >> ~/.bashrc
    export FFMPEG_PATH=$system_ffmpeg_path
fi

# Check if our local ffmpeg directory already exists and contains the binary
if [ -f "ffmpeg-4.4-amd64-static/ffmpeg" ] && [ -f "ffmpeg-4.4-amd64-static/ffprobe" ]; then
    echo "Local ffmpeg installation found."
    ffmpeg_installed=true
    echo "export FFMPEG_PATH=$(pwd)/ffmpeg-4.4-amd64-static" >> ~/.bashrc
    export FFMPEG_PATH=$(pwd)/ffmpeg-4.4-amd64-static
fi

# Download and setup ffmpeg if not found
if [ "$ffmpeg_installed" = false ]; then
    echo "Setting up ffmpeg..."
    mkdir -p ffmpeg-4.4-amd64-static
    wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -O ffmpeg.tar.xz
    tar -xf ffmpeg.tar.xz --strip-components=1 -C ffmpeg-4.4-amd64-static
    rm ffmpeg.tar.xz
    echo "ffmpeg setup complete."
    echo "export FFMPEG_PATH=$(pwd)/ffmpeg-4.4-amd64-static" >> ~/.bashrc
    export FFMPEG_PATH=$(pwd)/ffmpeg-4.4-amd64-static
fi

echo "FFMPEG_PATH set to $FFMPEG_PATH"

# Write environment activation to a separate file for convenience
cat > activate_env.sh << 'EOL'
#!/bin/bash
. $(pwd)/musetalk_env/bin/activate
export FFMPEG_PATH=$(pwd)/ffmpeg-4.4-amd64-static
echo "MuseTalk environment activated. FFMPEG_PATH set to $FFMPEG_PATH"
EOL
chmod +x activate_env.sh

# Create directory structure for models
echo "Creating model directory structure..."
mkdir -p models/musetalk
mkdir -p models/musetalkV15
mkdir -p models/dwpose
mkdir -p models/face-parse-bisent
mkdir -p models/sd-vae-ft-mse
mkdir -p models/whisper

# Download model weights
echo "Downloading model weights (this may take a while)..."
export HF_ENDPOINT=https://hf-mirror.com

echo "Downloading MuseTalk models..."
huggingface-cli download TMElyralab/MuseTalk --local-dir models/ 2>&1 | tee logs/model_download.log

# Download other components
echo "Downloading SD-VAE-FT-MSE..."
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir models/sd-vae-ft-mse --include "config.json" "diffusion_pytorch_model.bin" 2>&1 | tee -a logs/model_download.log

echo "Downloading Whisper tiny model..."
huggingface-cli download openai/whisper-tiny --local-dir models/whisper --include "config.json" "pytorch_model.bin" "preprocessor_config.json" 2>&1 | tee -a logs/model_download.log

echo "Downloading DWPose..."
mkdir -p temp_download
huggingface-cli download yzd-v/DWPose --local-dir temp_download 2>&1 | tee -a logs/model_download.log
cp temp_download/dw-ll_ucoco_384.pth models/dwpose/
rm -rf temp_download

echo "Downloading ResNet18 model for face parsing..."
wget -q https://download.pytorch.org/models/resnet18-5c106cde.pth -O models/face-parse-bisent/resnet18-5c106cde.pth

echo "Downloading Face-Parse-BiSeNet model..."
wget -q https://github.com/zllrunning/face-parsing.PyTorch/raw/master/res/cp/79999_iter.pth -O models/face-parse-bisent/79999_iter.pth 2>&1 | tee -a logs/model_download.log

# Deactivate virtual environment
# deactivate

echo "==============================================="
echo "MuseTalk environment setup completed!"
echo "==============================================="
echo ""
echo "To activate the virtual environment for MuseTalk, run:"
echo "bash activate_env.sh"
echo ""
echo "To run MuseTalk inference:"
echo "bash activate_env.sh"
echo "sh inference.sh v1.5 normal  # For MuseTalk 1.5 (recommended)"
echo "sh inference.sh v1.0 normal  # For MuseTalk 1.0"
echo ""
echo "For real-time inference:"
echo "bash activate_env.sh"
echo "sh inference.sh v1.5 realtime"
echo ""