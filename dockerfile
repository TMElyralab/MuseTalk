FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED=1
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch for CPU/MPS (Mac)
RUN pip3 install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2

# Clone MuseTalk
# RUN git clone https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk
RUN git clone https://github.com/rafipatel/MuseTalk.git /app/MuseTalk

WORKDIR /app/MuseTalk

# Install requirements
RUN pip3 install --no-cache-dir -r requirements.txt
# RUN pip3 install -r requirements.txt || true

# Install OpenMMLab packages (CPU version)
RUN pip3 install --no-cache-dir -U openmim && \
    mim install mmengine && \
    pip3 install mmcv==2.0.1 && \
    mim install "mmdet==3.1.0" && \
    mim install "mmpose==1.1.0"

# Download model weights
# RUN python3 -m pip install huggingface_hub && \
#     python3 -c "from huggingface_hub import snapshot_download; \
#     snapshot_download(repo_id='TMElyralab/MuseTalk', local_dir='./models', allow_patterns=['models/musetalkV15/*'])" || true

# Download additional model files


# RUN mkdir -p models/face-parse-bisent && \
#     pip3 install gdown && \
#     gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth && \
#     curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth -o models/face-parse-bisent/resnet18-5c106cde.pth


# # Set working directory
# WORKDIR /app/MuseTalk

# Create entrypoint script
# RUN echo '#!/bin/bash\n\
# python -m scripts.inference \\\n\
#   --inference_config ${INFERENCE_CONFIG:-configs/inference/test.yaml} \\\n\
#   --result_dir ${RESULT_DIR:-results/test} \\\n\
#   --unet_model_path ${UNET_MODEL_PATH:-models/musetalkV15/unet.pth} \\\n\
#   --unet_config ${UNET_CONFIG:-models/musetalkV15/musetalk.json} \\\n\
#   --version ${VERSION:-v15} \\\n\
#   --ffmpeg_path ${FFMPEG_PATH:-/usr/bin/ffmpeg}' > /app/run_inference.sh && \
# chmod +x /app/run_inference.sh

# RUN chmod +x /app/MuseTalk/download_weights.sh
# RUN  /app/MuseTalk/download_weights.sh

# COPY inference.sh /app/MuseTalk/inference.sh

# RUN chmod +x /app/MuseTalk/inference.sh
# CMD ["/app/MuseTalk/inference.sh", "v1.5","normal"]
# CMD ["uvicorn", "fastapi_service:app","--host", "0.0.0.0" ,"--port", "8000"]
CMD ["python3", "-m", "uvicorn", "fastapi_service:app", "--host", "0.0.0.0" ,"--port", "8000"]
# CMD ["/app/run_inference.sh"]
