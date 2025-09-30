# Use the official NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variable to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Add the deadsnakes PPA and install Python 3.11 and other necessary dependencies
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    curl \
    build-essential \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6

# Manually install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Update alternatives to ensure python3 points to python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3 1

# Copy the application code inside the container
WORKDIR /app

# Install Python dependencies (if you have a requirements.txt)
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN pip3 install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmcv==2.0.1" && \
    mim install "mmdet>=3.1.0" && \
    mim install "mmpose>=1.1.0"

ENV FFMPEG_PATH=/app/ffmpeg-7.0.2-amd64-static

COPY . .

# Specify the command to run your application (modify as needed)
CMD ["python3", "app.py"]