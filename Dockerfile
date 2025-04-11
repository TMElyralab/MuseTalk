FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and setup files
COPY requirements.txt ./
COPY setup_env.sh ./

# Create non-root user
RUN useradd -m -s /bin/bash musetalk
RUN chown -R musetalk:musetalk /app

# Switch to non-root user
USER musetalk

# Create virtual environment and install dependencies
RUN python3 -m venv musetalk_env && \
    . musetalk_env/bin/activate && \
    pip install --upgrade pip && \
    bash setup_env.sh

# Set default command
CMD ["/bin/bash"]
