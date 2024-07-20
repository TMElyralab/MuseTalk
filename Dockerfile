# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app/MuseTalk

# Copy the current directory contents into the container
COPY . /app/MuseTalk

# Install necessary Python packages
RUN pip install -r requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmcv>=2.0.1" && \
    mim install "mmdet>=3.1.0" && \
    mim install "mmpose>=1.1.0"

# Copy the weight download script into the container
COPY download_weights.sh /app/MuseTalk/download_weights.sh

# Run the weight download script
RUN chmod +x /app/MuseTalk/download_weights.sh && /app/MuseTalk/download_weights.sh

# Entrypoint to pass arguments
ENTRYPOINT ["python", "-m", "scripts.inference"]