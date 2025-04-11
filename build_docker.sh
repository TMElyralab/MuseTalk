#!/bin/bash

# Set the image name and tag
IMAGE_NAME="musetalk"
TAG="latest"

echo "Building Docker image for MuseTalk..."
echo "Image: ${IMAGE_NAME}:${TAG}"

# Build the Docker image
docker build -t ${IMAGE_NAME}:${TAG} .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Docker image built successfully!"
    echo "To run the container:"
    echo "docker run --gpus all -it ${IMAGE_NAME}:${TAG}"
else
    echo "Error: Docker build failed!"
    exit 1
fi
