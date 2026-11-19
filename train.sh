#!/bin/bash

# MuseTalk Training Script
# This script combines both training stages for the MuseTalk model
# Usage: sh train.sh [stage1|stage2]
# Example: sh train.sh stage1  # To run stage 1 training
# Example: sh train.sh stage2  # To run stage 2 training

# Check if stage argument is provided
if [ $# -ne 1 ]; then
    echo "Error: Please specify the training stage"
    echo "Usage: ./train.sh [stage1|stage2]"
    exit 1
fi

STAGE=$1

# Validate stage argument
if [ "$STAGE" != "stage1" ] && [ "$STAGE" != "stage2" ]; then
    echo "Error: Invalid stage. Must be either 'stage1' or 'stage2'"
    exit 1
fi

# Launch distributed training using accelerate
# --config_file: Path to the GPU configuration file
# --main_process_port: Port number for the main process, used for distributed training communication
# train.py: Training script
# --config: Path to the training configuration file
echo "Starting $STAGE training..."
accelerate launch --config_file ./configs/training/gpu.yaml \
                  --main_process_port 29502 \
                  train.py --config ./configs/training/$STAGE.yaml

echo "Training completed for $STAGE" 