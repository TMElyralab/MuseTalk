#!/bin/bash

# This script runs inference based on the version specified by the user.
# Usage:
# To run v1.0 inference: sh inference.sh v1.0
# To run v1.5 inference: sh inference.sh v1.5

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 v1.0 or $0 v1.5"
    exit 1
fi

# Get the version from the user input
version=$1
config_path="./configs/inference/test.yaml"

# Define the model paths based on the version
if [ "$version" = "v1.0" ]; then
    model_dir="./models/musetalk"
    unet_model_path="$model_dir/pytorch_model.bin"
elif [ "$version" = "v1.5" ]; then
    model_dir="./models/musetalkV15"
    unet_model_path="$model_dir/unet.pth"
else
    echo "Invalid version specified. Please use v1.0 or v1.5."
    exit 1
fi

# Run inference based on the version
if [ "$version" = "v1.0" ]; then
    python3 -m scripts.inference \
        --inference_config "$config_path" \
        --result_dir "./results/test" \
        --unet_model_path "$unet_model_path"
elif [ "$version" = "v1.5" ]; then
    python3 -m scripts.inference_alpha \
        --inference_config "$config_path" \
        --result_dir "./results/test" \
        --unet_model_path "$unet_model_path"
fi