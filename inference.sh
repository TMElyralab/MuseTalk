#!/bin/bash

# This script runs inference based on the version and mode specified by the user.
# Usage:
# To run v1.0 inference: sh inference.sh v1.0 [normal|realtime]
# To run v1.5 inference: sh inference.sh v1.5 [normal|realtime]

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <version> <mode>"
    echo "Example: $0 v1.0 normal or $0 v1.5 realtime"
    exit 1
fi

# Get the version and mode from the user input
version=$1
mode=$2

# Validate mode
if [ "$mode" != "normal" ] && [ "$mode" != "realtime" ]; then
    echo "Invalid mode specified. Please use 'normal' or 'realtime'."
    exit 1
fi

# Set config path based on mode
if [ "$mode" = "normal" ]; then
    config_path="./configs/inference/test.yaml"
    result_dir="./results/test"
else
    config_path="./configs/inference/realtime.yaml"
    result_dir="./results/realtime"
fi

# Define the model paths based on the version
if [ "$version" = "v1.0" ]; then
    model_dir="./models/musetalk"
    unet_model_path="$model_dir/pytorch_model.bin"
    unet_config="$model_dir/musetalk.json"
    version_arg="v1"
elif [ "$version" = "v1.5" ]; then
    model_dir="./models/musetalkV15"
    unet_model_path="$model_dir/unet.pth"
    unet_config="$model_dir/musetalk.json"
    version_arg="v15"
else
    echo "Invalid version specified. Please use v1.0 or v1.5."
    exit 1
fi

# Set script name based on mode
if [ "$mode" = "normal" ]; then
    script_name="scripts.inference"
else
    script_name="scripts.realtime_inference"
fi

# Base command arguments
cmd_args="--inference_config $config_path \
    --result_dir $result_dir \
    --unet_model_path $unet_model_path \
    --unet_config $unet_config \
    --version $version_arg"

# Add realtime-specific arguments if in realtime mode
if [ "$mode" = "realtime" ]; then
    cmd_args="$cmd_args \
    --fps 25 \
    --version $version_arg"
fi

# Run inference
python3 -m $script_name $cmd_args
