#!/bin/bash

# Start MuseTalk WebSocket Server
# This script activates the MuseTalk conda environment and starts the server

echo "🎭 Starting MuseTalk WebSocket Server..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found. Please install conda first."
    exit 1
fi

# Check if MuseTalk environment exists
if ! conda env list | grep -q "MuseTalk"; then
    echo "❌ Error: MuseTalk conda environment not found."
    echo "Please create the environment first:"
    echo "conda create -n MuseTalk python=3.10"
    echo "conda activate MuseTalk"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Activate MuseTalk environment and run server
echo "🔧 Activating MuseTalk environment..."
eval "$(conda shell.bash hook)"
conda activate MuseTalk

echo "🚀 Starting server..."
cd src
python server.py