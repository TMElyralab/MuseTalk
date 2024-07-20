# Lip-Syncing Audio to Video using MuseTalk

## Introduction
This project demonstrates how to lip-sync provided audio to a given video using the MuseTalk model.

## Requirements
- Docker
- Docker Compose

## Setup and Usage

### Step 1: Clone the Repository
Clone the MuseTalk repository to your local machine:
```bash
git clone https://github.com/elcaiseri/MuseTalk.git
cd MuseTalk
```


### Step 2: Configure the Inference File
Update the `configs/inference/test.yaml` file with the paths to your video and audio files:
```yaml
task_0:
 video_path: "path/to/13_K.mp4"
 audio_path: "path/to/96_A.wav"
  
task_1:
 video_path: "path/to/13_K.mp4"
 audio_path: "path/to/10_S.wav"
```
Replace `path/to/13_K.mp4` and `path/to/audio.wav` with the actual paths to your video and audio files.

### Step 3: Build the Docker Image
Build the Docker image using Docker Compose:
```bash
sudo docker-compose build
```

### Step 5: Run the Docker Container
Run the Docker container to start the inference process:
```bash
sudo docker-compose up
```

### Step 6: Access the Results
The generated video with synchronized lip-syncing will be saved in the specified output path defined in the `configs/inference/test.yaml` file.

## Troubleshooting
If you encounter any issues during the setup or usage, please ensure the following:
- Docker and Docker Compose are installed correctly on your system.
- The paths in the `configs/inference/test.yaml` file are correctly set to your input video and audio files.

For further assistance, please refer to the project documentation or open an issue in the repository.

## Acknowledgments
This project uses the MuseTalk model developed by TMElyralab. Special thanks to the developers and contributors of the MuseTalk project.