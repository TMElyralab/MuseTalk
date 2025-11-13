import requests
import json

# URL of your FastAPI service
url = "http://localhost:8000/inference"

# Paths to test video/audio files
video_path = "/Users/rafa/MscAi/curify/MuseTalk/data/video/sun.mp4"
audio_path = "/Users/rafa/MscAi/curify/MuseTalk/data/audio/eng.wav"

# Inference parameters (use defaults or specify your values as needed)
inference_params = {
    "ffmpeg_path": "./ffmpeg-4.4-amd64-static/",
    "gpu_id": 0,
    "vae_type": "sd-vae",
    "unet_config": "./models/musetalkV15/musetalk.json",
    "unet_model_path": "./models/musetalkV15/unet.pth",
    "whisper_dir": "./models/whisper",
    "inference_config": "configs/inference/test.yaml",
    "bbox_shift": 0,
    "result_dir": "./results",
    "extra_margin": 10,
    "fps": 25,
    "audio_padding_length_left": 2,
    "audio_padding_length_right": 2,
    "batch_size": 8,
    "output_vid_name": None,
    "use_saved_coord": False,
    "saved_coord": False,
    "use_float16": False,
    "parsing_mode": "jaw",
    "left_cheek_width": 90,
    "right_cheek_width": 90,
    "version": "v15"
}
# Many can be omitted if you use defaults!

files = {
    "video": open(video_path, "rb"),
    "audio": open(audio_path, "rb"),
}
# Must send JSON string for inference_params
data = {
    "inference_params": json.dumps(inference_params)
}

response = requests.post(url, files=files, data=data)

# Save result or print error
if response.ok:
    with open("result.mp4", "wb") as f:
        f.write(response.content)
    print("Success! Video saved as result.mp4")
else:
    print(response.status_code, response.text)
