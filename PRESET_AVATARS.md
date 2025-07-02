# Setting Up Preset Avatars

This guide explains how to prepare preset avatars for the WebSocket service demo.

## Quick Setup

1. **Prepare avatars** (this will take 5-10 minutes):
   ```bash
   python prepare_preset_avatars.py --copy-to-websocket
   ```

2. **Start the WebSocket service**:
   ```bash
   cd websocket_service
   python -m uvicorn src.server:app --host 0.0.0.0 --port 8000
   ```

3. **Open the demo**: http://localhost:8000

## What the preparation script does

The `prepare_preset_avatars.py` script:

1. **Preprocesses videos** using `realtime_inference.py` with the config in `configs/inference/preset_avatars.yaml`
2. **Generates intermediate data**:
   - `coords.pkl` - Face bounding box coordinates for each frame
   - `latents.pt` - VAE-encoded latent representations 
   - `avator_info.json` - Avatar metadata
   - `full_imgs/` - Preprocessed frames
   - `mask/` - Face masks for blending
3. **Copies the data** to `websocket_service/avatars/` for the demo

## Default avatars

The preset config includes:
- **demo_user**: Uses `data/video/yongen.mp4`
- **user_1**: Uses `data/video/yongen.mp4` 
- **user_2**: Uses `data/video/sun.mp4`

## Adding custom avatars

1. **Add your video** to `data/video/`
2. **Add audio sample** to `data/audio/` 
3. **Update config** `configs/inference/preset_avatars.yaml`:
   ```yaml
   my_avatar:
     preparation: True
     bbox_shift: 5
     video_path: "data/video/my_video.mp4"
     audio_clips:
       audio_0: "data/audio/my_audio.wav"
   ```
4. **Re-run preparation**:
   ```bash
   python prepare_preset_avatars.py --copy-to-websocket
   ```

## Checking avatar status

List prepared avatars and their status:
```bash
python prepare_preset_avatars.py --list
```

## Directory structure

After preparation, the structure will be:
```
websocket_service/avatars/
├── demo_user/
│   ├── avator_info.json
│   ├── coords.pkl
│   ├── latents.pt
│   ├── mask_coords.pkl
│   ├── full_imgs/
│   └── mask/
├── user_1/
│   └── ...
└── user_2/
    └── ...
```

## Troubleshooting

- **"No such file or directory"**: Make sure you're running from the StreamingMuseTalk root directory
- **Avatar not loading**: Check that all required files exist using `--list`
- **Poor quality**: Adjust `bbox_shift` parameter in the config (try values 3-7)
- **WebSocket connection fails**: Ensure the avatar data was copied to `websocket_service/avatars/`