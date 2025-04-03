import os
import cv2
import numpy as np
import torch

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print("please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

    
from musetalk.whisper.audio2feature import Audio2Feature
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet,PositionalEncoding


def load_all_model(
    unet_model_path="./models/musetalk/pytorch_model.bin",
    vae_type="sd-vae-ft-mse",
    unet_config="./models/musetalk/musetalk.json",
    device=None,
):
    vae = VAE(
        model_path = f"./models/{vae_type}/",
    )
    print(f"load unet model from {unet_model_path}")
    unet = UNet(
        unet_config=unet_config,
        model_path=unet_model_path,
        device=device
    )
    pe = PositionalEncoding(d_model=384)
    return vae, unet, pe

def get_file_type(video_path):
    _, ext = os.path.splitext(video_path)

    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        return 'image'
    elif ext.lower() in ['.avi', '.mp4', '.mov', '.flv', '.mkv']:
        return 'video'
    else:
        return 'unsupported'

def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def datagen(
    whisper_chunks,
    vae_encode_latents,
    batch_size=8,
    delay_frame=0,
    device="cuda:0",
):
    whisper_batch, latent_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            whisper_batch = torch.stack(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)
            yield whisper_batch, latent_batch
            whisper_batch, latent_batch  = [], []

    # the last batch may smaller than batch size
    if len(latent_batch) > 0:
        whisper_batch = torch.stack(whisper_batch)
        latent_batch = torch.cat(latent_batch, dim=0)

        yield whisper_batch.to(device), latent_batch.to(device)
