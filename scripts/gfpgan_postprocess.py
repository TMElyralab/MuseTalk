"""
Post-procesa un video de salida de MuseTalk con GFPGAN (restauracion facial),
para probar si mejora los dientes/boca. Extrae frames, corre GFPGAN por frame
y vuelve a armar el video con el audio original.

Uso:
    python3 scripts/gfpgan_postprocess.py <video_in.mp4> <video_out.mp4>
"""
import sys
import os
import cv2
import glob
import subprocess
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

MODEL_PATH = "models/gfpgan/GFPGANv1.4.pth"
BG_MODEL_PATH = "models/realesrgan/RealESRGAN_x2plus.pth"


def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    subprocess.run(["curl", "-fL", "-o", MODEL_PATH, url], check=True)


def download_bg_model_if_needed():
    if os.path.exists(BG_MODEL_PATH):
        return
    os.makedirs(os.path.dirname(BG_MODEL_PATH), exist_ok=True)
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    subprocess.run(["curl", "-fL", "-o", BG_MODEL_PATH, url], check=True)


def build_bg_upsampler():
    # Sharpening + reduccion de ruido de TODO el frame (no solo la cara),
    # via Real-ESRGAN -- GFPGANer llama a esto internamente para el fondo.
    # outscale=1 en el enhance() de mas abajo mantiene la resolucion
    # original (no achicamos ni agrandamos el video, solo lo limpiamos).
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    return RealESRGANer(
        scale=2,
        model_path=BG_MODEL_PATH,
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True,
    )


def main():
    video_in, video_out = sys.argv[1], sys.argv[2]
    work_dir = os.path.splitext(video_out)[0] + "_gfpgan_tmp"
    frames_in = os.path.join(work_dir, "in")
    frames_out = os.path.join(work_dir, "out")
    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    download_model_if_needed()
    download_bg_model_if_needed()

    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    subprocess.run(
        ["ffmpeg", "-y", "-v", "warning", "-i", video_in, os.path.join(frames_in, "%08d.png")],
        check=True,
    )

    restorer = GFPGANer(
        model_path=MODEL_PATH,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=build_bg_upsampler(),
    )

    frame_files = sorted(glob.glob(os.path.join(frames_in, "*.png")))
    for i, fpath in enumerate(frame_files):
        img = cv2.imread(fpath)
        _, _, restored_img = restorer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )
        cv2.imwrite(os.path.join(frames_out, os.path.basename(fpath)), restored_img)
        if i % 20 == 0:
            print(f"gfpgan frame {i+1}/{len(frame_files)}")

    temp_video = os.path.join(work_dir, "temp.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "warning",
            "-r", str(fps),
            "-f", "image2",
            "-i", os.path.join(frames_out, "%08d.png"),
            "-vcodec", "libx264", "-vf", "format=yuv420p", "-crf", "18",
            temp_video,
        ],
        check=True,
    )

    audio_extract = os.path.join(work_dir, "audio.aac")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "warning", "-i", video_in, "-vn", "-acodec", "copy", audio_extract],
        check=True,
    )

    subprocess.run(
        ["ffmpeg", "-y", "-v", "warning", "-i", temp_video, "-i", audio_extract,
         "-c:v", "copy", "-c:a", "aac", video_out],
        check=True,
    )

    print(f"Listo: {video_out}")


if __name__ == "__main__":
    main()
