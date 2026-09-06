"""
Post-procesa un video SOLO con Real-ESRGAN (nitidez + reduccion de ruido de
todo el frame), sin GFPGAN -- para probar si evita el corrimiento de
identidad que GFPGAN produce al regenerar la cara.

Uso:
    python3 scripts/realesrgan_only_postprocess.py <video_in.mp4> <video_out.mp4>
"""
import sys
import os
import cv2
import glob
import subprocess
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

BG_MODEL_PATH = "models/realesrgan/RealESRGAN_x2plus.pth"


def download_bg_model_if_needed():
    if os.path.exists(BG_MODEL_PATH):
        return
    os.makedirs(os.path.dirname(BG_MODEL_PATH), exist_ok=True)
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    subprocess.run(["curl", "-fL", "-o", BG_MODEL_PATH, url], check=True)


def main():
    video_in, video_out = sys.argv[1], sys.argv[2]
    work_dir = os.path.splitext(video_out)[0] + "_esrgan_tmp"
    frames_in = os.path.join(work_dir, "in")
    frames_out = os.path.join(work_dir, "out")
    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    download_bg_model_if_needed()

    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    subprocess.run(
        ["ffmpeg", "-y", "-v", "warning", "-i", video_in, os.path.join(frames_in, "%08d.png")],
        check=True,
    )

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(
        scale=2,
        model_path=BG_MODEL_PATH,
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True,
    )

    frame_files = sorted(glob.glob(os.path.join(frames_in, "*.png")))
    for i, fpath in enumerate(frame_files):
        img = cv2.imread(fpath)
        restored_img, _ = upsampler.enhance(img, outscale=1)
        cv2.imwrite(os.path.join(frames_out, os.path.basename(fpath)), restored_img)
        if i % 20 == 0:
            print(f"esrgan frame {i+1}/{len(frame_files)}")

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
