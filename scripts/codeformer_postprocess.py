"""
Post-procesa un video con CodeFormer (restauracion de cara con dial de
fidelidad de identidad) en vez de GFPGAN.

Uso:
    python3 scripts/codeformer_postprocess.py <video_in.mp4> <video_out.mp4> [fidelity]

fidelity: 0.0-1.0, default 0.7. Mas alto = mas fiel a tu cara real,
mas bajo = mas "restaurado"/generico (mismo tipo de riesgo que GFPGAN).
"""
import sys
import os
import cv2
import glob
import subprocess
from codeformer.app import inference_app


def main():
    video_in, video_out = sys.argv[1], sys.argv[2]
    fidelity = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7

    work_dir = os.path.splitext(video_out)[0] + "_codeformer_tmp"
    frames_in = os.path.join(work_dir, "in")
    frames_out = os.path.join(work_dir, "out")
    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    subprocess.run(
        ["ffmpeg", "-y", "-v", "warning", "-i", video_in, os.path.join(frames_in, "%08d.png")],
        check=True,
    )

    frame_files = sorted(glob.glob(os.path.join(frames_in, "*.png")))
    for i, fpath in enumerate(frame_files):
        # inference_app espera una RUTA DE ARCHIVO (str), no un array -- hace
        # cv2.imread(str(image)) el mismo. Y siempre guarda su resultado en
        # el mismo path fijo "output/out.png" sin importar la entrada, asi
        # que hay que copiarlo afuera antes de que el siguiente frame lo pise.
        result_path = inference_app(
            image=fpath,
            background_enhance=True,
            face_upsample=True,
            upscale=1,
            codeformer_fidelity=fidelity,
        )
        restored_bgr = cv2.imread(result_path)
        cv2.imwrite(os.path.join(frames_out, os.path.basename(fpath)), restored_bgr)
        if i % 20 == 0:
            print(f"codeformer frame {i+1}/{len(frame_files)}")

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
