"""
Post-procesa un video con look "camara de cine":
1. Desenfoque de fondo por profundidad (Depth Anything V2) -- vos en foco,
   fondo desenfocado segun distancia, como un lente rapido (f2-ish).
2. Grado de color cinematografico via filtros de ffmpeg (curva de contraste
   suave, correccion de balance, vignette leve) -- no un LUT externo de
   origen incierto ni pensado para otro perfil de color.

Uso:
    python3 scripts/cinematic_postprocess.py <video_in.mp4> <video_out.mp4> [blur_strength] [audio_source]

blur_strength: 0-1, cuanto mas alto mas desenfoque de fondo. Default 0.6.
audio_source: opcional, ruta a un audio limpio (ej. el .wav original de la
  voz clonada) para usar en el mux final en vez de extraer el audio de
  video_in. video_in normalmente ya trae el audio recodificado a AAC una vez
  (por el paso de lip-sync) -- si encima se re-extrae y re-codifica de ahi,
  es una SEGUNDA pasada de perdida AAC. Pasar el .wav original evita esa
  doble compresion, se codifica a AAC una sola vez con settings buenos para
  Instagram (estereo, 48kHz, 192kbps).
"""
import sys
import os
import cv2
import glob
import subprocess
import numpy as np
import torch
from transformers import pipeline
import mediapipe as mp

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def get_subject_sharp_mask(img_bgr, segmenter, feather=81):
    """
    Mascara de "esto es el sujeto (cabeza, pelo entero, hombros), queda 100%
    nitido" via segmentacion de persona completa (MediaPipe Selfie
    Segmentation). El face-parsing de MuseTalk (BiSeNet) solo segmenta piel
    de la cara y deja afuera el pelo -- el corte de nitidez pasaba justo por
    el nacimiento del pelo y se veia como si el desenfoque "se comiera" el
    pelo. Esto segmenta la persona entera, pelo incluido.
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = segmenter.process(img_rgb)
    mask = result.segmentation_mask.astype(np.float32)

    feather = feather if feather % 2 == 1 else feather + 1
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    return mask


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def make_disk_kernel(radius):
    size = radius * 2 + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
    kernel /= kernel.sum()
    return kernel


def disk_bokeh_blur(img_bgr, radius, downscale=4):
    """
    Desenfoque circular (bokeh real) en vez de gaussiano -- un lente de
    apertura circular difumina los puntos de luz de fondo en discos, no en
    manchas suaves. Se procesa en baja resolucion y se reescala porque un
    kernel circular grande via filter2D es caro en resolucion completa.
    """
    h, w = img_bgr.shape[:2]
    small_w, small_h = max(1, w // downscale), max(1, h // downscale)
    small = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_radius = max(1, radius // downscale)
    kernel = make_disk_kernel(small_radius)
    blurred_small = cv2.filter2D(small, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)


def main():
    video_in, video_out = sys.argv[1], sys.argv[2]
    blur_strength = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    audio_source = sys.argv[4] if len(sys.argv) > 4 else None

    work_dir = os.path.splitext(video_out)[0] + "_cinematic_tmp"
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_estimator = pipeline("depth-estimation", model=DEPTH_MODEL, device=0 if device == "cuda" else -1)

    mp_selfie = mp.solutions.selfie_segmentation
    segmenter = mp_selfie.SelfieSegmentation(model_selection=0)

    frame_files = sorted(glob.glob(os.path.join(frames_in, "*.png")))
    max_blur_radius = 55  # radio del disco de bokeh
    focus_threshold = 0.72  # por encima de esto (mas cerca) = nitido total
    falloff = 0.28  # rango de profundidad de la transicion nitido->desenfocado
    # (antes 0.12, corte muy duro -- un lente real desenfoca gradual, no de
    # golpe; se combina con smoothstep() para una curva en S, no una rampa
    # lineal, que es como cae el foco realmente)

    for i, fpath in enumerate(frame_files):
        img_bgr = cv2.imread(fpath)
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        from PIL import Image
        pil_img = Image.fromarray(img_rgb)
        depth_result = depth_estimator(pil_img)
        depth_map = np.array(depth_result["depth"].resize((w, h)))
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        # depth_norm: 1.0 = mas cerca (sujeto), 0.0 = mas lejos (fondo).
        # smoothstep en vez de rampa lineal -- la transicion nitido->blur de
        # un lente real acelera en el medio y se aplana en los extremos, no
        # es una rampa recta.
        t = (focus_threshold - depth_norm) / falloff
        blur_weight = smoothstep(t) * blur_strength

        # El sujeto entero (cabeza, pelo, hombros) SIEMPRE queda nitido por
        # su contorno real, no por el umbral de profundidad -- la profundidad
        # sola confundia objetos de fondo a distancia similar con la persona.
        subject_mask = get_subject_sharp_mask(img_bgr, segmenter)
        blur_weight = blur_weight * (1.0 - subject_mask)

        # Bokeh circular (disco), no gaussiano -- las luces de fondo se ven
        # como discos de luz difuminados, que es como desenfoca un lente
        # real de apertura circular.
        blurred = disk_bokeh_blur(img_bgr, max_blur_radius)

        blur_weight_3ch = np.stack([blur_weight] * 3, axis=-1)
        composited = (img_bgr.astype(np.float32) * (1 - blur_weight_3ch) +
                      blurred.astype(np.float32) * blur_weight_3ch).astype(np.uint8)

        cv2.imwrite(os.path.join(frames_out, os.path.basename(fpath)), composited)
        if i % 20 == 0:
            print(f"cinematic frame {i+1}/{len(frame_files)}")

    temp_video = os.path.join(work_dir, "temp.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "warning",
            "-r", str(fps),
            "-f", "image2",
            "-i", os.path.join(frames_out, "%08d.png"),
            "-vcodec", "libx264", "-vf",
            # rgbashift: aberracion cromatica sutil (corrimiento minimo de R
            # y B, detalle real de lente) -- curves/eq: curva de contraste
            # suave + saturacion, look cinematografico generico sin LUT
            # externo -- vignette: bordes levemente mas oscuros -- noise:
            # grano de sensor sutil, sin esto una imagen 100% limpia se ve
            # "plastica"/generada en vez de filmada
            "rgbashift=rh=-2:bh=2,"
            "curves=preset=medium_contrast,eq=saturation=1.15:contrast=1.05,"
            # grano minimo -- la camara de origen ya aporta su propio ruido,
            # sumarle grano sintetico fuerte encima solo lo empeora
            "vignette=PI/5,noise=alls=2:allf=t,format=yuv420p",
            # crf+maxrate/bufsize en vez de crf solo -- Instagram recomprime
            # igual, pero un bitrate mas contenido (~8Mbps tope para 1080x1920)
            # evita que ffmpeg elija un bitrate innecesariamente alto que
            # Instagram va a aplastar de todos modos al re-codificar
            "-crf", "18", "-maxrate", "8M", "-bufsize", "16M",
            "-profile:v", "high", "-level", "4.0",
            temp_video,
        ],
        check=True,
    )

    if audio_source is None:
        audio_source = os.path.join(work_dir, "audio_extracted.aac")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "warning", "-i", video_in, "-vn", "-acodec", "copy", audio_source],
            check=True,
        )

    subprocess.run(
        ["ffmpeg", "-y", "-v", "warning", "-i", temp_video, "-i", audio_source,
         "-c:v", "copy",
         # AAC-LC estereo 48kHz 192kbps -- especificacion recomendada de
         # Instagram para audio, y una sola pasada de compresion con
         # audio_source apuntando al .wav original (no una segunda pasada
         # sobre un AAC ya comprimido)
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
         "-movflags", "+faststart",
         "-shortest", video_out],
        check=True,
    )

    print(f"Listo: {video_out}")


if __name__ == "__main__":
    main()
