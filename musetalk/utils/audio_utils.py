import os, subprocess

def ensure_wav(input_path: str, target_path: str | None = None) -> str:
    """
    Convert any audio (mp3/ogg/m4a/wav/…) to 16kHz mono PCM WAV via ffmpeg.
    Returns path to the converted .wav (original if already correct).
    """
    if not isinstance(input_path, str) or not os.path.exists(input_path):
        return input_path
    base, ext = os.path.splitext(input_path)
    ext = ext.lower()
    
    if target_path is None:
        target_path = base + "_16k.wav"
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", target_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return target_path