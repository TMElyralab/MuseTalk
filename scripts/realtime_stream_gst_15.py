import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
import shutil
import threading
import queue
import time
from termcolor import colored # Add this import
import subprocess
import io
import concurrent.futures
import traceback
import logging
from dotenv import load_dotenv # Keep this at the very top!
from PIL import Image
import tempfile
import ffmpeg
import soundfile as sf # Added for sf.write, assumed to be imported
import cpuinfo # For detailed CPU information

print("Script started!")

# --- Load environment variables FIRST to make them available globally ---
load_dotenv()
logging.info("Environment variables loaded from .env file.")

# --- Global Configuration from .env or Hardcoded Fallbacks ---
# These values will now be sourced directly from the .env file if set,
# otherwise they will fall back to the provided default.
# This makes them accessible globally without needing 'args.' prefix.

# GStreamer related paths and parameters
GSTREAMER_LAUNCH_PATH = os.getenv("GSTREAMER_LAUNCH_PATH", "gst-launch-1.0") # Global path for gst-launch
# NEW - Add this sanitization immediately after
if "GSTREAMER_LAUNCH_PATH=" in GSTREAMER_LAUNCH_PATH:
    # Remove the erroneous key prefix if it exists
    GSTREAMER_LAUNCH_PATH = GSTREAMER_LAUNCH_PATH.split("GSTREAMER_LAUNCH_PATH=", 1)[-1]
    logging.warning(f"⚠️  Cleaned malformed GSTREAMER_LAUNCH_PATH value. Now: {GSTREAMER_LAUNCH_PATH}")

# Also add debug logging to see what was actually loaded
logging.info(f"🔍 Final GSTREAMER_LAUNCH_PATH: '{GSTREAMER_LAUNCH_PATH}'")
STREAM_PIPE_PATH = os.getenv("STREAM_PIPE_PATH", "./hot_file.opus")
TARGET_FPS = int(os.getenv("TARGET_FPS", "25")) # Use for general FPS calculations
FRAME_SKIP_THRESHOLD = int(os.getenv("FRAME_SKIP_THRESHOLD", "3")) # Use for consumer queue management
OVERLOAD_MULTIPLIER = float(os.getenv("OVERLOAD_MULTIPLIER", "1.2"))
MAX_FRAMES_TO_SKIP = int(os.getenv("MAX_FRAMES_TO_SKIP", "1"))

# MuseTalk Model Paths (primarily from .env)
MUSE_VERSION = os.getenv("MUSE_VERSION", "v15")
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "./ffmpeg-4.4-amd64-static/")
GPU_ID = int(os.getenv("GPU_ID", "0"))
VAE_TYPE = os.getenv("VAE_TYPE", "sd-vae")
UNET_CONFIG_PATH = os.getenv("UNET_CONFIG", "./models/musetalkV15/musetalk.json")
UNET_MODEL_PATH = os.getenv("UNET_MODEL_PATH", "./models/musetalkV15/unet.pth")
WHISPER_DIR = os.getenv("WHISPER_DIR", "./models/whisper")
RESULT_DIR = os.getenv("RESULT_DIR", './results')

# Avatar/Inference specific parameters (primarily from .env)
EXTRA_MARGIN = int(os.getenv("EXTRA_MARGIN", "10"))
AUDIO_PADDING_LENGTH_LEFT = int(os.getenv("AUDIO_PADDING_LEFT", "2"))
AUDIO_PADDING_LENGTH_RIGHT = int(os.getenv("AUDIO_PADDING_RIGHT", "2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
PARSING_MODE = os.getenv("PARSING_MODE", 'jaw')
LEFT_CHEEK_WIDTH = int(os.getenv("LEFT_CHEEK_WIDTH", "90"))
RIGHT_CHEEK_WIDTH = int(os.getenv("RIGHT_CHEEK_WIDTH", "90"))
AVATAR_CONFIG_PATH = os.getenv("AVATAR_CONFIG_PATH", "configs/inference/realtime.yaml")
AVATAR_ID_TO_USE = os.getenv("AVATAR_ID_TO_USE", "default_avatar_id")


# --- Set up basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True) # Add force=True if Python >= 3.8
print("Logging configured!")

# --- PyTorch Device Setup & System Information Block [CORRECTED] ---
logging.info("--- System & Hardware Information ---")

# PyTorch Version
try:
    torch_version = torch.__version__
    logging.info(colored(f"PyTorch Version: {torch_version}", 'cyan'))
except Exception as e:
    logging.warning(f"Could not determine PyTorch version: {e}")

# CPU Information
try:
    cpu_info = cpuinfo.get_cpu_info()
    cpu_brand = cpu_info.get('brand_raw', 'N/A')
    cpu_hz = cpu_info.get('hz_actual_friendly', 'N/A')
    logging.info(colored(f"CPU:             {cpu_brand} @ {cpu_hz}", 'cyan'))
except Exception as e:
    logging.warning(f"Could not determine CPU info: {e}")


# --- FIX: Define the 'device' object here, before it is used ---
cuda_available = torch.cuda.is_available()
if cuda_available:
    device = torch.device(f"cuda:{GPU_ID}")
else:
    device = torch.device("cpu")
# -------------------------------------------------------------


# CUDA and GPU Information
if cuda_available:
    try:
        # Get general device properties
        properties = torch.cuda.get_device_properties(device)
        gpu_name = properties.name
        total_vram_gb = properties.total_memory / (1024**3)
        
        # Get current memory usage
        free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info(device)
        used_vram_gb = (total_vram_bytes - free_vram_bytes) / (1024**3)
        available_vram_gb = free_vram_bytes / (1024**3)

        # CUDA Version
        cuda_version = torch.version.cuda
        
        # Log formatted, color-coded information
        logging.info(colored(f"CUDA Version:    {cuda_version}", 'green'))
        logging.info(colored(f"GPU:             {gpu_name} (ID: {GPU_ID})", 'green', attrs=['bold']))
        logging.info(colored(f"VRAM:            {used_vram_gb:.2f} GB Used / {available_vram_gb:.2f} GB Available / {total_vram_gb:.2f} GB Total", 'green'))

    except Exception as e:
        logging.warning(colored(f"⚠️ Could not retrieve detailed GPU/CUDA information: {e}", 'yellow'))
else:
    logging.info(colored("GPU:             ❌ CUDA (GPU) not available. Using CPU.", 'red', attrs=['bold']))

# Selected Device
logging.info(f"Selected Device: {device}")
logging.info("---------------------------------------")

def get_optimal_gstreamer_pipeline(gpu_name, width, height, fps, host, port, bitrate=5000):
    """
    Returns the optimal GStreamer pipeline string based on detected GPU.
    
    Args:
        gpu_name (str): Name of the GPU from torch.cuda.get_device_properties()
        width, height, fps: Video parameters
        host, port: Streaming destination
        bitrate: Target bitrate in kbps
    
    Returns:
        str: GStreamer pipeline string optimized for the detected GPU
    """
    
    # Detect RTX 50-series (Blackwell) - these need D3D11 pipeline
    if "RTX 50" in gpu_name or "5090" in gpu_name or "5080" in gpu_name:
        logging.info(f"🎮 Detected RTX 50-series GPU ({gpu_name}). Using D3D11 hardware acceleration.")
        return (
            f"fdsrc fd=0 do-timestamp=true is-live=true ! "
            f"videoparse format=bgr width={width} height={height} framerate={fps}/1 ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "videoconvert ! video/x-raw,format=NV12 ! "
            "d3d11upload ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            # FIXED: Use correct mfh264enc properties
            f"mfh264enc rc-mode=cbr bitrate={bitrate * 1000} low-latency=true max-bitrate={bitrate * 1000} ! "
            "h264parse ! rtph264pay pt=96 config-interval=1 ! "
            f"udpsink host={host} port={port} sync=true async=false"
        )

    
    # RTX 40-series and earlier - use CUDA pipeline
    elif "RTX" in gpu_name or "GTX" in gpu_name or "Tesla" in gpu_name or "Quadro" in gpu_name:
        logging.info(f"🎮 Detected NVIDIA GPU with CUDA support ({gpu_name}). Using CUDA pipeline.")
        return (
            f"fds qrc fd=0 do-timestamp=true is-live=true ! "
            f"videoparse format=bgr width={width} height={height} framerate={fps}/1 ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "cudaupload ! "
            "queue max-size-buffers=1 leaky=downstream ! cudaconvert ! videorate ! "
            "video/x-raw(memory:CUDAMemory),format=NV12 ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            f"nvh264enc preset=p1 tune=ultra-low-latency zerolatency=true rc-mode=cbr bitrate={bitrate} ! "
            "h264parse ! rtph264pay pt=96 config-interval=1 ! "
            f"udpsink host={host} port={port} sync=true async=false"
        )
    
    # Fallback to CPU encoding for unknown/non-NVIDIA GPUs
    else:
        logging.warning(f"⚠️ Unknown GPU type ({gpu_name}). Falling back to CPU-based x264 encoding.")
        return (
            f"fdsrc fd=0 do-timestamp=true is-live=true ! "
            f"videoparse format=bgr width={width} height={height} framerate={fps}/1 ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            "x264enc tune=zerolatency speed-preset=ultrafast bitrate=5000 ! "
            "h264parse ! rtph264pay pt=96 config-interval=1 ! "
            f"udpsink host={host} port={port} sync=true async=false"
        )

# --- Platform-specific imports for enhanced functionality ---
if sys.platform == "win32":
    try:
        import psutil
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        logging.info("INFO: Process priority set to HIGH on Windows (if psutil installed and permitted).")
    except ImportError:
        logging.warning("Warning: psutil not found. Cannot set process priority.")
    except Exception as e:
        logging.warning(f"Warning: Could not set process priority: {e}")


# --- MuseTalk Specific Imports (Ensure these are in your PYTHONPATH) ---
try:
    from transformers import WhisperModel
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.utils import datagen, load_all_model
    from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
    from musetalk.utils.blending import get_image_prepare_material, get_image_blending
    from musetalk.utils.audio_processor import AudioProcessor
except ImportError as e:
    logging.critical(f"Error importing MuseTalk utilities: {e}. Ensure the library is installed and 'musetalk' package is in your PYTHONPATH.", exc_info=True)
    sys.exit(1)

# --- Global Models and Device (initialized in __main__) ---
# These will be initialized once and made globally accessible for use by Avatar class and worker threads.
# They are declared here to be accessible, but will be assigned values in main.
vae = None
unet = None
pe = None
timesteps = None
audio_processor = None
whisper = None
fp = None # FaceParsing instance
weight_dtype = torch.float16 # Default to FP16 as per MuseTalk 1.5 optimizations

# --- Utility Functions ---
def fast_check_ffmpeg():
    """Checks if ffmpeg is accessible from the system's PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

def video2imgs(vid_path, save_path):
    """Extracts frames from a video file and saves them as PNG images."""
    logging.info(f"Extracting frames from {vid_path} to {save_path}...")
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file: {vid_path}")
        return
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(os.path.join(save_path, f"{str(count).zfill(8)}.png"), frame)
        count += 1
    cap.release()
    logging.info(f"Finished extracting {count} frames.")

def osmakedirs(path_list):
    """Creates directories if they don't exist, using exist_ok=True for robustness."""
    for path in path_list:
        os.makedirs(path, exist_ok=True)

def _log_subprocess_output(pipe, logger_func, prefix):
    """Reads output from a pipe line by line and logs it."""
    try:
        for line_bytes in iter(pipe.readline, b''):
            logger_func(f"[{prefix}]: {line_bytes.decode(errors='ignore').rstrip()}")
    except ValueError: # Pipe might close during readline
        pass
    except Exception as e:
        logging.error(f"Error reading from {prefix} pipe: {e}", exc_info=True)
    finally:
        if pipe and not pipe.closed:
            pipe.close()

# --- FFmpeg Audio Reader (User's custom class for flexible audio input) ---
class FFmpegAudioReader:
    """Uses FFmpeg to read an audio source (file path or bytes) and convert it to raw PCM."""
    def __init__(self, audio_source):
        self.audio_source = audio_source
        self.is_file_path = isinstance(audio_source, str)

    def read_full_audio(self):
        """Reads the entire audio source and converts it to PCM s16le, 48kHz, Stereo."""
        logging.info(f"Reading and converting audio from {'file' if self.is_file_path else 'memory'}...")
        target_sr, target_ac, target_format = 48000, 2, "s16le" # Target format for GStreamer audio pipeline
        input_data = None
        
        ffmpeg_input_args = {}
        if not self.is_file_path:
            input_filename = 'pipe:0' # Read from stdin if bytes are provided
            input_data = self.audio_source
        else:
            input_filename = self.audio_source

        try:
            out, err = (
                ffmpeg
                .input(input_filename, **ffmpeg_input_args)
                .output('pipe:', format=target_format, ac=target_ac, ar=target_sr)
                .run(capture_stdout=True, capture_stderr=True, input=input_data, quiet=True)
            )
            if err:
                logging.debug(f"FFmpeg stderr: {err.decode(errors='ignore')}")
        except ffmpeg.Error as e:
            logging.error(f"❌ FFmpeg error during audio conversion: {e.stderr.decode(errors='ignore') if e.stderr else 'Unknown FFmpeg error'}")
            return None
        except Exception as e:
            logging.error(f"❌ Unexpected error during FFmpeg execution for audio: {e}", exc_info=True)
            return None

        if not out:
            logging.error("❌ Failed to read audio: FFmpeg produced no PCM data!")
            return None

        audio_data = np.frombuffer(out, dtype=np.int16).reshape(-1, target_ac)
        logging.info(f"✅ Read and converted audio: {len(audio_data)} samples at {target_sr}Hz, {target_ac}ch.")
        return audio_data

# --- GStreamer Classes (User's custom classes for real-time streaming) ---
class GStreamerPipeline:
    """Manages the GStreamer video pipeline subprocess."""
    def __init__(self, width=1280, height=720, fps=TARGET_FPS, host="127.0.0.1", port=5000):
        self.width, self.height, self.fps, self.host, self.port = width, height, fps, host, port
        self.process = None
        self.stdout_thread = None
        self.stderr_thread = None
        self.is_running = False

        # --- DYNAMIC PIPELINE SELECTION ---
        # Get GPU name from global detection
        gpu_name = ""
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(device)
            except:
                gpu_name = "Unknown"
        
        # Generate optimal pipeline for detected hardware
        pipeline_str = get_optimal_gstreamer_pipeline(
            gpu_name=gpu_name,
            width=self.width,
            height=self.height,
            fps=self.fps,
            host=self.host,
            port=self.port,
            bitrate=5000
        )
        
        logging.info(f"Attempting to start GStreamer video pipeline ({self.width}x{self.height}@{self.fps}fps) to {self.host}:{self.port}...")
        env_vars = os.environ.copy()



        try:
            # NEW - FIXED
            command_to_run = f'"{GSTREAMER_LAUNCH_PATH}" -v {pipeline_str}'

            logging.info(f"DEBUG: GStreamer VIDEO command (Popen string): {command_to_run}")

            self.process = subprocess.Popen(
                command_to_run,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                bufsize=0,
                env=env_vars
            )
            
            # Check if process actually started (pid is assigned)
            if self.process.pid:
                logging.info(colored(f"✅ GStreamer video process launched (PID: {self.process.pid}).", 'green', attrs=['bold']))
                self.is_running = True
            else:
                logging.error(colored(f"❌ GStreamer video process failed to launch, PID not assigned.", 'red', attrs=['bold']))
                self.process = None # Ensure it's None if not truly launched
                self.is_running = False

            # Start threads to read stdout and stderr asynchronously *only if process launched*
            if self.is_running:
                self.stdout_thread = threading.Thread(
                    target=_log_subprocess_output,
                    args=(self.process.stdout, logging.info, "GST_VIDEO_STDOUT"),
                    daemon=True
                )
                self.stderr_thread = threading.Thread(
                    target=_log_subprocess_output,
                    args=(self.process.stderr, logging.error, "GST_VIDEO_STDERR"),
                    daemon=True
                )
                self.stdout_thread.start()
                self.stderr_thread.start()

        except Exception as e:
            logging.error(colored(f"❌ Failed to start GStreamer video pipeline: {e}", 'red', attrs=['bold']), exc_info=True)
            self.process = None
            self.is_running = False

    def send_frame(self, frame):
        """Sends a NumPy array frame to the GStreamer pipeline's stdin."""
        if not self.process or not self.is_running or self.process.stdin.closed: # Check is_running here too
            logging.debug("GStreamer video pipeline not running or stdin closed. Cannot send frame.")
            return False
        try:
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
            logging.debug(colored("Sent video frame to GStreamer.", 'cyan')) # Log successful send
            return True
        except (BrokenPipeError, OSError):
            logging.error(colored("❌ GStreamer video pipeline: Broken pipe. The process may have crashed.", 'red', attrs=['bold']))
            self.stop()
            return False
        except Exception as e:
            logging.error(colored(f"❌ Error pushing video frame: {e}", 'red', attrs=['bold']), exc_info=True)
            return False

    def stop(self):
        """Stops the GStreamer video subprocess gracefully."""
        if self.process:
            logging.info(f"Stopping GStreamer video pipeline (PID: {self.process.pid})...")
            proc_to_stop, self.process = self.process, None
            if proc_to_stop.stdin and not proc_to_stop.stdin.closed:
                try: proc_to_stop.stdin.close()
                except Exception: pass
            proc_to_stop.terminate()
            try:
                proc_to_stop.wait(timeout=3)
                logging.info(colored(f"✅ GStreamer video process terminated.", 'green'))
            except subprocess.TimeoutExpired:
                logging.warning(colored(f"⚠️ GStreamer video process did not terminate gracefully, killing...", 'yellow'))
                proc_to_stop.kill()

        if self.stdout_thread and self.stdout_thread.is_alive():
            self.stdout_thread.join(timeout=1)
        if self.stderr_thread and self.stderr_thread.is_alive():
            self.stderr_thread.join(timeout=1)
        logging.info("GStreamer video pipeline stop complete.")


class GStreamerAudio:
    """Manages the GStreamer audio pipeline subprocess."""
    def __init__(self, host="127.0.0.1", port=5001, sample_rate=48000, channels=2):
        self.host, self.port, self.sample_rate, self.channels = host, port, sample_rate, channels
        self.process = None
        self.stdout_thread = None
        self.stderr_thread = None
        self.is_running = False # New flag

        pipeline_str = (
            f"fdsrc fd=0 do-timestamp=true is-live=true ! "
            "queue max-size-buffers=2 leaky=downstream ! "
            f"audio/x-raw,format=S16LE,channels={self.channels},rate={self.sample_rate},layout=interleaved ! "
            "audioconvert ! audioresample ! "
            "opusenc bitrate=64000 ! rtpopuspay pt=97 ! "
            f"udpsink host={self.host} port={self.port} sync=true"
        )
        logging.info(f"Attempting to start GStreamer audio pipeline ({self.sample_rate}Hz, {self.channels}ch) to {self.host}:{self.port}...")

        env_vars = os.environ.copy()
        env_vars['GST_DEBUG'] = '3'
        
        # In GStreamerAudio.__init__(), around line 378:
        try:
            command_to_run = f'"{GSTREAMER_LAUNCH_PATH}" -v {pipeline_str}'
            logging.debug(f"DEBUG: GStreamer AUDIO command (Popen string): {command_to_run}")
            
            self.process = subprocess.Popen(
                command_to_run,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                bufsize=0,
                env=env_vars
    )

            
            if self.process.pid:
                logging.info(colored(f"✅ GStreamer audio process launched (PID: {self.process.pid}).", 'green', attrs=['bold']))
                self.is_running = True
            else:
                logging.error(colored(f"❌ GStreamer audio process failed to launch, PID not assigned.", 'red', attrs=['bold']))
                self.process = None
                self.is_running = False

            if self.is_running:
                self.stdout_thread = threading.Thread(
                    target=_log_subprocess_output,
                    args=(self.process.stdout, logging.info, "GST_AUDIO_STDOUT"),
                    daemon=True
                )
                self.stderr_thread = threading.Thread(
                    target=_log_subprocess_output,
                    args=(self.process.stderr, logging.error, "GST_AUDIO_STDERR"),
                    daemon=True
                )
                self.stdout_thread.start()
                self.stderr_thread.start()

        except Exception as e:
            logging.error(colored(f"❌ Failed to start GStreamer audio pipeline: {e}", 'red', attrs=['bold']), exc_info=True)
            self.process = None
            self.is_running = False

    def send_audio(self, audio_data_pcm):
        """Sends raw PCM audio data to the GStreamer pipeline's stdin."""
        if not self.process or not self.is_running or self.process.stdin.closed: # Check is_running here too
            logging.debug("GStreamer audio pipeline not running or stdin closed. Cannot send audio.")
            return False
        try:
            self.process.stdin.write(audio_data_pcm.tobytes())
            self.process.stdin.flush()
            logging.debug(colored("Sent audio chunk to GStreamer.", 'cyan')) # Log successful send
            return True
        except (BrokenPipeError, OSError):
            logging.error(colored("❌ GStreamer audio pipeline: Broken pipe.", 'red', attrs=['bold']))
            self.stop()
            return False
        except Exception as e:
            logging.error(colored(f"❌ Error pushing audio chunk: {e}", 'red', attrs=['bold']), exc_info=True)
            return False

    def stop(self):
        """Stops the GStreamer audio subprocess gracefully."""
        if self.process:
            logging.info(f"Stopping GStreamer audio pipeline (PID: {self.process.pid})...")
            proc_to_stop, self.process = self.process, None
            if proc_to_stop.stdin and not proc_to_stop.stdin.closed:
                try: proc_to_stop.stdin.close()
                except Exception: pass
            proc_to_stop.terminate()
            try:
                proc_to_stop.wait(timeout=3)
                logging.info(colored("✅ GStreamer audio process terminated.", 'green'))
            except subprocess.TimeoutExpired:
                logging.warning(colored("⚠️ GStreamer audio process did not terminate gracefully, killing...", 'yellow'))
                proc_to_stop.kill()

        if self.stdout_thread and self.stdout_thread.is_alive():
            self.stdout_thread.join(timeout=1)
        if self.stderr_thread and self.stderr_thread.is_alive():
            self.stderr_thread.join(timeout=1)
        logging.info("GStreamer audio pipeline stop complete.")


# --- Avatar Class (Unified logic from both original and user's script) ---
@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation, version_str, extra_margin=0, parsing_mode='jaw'):
        logging.info(f"Initializing Avatar: {avatar_id}")
        self.avatar_id = str(avatar_id)
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.batch_size = batch_size
        self.preparation = preparation
        self.version_str = version_str
        self.extra_margin = extra_margin
        self.parsing_mode = parsing_mode

        # Define paths (adapted for v15 structure based on version_str)
        if self.version_str == "v15":
            self.avatar_base_path = os.path.join(RESULT_DIR, self.version_str, "avatars", self.avatar_id)
        else: # v1
            self.avatar_base_path = os.path.join(RESULT_DIR, "avatars", self.avatar_id)
            
        self.full_imgs_path = os.path.join(self.avatar_base_path, "full_imgs")
        self.mask_out_path = os.path.join(self.avatar_base_path, "masks")
        self.coords_path = os.path.join(self.avatar_base_path, "coords.pkl")
        self.latents_out_path = os.path.join(self.avatar_base_path, "latents.pt")
        self.mask_coords_path = os.path.join(self.avatar_base_path, "mask_coords.pkl")
        self.avatar_info_path = os.path.join(self.avatar_base_path, "avatar_info.json")

        self.input_latent_list_cycle = []
        self.coord_list_cycle = []
        self.frame_list_cycle = []
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []
        
        self.idx = 0
        
        self.init_avatar_data()
        # --- NEW OPTIMIZATION: PRE-PROCESSING STEP ---
        logging.info("Pre-processing avatar data for optimized performance...")
        
        # Pre-convert frames from BGR NumPy arrays to RGB PIL Images
        self.body_pil_cycle = [
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for frame in self.frame_list_cycle
        ]
        
        # Pre-convert masks to grayscale PIL Images
        self.mask_pil_cycle = [
            Image.fromarray(mask).convert("L")
            for mask in self.mask_list_cycle
        ]
        
        logging.info("✅ Avatar data pre-processing complete.")
        logging.info(f"✅ Avatar '{self.avatar_id}' initialized with {len(self.frame_list_cycle)} reference frames.")

    def init_avatar_data(self):
        """Initializes or reloads avatar data based on 'preparation' flag."""
        if self.preparation:
            if os.path.exists(self.avatar_base_path):
                response = input(f"Avatar '{self.avatar_id}' data exists. Re-create all material? (y/n): ").strip().lower()
                if response == "y":
                    logging.info(f"User chose to re-create. Removing: {self.avatar_base_path}")
                    shutil.rmtree(self.avatar_base_path)
                    self._prepare_material_core()
                else:
                    logging.info("Loading existing data as per user request.")
                    self._reload_prepared_data()
            else:
                self._prepare_material_core()
        else:
            logging.info("Preparation=False. Loading existing prepared data...")
            self._reload_prepared_data()

    def _reload_prepared_data(self):
        """Loads pre-processed avatar data from disk."""
        logging.info(f"Reloading prepared data from: {self.avatar_base_path}")
        try:
            loaded_latents = torch.load(self.latents_out_path, map_location='cpu')
            self.input_latent_list_cycle = list(loaded_latents) if isinstance(loaded_latents, torch.Tensor) else loaded_latents
            with open(self.coords_path, 'rb') as f: self.coord_list_cycle = pickle.load(f)
            with open(self.mask_coords_path, 'rb') as f: self.mask_coords_list_cycle = pickle.load(f)

            num_items = len(self.coord_list_cycle)
            if num_items == 0: raise ValueError("Loaded coordinate data is empty.")
            
            frame_files = [os.path.join(self.full_imgs_path, f"{str(i).zfill(8)}.png") for i in range(num_items)]
            self.frame_list_cycle = read_imgs(frame_files)
            mask_files = [os.path.join(self.mask_out_path, f"{str(i).zfill(8)}.png") for i in range(num_items)]
            self.mask_list_cycle = read_imgs(mask_files)
            
            data_map = {
                "Latents": self.input_latent_list_cycle, "Coords": self.coord_list_cycle, 
                "Frames": self.frame_list_cycle, "Masks": self.mask_list_cycle, "MaskCoords": self.mask_coords_list_cycle
            }
            if not all(len(lst) == num_items for lst in data_map.values()):
                lengths = {name: len(lst) for name, lst in data_map.items()}
                raise ValueError(f"Data lists have mismatched lengths after loading: {lengths}")

            with open(self.avatar_info_path, "r") as f:
                avatar_info_loaded = json.load(f)
            if avatar_info_loaded.get('bbox_shift') != self.bbox_shift or \
               avatar_info_loaded.get('version') != self.version_str or \
               avatar_info_loaded.get('extra_margin') != self.extra_margin or \
               avatar_info_loaded.get('parsing_mode') != self.parsing_mode:
                
                logging.warning("Avatar config (bbox_shift, version, extra_margin, or parsing_mode) has changed.")
                response = input(f"Config change detected. Re-create avatar materials? (y/n) (Old: {avatar_info_loaded}, New: {{'bbox_shift': {self.bbox_shift}, 'version': '{self.version_str}', 'extra_margin': {self.extra_margin}, 'parsing_mode': '{self.parsing_mode}'}}): ").strip().lower()
                if response == "y":
                    logging.info("User chose to re-create due to config change.")
                    shutil.rmtree(self.avatar_base_path)
                    self._prepare_material_core()
                else:
                    logging.info("Continuing with old avatar data despite config change. This might lead to unexpected results.")

        except FileNotFoundError:
            logging.critical(f"❌ Prepared data not found for avatar '{self.avatar_id}'. You must run with preparation=True first.", exc_info=True)
            raise SystemExit(f"Exiting: Prepared data not found for {self.avatar_id}.")
        except Exception as e:
            logging.critical(f"❌ Error reloading prepared data for avatar '{self.avatar_id}'. You may need to run with preparation=True.", exc_info=True)
            raise SystemExit(f"Exiting: Failed to reload data for {self.avatar_id}.")

    @torch.no_grad()
    def _prepare_material_core(self):
        logging.info(f"--- Preparing new material for avatar: {self.avatar_id} ---")
        osmakedirs([self.avatar_base_path, self.full_imgs_path, self.mask_out_path])

        # Store avatar info (unchanged)
        avatar_info_data = {
            "avatar_id": self.avatar_id,
            "video_path": self.video_path,
            "bbox_shift": self.bbox_shift,
            "version": self.version_str,
            "extra_margin": self.extra_margin,
            "parsing_mode": self.parsing_mode
        }

        with open(self.avatar_info_path, "w") as f:
            json.dump(avatar_info_data, f)

        # 1. Extract frames from video or copy from image folder
        if os.path.isfile(self.video_path):
            logging.debug(f"DEBUG: Starting video frame extraction from {self.video_path} to {self.full_imgs_path}...")
            video2imgs(self.video_path, self.full_imgs_path)
        elif os.path.isdir(self.video_path):
            logging.debug(f"DEBUG: Starting image frame copying from {self.video_path} to {self.full_imgs_path}...")
            source_files = sorted([f for f in os.listdir(self.video_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for i, filename in enumerate(tqdm(source_files, desc="Copying frames")):
                shutil.copy(os.path.join(self.video_path, filename), os.path.join(self.full_imgs_path, f"{i:08d}.png"))
        else:
            raise FileNotFoundError(f"video_path '{self.video_path}' is not a valid file or directory.")

        logging.debug("DEBUG: Frame extraction/copy completed. Proceeding to landmark extraction.")

        # 2. Get landmarks and filter out invalid frames
        source_images = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.png')))
        logging.debug(f"DEBUG: Starting face landmark and bbox extraction using {len(source_images)} images...")
        
        initial_coords, initial_frames = [], []
        try:
            # Call the original get_landmark_and_bbox
            initial_coords, initial_frames = get_landmark_and_bbox(source_images, self.bbox_shift)
            
            # Count valid/invalid frames
            valid_frame_count = sum(1 for bbox in initial_coords if bbox != (0.0, 0.0, 0.0, 0.0))
            logging.info(f"DEBUG: get_landmark_and_bbox returned {len(initial_coords)} frames.")
            logging.info(f"DEBUG: Detected {valid_frame_count} valid frames (non-placeholder bbox).")
            
            if valid_frame_count == 0:
                logging.error("CRITICAL: No valid bounding boxes were detected in any frame. Check video content, bbox_shift, or face detection model.")
                # Save first few problematic frames for visual inspection
                debug_output_dir = os.path.join(self.avatar_base_path, "debug_invalid_frames")
                os.makedirs(debug_output_dir, exist_ok=True)
                saved_debug_frames = 0
                for i, (bbox, frame) in enumerate(zip(initial_coords, initial_frames)):
                    if bbox == (0.0, 0.0, 0.0, 0.0): # This is the placeholder for no face detected
                        if saved_debug_frames < 20: # Save up to 20 debug frames
                            debug_frame_path = os.path.join(debug_output_dir, f"invalid_frame_{i:08d}.png")
                            cv2.imwrite(debug_frame_path, frame)
                            logging.debug(f"Saved debug image for invalid frame {i} to {debug_frame_path}")
                            saved_debug_frames += 1
                        else:
                            break # Stop saving debug frames after 20
                
                raise RuntimeError("No valid frames survived the preparation process.")

        except Exception as e:
            logging.critical(f"❌ Error during get_landmark_and_bbox: {e}", exc_info=True)
            raise # Re-raise to ensure the main error propagates

        logging.debug("DEBUG: Face landmark and bbox extraction completed. Starting VAE encoding and mask generation.")

        # 3. Process valid frames: VAE encoding and mask generation
        valid_latents, valid_coords, valid_frames, valid_masks, valid_mask_coords = [], [], [], [], []
        logging.debug("DEBUG: Entering VAE Encoding & Masking loop.")

        for i, (bbox, frame) in enumerate(tqdm(zip(initial_coords, initial_frames), total=len(initial_coords), desc="VAE Encoding & Masking")):
            if bbox == (0.0, 0.0, 0.0, 0.0):
                continue

            # --- THE FIX: Do NOT convert bbox to integers. Keep them as floats. ---
            x1, y1, x2, y2 = bbox

            # For V15, the enlarged bbox is used for BOTH the parser and the VAE.
            if self.version_str == "v15":
                y2_adjusted = y2 + self.extra_margin
                y2_adjusted = min(y2_adjusted, frame.shape[0])
                final_bbox = [x1, y1, x2, y2_adjusted]
            else:
                final_bbox = [x1, y1, x2, y2]

            # --- Step 1: Face Parsing with original float precision ---
            if self.version_str == "v15":
                mode = self.parsing_mode
            else:
                mode = "raw"

            # This call now uses the original float coordinates and should succeed.
            parsing_result = get_image_prepare_material(frame, final_bbox, fp=fp, mode=mode)

            # The robustness check remains as a safeguard for genuinely bad frames.
            if parsing_result is None:
                logging.warning(f"DIAGNOSTIC: Frame {i} is a genuinely problematic frame and was skipped.")
                continue

            mask, mask_crop_box = parsing_result

            # --- Step 2: VAE Encoding ---
            # Cropping with floats is fine; array slicing will implicitly convert to integers.
            vae_x1, vae_y1, vae_x2, vae_y2 = final_bbox
            crop_frame = frame[int(vae_y1):int(vae_y2), int(vae_x1):int(vae_x2)]

            if crop_frame.size == 0:
                logging.warning(f"Skipping frame {i} as VAE cropped frame is empty. Bbox: {final_bbox}")
                continue

            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)

            # --- Step 3: Append all data from the successful frame ---
            valid_latents.append(latents)
            valid_coords.append(final_bbox)
            valid_frames.append(frame)
            valid_masks.append(mask)
            valid_mask_coords.append(mask_crop_box)

        logging.debug("DEBUG: VAE Encoding & Masking completed.")

        if not valid_frames:
            logging.error("All frames failed the parsing step. This could indicate a problem with the source video or a deeper issue with the face parsing model's environment/dependencies.")
            raise RuntimeError("No valid frames survived the preparation process.")



        # 4. Create looping cycle (forward and reverse) and save all data
        self.frame_list_cycle = valid_frames + valid_frames[::-1]
        self.coord_list_cycle = valid_coords + valid_coords[::-1]
        self.input_latent_list_cycle = valid_latents + valid_latents[::-1]
        self.mask_list_cycle = valid_masks + valid_masks[::-1]
        self.mask_coords_list_cycle = valid_mask_coords + valid_mask_coords[::-1]

        logging.debug("DEBUG: Saving final cycle data to disk.")
        shutil.rmtree(self.full_imgs_path, ignore_errors=True)
        os.makedirs(self.full_imgs_path)
        shutil.rmtree(self.mask_out_path, ignore_errors=True)
        os.makedirs(self.mask_out_path)

        for i, (frame, mask) in enumerate(tqdm(zip(self.frame_list_cycle, self.mask_list_cycle), total=len(self.frame_list_cycle), desc="Saving final cycle data")):
            cv2.imwrite(os.path.join(self.full_imgs_path, f"{i:08d}.png"), frame)
            cv2.imwrite(os.path.join(self.mask_out_path, f"{i:08d}.png"), mask)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)
        torch.save(torch.stack(self.input_latent_list_cycle), self.latents_out_path)

        logging.info(f"--- Material prep complete. Final cycle length: {len(self.frame_list_cycle)} frames. ---")
        logging.debug("DEBUG: Avatar material preparation function finished.")
    
    @torch.no_grad()
    def inference(self, audio_source, target_fps):
        """
        Main inference loop for real-time streaming.
        Processes audio, generates frames, and sends them via GStreamer.
        """
        run_id = f"stream_{int(time.time())}"
        logging.info(f"🎬 Starting inference run ID: {run_id}")

        vae_to_blend_queue = queue.Queue(maxsize=self.batch_size * 2)
        gst_video_pipeline, gst_audio_pipeline, frame_processor_thread = None, None, None
        start_time = time.time()
        
        try:
            # 1. Setup GStreamer pipelines
            if self.frame_list_cycle:
                h, w, _ = self.frame_list_cycle[0].shape
                gst_video_pipeline = GStreamerPipeline(width=w, height=h, fps=target_fps)
            else:
                logging.warning("Avatar reference frames not loaded, using default GStreamer resolution (1280x720).")
                gst_video_pipeline = GStreamerPipeline(width=1280, height=720, fps=target_fps)

            gst_audio_pipeline = GStreamerAudio(sample_rate=48000)
            if not gst_video_pipeline.is_running or not gst_audio_pipeline.is_running: # CHECK is_running flag
                raise RuntimeError("GStreamer pipeline(s) failed to initialize. Check GStreamer installation and plugins.")
	    
	    # --- NEW: PIPELINE WARM-UP / PRE-ROLL ---
            # This is the critical fix for the initial scrambled/sped-up quarter-second.
            # We send a few silent, black frames to stabilize the entire pipeline
            # (Python -> OS buffers -> GStreamer) before the real content starts.
            logging.info("Warming up streaming pipeline with pre-roll frames...")
            frame_duration = 1.0 / target_fps
            
            # Create a single black frame to reuse.
            black_frame = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Pre-roll for a quarter-second to stabilize the connection.
            num_preroll_frames = int(target_fps / 4) 

            for _ in range(num_preroll_frames):
                if not gst_video_pipeline.send_frame(black_frame):
                    raise RuntimeError("GStreamer video pipe broke during pre-roll.")
                # We use a simple sleep here for the pre-roll, as precision is less critical
                # than just giving the pipeline time to initialize.
                time.sleep(frame_duration)
            logging.info("Pre-roll complete. Starting main content stream.")
            # --- END NEW SECTION ---

            # 2. Process audio input
            audio_reader = FFmpegAudioReader(audio_source)
            full_audio_pcm = audio_reader.read_full_audio()
            if full_audio_pcm is None or full_audio_pcm.size == 0:
                raise ValueError("PCM audio is empty after FFmpeg conversion. Cannot proceed with inference.")
            
            feature_extraction_path = audio_source if isinstance(audio_source, str) else None
            temp_file_handle = None
            if not feature_extraction_path:
                temp_file_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                # import soundfile as sf # Already imported globally now
                sf.write(temp_file_handle.name, full_audio_pcm, gst_audio_pipeline.sample_rate)
                feature_extraction_path = temp_file_handle.name
                temp_file_handle.close()

            whisper_input_features, librosa_length = audio_processor.get_audio_feature(feature_extraction_path, weight_dtype=weight_dtype)
            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features,
                device,
                weight_dtype,
                whisper,
                librosa_length,
                fps=target_fps,
                audio_padding_length_left=AUDIO_PADDING_LENGTH_LEFT,
                audio_padding_length_right=AUDIO_PADDING_LENGTH_RIGHT,
            )
            
            if temp_file_handle:
                os.unlink(feature_extraction_path)

            num_frames_to_generate = len(whisper_chunks)
            if num_frames_to_generate == 0:
                logging.warning("No frames to generate based on audio features. Skipping inference.")
                return
            
            num_vae_batches = (num_frames_to_generate + self.batch_size - 1) // self.batch_size
            logging.info(f"Audio processed. Planning {num_frames_to_generate} frames in {num_vae_batches} VAE/UNet batches.")

            # 3. Start the consumer thread
            self.idx = 0
            frame_processor_thread = threading.Thread(
                target=self.process_and_send_frames,
                args=(vae_to_blend_queue, gst_video_pipeline, gst_audio_pipeline, num_frames_to_generate, FRAME_SKIP_THRESHOLD),
                daemon=True, name=f"FrameProcessor_{run_id}"
            )
            frame_processor_thread.start()

            # 4. Main Generation Loop (Producer: VAE/UNet inference)
            data_gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
            total_audio_samples = len(full_audio_pcm)
            audio_samples_sent = 0

            vae_pbar = tqdm(data_gen, total=num_vae_batches, desc=f"VAE/UNET [{run_id}]", unit="batch")
            for i, batch_data in enumerate(vae_pbar):
                current_batch_start_time = time.perf_counter() # Mark start of current batch processing

                if not frame_processor_thread.is_alive():
                    logging.error(f"Frame processor thread died unexpectedly. Halting generation.")
                    break
                if not batch_data or len(batch_data) != 2:
                    logging.warning(f"Skipping malformed batch {i}.")
                    continue
                
                whisper_batch, latent_batch = batch_data
                num_frames_in_batch = len(latent_batch)

                # --- Core AI Inference (THIS WAS THE MISSING PART) ---
                audio_feature = pe(whisper_batch.to(device, dtype=weight_dtype))
                latent_input = latent_batch.to(device, dtype=unet.model.dtype)
                
                pred_latents = unet.model(latent_input, timesteps, encoder_hidden_states=audio_feature).sample
                pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
                recon_frames = vae.decode_latents(pred_latents)

                if recon_frames is None or len(recon_frames) == 0:
                    logging.warning(f"Skipping empty VAE output batch {i}.")
                    continue
                
                # --- FPS Calculation and Display ---
                current_batch_end_time = time.perf_counter()
                batch_processing_time = current_batch_end_time - current_batch_start_time
                
                if batch_processing_time > 0:
                    current_fps = num_frames_in_batch / batch_processing_time
                else:
                    current_fps = 0.0

                fps_color = 'green' if current_fps >= target_fps * 0.9 else ('yellow' if current_fps >= target_fps * 0.5 else 'red')
                fps_string = colored(f"{current_fps:.2f} FPS", fps_color, attrs=['bold'])

                vae_pbar.set_postfix_str(f"FPS: {fps_string} | Batch: {batch_processing_time:.2f}s", refresh=True)

                # --- DYNAMIC A/V SYNC ---
                audio_duration_of_batch = num_frames_in_batch / target_fps
                num_audio_samples_for_batch = int(audio_duration_of_batch * gst_audio_pipeline.sample_rate)

                start_audio_idx = audio_samples_sent
                end_audio_idx = min(start_audio_idx + num_audio_samples_for_batch, total_audio_samples)
                audio_chunk_pcm = full_audio_pcm[start_audio_idx:end_audio_idx]
                audio_samples_sent = end_audio_idx
                
                try:
                    # Put both the recon_frames (list) and audio_chunk_pcm (numpy array) into the queue
                    vae_to_blend_queue.put((list(recon_frames), audio_chunk_pcm), timeout=5.0)
                except queue.Full:
                    logging.error(f"VAE-to-Blend queue is full. Consumer can't keep up. Halting generation.")
                    break

        except Exception as e:
            logging.critical(f"CRITICAL ERROR in inference run '{run_id}'.", exc_info=True)
        finally:
            logging.info(f"\n--- [{run_id}] Final Cleanup ---")
            if 'vae_to_blend_queue' in locals():
                try: vae_to_blend_queue.put(None) # Sentinel value for graceful shutdown
                except Exception: pass
            
            # Wait for consumer thread to finish
            if frame_processor_thread and frame_processor_thread.is_alive():
                logging.info("Waiting for frame processor thread to finish...")
                frame_processor_thread.join(timeout=30.0)
                if frame_processor_thread.is_alive():
                    logging.warning("Frame processor thread did not terminate gracefully.")
            
            # Stop GStreamer pipelines
            if gst_video_pipeline: gst_video_pipeline.stop()
            if gst_audio_pipeline: gst_audio_pipeline.stop()
            
            elapsed = time.time() - start_time
            logging.info(f">>> Inference run '{run_id}' finished in {elapsed:.2f}s. <<<")



    def process_and_send_frames(self, vae_to_blend_q, gst_video, gst_audio, total_frames_planned, frame_skip_threshold_val):
        '''
        [ENHANCED VERSION] Consumer thread: blends, paces, interleaves audio, and sends to GStreamer.

        This function is the heart of the real-time playback logic. It performs three critical tasks:
        1.  **Real-Time Pacing:** Ensures the stream does not play faster than the target FPS on fast hardware,
            while having no negative performance impact on slower hardware.
        2.  **Dynamic Frame Skipping:** If the AI model (producer) generates frames much faster than they can be
            streamed, this function will intelligently drop the oldest batches to "catch up" to real-time,
            preventing runaway latency.
        3.  **Frame-by-Frame A/V Interleaving:** To ensure smooth audio playback without choppiness, this function
            sends a single video frame and then IMMEDIATELY sends its corresponding small slice of audio.
            This tight interleaving prevents audio buffer underruns on the receiving end.

        Args:
            vae_to_blend_q (queue.Queue): The queue from which to get generated face frames and audio chunks.
            gst_video (GStreamerPipeline): The GStreamer video pipeline manager.
            gst_audio (GStreamerAudio): The GStreamer audio pipeline manager.
            total_frames_planned (int): The total number of frames expected for the entire audio clip.
            frame_skip_threshold_val (int): The queue size at which we start dropping frames to reduce latency.
        '''
        # --- PACER INITIALIZATION ---
        target_fps = gst_video.fps
        frame_duration = 1.0 / target_fps
        start_time = None  # <<< FIX: Initialize start_time to None.
        frames_sent = 0
        # --------------------------
        total_frames_processed = 0
        total_frames_skipped = 0

        # --- DYNAMIC FRAME SKIPPING ---
        # (This section remains unchanged)
        while vae_to_blend_q.qsize() > frame_skip_threshold_val:
            try:
                skipped_item = vae_to_blend_q.get_nowait()
                if skipped_item and isinstance(skipped_item, tuple) and len(skipped_item[0]) > 0:
                    num_skipped_in_batch = len(skipped_item[0])
                    total_frames_skipped += num_skipped_in_batch
                    total_frames_processed += num_skipped_in_batch
                    logging.warning(
                        colored(f"Queue overloaded ({vae_to_blend_q.qsize()}). Skipping a batch of {num_skipped_in_batch} frames to catch up.", 'yellow')
                    )
                vae_to_blend_q.task_done()
            except queue.Empty:
                break

        # --- MAIN PROCESSING LOOP ---
        while total_frames_processed < total_frames_planned:
            try:
                batch_data = vae_to_blend_q.get(block=True, timeout=10.0)
            except queue.Empty:
                logging.error("Timeout waiting for frames from VAE producer. Ending processor thread.")
                break

            if batch_data is None:
                logging.info("Received sentinel. Frame processor shutting down.")
                break

            # <<< FIX: Start the pacer's clock at the exact moment the first batch arrives.
            if start_time is None:
                start_time = time.perf_counter()
                logging.info("First data batch received. Starting real-time pacer clock.")
            # <<< END FIX

            vae_frames, audio_chunk = batch_data
            num_frames_in_batch = len(vae_frames)
            if num_frames_in_batch == 0:
                vae_to_blend_q.task_done()
                continue

            samples_per_frame = len(audio_chunk) // num_frames_in_batch
            audio_cursor = 0

            for i, frame in enumerate(vae_frames):
                # --- REAL-TIME PACER ---
                next_frame_target_time = start_time + (frames_sent + 1) * frame_duration
                sleep_duration = next_frame_target_time - time.perf_counter()
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                # --- END PACER ---

                # 1. Blend the generated face onto the background video frame
                blended_frame = self._blend_single_frame(frame, gst_video.width, gst_video.height)

                if blended_frame is not None:
                    # 2. Send the final video frame to the video pipeline
                    if gst_video.send_frame(blended_frame):
                        frames_sent += 1
                    else:
                        logging.error("Video pipe broken. Halting all frame processing.")
                        vae_to_blend_q.task_done()
                        return

                # 3. Determine the precise audio slice for THIS video frame
                start_idx = audio_cursor
                end_idx = (audio_cursor + samples_per_frame) if (i < num_frames_in_batch - 1) else len(audio_chunk)
                audio_slice = audio_chunk[start_idx:end_idx]
                audio_cursor = end_idx

                # 4. Send the corresponding audio slice
                if audio_slice.size > 0:
                    if not gst_audio.send_audio(audio_slice):
                        logging.error("Audio pipe broken. Stopping audio sends for this run.")

            total_frames_processed += num_frames_in_batch
            vae_to_blend_q.task_done()

        logging.info(colored(f"--- Frame processor finished. Sent: {frames_sent}, Skipped: {total_frames_skipped} ---", "green"))
        
    def _blend_single_frame(self, res_frame, target_width, target_height):
        """
        [OPTIMIZED] Blends a single generated face frame onto the original background.
        This version uses pre-converted PIL objects to reduce real-time overhead.
        """
        try:
            # 1. --- Retrieve pre-processed data for the current frame ---
            cycle_len = len(self.coord_list_cycle)
            if cycle_len == 0:
                return None # No need to log error every frame

            current_idx = self.idx % cycle_len

            # Retrieve the pre-calculated and pre-converted PIL images
            body_pil = self.body_pil_cycle[current_idx]
            mask_pil = self.mask_pil_cycle[current_idx]
            
            # These are still needed from the original lists
            face_bbox = self.coord_list_cycle[current_idx]
            crop_box = self.mask_coords_list_cycle[current_idx]

            # 2. --- Perform necessary real-time conversions and operations ---

            # Resize the AI-generated face (this must be done in real-time)
            face_w = int(face_bbox[2] - face_bbox[0])
            face_h = int(face_bbox[3] - face_bbox[1])
            
            # OPTIMIZATION: Switched to a faster interpolation method.
            # INTER_LINEAR is a great balance of speed and quality.
            res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (face_w, face_h), interpolation=cv2.INTER_LINEAR)
            
            # This is the only BGR->RGB conversion left in the hot-loop
            face_pil = Image.fromarray(res_frame_resized[:, :, ::-1])

            # 3. --- Replicate the Original Library's Paste Logic ---
            x_s, y_s, _, _ = [int(p) for p in crop_box]
            x_f, y_f, _, _ = [int(p) for p in face_bbox]

            # This crop operation is faster now as it's on an already-loaded PIL image
            face_large_pil = body_pil.crop(tuple(int(p) for p in crop_box))

            paste_position = (x_f - x_s, y_f - y_s)
            face_large_pil.paste(face_pil, paste_position)
            body_pil.paste(face_large_pil, (x_s, y_s), mask_pil)

            # 4. --- Convert back to numpy for GStreamer ---
            # This final conversion is unavoidable
            final_frame = np.array(body_pil, dtype=np.uint8)[:, :, ::-1]

            # 5. --- Increment index and return the final frame ---
            self.idx += 1
            
            # This final resize for GStreamer is also unavoidable
            if final_frame.shape[0] != target_height or final_frame.shape[1] != target_width:
                 return cv2.resize(final_frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            else:
                 return final_frame

        except Exception as e:
            # Reduced logging level for performance
            if self.idx % 100 == 0: # Log only every 100 frames to avoid spam
                logging.warning(f"Error blending frame at index {self.idx}: {e}")
            self.idx += 1
            return None

# --- Main Application Entry Point ---
inference_lock = threading.Lock() # Ensures only one inference run processes audio at a time
audio_queue = queue.Queue(maxsize=10) # Queue for raw audio data received by file watcher

# NEW: Load environment from powershell_env.txt (GLOBAL SCOPE)
POWERSHELL_ENV_FILE = "C:\\temp\\powershell_env.txt"

env_from_file = os.environ.copy() 

if os.path.exists(POWERSHELL_ENV_FILE):
    try:
        with open(POWERSHELL_ENV_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_from_file[key] = value
        logging.info(f"Successfully loaded environment from {POWERSHELL_ENV_FILE}.")
    except Exception as e:
        logging.warning(f"Failed to load environment from {POWERSHELL_ENV_FILE}: {e}", exc_info=True)
else:
    logging.warning(f"PowerShell environment file not found at {POWERSHELL_ENV_FILE}. Proceeding with inherited environment.")
# --- END NEW GLOBAL BLOCK ---


def inference_worker(avatar_instance):
    """
    Worker thread that waits for audio data from a queue and initiates an inference run.
    Uses a non-blocking lock to skip audio if inference is already in progress.
    """
    logging.info("🚀 Inference worker started. Waiting for audio data...")
    while True:
        audio_data = audio_queue.get() # Blocks until audio is available or None is received
        if audio_data is None: # Shutdown signal
            logging.info("Inference worker received shutdown signal. Exiting.")
            break
        
        # Try to acquire lock non-blocking. If busy, skip this audio chunk.
        if not inference_lock.acquire(blocking=False):
            logging.warning("Inference already in progress. Skipping newly received audio to prevent backlog.")
            audio_queue.task_done() # Mark this item as done even if skipped
            continue # Continue to next item in queue

        try:
            logging.info(f"Inference lock acquired. Processing {len(audio_data)} bytes of audio.")
            # Delegate to Avatar's inference method, which now handles GStreamer internally
            avatar_instance.inference(audio_data, TARGET_FPS)
        except Exception as e:
            logging.critical(f"Unhandled exception during avatar inference: {e}", exc_info=True)
        finally:
            inference_lock.release() # Release lock whether successful or not
            audio_queue.task_done() # Mark this item as done
            logging.info("Inference lock released.")

def file_watcher():
    """
    Monitors a specified file for changes (e.g., new audio data written to it).
    When a change is detected, reads the file's content and puts it into the audio queue.
    Designed for use with a "hot file" or named pipe for continuous audio input.
    """
    logging.info(f"👀 Starting file watcher for: {STREAM_PIPE_PATH}") # Use global STREAM_PIPE_PATH
    last_processed_mtime = 0 # Tracks last modification time to detect new data
    file_watcher_started_log = False # Flag to log only once

    while True:
        if not file_watcher_started_log:
            logging.info("✅ File watcher thread is actively monitoring.")
            file_watcher_started_log = True

        try:
            if os.path.isfile(STREAM_PIPE_PATH):
                current_mtime = os.path.getmtime(STREAM_PIPE_PATH)
                if current_mtime > last_processed_mtime:
                    logging.info("New audio file detected. Reading content...")
                    # Small delay to ensure file is fully written by external process
                    time.sleep(0.1) 
                    with open(STREAM_PIPE_PATH, "rb") as f:
                        incoming_audio_data = f.read()
                    
                    if incoming_audio_data:
                        audio_queue.put(incoming_audio_data) # Add to queue for inference worker
                        last_processed_mtime = current_mtime # Update mtime only if data was processed
                    else:
                        logging.warning("File modification detected, but file was empty. Skipping.")
                        last_processed_mtime = current_mtime # Still update mtime to avoid re-processing empty file
            time.sleep(0.2) # Polling interval for file changes
        except FileNotFoundError:
            # This is normal if the file is deleted and recreated (e.g., for continuous streaming)
            last_processed_mtime = 0 # Reset mtime to ensure next file is processed
            time.sleep(0.5) # Wait a bit longer before re-checking for file
        except Exception as e:
            logging.error(f"Error in file watcher loop: {e}", exc_info=True)
            time.sleep(2)


if __name__ == "__main__":
    logging.info("🎬 Starting MuseTalk Realtime Stream Sync Application (v1.5 compatible)...")

    # 1. Load environment variables from .env file (must be at the top)
    # load_dotenv() # Already done globally at the very top of script

    # --- environment loading (now outside of __main__ but used by GStreamer classes) ---
    # The `env_from_file` variable is already defined globally now.
    
    # 2. Argument Parsing (incorporating 1.5 defaults and user's .env/cmd-line options)
    parser = argparse.ArgumentParser(description="MuseTalk Real-Time Streaming Script")
    parser.add_argument("--version", type=str, default=MUSE_VERSION, choices=["v1", "v15"], help="MuseTalk version (from .env MUSE_VERSION)")
    parser.add_argument("--ffmpeg_path", type=str, default=FFMPEG_PATH, help="Path to ffmpeg executable (from .env FFMPEG_PATH)")
    parser.add_argument("--gpu_id", type=int, default=GPU_ID, help="GPU ID to use (from .env GPU_ID)")
    parser.add_argument("--vae_type", type=str, default=VAE_TYPE, help="Type of VAE model (from .env VAE_TYPE)")
    parser.add_argument("--unet_config", type=str, default=UNET_CONFIG_PATH, help="Path to UNet configuration file (from .env UNET_CONFIG)")
    parser.add_argument("--unet_model_path", type=str, default=UNET_MODEL_PATH, help="Path to UNet model weights (from .env UNET_MODEL_PATH)")
    parser.add_argument("--whisper_dir", type=str, default=WHISPER_DIR, help="Directory containing Whisper model (from .env WHISPER_DIR)")
    parser.add_argument("--result_dir", default=RESULT_DIR, help="Directory for output results (from .env RESULT_DIR)")
    parser.add_argument("--extra_margin", type=int, default=EXTRA_MARGIN, help="Extra margin for face cropping (from .env EXTRA_MARGIN)")
    parser.add_argument("--fps", type=int, default=TARGET_FPS, help="Video frames per second (from .env TARGET_FPS)")
    parser.add_argument("--audio_padding_length_left", type=int, default=AUDIO_PADDING_LENGTH_LEFT, help="Left padding length for audio (from .env AUDIO_PADDING_LEFT)")
    parser.add_argument("--audio_padding_length_right", type=int, default=AUDIO_PADDING_LENGTH_RIGHT, help="Right padding length for audio (from .env AUDIO_PADDING_RIGHT)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for inference (from .env BATCH_SIZE)")
    parser.add_argument("--parsing_mode", default=PARSING_MODE, help="Face blending parsing mode (from .env PARSING_MODE)")
    parser.add_argument("--left_cheek_width", type=int, default=LEFT_CHEEK_WIDTH, help="Width of left cheek region (from .env LEFT_CHEEK_WIDTH)")
    parser.add_argument("--right_cheek_width", type=int, default=RIGHT_CHEEK_WIDTH, help="Width of right cheek region (from .env RIGHT_CHEEK_WIDTH)")
    parser.add_argument("--avatar_config_path", type=str, default=AVATAR_CONFIG_PATH, help="Path to avatar configuration YAML (from .env AVATAR_CONFIG_PATH)")
    parser.add_argument("--avatar_id_to_use", type=str, default=AVATAR_ID_TO_USE, help="ID of the avatar to use from config (from .env AVATAR_ID_TO_USE)")
    parser.add_argument("--stream_pipe_path", type=str, default=STREAM_PIPE_PATH, help="Path to watch for audio input (from .env STREAM_PIPE_PATH)")
    parser.add_argument("--gstreamer_launch_path", type=str, default=GSTREAMER_LAUNCH_PATH, help="Full path to gst-launch-1.0.exe (from .env GSTREAMER_LAUNCH_PATH)")

    args = parser.parse_args()
    logging.info(f"INFO: GStreamer launch path resolved to: {GSTREAMER_LAUNCH_PATH}") 

    # 3. Configure ffmpeg path and verify
    if not fast_check_ffmpeg():
        logging.info("Attempting to add ffmpeg to PATH...")
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{FFMPEG_PATH}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            logging.critical("❌ Critical: Unable to find ffmpeg even after attempting to add to PATH. Please ensure ffmpeg is properly installed and accessible. Exiting.")
            sys.exit(1) # Exit if ffmpeg is not found

    # 4. Set computing device and print GPU info
    # 'device' is already set globally based on GPU_ID from .env
    logging.info(f"✅ Selected device: {device}")
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(device)
            gpu_capability = torch.cuda.get_device_capability(device)
            logging.info(f"GPU Name: {gpu_name}")
            logging.info(f"CUDA Capability: {gpu_capability}")
        except Exception as e:
            logging.warning(f"Warning: Could not retrieve detailed GPU information: {e}")

    # 5. Load all MuseTalk models (following MuseTalk 1.5's new loading pattern)
    logging.info("Loading MuseTalk models (VAE, UNet, Positional Encoding)...")
    try:
        vae, unet, pe = load_all_model(
            unet_model_path=UNET_MODEL_PATH,
            vae_type=VAE_TYPE,
            unet_config=UNET_CONFIG_PATH,
            device=device
        )
        # --- NEW OPTIMIZATION --- 
        '''
        logging.info("Applying torch.compile() for maximum performance...")
        if torch.__version__ >= "2.0":
            unet.model = torch.compile(unet.model, mode="reduce-overhead")
            vae.vae = torch.compile(vae.vae, mode="reduce-overhead")
            pe = torch.compile(pe)
            logging.info("✅ Models successfully compiled.")
        else:
            logging.warning("torch.compile() requires PyTorch 2.0+. Skipping.")
        '''
        # ------------------------
        timesteps = torch.tensor([0], device=device) # Initialize global timesteps tensor
        
        pe = pe.half().to(device)
        vae.vae = vae.vae.half().to(device)
        unet.model = unet.model.half().to(device)
        
        logging.info("Initializing AudioProcessor and loading Whisper model...")
        audio_processor = AudioProcessor(feature_extractor_path=WHISPER_DIR)
        weight_dtype = unet.model.dtype # Use model's dtype for Whisper (usually FP16 after .half())
        whisper = WhisperModel.from_pretrained(WHISPER_DIR)
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False) # Freeze Whisper model parameters

        logging.info("Initializing FaceParsing model...")
        if MUSE_VERSION == "v15":
            fp = FaceParsing(
                left_cheek_width=LEFT_CHEEK_WIDTH,
                right_cheek_width=RIGHT_CHEEK_WIDTH
            )
        else: # v1 fallback
            fp = FaceParsing()
        
        logging.info("✅ All MuseTalk 1.5 core models loaded and configured successfully.")
    except Exception as e:
        logging.critical("❌ Fatal error loading MuseTalk models or setting up device/precision. Check model paths, CUDA, and dependencies.", exc_info=True)
        sys.exit(1)

    main_avatar = None
    try:
        # 6. Validate configuration paths and avatar ID
        if not all([AVATAR_CONFIG_PATH, AVATAR_ID_TO_USE, STREAM_PIPE_PATH]):
            raise ValueError("A required configuration path (AVATAR_CONFIG_PATH, AVATAR_ID_TO_USE, or STREAM_PIPE_PATH) is missing. Check your .env file or command-line arguments.")

        logging.info(f"Attempting to load Avatar ID '{AVATAR_ID_TO_USE}' from config file '{AVATAR_CONFIG_PATH}'...")
        config = OmegaConf.load(AVATAR_CONFIG_PATH)
        if AVATAR_ID_TO_USE not in config:
            raise ValueError(f"Avatar ID '{AVATAR_ID_TO_USE}' was not found as a key in the YAML file '{AVATAR_CONFIG_PATH}'")
        
        avatar_config = config[AVATAR_ID_TO_USE]

        # 7. Instantiate the Avatar with the loaded configuration
        main_avatar = Avatar(
            avatar_id=AVATAR_ID_TO_USE,
            video_path=avatar_config.video_path,
            bbox_shift=avatar_config.bbox_shift,
            batch_size=BATCH_SIZE, # Use global BATCH_SIZE
            preparation=avatar_config.preparation,
            version_str=MUSE_VERSION, # Use global MUSE_VERSION
            extra_margin=EXTRA_MARGIN, # Use global EXTRA_MARGIN
            parsing_mode=PARSING_MODE, # Use global PARSING_MODE
        )
        logging.debug("DEBUG: Avatar instance successfully created. Proceeding to thread initialization.") 
    except Exception as e:
        logging.critical("❌ CRITICAL ERROR during Avatar setup. Check your avatar config YAML and video paths.", exc_info=True)
        sys.exit(1)

    # 8. Start the file watcher and inference worker threads
    logging.info("Starting file watcher and inference worker threads...")
    inference_thread = threading.Thread(target=inference_worker, args=(main_avatar,), daemon=True, name="InferenceWorker")
    watcher_thread = threading.Thread(target=file_watcher, daemon=True, name="FileWatcher")

    inference_thread.start()
    watcher_thread.start()

    # 9. Keep the main thread alive and handle graceful shutdown
    logging.info("✅ Application is running. Press Ctrl+C to shut down gracefully.")
    try:
        while True:
            # Check if critical worker threads are still alive
            if not watcher_thread.is_alive():
                logging.error("File watcher thread died unexpectedly. Initiating shutdown.")
                break
            if not inference_thread.is_alive():
                logging.error("Inference worker thread died unexpectedly. Initiating shutdown.")
                break
            time.sleep(1.0) # Main thread sleeps, allowing worker threads to run
    except KeyboardInterrupt:
        logging.info("\n🛑 KeyboardInterrupt received. Initiating graceful shutdown...")
    finally:
        # Signal worker threads to stop by putting a sentinel value into the queue
        audio_queue.put(None) 
        
        logging.info("Attempting to join worker threads for clean shutdown...")
        
        # Give watcher thread a moment to finish
        if watcher_thread.is_alive():
            watcher_thread.join(timeout=2) 
            if watcher_thread.is_alive():
                logging.warning("File watcher thread did not terminate gracefully. It might be stuck.")
        
        # Give inference thread more time as it might be processing a batch
        if inference_thread.is_alive():
            inference_thread.join(timeout=30) 
            if inference_thread.is_alive():
                logging.warning("Inference worker thread did not terminate gracefully. It might be stuck.")

        logging.info("✅ Application shutdown complete.")