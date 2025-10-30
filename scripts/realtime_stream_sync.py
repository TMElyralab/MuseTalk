# -*- coding: utf-8 -*-
import ffmpeg
import argparse
import os
import concurrent.futures # For ThreadPoolExecutor in Avatar preparation
import threading
import queue
import io
from omegaconf import OmegaConf
import subprocess
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
import traceback          # For detailed error printing
import time
from PIL import Image
import tempfile
import logging
from dotenv import load_dotenv 
# --- Platform-specific imports for future use if needed ---
if sys.platform == "win32":
    try:
        import psutil
        # Optional: Set high process priority on Windows
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("INFO: Process priority set to HIGH on Windows.")
    except ImportError:
        print("Warning: psutil not found. Cannot set process priority.")
    except Exception as e:
        print(f"Warning: Could not set process priority: {e}")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- MuseTalk Specific Imports (Ensure these are in your PYTHONPATH) ---
try:
    from musetalk.utils.utils import get_file_type, get_video_fps, datagen
    from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
    from musetalk.utils.blending import get_image_prepare_material
    from musetalk.utils.utils import load_all_model
except ImportError as e:
    logging.critical(f"Error importing MuseTalk utilities: {e}. Ensure the library is installed and in your PYTHONPATH.", exc_info=True)
    sys.exit(1)

import shutil

# --- Configuration & Global Variables ---
# Recommended to be set via environment variables or a config file
TARGET_FPS = int(os.getenv("TARGET_FPS", "25"))
FRAME_SKIP_THRESHOLD = int(os.getenv("FRAME_SKIP_THRESHOLD", "3")) # Drop frames if queue size exceeds this
AVATAR_CONFIG_PATH = os.getenv("AVATAR_CONFIG_PATH", "configs/inference/realtime.yaml")
AVATAR_ID_TO_USE = os.getenv("AVATAR_ID_TO_USE", "default_avatar_id") # CHANGE THIS
STREAM_PIPE_PATH = os.getenv("STREAM_PIPE_PATH", "./hot_file.opus") # File to watch for audio

# --- PyTorch Device Setup ---
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
logging.info("--- PyTorch Device Information ---")
if cuda_available:
    try:
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"✅ CUDA (GPU) detected: {gpu_name}")
    except Exception as e:
        logging.warning(f"⚠️ Could not retrieve GPU name: {e}")
else:
    logging.info("❌ CUDA (GPU) not available. Using CPU.")
logging.info(f"✅ Selected device: {device}")
logging.info("-------------------------------")

# --- Load Models ---
logging.info("Loading models...")
try:
    audio_processor, vae, unet, pe = load_all_model()
    logging.info("✅ Models loaded successfully.")
except Exception as e:
    logging.critical("Fatal error loading models.", exc_info=True)
    sys.exit(1)

# --- Set Model Precision (FP16) ---
logging.info("Setting model precision to half (FP16) where applicable...")
try:
    if hasattr(pe, 'half'): pe = pe.to(device).half()
    if hasattr(vae, 'vae') and hasattr(vae.vae, 'half'): vae.vae = vae.vae.to(device).half()
    if hasattr(unet, 'model') and hasattr(unet.model, 'half'): unet.model = unet.model.to(device).half()
    timesteps = torch.tensor([0], device=device)
    logging.info("✅ Model precision set.")
except Exception as e:
    logging.warning(f"⚠️ Error setting model precision: {e}. Performance may be affected.", exc_info=True)


# --- FFmpeg Audio Reader ---
class FFmpegAudioReader:
    """Uses FFmpeg to read an audio file (from path or bytes) and convert it to raw PCM."""
    def __init__(self, audio_source):
        self.audio_source = audio_source
        self.is_file_path = isinstance(audio_source, str)

    def read_full_audio(self):
        """Reads the entire audio source and converts it to PCM s16le, 48kHz, Stereo."""
        logging.info(f"Reading and converting audio from {'file' if self.is_file_path else 'memory'}...")
        target_sr, target_ac, target_format = 48000, 2, "s16le"
        input_data = None
        
        ffmpeg_input_args = {}
        if not self.is_file_path:
            # If source is bytes, we'll pipe it to ffmpeg's stdin
            input_filename = 'pipe:0'
            input_data = self.audio_source
        else:
            input_filename = self.audio_source

        try:
            # Use ffmpeg-python for a cleaner interface
            out, err = (
                ffmpeg
                .input(input_filename, **ffmpeg_input_args)
                .output('pipe:', format=target_format, ac=target_ac, ar=target_sr)
                .run(capture_stdout=True, capture_stderr=True, input=input_data)
            )
            if err:
                logging.debug(f"FFmpeg stderr: {err.decode(errors='ignore')}")
        except ffmpeg.Error as e:
            logging.error(f"❌ FFmpeg error during audio conversion: {e.stderr.decode(errors='ignore') if e.stderr else 'Unknown FFmpeg error'}")
            return None
        except Exception as e:
            logging.error(f"❌ Unexpected error during FFmpeg execution: {e}", exc_info=True)
            return None

        if not out:
            logging.error("❌ Failed to read audio: FFmpeg produced no PCM data!")
            return None

        audio_data = np.frombuffer(out, dtype=np.int16).reshape(-1, target_ac)
        logging.info(f"✅ Read and converted audio: {len(audio_data)} samples at {target_sr}Hz, {target_ac}ch.")
        return audio_data

# --- GStreamer Classes ---
class GStreamerPipeline:
    """Manages the GStreamer video pipeline subprocess."""
    def __init__(self, width=1280, height=720, fps=TARGET_FPS, host="127.0.0.1", port=5000):
        self.width, self.height, self.fps, self.host, self.port = width, height, fps, host, port
        self.process = None
        self.stdout_thread = None
        self.stderr_thread = None

        # A robust pipeline with leaky queues (for backpressure) and nvenc for performance
        pipeline_str = (
            f"fdsrc fd=0 do-timestamp=true is-live=true ! videoparse format=bgr width={self.width} height={self.height} framerate={self.fps}/1 ! "
            "queue ! videoconvert ! videorate ! "
            f"video/x-raw,format=NV12 ! " # Assuming NV12 is desired for nvh265enc
            "queue ! "
            # Reverting bitrate to 4000 as per your old working code, and keep preset for testing
            f"nvh265enc preset=low-latency-hq rc-mode=cbr bitrate=4000 gop-size=30 ! "
            "h265parse ! rtph265pay pt=96 config-interval=1 ! "
            f"udpsink host={self.host} port={self.port} sync=true async=false"
        )
        logging.info(f"Starting GStreamer video pipeline with resolution {self.width}x{self.height}@{self.fps}fps...")
        
        # Prepare environment variables for the subprocess
        env_vars = os.environ.copy()
        env_vars['GST_DEBUG'] = '3' # Set GStreamer debug level

        try:
            # Construct the command string without explicit quoting for GSTREAMER_LAUNCH_PATH here.
            # This relies on the shell's PATH to find gst-launch-1.0.
            # If GSTREAMER_LAUNCH_PATH itself has a full path, this will override PATH.
            command_to_run = f"{GSTREAMER_LAUNCH_PATH} -v {pipeline_str}"

            logging.debug(f"DEBUG: GStreamer VIDEO command (old way): {command_to_run}")

            self.process = subprocess.Popen(
                command_to_run,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, # Capture stdout
                stderr=subprocess.PIPE, # Capture stderr
                shell=True, # Keep shell=True
                bufsize=0,
                env=env_vars
            )
            logging.info(f"✅ GStreamer video process launched (PID: {self.process.pid}).")

            # --- Start threads to read stdout and stderr asynchronously ---
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
            logging.error(f"❌ Failed to start GStreamer video pipeline: {e}", exc_info=True)
            self.process = None 

    def send_frame(self, frame):
        """Sends a NumPy array frame to the GStreamer pipeline's stdin."""
        if not self.process or self.process.stdin.closed:
            logging.error("❌ GStreamer video pipeline process is not running or stdin is closed.")
            return False
        try:
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            logging.error("❌ GStreamer video pipeline: Broken pipe. The process may have crashed.")
            self.stop()
            return False
        except Exception as e:
            logging.error(f"❌ Error pushing video frame: {e}", exc_info=True)
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
                logging.info(f"✅ GStreamer video process terminated.")
            except subprocess.TimeoutExpired:
                logging.warning(f"⚠️ GStreamer video process did not terminate gracefully, killing...")
                proc_to_stop.kill()

        if self.stdout_thread and self.stdout_thread.is_alive():
            logging.info("Joining GST_VIDEO_STDOUT thread...")
            self.stdout_thread.join(timeout=1)
        if self.stderr_thread and self.stderr_thread.is_alive():
            logging.info("Joining GST_VIDEO_STDERR thread...")
            self.stderr_thread.join(timeout=1)
        logging.info("GStreamer video pipeline stop complete.")


class GStreamerAudio:
    """Manages the GStreamer audio pipeline subprocess."""
    def __init__(self, host="127.0.0.1", port=5001, sample_rate=48000, channels=2):
        self.host, self.port, self.sample_rate, self.channels = host, port, sample_rate, channels
        self.process = None
        self.stdout_thread = None
        self.stderr_thread = None

        pipeline_str = (
            f"fdsrc fd=0 do-timestamp=true is-live=true ! "
            "queue ! "
            f"audio/x-raw,format=S16LE,channels={self.channels},rate={self.sample_rate},layout=interleaved ! "
            "audioconvert ! audioresample ! "
            "opusenc bitrate=96000 ! rtpopuspay pt=97 ! "
            f"udpsink host={self.host} port={self.port} sync=true"
        )
        logging.info("Starting GStreamer audio pipeline...")

        env_vars = os.environ.copy()
        env_vars['GST_DEBUG'] = '3'
        
        try:
            # Construct the command string without explicit quoting for GSTREAMER_LAUNCH_PATH here.
            command_to_run = f"{GSTREAMER_LAUNCH_PATH} -v {pipeline_str}"

            logging.debug(f"DEBUG: GStreamer AUDIO command (old way): {command_to_run}")

            self.process = subprocess.Popen(
                command_to_run,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                bufsize=0,
                env=env_vars
            )
            logging.info(f"✅ GStreamer audio process launched (PID: {self.process.pid}).")

            # --- Start threads to read stdout and stderr asynchronously ---
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
            logging.error(f"❌ Failed to start GStreamer audio pipeline: {e}", exc_info=True)
            self.process = None

    def send_audio(self, audio_data_pcm):
        """Sends raw PCM audio data to the GStreamer pipeline's stdin."""
        if not self.process or self.process.stdin.closed:
            logging.error("❌ GStreamer audio pipeline process is not running or stdin is closed.")
            return False
        try:
            self.process.stdin.write(audio_data_pcm.tobytes())
            self.process.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            logging.error("❌ GStreamer audio pipeline: Broken pipe.")
            self.stop()
            return False
        except Exception as e:
            logging.error(f"❌ Error pushing audio chunk: {e}", exc_info=True)
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
                logging.info("✅ GStreamer audio process terminated.")
            except subprocess.TimeoutExpired:
                logging.warning("⚠️ GStreamer audio process did not terminate gracefully, killing...")
                proc_to_stop.kill()

        if self.stdout_thread and self.stdout_thread.is_alive():
            logging.info("Joining GST_AUDIO_STDOUT thread...")
            self.stdout_thread.join(timeout=1)
        if self.stderr_thread and self.stderr_thread.is_alive():
            logging.info("Joining GST_AUDIO_STDERR thread...")
            self.stderr_thread.join(timeout=1)
        logging.info("GStreamer audio pipeline stop complete.")


    def _log_stream(self, stream, prefix):
        try:
            for line_bytes in iter(stream.readline, b''):
                logging.info(f"[{prefix}]: {line_bytes.decode(errors='ignore').strip()}")
        finally:
            stream.close()

    def send_audio(self, audio_data_pcm):
        if not self.process or self.process.stdin.closed:
            return False
        try:
            self.process.stdin.write(audio_data_pcm.tobytes())
            self.process.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            logging.error("❌ GStreamer audio pipeline: Broken pipe.")
            self.stop()
            return False
        except Exception as e:
            logging.error(f"❌ Error pushing audio chunk: {e}", exc_info=True)
            return False

    def stop(self):
        if self.process:
            logging.info(f"Stopping GStreamer audio pipeline (PID: {self.process.pid})...")
            proc_to_stop, self.process = self.process, None
            if proc_to_stop.stdin and not proc_to_stop.stdin.closed:
                try: proc_to_stop.stdin.close()
                except Exception: pass
            proc_to_stop.terminate()
            try:
                proc_to_stop.wait(timeout=2)
                logging.info("✅ GStreamer audio process terminated.")
            except subprocess.TimeoutExpired:
                logging.warning("⚠️ GStreamer audio process did not terminate gracefully, killing...")
                proc_to_stop.kill()

# --- Helper Functions ---
def video2imgs(vid_path, save_path):
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
    for path in path_list:
        os.makedirs(path, exist_ok=True)

# --- Avatar Class ---
class Avatar:
    @torch.no_grad()
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        logging.info(f"Initializing Avatar: {avatar_id}")
        self.avatar_id = str(avatar_id)
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.batch_size = batch_size
        self.preparation = preparation
        
        # Define paths
        self.avatar_base_path = os.path.join("./results/avatars", self.avatar_id)
        self.full_imgs_path = os.path.join(self.avatar_base_path, "full_imgs")
        self.mask_out_path = os.path.join(self.avatar_base_path, "masks")
        self.coords_path = os.path.join(self.avatar_base_path, "coords.pkl")
        self.latents_out_path = os.path.join(self.avatar_base_path, "latents.pt")
        self.mask_coords_path = os.path.join(self.avatar_base_path, "mask_coords.pkl")
        self.avatar_info_path = os.path.join(self.avatar_base_path, "avatar_info.json")

        # Initialize data stores
        self.input_latent_list_cycle = []
        self.coord_list_cycle = []
        self.frame_list_cycle = []
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []
        
        self.idx = 0 # Tracks current position in the reference video cycle
        
        self.init_avatar_data()
        logging.info(f"✅ Avatar '{self.avatar_id}' initialized with {len(self.frame_list_cycle)} reference frames.")

    def init_avatar_data(self):
        # If preparation is needed, do it. Otherwise, load existing data.
        if self.preparation:
            if os.path.exists(self.avatar_base_path):
                # Prompt user before overwriting existing data
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
        logging.info(f"Reloading prepared data from: {self.avatar_base_path}")
        try:
            # Load all data from disk
            loaded_latents = torch.load(self.latents_out_path, map_location='cpu')
            self.input_latent_list_cycle = list(loaded_latents) if isinstance(loaded_latents, torch.Tensor) else loaded_latents
            with open(self.coords_path, 'rb') as f: self.coord_list_cycle = pickle.load(f)
            with open(self.mask_coords_path, 'rb') as f: self.mask_coords_list_cycle = pickle.load(f)

            # Read corresponding images and masks
            num_items = len(self.coord_list_cycle)
            if num_items == 0: raise ValueError("Loaded coordinate data is empty.")
            
            frame_files = [os.path.join(self.full_imgs_path, f"{str(i).zfill(8)}.png") for i in range(num_items)]
            self.frame_list_cycle = read_imgs(frame_files)
            mask_files = [os.path.join(self.mask_out_path, f"{str(i).zfill(8)}.png") for i in range(num_items)]
            self.mask_list_cycle = read_imgs(mask_files)
            
            # Validate that all lists have the same, non-zero length
            data_map = {
                "Latents": self.input_latent_list_cycle, "Coords": self.coord_list_cycle, 
                "Frames": self.frame_list_cycle, "Masks": self.mask_list_cycle, "MaskCoords": self.mask_coords_list_cycle
            }
            if not all(len(lst) == num_items for lst in data_map.values()):
                lengths = {name: len(lst) for name, lst in data_map.items()}
                raise ValueError(f"Data lists have mismatched lengths after loading: {lengths}")

        except Exception as e:
            logging.critical(f"Error reloading prepared data. You may need to run with preparation=True.", exc_info=True)
            raise SystemExit(f"Exiting: Failed to reload data for {self.avatar_id}.")

    @torch.no_grad()
    def _prepare_material_core(self):
        logging.info(f"--- Preparing new material for avatar: {self.avatar_id} ---")
        osmakedirs([self.avatar_base_path, self.full_imgs_path, self.mask_out_path])
        with open(self.avatar_info_path, "w") as f: json.dump({"avatar_id": self.avatar_id}, f)
        
        # 1. Extract frames from video or copy from image folder
        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path)
        elif os.path.isdir(self.video_path):
            # Handles copying and renaming from a folder of images
            source_files = sorted([f for f in os.listdir(self.video_path) if f.lower().endswith(('.png','.jpg','.jpeg'))])
            for i, filename in enumerate(tqdm(source_files, desc="Copying frames")):
                shutil.copy(os.path.join(self.video_path, filename), os.path.join(self.full_imgs_path, f"{i:08d}.png"))
        else:
            raise FileNotFoundError(f"video_path '{self.video_path}' is not a valid file or directory.")

        # 2. Get landmarks and filter out invalid frames
        source_images = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.png')))
        initial_coords, initial_frames = get_landmark_and_bbox(source_images, self.bbox_shift)
        
        # 3. Process valid frames: VAE encoding and mask generation
        valid_latents, valid_coords, valid_frames, valid_masks, valid_mask_coords = [], [], [], [], []
        coord_ph_val = coord_placeholder

        for i, (bbox, frame) in enumerate(tqdm(zip(initial_coords, initial_frames), total=len(initial_coords), desc="VAE Encoding & Masking")):
            if bbox is None or np.array_equal(bbox, coord_ph_val) or frame is None:
                continue
            
            x1c, y1c, x2c, y2c = bbox
            crop = frame[int(y1c):int(y2c), int(x1c):int(x2c)]
            if crop.size == 0: continue

            try:
                # VAE Encoding
                resized_crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                latents = vae.get_latents_for_unet(resized_crop).cpu()
                
                # Mask Generation
                mask, crop_box = get_image_prepare_material(frame, bbox)
                if mask is None or crop_box is None: continue

                # Add all valid data together
                valid_latents.append(latents)
                valid_coords.append(bbox)
                valid_frames.append(frame)
                valid_masks.append(mask)
                valid_mask_coords.append(crop_box)
            except Exception as e:
                logging.warning(f"Skipping frame {i} due to processing error: {e}")

        if not valid_frames:
            raise RuntimeError("No valid frames survived the preparation process.")

        # 4. Create looping cycle (forward and reverse) and save all data
        self.frame_list_cycle = valid_frames + valid_frames[::-1]
        self.coord_list_cycle = valid_coords + valid_coords[::-1]
        self.input_latent_list_cycle = valid_latents + valid_latents[::-1]
        self.mask_list_cycle = valid_masks + valid_masks[::-1]
        self.mask_coords_list_cycle = valid_mask_coords + valid_mask_coords[::-1]

        # Overwrite content with the final, filtered, and cycled data
        shutil.rmtree(self.full_imgs_path); os.makedirs(self.full_imgs_path)
        shutil.rmtree(self.mask_out_path); os.makedirs(self.mask_out_path)
        for i, (frame, mask) in enumerate(tqdm(zip(self.frame_list_cycle, self.mask_list_cycle), total=len(self.frame_list_cycle), desc="Saving final cycle data")):
            cv2.imwrite(os.path.join(self.full_imgs_path, f"{i:08d}.png"), frame)
            cv2.imwrite(os.path.join(self.mask_out_path, f"{i:08d}.png"), mask)

        with open(self.coords_path, 'wb') as f: pickle.dump(self.coord_list_cycle, f)
        with open(self.mask_coords_path, 'wb') as f: pickle.dump(self.mask_coords_list_cycle, f)
        torch.save(torch.stack(self.input_latent_list_cycle), self.latents_out_path)

        logging.info(f"--- Material prep complete. Final cycle length: {len(self.frame_list_cycle)} frames. ---")
    
    @torch.no_grad()
    def inference(self, audio_source, target_fps):
        # This is the main inference producer loop
        run_id = f"stream_{int(time.time())}"
        logging.info(f"🎬 Starting inference run ID: {run_id}")

        # This queue passes generated frames and audio to the processing/sending thread
        vae_to_blend_queue = queue.Queue(maxsize=self.batch_size * 2)
        gst_video_pipeline, gst_audio_pipeline, frame_processor_thread = None, None, None
        start_time = time.time()
        
        try:
            # 1. Setup GStreamer pipelines
            gst_video_pipeline = GStreamerPipeline(fps=target_fps)
            gst_audio_pipeline = GStreamerAudio()
            if not gst_video_pipeline.process or not gst_audio_pipeline.process:
                raise RuntimeError("GStreamer pipeline(s) failed to initialize.")

            # 2. Process audio input
            audio_reader = FFmpegAudioReader(audio_source)
            full_audio_pcm = audio_reader.read_full_audio()
            if full_audio_pcm is None or full_audio_pcm.size == 0:
                raise ValueError("PCM audio is empty after FFmpeg conversion.")
            
            # Use a temporary file for feature extraction if audio source is in memory
            feature_extraction_path = audio_source if isinstance(audio_source, str) else None
            temp_file_handle = None
            if not feature_extraction_path:
                temp_file_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".opus")
                temp_file_handle.write(audio_source)
                feature_extraction_path = temp_file_handle.name
                temp_file_handle.close() # Close handle so audio_processor can open it

            whisper_feature = audio_processor.audio2feat(feature_extraction_path)
            whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=target_fps)
            
            if temp_file_handle:
                os.unlink(feature_extraction_path)

            num_frames_to_generate = len(whisper_chunks)
            if num_frames_to_generate == 0:
                logging.warning("No frames to generate based on audio features. Skipping.")
                return
                
            num_vae_batches = (num_frames_to_generate + self.batch_size - 1) // self.batch_size
            logging.info(f"Audio processed. Planning {num_frames_to_generate} frames in {num_vae_batches} VAE batches.")

            # 3. Start the consumer thread (frame processing and sending)
            self.idx = 0 # Reset reference frame index for each run
            frame_processor_thread = threading.Thread(
                target=self.process_and_send_frames,
                args=(vae_to_blend_queue, gst_video_pipeline, gst_audio_pipeline, num_frames_to_generate),
                daemon=True, name=f"FrameProcessor_{run_id}"
            )
            frame_processor_thread.start()

            # 4. Main Generation Loop (Producer)
            data_gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
            total_audio_samples = len(full_audio_pcm)
            audio_samples_sent = 0

            for i, batch_data in enumerate(tqdm(data_gen, total=num_vae_batches, desc=f"VAE/UNET [{run_id}]")):
                if not frame_processor_thread.is_alive():
                    logging.error(f"Frame processor thread died unexpectedly. Halting generation.")
                    break
                if not batch_data or len(batch_data) != 2: continue
                
                whisper_batch, latent_batch = batch_data
                
                # --- Core AI Inference ---
                audio_feature = pe(torch.from_numpy(whisper_batch).to(device, dtype=unet.model.dtype))
                # The 'datagen' utility already stacks the latents into a batch tensor.
                # We just need to move this tensor to the correct device and set its data type.
                if not isinstance(latent_batch, torch.Tensor):
                    raise TypeError(f"The 'datagen' utility should yield a Tensor, but got {type(latent_batch)}")
                latent_input = latent_batch.to(device, dtype=unet.model.dtype)
                pred_latents = unet.model(latent_input, timesteps, encoder_hidden_states=audio_feature).sample
                vae_output = vae.decode_latents(pred_latents) # Returns list/np.array of frames
                # --- End Core AI ---

                if vae_output is None or len(vae_output) == 0: continue
                
                # --- DYNAMIC A/V SYNC ---
                # Calculate how much audio this batch of video frames represents
                num_frames_in_batch = len(vae_output)
                audio_duration_of_batch = num_frames_in_batch / target_fps
                num_audio_samples_for_batch = int(audio_duration_of_batch * gst_audio_pipeline.sample_rate)

                # Slice the exact audio chunk from the full PCM data
                start_audio_idx = audio_samples_sent
                end_audio_idx = min(start_audio_idx + num_audio_samples_for_batch, total_audio_samples)
                audio_chunk_pcm = full_audio_pcm[start_audio_idx:end_audio_idx]
                audio_samples_sent = end_audio_idx
                # --- End A/V SYNC ---
                
                try:
                    # Put the generated frames and their corresponding audio chunk into the queue
                    vae_to_blend_queue.put((list(vae_output), audio_chunk_pcm), timeout=5.0)
                except queue.Full:
                    logging.error(f"VAE-to-Blend queue is full. Consumer can't keep up. Halting generation.")
                    break

        except Exception as e:
            logging.critical(f"CRITICAL ERROR in inference run '{run_id}'.", exc_info=True)
        finally:
            logging.info(f"\n--- [{run_id}] Final Cleanup ---")
            # Signal consumer thread to finish
            if 'vae_to_blend_queue' in locals():
                try: vae_to_blend_queue.put(None) # Sentinel value
                except Exception: pass
            
            # Wait for consumer thread to finish its work
            if frame_processor_thread and frame_processor_thread.is_alive():
                logging.info("Waiting for frame processor thread to finish...")
                frame_processor_thread.join(timeout=15.0)
            
            # Stop GStreamer pipelines
            if gst_video_pipeline: gst_video_pipeline.stop()
            if gst_audio_pipeline: gst_audio_pipeline.stop()
            
            elapsed = time.time() - start_time
            logging.info(f">>> Inference run '{run_id}' finished in {elapsed:.2f}s. <<<")

    def process_and_send_frames(self, vae_to_blend_q, gst_video, gst_audio, total_frames_planned):
        # This is the consumer loop
        total_frames_processed = 0
        while total_frames_processed < total_frames_planned:
            
            # --- FRAME SKIPPING LOGIC ---
            # If the queue is getting too full, processing is falling behind.
            # Discard older frames to catch up to the most recent ones.
            while vae_to_blend_q.qsize() > FRAME_SKIP_THRESHOLD:
                try:
                    logging.warning(f"Queue high ({vae_to_blend_q.qsize()}). Skipping a frame batch to catch up.")
                    skipped_item = vae_to_blend_q.get_nowait()
                    if skipped_item is not None:
                        total_frames_processed += len(skipped_item[0]) # Add skipped frames to total
                    vae_to_blend_q.task_done()
                except queue.Empty:
                    break # Queue emptied, proceed normally
            # --- END FRAME SKIPPING ---

            try:
                # Block and wait for the next item from the generator
                batch_data = vae_to_blend_q.get(block=True, timeout=10.0)
            except queue.Empty:
                logging.error("Timeout waiting for frames from VAE. Ending processor thread.")
                break

            if batch_data is None: # Sentinel value means we're done
                logging.info("Received sentinel. Frame processor shutting down.")
                break
                
            vae_frames, audio_chunk = batch_data
            
            # Process each frame in the batch (simplified, no inner parallelization)
            for frame in vae_frames:
                blended_frame = self._blend_single_frame(frame, gst_video.width, gst_video.height)
                if blended_frame is not None:
                    if not gst_video.send_frame(blended_frame):
                        logging.error("Failed to send video frame to GStreamer. Stopping video sends.")
                        vae_to_blend_q.task_done()
                        return # Exit thread if pipe is broken
            
            # After sending all video frames for the batch, send the corresponding audio
            if audio_chunk is not None and audio_chunk.size > 0:
                if not gst_audio.send_audio(audio_chunk):
                    logging.error("Failed to send audio chunk to GStreamer.")

            total_frames_processed += len(vae_frames)
            vae_to_blend_q.task_done()
            
        logging.info("--- Frame processor finished its work. ---")
        
    def _blend_single_frame(self, res_frame, target_width, target_height):
        try:
            cycle_len = len(self.coord_list_cycle)
            if cycle_len == 0:
                logging.error("Cannot blend frame, avatar cycle data is empty.")
                return None

            current_idx = self.idx % cycle_len

            # Retrieve all necessary data for the current reference frame
            bbox = self.coord_list_cycle[current_idx]
            ori_frame = self.frame_list_cycle[current_idx]
            mask = self.mask_list_cycle[current_idx]

            # Check for corrupt/missing data before processing
            if mask is None or ori_frame is None or bbox is None:
                logging.warning(f"Skipping frame blend at index {self.idx} due to missing or corrupt reference data for cycle index {current_idx}.")
                self.idx += 1
                return None

            x, y, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            face_w, face_h = x1 - x, y1 - y

            if face_w <= 0 or face_h <= 0:
                self.idx += 1
                return None

            resized_face = cv2.resize(res_frame.astype(np.uint8), (face_w, face_h), interpolation=cv2.INTER_LINEAR)

            if mask.shape[:2] != (face_h, face_w):
                mask = cv2.resize(mask, (face_w, face_h), interpolation=cv2.INTER_LINEAR)

            # --- THIS IS THE CORRECTED LOGIC ---
            # First, check if the mask is a 3-channel BGR image.
            if len(mask.shape) == 3 and mask.shape[2] == 3:
                # If it is, convert it to grayscale.
                alpha_mask = (cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255.0)[..., np.newaxis]
            else:
                # Otherwise, assume it's already the 1-channel grayscale image we need.
                alpha_mask = (mask / 255.0)[..., np.newaxis]
            # --- END OF CORRECTION ---

            blended_frame = ori_frame.copy()
            face_region = blended_frame[y:y1, x:x1]

            # Ensure the alpha mask can be broadcast to the face region shape for blending
            if face_region.shape != alpha_mask.shape:
                alpha_mask = np.repeat(alpha_mask, 3, axis=2)

            blended_face = face_region.astype(np.float32) * (1.0 - alpha_mask) + resized_face.astype(np.float32) * alpha_mask
            blended_frame[y:y1, x:x1] = blended_face.astype(np.uint8)

            self.idx += 1
            return cv2.resize(blended_frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        except Exception as e:
            logging.error(f"Error blending frame at index {self.idx}: {e}", exc_info=True)
            self.idx += 1
            return None


# --- Main Application Logic ---
inference_lock = threading.Lock()
audio_queue = queue.Queue(maxsize=10)

def inference_worker(avatar, fps):
    """Worker thread that waits for audio data from a queue and runs inference."""
    logging.info("🚀 Inference worker started. Waiting for audio data...")
    while True:
        audio_data = audio_queue.get()
        if audio_data is None: # Shutdown signal
            break
        
        if not inference_lock.acquire(blocking=False):
            logging.warning("Inference already in progress. Skipping newly received audio.")
            audio_queue.task_done()
            continue
        
        try:
            logging.info(f"Inference lock acquired. Processing {len(audio_data)} bytes of audio.")
            avatar.inference(audio_data, fps)
        finally:
            inference_lock.release()
            audio_queue.task_done()
            logging.info("Inference lock released.")

def file_watcher(pipe_path):
    """Monitors a file for changes and puts its content into the audio queue."""
    logging.info(f"👀 Starting file watcher for: {pipe_path}")
    last_processed_mtime = 0
    while True:
        try:
            if os.path.isfile(pipe_path):
                current_mtime = os.path.getmtime(pipe_path)
                if current_mtime > last_processed_mtime:
                    logging.info("New audio file detected. Reading content...")
                    # Add a small delay and retry to handle partially written files
                    time.sleep(0.1)
                    with open(pipe_path, "rb") as f:
                        opus_data = f.read()
                    
                    if opus_data:
                        audio_queue.put(opus_data)
                        last_processed_mtime = current_mtime
                    else:
                        logging.warning("File modification detected, but file was empty.")
                        last_processed_mtime = current_mtime
            time.sleep(0.2) # Poll interval
        except FileNotFoundError:
            # This is okay, the file might be deleted and recreated
            last_processed_mtime = 0
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"Error in file watcher loop.", exc_info=True)
            time.sleep(2)


if __name__ == "__main__":
    logging.info("🎬 Starting Realtime Stream Sync Application...")

    # 1. Load environment variables from .env file
    # This must be at the top of the execution block.
    from dotenv import load_dotenv
    load_dotenv()

    # 2. Read configuration exclusively from the loaded environment variables
    AVATAR_CONFIG_PATH = os.getenv("AVATAR_CONFIG_PATH")
    AVATAR_ID_TO_USE = os.getenv("AVATAR_ID_TO_USE")
    TARGET_FPS = int(os.getenv("TARGET_FPS", "25"))
    STREAM_PIPE_PATH = os.getenv("STREAM_PIPE_PATH")

    main_avatar = None
    try:
        # 3. Validate that the required variables were found in the .env file
        if not all([AVATAR_CONFIG_PATH, AVATAR_ID_TO_USE, STREAM_PIPE_PATH]):
            raise ValueError("A required variable (AVATAR_CONFIG_PATH, AVATAR_ID_TO_USE, or STREAM_PIPE_PATH) is missing from your .env file.")

        logging.info(f"Attempting to load Avatar ID '{AVATAR_ID_TO_USE}' from config file '{AVATAR_CONFIG_PATH}'...")

        # 4. Load the YAML config and select the specified avatar
        config = OmegaConf.load(AVATAR_CONFIG_PATH)
        if AVATAR_ID_TO_USE not in config:
            raise ValueError(f"Avatar ID '{AVATAR_ID_TO_USE}' was not found as a key in the YAML file '{AVATAR_CONFIG_PATH}'")
        
        avatar_config = config[AVATAR_ID_TO_USE]

        # 5. Instantiate the Avatar with the loaded configuration
        main_avatar = Avatar(
            avatar_id=AVATAR_ID_TO_USE,
            video_path=avatar_config.video_path,
            bbox_shift=avatar_config.bbox_shift,
            batch_size=avatar_config.get("batch_size", 4),
            preparation=avatar_config.preparation
        )
    except Exception as e:
        logging.critical("❌ CRITICAL ERROR during setup.", exc_info=True)
        sys.exit(1)

    # 6. Start the watcher and inference worker threads
    inference_thread = threading.Thread(target=inference_worker, args=(main_avatar, TARGET_FPS), daemon=True)
    watcher_thread = threading.Thread(target=file_watcher, args=(STREAM_PIPE_PATH,), daemon=True)

    inference_thread.start()
    watcher_thread.start()

    # 7. Keep the main thread alive and handle graceful shutdown
    logging.info("✅ Application is running. Press Ctrl+C to shut down.")
    try:
        while True:
            if not watcher_thread.is_alive() or not inference_thread.is_alive():
                logging.error("A critical worker thread has died. Shutting down.")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        logging.info("\n🛑 KeyboardInterrupt received. Initiating graceful shutdown...")
    finally:
        audio_queue.put(None)
        if watcher_thread.is_alive(): watcher_thread.join(timeout=2)
        if inference_thread.is_alive(): inference_thread.join(timeout=30)
        logging.info("✅ Application shutdown complete.")