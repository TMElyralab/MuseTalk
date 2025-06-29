#!/usr/bin/env python3
"""
Real-time MuseTalk inference with web streaming
Streams generated frames to web browser in real-time
"""

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
from transformers import WhisperModel
import threading
import queue
import time
import base64
import io
from flask import Flask, render_template, Response, request, jsonify
from PIL import Image

# Import MuseTalk modules
sys.path.append('/workspace/MuseTalk')
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor

# Flask app for web streaming
app = Flask(__name__)

def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    """Extract frames from video"""
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break
    cap.release()

def osmakedirs(path_list):
    """Create directories if they don't exist"""
    for path in path_list:
        os.makedirs(path, exist_ok=True)

# Global variables for streaming
frame_queue = queue.Queue(maxsize=5)
is_generating = False
current_stats = {"frame": 0, "total": 0, "fps": 0}
avatar_instance = None
current_video_path = None
video_processed = False

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>MuseTalk Real-Time Stream</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; background: #1a1a1a; color: white; }
            #video { max-width: 90%; border: 2px solid #00ff00; margin: 20px; }
            #stats { margin: 20px; font-size: 18px; color: #00ff00; }
            .controls { margin: 20px; }
            .upload-section { background: #2a2a2a; padding: 20px; margin: 20px auto; max-width: 500px; border-radius: 10px; }
            button { padding: 10px 20px; font-size: 16px; margin: 5px; background: #00ff00; color: black; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #00cc00; }
            button:disabled { background: #666; cursor: not-allowed; }
            input[type="file"] { margin: 10px; padding: 10px; background: #3a3a3a; color: white; border: 1px solid #00ff00; border-radius: 5px; }
            #upload-status { margin: 10px; color: #ffff00; }
        </style>
    </head>
    <body>
        <h1>🎬 MuseTalk Real-Time Generation</h1>
        
        <div class="upload-section">
            <h2>🎬 Step 1: Upload Video</h2>
            <input type="file" id="videoFile" accept="video/*" />
            <button onclick="uploadVideo()" id="videoBtn">📤 Process Video</button>
            <div id="video-status">Upload a video of a person's face (processing takes 1-2 minutes)</div>
        </div>
        
        <div class="upload-section" id="audioSection" style="opacity: 0.5;">
            <h2>🎤 Step 2: Upload Audio</h2>
            <input type="file" id="audioFile" accept="audio/*" disabled />
            <button onclick="startGeneration()" id="generateBtn" disabled>🚀 Start Generation</button>
            <div id="upload-status">Video must be processed first (1-2 min)</div>
        </div>
        
        <div id="stats">Waiting for video upload...</div>
        <img id="video" src="/video_feed" alt="Real-time video stream will appear here">
        
        <div class="controls">
            <p>🎮 Instructions:</p>
            <p>1. Upload a video file (MP4, AVI, etc.) with a person's face</p>
            <p>2. Wait for video processing (face detection, landmarks)</p>
            <p>3. Upload an audio file (WAV, MP3, etc.)</p>
            <p>4. Click "Start Generation" to begin real-time lip-sync</p>
            <p>5. Watch the live stream as frames are generated!</p>
        </div>
        
        <script>
            let isGenerating = false;
            let videoProcessed = false;
            
            function uploadVideo() {
                const fileInput = document.getElementById('videoFile');
                const statusDiv = document.getElementById('video-status');
                const videoBtn = document.getElementById('videoBtn');
                
                if (!fileInput.files[0]) {
                    statusDiv.innerHTML = '❌ Please select a video file first!';
                    return;
                }
                
                const formData = new FormData();
                formData.append('video', fileInput.files[0]);
                
                statusDiv.innerHTML = '📤 Uploading video...';
                videoBtn.disabled = true;
                
                fetch('/upload_video', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        statusDiv.innerHTML = '⏳ Processing video... Please wait...';
                        // Don't enable audio section yet - wait for processing to complete
                    } else {
                        statusDiv.innerHTML = '❌ Error: ' + data.error;
                        videoBtn.disabled = false;
                    }
                })
                .catch(error => {
                    statusDiv.innerHTML = '❌ Upload failed: ' + error;
                    videoBtn.disabled = false;
                });
            }
            
            function enableAudioSection() {
                const audioSection = document.getElementById('audioSection');
                const audioFile = document.getElementById('audioFile');
                const generateBtn = document.getElementById('generateBtn');
                const uploadStatus = document.getElementById('upload-status');
                
                audioSection.style.opacity = '1.0';
                audioFile.disabled = false;
                generateBtn.disabled = false;
                uploadStatus.innerHTML = '🎤 Ready for audio upload!';
                
                document.getElementById('stats').innerHTML = 'Video ready! Upload audio to start generation...';
            }
            
            function startGeneration() {
                const fileInput = document.getElementById('audioFile');
                const statusDiv = document.getElementById('upload-status');
                const generateBtn = document.getElementById('generateBtn');
                
                if (!videoProcessed) {
                    statusDiv.innerHTML = '❌ Please upload and process a video first!';
                    return;
                }
                
                if (!fileInput.files[0]) {
                    statusDiv.innerHTML = '❌ Please select an audio file first!';
                    return;
                }
                
                if (isGenerating) {
                    statusDiv.innerHTML = '⚠️ Generation already in progress!';
                    return;
                }
                
                const formData = new FormData();
                formData.append('audio', fileInput.files[0]);
                
                statusDiv.innerHTML = '📤 Uploading audio...';
                generateBtn.disabled = true;
                isGenerating = true;
                
                fetch('/upload_audio', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        statusDiv.innerHTML = '🎬 Generation started! Watch the stream below...';
                    } else {
                        statusDiv.innerHTML = '❌ Error: ' + data.error;
                        generateBtn.disabled = false;
                        isGenerating = false;
                    }
                })
                .catch(error => {
                    statusDiv.innerHTML = '❌ Upload failed: ' + error;
                    generateBtn.disabled = false;
                    isGenerating = false;
                });
            }
            
            // Auto-refresh stats every second
            setInterval(function() {
                fetch('/stats')
                    .then(response => response.json())
                    .then(data => {
                        const statsDiv = document.getElementById('stats');
                        
                        // Check if video processing just completed
                        if (data.video_processed && !videoProcessed) {
                            videoProcessed = true;
                            enableAudioSection();
                            document.getElementById('video-status').innerHTML = '✅ Video processed! Now upload audio...';
                            document.getElementById('videoBtn').disabled = false;
                        }
                        
                        if (data.status === "Ready") {
                            if (videoProcessed) {
                                statsDiv.innerHTML = 'Ready for next generation...';
                                document.getElementById('generateBtn').disabled = false;
                            } else {
                                statsDiv.innerHTML = 'Upload a video to get started...';
                            }
                            isGenerating = false;
                        } else if (data.status === "Processing Video") {
                            statsDiv.innerHTML = 'Processing video: extracting faces and landmarks...';
                        } else {
                            statsDiv.innerHTML = 
                                `Frame: ${data.frame}/${data.total} | Generation FPS: ${data.fps.toFixed(2)} | Status: ${data.status}`;
                        }
                    });
            }, 1000);
        </script>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                # Get frame from queue
                frame_data = frame_queue.get(timeout=1.0)
                if frame_data is None:  # End signal
                    break
                    
                frame, frame_idx, total_frames = frame_data
                
                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except queue.Empty:
                # Send a black frame if no new frames
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "Waiting for frames...", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', black_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    global current_stats, is_generating, video_processed
    
    if is_generating and not video_processed:
        status = "Processing Video"
    elif is_generating:
        status = "Generating..."
    else:
        status = "Ready"
    
    return {
        "frame": current_stats["frame"],
        "total": current_stats["total"],
        "fps": current_stats["fps"],
        "status": status,
        "video_processed": video_processed
    }

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global avatar_instance, current_video_path, video_processed, current_stats, is_generating
    
    try:
        if 'video' not in request.files:
            return jsonify({"success": False, "error": "No video file provided"})
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({"success": False, "error": "No video file selected"})
        
        # Save uploaded video
        upload_dir = "/tmp/musetalk_uploads"
        os.makedirs(upload_dir, exist_ok=True)
        video_path = os.path.join(upload_dir, f"video_{int(time.time())}.mp4")
        file.save(video_path)
        current_video_path = video_path
        
        # Update status
        current_stats = {"frame": 0, "total": 0, "fps": 0}
        is_generating = True  # Use this to show "Processing Video" status
        
        # Process video in background thread
        def process_video():
            global avatar_instance, video_processed, is_generating
            try:
                print(f"🎬 Processing uploaded video: {video_path}")
                
                # Create new avatar with uploaded video
                avatar_id = f"uploaded_{int(time.time())}"
                avatar_instance = Avatar(
                    avatar_id=avatar_id,
                    video_path=video_path,
                    bbox_shift=0,
                    batch_size=4,
                    preparation=True  # Process the new video
                )
                
                video_processed = True
                is_generating = False
                print(f"✅ Video processing complete for: {video_path}")
                
            except Exception as e:
                print(f"❌ Video processing error: {e}")
                video_processed = False
                is_generating = False
        
        processing_thread = threading.Thread(target=process_video)
        processing_thread.daemon = True
        processing_thread.start()
        
        return jsonify({"success": True, "message": "Video processing started"})
        
    except Exception as e:
        is_generating = False
        return jsonify({"success": False, "error": str(e)})

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    global avatar_instance, video_processed
    
    try:
        if not video_processed or avatar_instance is None:
            return jsonify({"success": False, "error": "Please upload and process a video first"})
            
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"})
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({"success": False, "error": "No audio file selected"})
        
        # Save uploaded file
        upload_dir = "/tmp/musetalk_uploads"
        os.makedirs(upload_dir, exist_ok=True)
        audio_path = os.path.join(upload_dir, f"audio_{int(time.time())}.wav")
        file.save(audio_path)
        
        # Start generation in background thread
        def run_generation():
            try:
                avatar_instance.inference_web_stream(audio_path, fps=15)
            except Exception as e:
                print(f"Generation error: {e}")
            finally:
                # Clean up uploaded file
                try:
                    os.remove(audio_path)
                except:
                    pass
        
        generation_thread = threading.Thread(target=run_generation)
        generation_thread.daemon = True
        generation_thread.start()
        
        return jsonify({"success": True, "message": "Generation started"})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.base_path = f"./results/v15/avatars/{avatar_id}"
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.batch_size = batch_size
        self.idx = 0
        
        if preparation:
            self.prepare_material()
        else:
            self.load_existing_avatar()

    def load_existing_avatar(self):
        """Load pre-computed avatar data"""
        print("Loading existing avatar data...")
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        
        # Load images
        full_imgs_path = f"{self.avatar_path}/full_imgs"
        input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)
        
        # Load masks
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        mask_out_path = f"{self.avatar_path}/mask"
        input_mask_list = glob.glob(os.path.join(mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(input_mask_list)
        print("Avatar data loaded successfully!")

    def prepare_material(self):
        """Process uploaded video to create avatar materials"""
        print(f"🎬 Processing video: {self.video_path}")
        
        # Create directories
        osmakedirs([self.avatar_path, self.full_imgs_path, self.mask_out_path])
        
        # Extract frames from video
        print("📹 Extracting frames from video...")
        video2imgs(self.video_path, self.full_imgs_path, ext='png')
        
        # Get image list
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        print(f"📊 Extracted {len(input_img_list)} frames")
        
        # Extract landmarks and bounding boxes
        print("🎯 Extracting landmarks and bounding boxes...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        
        # Create latent representations
        print("🧠 Creating latent representations...")
        input_latent_list = []
        idx = -1
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            # Add extra margin for v15
            y2 = y2 + 10  # extra_margin
            y2 = min(y2, frame.shape[0])
            coord_list[idx] = [x1, y1, x2, y2]
            
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)
        
        # Create cyclic lists
        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []
        
        # Generate masks for each frame
        print("🎭 Generating masks...")
        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)
            
            x1, y1, x2, y2 = self.coord_list_cycle[i]
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode="jaw")
            
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle.append(crop_box)
            self.mask_list_cycle.append(mask)
        
        # Save processed data
        print("💾 Saving processed data...")
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)
        
        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
        
        torch.save(self.input_latent_list_cycle, self.latents_out_path)
        
        print(f"✅ Video processing complete! Created {len(self.frame_list_cycle)} frames")

    def process_frames_web_stream(self, res_frame_queue, video_len, fps):
        """Process frames and send to web stream"""
        global frame_queue, current_stats, is_generating
        
        print(f"Starting web stream for {video_len} frames at {fps} FPS")
        is_generating = True
        
        frame_count = 0
        start_time = time.time()
        frames_processed = 0
        
        while frames_processed < video_len:
            try:
                res_frame = res_frame_queue.get(block=True, timeout=3)
                
                # Process frame
                bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
                ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
                x1, y1, x2, y2 = bbox
                
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except:
                    self.idx += 1
                    frames_processed += 1
                    continue
                
                # Blend frames
                mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
                mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
                combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                
                # Convert to BGR for web display
                display_frame = cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR)
                
                # Add overlay info
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                
                info_text = f"Frame: {frames_processed + 1}/{video_len} | FPS: {current_fps:.1f}"
                cv2.putText(display_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Update global stats
                current_stats.update({
                    "frame": frames_processed + 1,
                    "total": video_len,
                    "fps": current_fps
                })
                
                # Send to web stream (non-blocking)
                try:
                    frame_queue.put((display_frame, frames_processed + 1, video_len), timeout=0.1)
                except queue.Full:
                    # Remove old frame and add new one
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        frame_queue.put((display_frame, frames_processed + 1, video_len), timeout=0.1)
                    except queue.Full:
                        pass  # Skip this frame if still full
                
                self.idx += 1
                frames_processed += 1
                frame_count += 1
                
                # Print progress
                if frame_count % 25 == 0:
                    print(f"🎬 Streamed {frame_count} frames, Current FPS: {current_fps:.2f}")
                
            except queue.Empty:
                print(f"⏰ Timeout waiting for frame {frames_processed + 1}/{video_len}")
                # If we timeout but haven't processed all frames, something went wrong
                if frames_processed < video_len:
                    print(f"⚠️  Generation may have stopped early. Expected {video_len}, got {frames_processed}")
                    break
        
        # Send end signal
        frame_queue.put(None)
        is_generating = False
        print(f"✅ Web streaming finished. Processed: {frames_processed}/{video_len} frames, Displayed: {frame_count}")

    def inference_web_stream(self, audio_path, fps=15):
        """Real-time inference with web streaming"""
        global current_stats, is_generating
        
        print("🌐 Starting web stream inference!")
        
        # Reset stats
        current_stats = {"frame": 0, "total": 0, "fps": 0}
        
        # Extract audio features
        start_time = time.time()
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path, weight_dtype=weight_dtype)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features, device, weight_dtype, whisper,
            librosa_length, fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
        print(f"Audio processing: {(time.time() - start_time) * 1000:.0f}ms")
        
        # Setup streaming
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue(maxsize=20)  # Increased buffer size
        self.idx = 0
        
        current_stats["total"] = video_num
        
        # Start streaming thread
        stream_thread = threading.Thread(
            target=self.process_frames_web_stream,
            args=(res_frame_queue, video_num, fps)
        )
        stream_thread.start()
        
        # Generate frames
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        
        print(f"🚀 Generating {video_num} frames for web stream...")
        generated_count = 0
        
        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            audio_feature_batch = pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)
            
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
            recon = vae.decode_latents(pred_latents)
            
            for res_frame in recon:
                # Ensure we don't exceed expected frame count
                if generated_count >= video_num:
                    break
                    
                # Try to put frame in queue, wait if full
                while True:
                    try:
                        res_frame_queue.put(res_frame, timeout=0.5)
                        generated_count += 1
                        break
                    except queue.Full:
                        # If queue is full, wait a bit and try again
                        time.sleep(0.1)
                        continue
            
            # Break if we've generated enough frames
            if generated_count >= video_num:
                break
        
        print(f"📊 Generated {generated_count}/{video_num} frames")
        stream_thread.join()
        print("🎬 Web streaming complete!")

# Global variables for models
device = None
vae = unet = pe = None
audio_processor = whisper = fp = None
weight_dtype = None
timesteps = None

def initialize_models():
    """Initialize all models"""
    global device, vae, unet, pe, audio_processor, whisper, fp, weight_dtype, timesteps
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    vae, unet, pe = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        device=device
    )
    timesteps = torch.tensor([0], device=device)
    
    # Use half precision
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    weight_dtype = unet.model.dtype
    
    # Initialize processors
    audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
    whisper = WhisperModel.from_pretrained("./models/whisper")
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    
    fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    
    print("✅ All models initialized!")

def start_web_stream():
    """Start the web streaming demo"""
    global avatar_instance
    
    print("🌐 Starting MuseTalk Interactive Web Stream Server...")
    
    port = 7777
    # Initialize models
    initialize_models()
    
    # Avatar will be created when user uploads video
    avatar_instance = None
    print("⏳ Waiting for user to upload video...")
    
    print("🌐 Web server starting at:")
    print(f"   - Local: http://localhost:{port}") 
    print(f"   - Network: http://0.0.0.0:{port}")
    print("🎤 Interactive mode: Upload audio files to start generation!")
    print(f"💡 For external access, use: http://YOUR-SERVER-IP:{port}")
    print("📋 For SSH port forwarding: ssh -L 7777:localhost:7777 user@server")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == "__main__":
    start_web_stream() 