#!/usr/bin/env python3
"""
Real-time MuseTalk inference with Gradio web interface
Streams generated frames to web browser in real-time using Gradio
"""

import argparse
import os
import sys
import numpy as np
import cv2
import torch
import glob
import pickle
import copy
import json
import threading
import queue
import time
import gradio as gr
from PIL import Image
from tqdm import tqdm
import imageio
from transformers import WhisperModel

# Import MuseTalk modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.audio_processor import AudioProcessor

# Global variables for streaming
is_generating = False
current_stats = {"frame": 0, "total": 0, "fps": 0, "status": "Ready"}
avatar_instance = None
current_frame = None
all_frames = []
frame_buffer = []
temp_files = []
buffer_lock = threading.Lock()

# Global variables for models
device = None
vae = unet = pe = None
audio_processor = whisper = fp = None
weight_dtype = None
timesteps = None

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

def create_video_chunk(frames, fps, chunk_id):
    """Create a video chunk from frames for streaming"""
    global temp_files
    
    if not frames:
        return None
    
    # Create directory in Gradio-accessible location
    chunk_dir = "./results/streaming_chunks"
    os.makedirs(chunk_dir, exist_ok=True)
    
    # Try MP4 format first (more compatible)
    chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_id}_{int(time.time()*1000)}.mp4")
    
    try:
        # Convert frames to proper format if needed
        processed_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                # Ensure RGB format and proper data type
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    processed_frames.append(frame)
                else:
                    print(f"Warning: Skipping frame with shape {frame.shape}")
            else:
                print(f"Warning: Frame is not numpy array: {type(frame)}")
        
        if not processed_frames:
            print("No valid frames to write")
            return None
        
        # Write video chunk with simpler parameters
        imageio.mimwrite(
            chunk_path, 
            processed_frames, 
            fps=fps, 
            codec='libx264',
            quality=8,
            pixelformat='yuv420p'
        )
        
        # Track temp file for cleanup
        temp_files.append(chunk_path)
        
        return chunk_path
    except Exception as e:
        print(f"Error creating video chunk: {e}")
        # Try alternative approach with opencv
        try:
            return create_video_chunk_opencv(frames, fps, chunk_id)
        except Exception as e2:
            print(f"OpenCV fallback also failed: {e2}")
            return None

def create_video_chunk_opencv(frames, fps, chunk_id):
    """Fallback method using OpenCV to create video chunks"""
    global temp_files
    
    if not frames:
        return None
    
    chunk_dir = "./results/streaming_chunks"
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_path = os.path.join(chunk_dir, f"chunk_cv_{chunk_id}_{int(time.time()*1000)}.mp4")
    
    # Get frame dimensions
    first_frame = frames[0]
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(chunk_path, fourcc, fps, (width, height))
    
    for frame in frames:
        if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
    
    out.release()
    
    # Track temp file for cleanup
    temp_files.append(chunk_path)
    
    return chunk_path

def cleanup_temp_files():
    """Clean up temporary video chunk files"""
    global temp_files
    
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error removing temp file {file_path}: {e}")
    
    temp_files.clear()
    
    # Also clean up chunk directory if empty
    chunk_dir = "./results/streaming_chunks"
    try:
        if os.path.exists(chunk_dir) and not os.listdir(chunk_dir):
            os.rmdir(chunk_dir)
    except:
        pass

def delayed_cleanup():
    """Clean up files after a delay to let Gradio serve them"""
    def cleanup_after_delay():
        time.sleep(10)  # Wait 10 seconds
        cleanup_temp_files()
        print("🧹 Cleaned up temporary streaming files")
    
    cleanup_thread = threading.Thread(target=cleanup_after_delay, daemon=True)
    cleanup_thread.start()

class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, extra_margin=10, 
                 parsing_mode="jaw", left_cheek_width=90, right_cheek_width=90):
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
        self.extra_margin = extra_margin
        self.parsing_mode = parsing_mode
        self.left_cheek_width = left_cheek_width
        self.right_cheek_width = right_cheek_width
        
        # Always prepare material for new avatars
        self.prepare_material()

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
        coord_placeholder_val = (0.0, 0.0, 0.0, 0.0)
        
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder_val:
                continue
            x1, y1, x2, y2 = bbox
            # Add extra margin
            y2 = y2 + self.extra_margin
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
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=self.parsing_mode)
            
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

    def inference_stream(self, audio_path, fps=15, progress_callback=None):
        """Real-time inference with streaming"""
        global current_stats, is_generating, current_frame, all_frames, frame_buffer
        
        print("🌐 Starting stream inference!")
        
        # Reset stats
        current_stats = {"frame": 0, "total": 0, "fps": 0, "status": "Processing audio..."}
        is_generating = True
        all_frames = []
        
        # Thread-safe buffer reset
        with buffer_lock:
            frame_buffer.clear()
        
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
        self.idx = 0
        
        current_stats["total"] = video_num
        current_stats["status"] = "Generating..."
        
        # Generate frames
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        
        print(f"🚀 Generating {video_num} frames...")
        generated_count = 0
        frame_start_time = time.time()
        
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
                
                # Process frame
                bbox = self.coord_list_cycle[self.idx % len(self.coord_list_cycle)]
                ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % len(self.frame_list_cycle)])
                x1, y1, x2, y2 = bbox
                
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except:
                    self.idx += 1
                    generated_count += 1
                    continue
                
                # Blend frames
                mask = self.mask_list_cycle[self.idx % len(self.mask_list_cycle)]
                mask_crop_box = self.mask_coords_list_cycle[self.idx % len(self.mask_coords_list_cycle)]
                combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                
                # Update current frame and store
                current_frame = combine_frame
                all_frames.append(combine_frame)
                
                # Thread-safe buffer update
                with buffer_lock:
                    frame_buffer.append(combine_frame)
                
                # Update stats
                elapsed = time.time() - frame_start_time
                current_fps = generated_count / elapsed if elapsed > 0 else 0
                current_stats.update({
                    "frame": generated_count + 1,
                    "fps": current_fps,
                    "status": "Generating..."
                })
                
                # Update progress
                if progress_callback:
                    progress_callback(generated_count / video_num, f"Frame {generated_count + 1}/{video_num}")
                
                self.idx += 1
                generated_count += 1
            
            # Break if we've generated enough frames
            if generated_count >= video_num:
                break
        
        # Update final stats
        current_stats["status"] = "Complete"
        is_generating = False
        print(f"✅ Generation complete! Generated {generated_count}/{video_num} frames")
        
        return generated_count

def initialize_models(use_float16=True):
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
    
    # Use half precision if requested
    if use_float16:
        pe = pe.half().to(device)
        vae.vae = vae.vae.half().to(device)
        unet.model = unet.model.half().to(device)
        weight_dtype = torch.float16
    else:
        pe = pe.to(device)
        vae.vae = vae.vae.to(device)
        unet.model = unet.model.to(device)
        weight_dtype = torch.float32
    
    # Initialize processors
    audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
    whisper = WhisperModel.from_pretrained("./models/whisper")
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    
    fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    
    print("✅ All models initialized!")

def process_video(video_path, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width):
    """Process uploaded video"""
    global avatar_instance, fp
    
    try:
        # Update face parsing parameters
        fp = FaceParsing(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)
        
        # Create new avatar
        avatar_id = f"uploaded_{int(time.time())}"
        avatar_instance = Avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=bbox_shift,
            batch_size=4,
            extra_margin=extra_margin,
            parsing_mode=parsing_mode,
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width
        )
        
        return "✅ Video processed successfully! Now upload an audio file to start generation."
    except Exception as e:
        return f"❌ Error processing video: {str(e)}"

def generate_video_stream(audio_path, fps, progress=gr.Progress()):
    """Generate lip-sync video with streaming"""
    global avatar_instance, current_stats, is_generating, frame_buffer
    
    if avatar_instance is None:
        yield None, "❌ Please upload and process a video first!", "🔴 Error"
        return
    
    if is_generating:
        yield None, "⚠️ Generation already in progress!", "🟡 Busy"
        return
    
    try:
        # Clean up any previous temp files
        cleanup_temp_files()
        
        # Start generation in background thread
        generation_thread = threading.Thread(
            target=avatar_instance.inference_stream,
            args=(audio_path, fps),
            daemon=True
        )
        generation_thread.start()
        
        # Stream video chunks as they're generated
        chunk_id = 0
        frames_per_chunk = max(int(fps * 1.0), 10)  # 1 second chunks, minimum 10 frames
        last_buffer_size = 0
        
        while is_generating or len(frame_buffer) > last_buffer_size:
            # Thread-safe buffer access
            with buffer_lock:
                current_buffer_size = len(frame_buffer)
                
                # Check if we have enough frames for a chunk
                if current_buffer_size >= frames_per_chunk:
                    # Extract frames for chunk
                    chunk_frames = frame_buffer[:frames_per_chunk]
                    frame_buffer[:] = frame_buffer[frames_per_chunk:]  # In-place modification
                    
                    print(f"📹 Creating chunk {chunk_id} with {len(chunk_frames)} frames")
                    
                    # Create video chunk outside the lock
                    chunk_path = create_video_chunk(chunk_frames, fps, chunk_id)
                    if chunk_path:
                        status = f"🎬 Frame {current_stats['frame']}/{current_stats['total']} | FPS: {current_stats['fps']:.1f}"
                        print(f"✅ Yielding chunk: {chunk_path}")
                        yield chunk_path, "", status
                        chunk_id += 1
                    else:
                        print("❌ Failed to create video chunk")
                    
                    last_buffer_size = len(frame_buffer)
                else:
                    last_buffer_size = current_buffer_size
            
            time.sleep(0.2)  # Check for updates every 200ms
        
        # Process remaining frames in buffer
        with buffer_lock:
            remaining_frames = frame_buffer.copy()
            frame_buffer.clear()
        
        if len(remaining_frames) > 0:
            print(f"📹 Creating final chunk with {len(remaining_frames)} frames")
            chunk_path = create_video_chunk(remaining_frames, fps, chunk_id)
            if chunk_path:
                status = f"✅ Complete - {current_stats['frame']} frames"
                yield chunk_path, f"✅ Generation complete! Generated {current_stats['frame']} frames.", status
        
        # Schedule delayed cleanup to let Gradio serve the files
        delayed_cleanup()
        print("🎬 Streaming completed. Files will be cleaned up in 10 seconds.")
        
    except Exception as e:
        current_stats["status"] = "Error"
        print(f"❌ Streaming error: {e}")
        yield None, f"❌ Generation error: {str(e)}", "🔴 Error"

def save_video(fps):
    """Save generated frames as video"""
    global all_frames
    
    if not all_frames:
        return None, "No frames to save!"
    
    try:
        output_path = f"./results/output/generated_{int(time.time())}.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert frames to video
        imageio.mimwrite(output_path, all_frames, fps=fps, codec='h264', quality=8)
        
        return output_path, f"✅ Video saved to: {output_path}"
    except Exception as e:
        return None, f"❌ Error saving video: {str(e)}"

def create_gradio_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="MuseTalk Real-Time Stream") as demo:
        gr.Markdown(
            """
            # 🎬 MuseTalk Real-Time Video Generation
            
            Generate lip-sync videos with real-time streaming preview!
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📹 Step 1: Upload Reference Video")
                video_input = gr.Video(label="Reference Video", sources=['upload'])
                
                with gr.Accordion("Advanced Parameters", open=False):
                    bbox_shift = gr.Number(label="BBox Shift (px)", value=0)
                    extra_margin = gr.Slider(label="Extra Margin", minimum=0, maximum=40, value=10, step=1)
                    parsing_mode = gr.Radio(label="Parsing Mode", choices=["jaw", "raw"], value="jaw")
                    left_cheek_width = gr.Slider(label="Left Cheek Width", minimum=20, maximum=160, value=90, step=5)
                    right_cheek_width = gr.Slider(label="Right Cheek Width", minimum=20, maximum=160, value=90, step=5)
                
                process_btn = gr.Button("Process Video", variant="primary")
                process_status = gr.Textbox(label="Processing Status", interactive=False)
                
                gr.Markdown("### 🎤 Step 2: Upload Audio")
                audio_input = gr.Audio(label="Driving Audio", type="filepath")
                fps_slider = gr.Slider(label="Generation FPS", minimum=10, maximum=30, value=15, step=1)
                
                generate_btn = gr.Button("Start Generation", variant="primary")
                
                gr.Markdown("### 💾 Step 3: Save Result")
                save_btn = gr.Button("Save as Video", variant="secondary")
                save_status = gr.Textbox(label="Save Status", interactive=False)
            
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Real-Time Preview")
                status_display = gr.Textbox(label="Status", value="🟢 Ready to start", interactive=False)
                video_output = gr.Video(label="Live Stream", streaming=True, autoplay=True)
                generation_status = gr.Textbox(label="Generation Log", interactive=False)
        
        # Event handlers
        process_btn.click(
            fn=process_video,
            inputs=[video_input, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
            outputs=[process_status]
        )
        
        # Streaming generation
        generate_btn.click(
            fn=generate_video_stream,
            inputs=[audio_input, fps_slider],
            outputs=[video_output, generation_status, status_display]
        )
        
        save_btn.click(
            fn=save_video,
            inputs=[fps_slider],
            outputs=[gr.File(label="Download Video"), save_status]
        )
        
        gr.Markdown(
            """
            ### 📝 Instructions:
            1. Upload a video file with a person's face
            2. Click "Process Video" and wait for processing to complete
            3. Upload an audio file (WAV, MP3, etc.)
            4. Click "Start Generation" to begin real-time lip-sync
            5. Watch the live preview as frames are generated!
            6. Save the result as a video file when complete
            
            ### ⚙️ Tips:
            - Adjust parameters in "Advanced Parameters" for better results
            - Use 25 FPS videos for optimal quality
            - The preview updates in real-time as frames are generated
            - Processing may take 1-2 minutes for longer videos
            """
        )
    
    return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--use_float16", action="store_true", help="Use float16 for faster inference")
    args = parser.parse_args()
    
    print("🌐 Starting MuseTalk Real-Time Gradio Interface...")
    
    # Initialize models
    initialize_models(use_float16=args.use_float16)
    
    # Create and launch interface
    demo = create_gradio_interface()
    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        show_error=True
    )

if __name__ == "__main__":
    main()