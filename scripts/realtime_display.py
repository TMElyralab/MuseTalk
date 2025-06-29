#!/usr/bin/env python3
"""
Real-time MuseTalk inference with live display
Modified from realtime_inference.py to show frames as they're generated
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

from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor

import shutil
import threading
import queue
import time
import subprocess


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
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


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        # Version-based path setup
        if args.version == "v15":
            self.base_path = f"./results/{args.version}/avatars/{avatar_id}"
        else:  # v1
            self.base_path = f"./results/avatars/{avatar_id}"
            
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": args.version
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        self.init()

    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n)")
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.load_existing_avatar()
            else:
                print("*********************************")
                print(f"  creating avator: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()
            self.load_existing_avatar()

    def load_existing_avatar(self):
        """Load pre-computed avatar data"""
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(input_mask_list)

    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if args.version == "v15":
                y2 = y2 + args.extra_margin
                y2 = min(y2, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            x1, y1, x2, y2 = self.coord_list_cycle[i]
            if args.version == "v15":
                mode = args.parsing_mode
            else:
                mode = "raw"
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)

            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

    def process_frames_live_display(self, res_frame_queue, video_len, fps):
        """Process frames with live OpenCV display"""
        print(f"Starting live display for {video_len} frames at {fps} FPS")
        
        # Calculate frame delay for target FPS
        frame_delay = 1.0 / fps
        
        cv2.namedWindow('MuseTalk Real-Time', cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0
        start_display_time = time.time()
        
        while True:
            if self.idx >= video_len - 1:
                break
                
            try:
                # Get generated frame from queue
                res_frame = res_frame_queue.get(block=True, timeout=1)
                frame_start_time = time.time()
                
                # Get original frame and coordinates
                bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
                ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
                x1, y1, x2, y2 = bbox
                
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except:
                    continue
                    
                # Blend with original frame
                mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
                mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
                combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                
                # Convert BGR to RGB for display (OpenCV uses BGR)
                display_frame = cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR)
                
                # Add frame info overlay
                fps_text = f"Frame: {self.idx + 1}/{video_len} | Target FPS: {fps}"
                cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('MuseTalk Real-Time', display_frame)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC to quit
                    print("Live display stopped by user")
                    break
                
                # Frame timing
                processing_time = time.time() - frame_start_time
                if processing_time < frame_delay:
                    time.sleep(frame_delay - processing_time)
                
                self.idx += 1
                frame_count += 1
                
                # Print performance stats every 25 frames
                if frame_count % 25 == 0:
                    elapsed = time.time() - start_display_time
                    actual_fps = frame_count / elapsed
                    print(f"Processed {frame_count} frames, Actual FPS: {actual_fps:.2f}")
                
            except queue.Empty:
                continue
                
        cv2.destroyAllWindows()
        print(f"Live display finished. Processed {frame_count} frames.")

    def inference_live(self, audio_path, fps=25):
        """Real-time inference with live display"""
        print("🎬 Starting LIVE real-time inference with display!")
        print("Controls: Press 'q' or ESC to stop")
        
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path, weight_dtype=weight_dtype)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
        print(f"Audio processing took: {(time.time() - start_time) * 1000:.0f}ms")
        
        ############################################## real-time inference with display ##############################################
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue(maxsize=10)  # Limit queue size for real-time feel
        self.idx = 0
        
        # Start display thread
        display_thread = threading.Thread(
            target=self.process_frames_live_display, 
            args=(res_frame_queue, video_num, fps)
        )
        display_thread.start()
        
        # Generate frames
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        generation_start_time = time.time()
        
        print(f"🚀 Generating {video_num} frames in real-time...")
        
        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            audio_feature_batch = pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)
            
            # Generate frames
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
            recon = vae.decode_latents(pred_latents)
            
            # Add frames to display queue
            for res_frame in recon:
                try:
                    res_frame_queue.put(res_frame, timeout=0.1)
                except queue.Full:
                    # Skip frame if display can't keep up
                    print("⚠️  Display queue full, skipping frame for real-time feel")
                    pass
        
        # Wait for display to finish
        display_thread.join()
        
        total_time = time.time() - generation_start_time
        actual_fps = video_num / total_time
        print(f"✅ Generation complete!")
        print(f"📊 Performance: {video_num} frames in {total_time:.2f}s = {actual_fps:.2f} FPS")


if __name__ == "__main__":
    print("🎬 MuseTalk Real-Time Display Demo")
    print("This shows generated frames as they're created!")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--vae_type", type=str, default="sd-vae")
    parser.add_argument("--unet_config", type=str, default="./models/musetalkV15/musetalk.json")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper")
    parser.add_argument("--inference_config", type=str, default="configs/inference/realtime.yaml")
    parser.add_argument("--extra_margin", type=int, default=10)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--audio_padding_length_left", type=int, default=2)
    parser.add_argument("--audio_padding_length_right", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)  # Smaller for more responsive real-time
    parser.add_argument("--parsing_mode", default='jaw')
    parser.add_argument("--left_cheek_width", type=int, default=90)
    parser.add_argument("--right_cheek_width", type=int, default=90)

    args = parser.parse_args()

    # Set computing device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)

    # Use half precision for speed
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    # Initialize processors
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # Initialize face parser
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width
    )

    # Load config and run
    inference_config = OmegaConf.load(args.inference_config)
    
    for avatar_id in inference_config:
        data_preparation = inference_config[avatar_id]["preparation"]
        video_path = inference_config[avatar_id]["video_path"]
        
        # Create avatar (preparation=False for live demo, assuming it's already prepared)
        avatar = Avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=0,
            batch_size=args.batch_size,
            preparation=False  # Set to False for live demo
        )

        audio_clips = inference_config[avatar_id]["audio_clips"]
        for audio_num, audio_path in audio_clips.items():
            print(f"\n🎤 Processing audio: {audio_path}")
            avatar.inference_live(audio_path, fps=args.fps)
            
            response = input("\n Continue with next audio clip? (y/n): ")
            if response.lower() != 'y':
                break
    
    print("🎬 Real-time demo complete!") 