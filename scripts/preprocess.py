import os
import argparse
import subprocess
from omegaconf import OmegaConf
from typing import Tuple, List, Union
import decord
import json
import cv2
from musetalk.utils.face_detection import FaceAlignment,LandmarksType
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import torch
import numpy as np
from tqdm import tqdm
import sys

def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

ffmpeg_path = "./ffmpeg-4.4-amd64-static/"
if not fast_check_ffmpeg():
    print("Adding ffmpeg to PATH")
    # Choose path separator based on operating system
    path_separator = ';' if sys.platform == 'win32' else ':'
    os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

class AnalyzeFace:
    def __init__(self, device: Union[str, torch.device], config_file: str, checkpoint_file: str):
        """
        Initialize the AnalyzeFace class with the given device, config file, and checkpoint file.

        Parameters:
        device (Union[str, torch.device]): The device to run the models on ('cuda' or 'cpu').
        config_file (str): Path to the mmpose model configuration file.
        checkpoint_file (str): Path to the mmpose model checkpoint file.
        """
        self.device = device
        self.dwpose = init_model(config_file, checkpoint_file, device=self.device)
        self.facedet = FaceAlignment(LandmarksType._2D, flip_input=False, device=self.device)

    def __call__(self, im: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Detect faces and keypoints in the given image.

        Parameters:
        im (np.ndarray): The input image.
        maxface (bool): Whether to detect the maximum face. Default is True.

        Returns:
        Tuple[List[np.ndarray], np.ndarray]: A tuple containing the bounding boxes and keypoints.
        """
        try:
            # Ensure the input image has the correct shape
            if im.ndim == 3:
                im = np.expand_dims(im, axis=0)
            elif im.ndim != 4 or im.shape[0] != 1:
                raise ValueError("Input image must have shape (1, H, W, C)")
            
            bbox = self.facedet.get_detections_for_batch(np.asarray(im))
            results = inference_topdown(self.dwpose, np.asarray(im)[0])
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_land_mark= keypoints[0][23:91]
            face_land_mark = face_land_mark.astype(np.int32)

            return face_land_mark, bbox
        
        except Exception as e:
            print(f"Error during face analysis: {e}")
            return np.array([]),[] 

def convert_video(org_path: str, dst_path: str, vid_list: List[str]) -> None:

    """
    Convert video files to a specified format and save them to the destination path.

    Parameters:
    org_path (str): The directory containing the original video files.
    dst_path (str): The directory where the converted video files will be saved.
    vid_list (List[str]): A list of video file names to process.

    Returns:
    None
    """
    for idx, vid in enumerate(vid_list):
        if vid.endswith('.mp4'):
            org_vid_path = os.path.join(org_path, vid)
            dst_vid_path = os.path.join(dst_path, vid)
                
            if org_vid_path != dst_vid_path:
                cmd = [
                    "ffmpeg", "-hide_banner", "-y", "-i", org_vid_path, 
                    "-r", "25", "-crf", "15", "-c:v", "libx264", 
                    "-pix_fmt", "yuv420p", dst_vid_path
                ]
                subprocess.run(cmd, check=True)

            if idx % 1000 == 0:
                print(f"### {idx} videos converted ###")

def segment_video(org_path: str, dst_path: str, vid_list: List[str], segment_duration: int = 30) -> None:
    """
    Segment video files into smaller clips of specified duration.

    Parameters:
    org_path (str): The directory containing the original video files.
    dst_path (str): The directory where the segmented video files will be saved.
    vid_list (List[str]): A list of video file names to process.
    segment_duration (int): The duration of each segment in seconds. Default is 30 seconds.

    Returns:
    None
    """
    for idx, vid in enumerate(vid_list):
        if vid.endswith('.mp4'):
            input_file = os.path.join(org_path, vid)
            original_filename = os.path.basename(input_file)

            command = [
                'ffmpeg', '-i', input_file, '-c', 'copy', '-map', '0',
                '-segment_time', str(segment_duration), '-f', 'segment',
                '-reset_timestamps', '1',
                os.path.join(dst_path, f'clip%03d_{original_filename}')
            ]

            subprocess.run(command, check=True)

def extract_audio(org_path: str, dst_path: str, vid_list: List[str]) -> None:
    """
    Extract audio from video files and save as WAV format.

    Parameters:
    org_path (str): The directory containing the original video files.
    dst_path (str): The directory where the extracted audio files will be saved.
    vid_list (List[str]): A list of video file names to process.

    Returns:
    None
    """
    for idx, vid in enumerate(vid_list):
        if vid.endswith('.mp4'):
            video_path = os.path.join(org_path, vid)
            audio_output_path = os.path.join(dst_path, os.path.splitext(vid)[0] + ".wav")
            try:
                command = [
                    'ffmpeg', '-hide_banner', '-y', '-i', video_path,
                    '-vn', '-acodec', 'pcm_s16le', '-f', 'wav',
                    '-ar', '16000', '-ac', '1', audio_output_path,
                ]
                
                subprocess.run(command, check=True)
                print(f"Audio saved to: {audio_output_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error extracting audio from {vid}: {e}")

def split_data(video_files: List[str], val_list_hdtf: List[str]) -> (List[str], List[str]):
    """
    Split video files into training and validation sets based on val_list_hdtf.

    Parameters:
    video_files (List[str]): A list of video file names.
    val_list_hdtf (List[str]): A list of validation file identifiers.

    Returns:
    (List[str], List[str]): A tuple containing the training and validation file lists.
    """
    val_files = [f for f in video_files if any(val_id in f for val_id in val_list_hdtf)]
    train_files = [f for f in video_files if f not in val_files]
    return train_files, val_files

def save_list_to_file(file_path: str, data_list: List[str]) -> None:
    """
    Save a list of strings to a file, each string on a new line.

    Parameters:
    file_path (str): The path to the file where the list will be saved.
    data_list (List[str]): The list of strings to save.

    Returns:
    None
    """
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(f"{item}\n")

def generate_train_list(cfg):
    train_file_path = cfg.video_clip_file_list_train
    val_file_path = cfg.video_clip_file_list_val
    val_list_hdtf = cfg.val_list_hdtf

    meta_list = os.listdir(cfg.meta_root)

    sorted_meta_list = sorted(meta_list)
    train_files, val_files = split_data(meta_list, val_list_hdtf)

    save_list_to_file(train_file_path, train_files)
    save_list_to_file(val_file_path, val_files)

    print(val_list_hdtf)    

def analyze_video(org_path: str, dst_path: str, vid_list: List[str]) -> None:
    """
    Convert video files to a specified format and save them to the destination path.

    Parameters:
    org_path (str): The directory containing the original video files.
    dst_path (str): The directory where the meta json will be saved.
    vid_list (List[str]): A list of video file names to process.

    Returns:
    None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
    checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'

    analyze_face = AnalyzeFace(device, config_file, checkpoint_file)
    
    for vid in tqdm(vid_list, desc="Processing videos"):
        #vid = "clip005_WDA_BernieSanders_000.mp4"
        #print(vid)
        if vid.endswith('.mp4'):
            vid_path = os.path.join(org_path, vid)
            wav_path = vid_path.replace(".mp4",".wav")
            vid_meta = os.path.join(dst_path, os.path.splitext(vid)[0] + ".json")
            if os.path.exists(vid_meta):
                continue
            print('process video {}'.format(vid))

            total_bbox_list = []
            total_pts_list = []
            isvalid = True

            # process
            try:
                cap = decord.VideoReader(vid_path, fault_tol=1)
            except Exception as e:
                print(e)
                continue

            total_frames = len(cap)
            for frame_idx in range(total_frames):
                frame = cap[frame_idx]
                if frame_idx==0:
                    video_height,video_width,_ = frame.shape
                frame_bgr = cv2.cvtColor(frame.asnumpy(), cv2.COLOR_BGR2RGB)
                pts_list, bbox_list = analyze_face(frame_bgr)

                if len(bbox_list)>0 and None not in bbox_list:
                    bbox = bbox_list[0]
                else:
                    isvalid = False
                    bbox = []
                    print(f"set isvalid to False as broken img in {frame_idx} of {vid}")
                    break

                #print(pts_list)
                if len(pts_list)>0 and pts_list is not None:
                    pts = pts_list.tolist()
                else:
                    isvalid = False
                    pts = []
                    break

                if frame_idx==0:
                    x1,y1,x2,y2 = bbox 
                    face_height, face_width = y2-y1,x2-x1

                total_pts_list.append(pts)
                total_bbox_list.append(bbox)

            meta_data = {
                    "mp4_path": vid_path,
                     "wav_path": wav_path,
                     "video_size": [video_height, video_width],
                     "face_size": [face_height, face_width],
                     "frames": total_frames,
                     "face_list": total_bbox_list,
                     "landmark_list": total_pts_list,
                     "isvalid":isvalid,
            }        
            with open(vid_meta, 'w') as f:
                json.dump(meta_data, f, indent=4)
            


def main(cfg):
    # Ensure all necessary directories exist
    os.makedirs(cfg.video_root_25fps, exist_ok=True)
    os.makedirs(cfg.video_audio_clip_root, exist_ok=True)
    os.makedirs(cfg.meta_root, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_file_list), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_clip_file_list_train), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_clip_file_list_val), exist_ok=True)

    vid_list = os.listdir(cfg.video_root_raw)
    sorted_vid_list = sorted(vid_list)
 
    # Save video file list
    with open(cfg.video_file_list, 'w') as file:
        for vid in sorted_vid_list:
            file.write(vid + '\n')

    # 1. Convert videos to 25 FPS
    convert_video(cfg.video_root_raw, cfg.video_root_25fps, sorted_vid_list)
    
    # 2. Segment videos into 30-second clips
    segment_video(cfg.video_root_25fps, cfg.video_audio_clip_root, vid_list, segment_duration=cfg.clip_len_second)
    
    # 3. Extract audio
    clip_vid_list = os.listdir(cfg.video_audio_clip_root)
    extract_audio(cfg.video_audio_clip_root, cfg.video_audio_clip_root, clip_vid_list)
    
    # 4. Generate video metadata
    analyze_video(cfg.video_audio_clip_root, cfg.meta_root, clip_vid_list)
    
    # 5. Generate training and validation set lists
    generate_train_list(cfg)
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/preprocess.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    main(config)
    