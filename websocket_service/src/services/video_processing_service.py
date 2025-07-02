"""
Video processing service for automatic avatar video generation.
Creates multiple avatar videos from a single source video.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import subprocess
import tempfile

# Required dependencies
import numpy as np
import librosa
import torch
import pickle

# OpenCV is required
import cv2

# Import MuseTalk utilities
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Change working directory to MuseTalk root for proper imports
original_cwd = os.getcwd()
musetalk_root = Path(__file__).parent.parent.parent.parent
os.chdir(musetalk_root)

try:
    # Import MuseTalk modules - fail fast if not available
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.blending import get_image_prepare_material, get_image_blending
    from musetalk.utils.utils import load_all_model
    from musetalk.utils.audio_processor import AudioProcessor
    print("Info: MuseTalk modules loaded successfully")
finally:
    # Restore original working directory
    os.chdir(original_cwd)


class VideoSegment:
    """Represents a video segment with metadata."""
    
    def __init__(self, start_frame: int, end_frame: int, segment_type: str, 
                 quality_score: float = 0.0, metadata: Dict = None):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.segment_type = segment_type  # 'idle', 'speaking', 'action'
        self.quality_score = quality_score
        self.metadata = metadata or {}
        
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame
        
    def to_dict(self) -> Dict:
        return {
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'segment_type': self.segment_type,
            'quality_score': self.quality_score,
            'metadata': self.metadata
        }


class VideoProcessingService:
    """Service for processing single video into multiple avatar videos."""
    
    def __init__(self, 
                 temp_dir: str = "/tmp/musetalk_processing",
                 target_fps: int = 25,
                 target_resolution: Tuple[int, int] = (512, 512)):
        """
        Initialize video processing service.
        
        Args:
            temp_dir: Temporary directory for processing
            target_fps: Target FPS for output videos
            target_resolution: Target resolution (width, height)
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        self.musetalk_root = Path(__file__).parent.parent.parent.parent
        
        # Video requirements
        self.required_videos = {
            'idle': 7,      # idle_0 to idle_6
            'speaking': 8,  # speaking_0 to speaking_7
            'action': 2     # action_1, action_2
        }
        
        # Initialize MuseTalk models and services
        try:
            # Change to MuseTalk root directory for initialization
            original_cwd = os.getcwd()
            os.chdir(self.musetalk_root)
            
            # Initialize models
            self.vae, self.unet, self.pe = load_all_model(
                unet_model_path="models/musetalkV15/unet.pth",
                vae_type="sd-vae",
                unet_config="models/musetalkV15/musetalk.json",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Initialize face parser for v1.5
            self.face_parser = FaceParsing()
            
            print("MuseTalk models and face parsing initialized successfully")
            
            # Restore working directory
            os.chdir(original_cwd)
        except Exception as e:
            print(f"FATAL: Error initializing MuseTalk models: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Cannot initialize VideoProcessingService without MuseTalk models: {e}")
    
    async def process_video(self, video_path: str, user_id: str, 
                          output_dir: str) -> Dict[str, any]:
        """
        Process single video into multiple avatar videos.
        
        Args:
            video_path: Path to input video
            user_id: User identifier
            output_dir: Output directory for avatar videos
            
        Returns:
            Dictionary with processing results
        """
        try:
            print(f"Starting video processing for user {user_id}")
            
            # Create user output directory
            user_output_dir = Path(output_dir) / user_id
            user_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Extract and analyze video
            video_info = await self._extract_video_info(video_path)
            print(f"Video info: {video_info}")
            
            # Step 2: Detect and track faces
            face_data = await self._detect_faces(video_path)
            print(f"Face detection complete: {len(face_data['frames'])} frames processed")
            
            # Step 3: Extract and analyze audio
            audio_data = await self._extract_audio_features(video_path)
            print(f"Audio analysis complete: {len(audio_data['segments'])} segments found")
            
            # Step 4: Segment video by content type
            segments = await self._segment_video(video_path, face_data, audio_data)
            print(f"Video segmentation complete: {len(segments)} segments")
            
            # Step 5: Generate avatar videos
            avatar_videos = await self._generate_avatar_videos(
                video_path, segments, str(user_output_dir)
            )
            print(f"Avatar video generation complete: {len(avatar_videos)} videos")
            
            # Step 6: Generate MuseTalk avatar data (following realtime_inference.py workflow)
            model_data = await self._prepare_avatar_musetalk_style(
                video_path, user_id, str(user_output_dir)
            )
            print("MuseTalk avatar preparation complete")
            
            # Step 7: Save metadata
            metadata = {
                'user_id': user_id,
                'source_video': video_path,
                'processed_at': datetime.utcnow().isoformat(),
                'video_info': video_info,
                'segments': [seg.to_dict() for seg in segments],
                'avatar_videos': avatar_videos,
                'model_data': model_data
            }
            
            metadata_path = user_output_dir / 'avatar_info.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'success': True,
                'user_id': user_id,
                'output_dir': str(user_output_dir),
                'avatar_videos': avatar_videos,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _extract_video_info(self, video_path: str) -> Dict:
        """Extract basic video information."""
        if not CV2_AVAILABLE:
            # Return mock video info
            return {
                'width': 1920,
                'height': 1080,
                'fps': 25.0,
                'frame_count': 750,  # 30 seconds at 25fps
                'duration': 30.0
            }
        
        cap = cv2.VideoCapture(video_path)
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    async def _detect_faces(self, video_path: str) -> Dict:
        """Detect and track faces in video."""
        if not CV2_AVAILABLE or not self.musetalk_available:
            # Return mock face detection data
            total_frames = 750  # 30 seconds at 25fps
            frames_data = []
            
            for frame_idx in range(total_frames):
                # Mock: assume face detected in most frames with good quality
                face_detected = frame_idx % 10 != 0  # Miss every 10th frame
                quality_score = 75.0 + (frame_idx % 20)  # Vary quality 75-95
                
                frames_data.append({
                    'frame_idx': frame_idx,
                    'face_detected': face_detected,
                    'face_bbox': [100, 100, 300, 300] if face_detected else None,
                    'quality_score': quality_score,
                    'frame_shape': [1080, 1920, 3]
                })
            
            return {
                'total_frames': total_frames,
                'frames': frames_data,
                'face_detection_rate': 0.9  # 90% detection rate
            }
        
        cap = cv2.VideoCapture(video_path)
        
        frames_data = []
        frame_idx = 0
        
        # Face detector (using OpenCV for simplicity)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Calculate face quality metrics
            quality_score = 0.0
            face_bbox = None
            
            if len(faces) > 0:
                # Use largest face
                face_bbox = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = face_bbox
                
                # Quality based on face size and position
                face_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                size_ratio = face_area / frame_area
                
                # Center position score
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2
                
                center_distance = (((face_center_x - frame_center_x)**2 + 
                                  (face_center_y - frame_center_y)**2) ** 0.5)
                max_distance = ((frame_center_x**2 + frame_center_y**2) ** 0.5)
                center_score = 1.0 - (center_distance / max_distance)
                
                # Combine scores
                quality_score = (size_ratio * 0.7 + center_score * 0.3) * 100
            
            frames_data.append({
                'frame_idx': frame_idx,
                'face_detected': len(faces) > 0,
                'face_bbox': face_bbox.tolist() if face_bbox is not None else None,
                'quality_score': quality_score,
                'frame_shape': frame.shape
            })
            
            frame_idx += 1
        
        cap.release()
        
        return {
            'total_frames': frame_idx,
            'frames': frames_data,
            'face_detection_rate': sum(1 for f in frames_data if f['face_detected']) / frame_idx
        }
    
    async def _extract_audio_features(self, video_path: str) -> Dict:
        """Extract and analyze audio features."""
        try:
            # Extract audio using librosa
            y, sr = librosa.load(video_path, sr=16000)
            
            # Voice activity detection
            intervals = librosa.effects.split(y, top_db=20)
            
            # Calculate RMS energy for each frame
            frame_length = int(sr * 0.025)  # 25ms frames
            hop_length = int(sr * 0.010)    # 10ms hop
            
            rms = librosa.feature.rms(
                y=y, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            # Detect speech segments
            segments = []
            for start_sample, end_sample in intervals:
                start_time = start_sample / sr
                end_time = end_sample / sr
                duration = end_time - start_time
                
                if duration > 0.5:  # Minimum 0.5s segments
                    segment_y = y[start_sample:end_sample]
                    energy = np.mean(np.abs(segment_y))
                    
                    segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'energy': float(energy),
                        'is_speech': energy > np.mean(rms) * 0.5
                    })
            
            return {
                'sample_rate': sr,
                'duration': len(y) / sr,
                'segments': segments,
                'rms_energy': rms.tolist(),
                'speech_ratio': sum(seg['duration'] for seg in segments if seg['is_speech']) / (len(y) / sr)
            }
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            # Return mock data if audio processing fails
            return {
                'sample_rate': 16000,
                'duration': 30.0,
                'segments': [],
                'rms_energy': [],
                'speech_ratio': 0.5
            }
    
    async def _segment_video(self, video_path: str, face_data: Dict, 
                           audio_data: Dict) -> List[VideoSegment]:
        """Segment video into different types based on content analysis."""
        total_frames = face_data['total_frames']
        video_info = await self._extract_video_info(video_path)
        fps = video_info['fps']
        
        segments = []
        
        # Convert audio segments to frame indices
        audio_segments_frames = []
        for seg in audio_data['segments']:
            start_frame = int(seg['start_time'] * fps)
            end_frame = int(seg['end_time'] * fps)
            audio_segments_frames.append((start_frame, end_frame, seg['is_speech']))
        
        # Identify idle segments (silent periods with good face detection)
        idle_segments = []
        for i in range(0, total_frames, int(fps * 2)):  # 2-second windows
            end_frame = min(i + int(fps * 3), total_frames)  # 3-second segments
            
            # Check if this period is mostly silent
            is_silent = True
            for start_f, end_f, is_speech in audio_segments_frames:
                if start_f <= i <= end_f or start_f <= end_frame <= end_f:
                    if is_speech:
                        is_silent = False
                        break
            
            if is_silent:
                # Calculate average face quality in this segment
                frame_qualities = []
                for frame_idx in range(i, end_frame):
                    if frame_idx < len(face_data['frames']):
                        frame_info = face_data['frames'][frame_idx]
                        if frame_info['face_detected']:
                            frame_qualities.append(frame_info['quality_score'])
                
                if frame_qualities:
                    avg_quality = np.mean(frame_qualities)
                    if avg_quality > 30:  # Good enough quality
                        idle_segments.append(VideoSegment(
                            i, end_frame, 'idle', avg_quality
                        ))
        
        # Identify speaking segments (speech periods with face detection)
        speaking_segments = []
        for start_f, end_f, is_speech in audio_segments_frames:
            if is_speech and end_f - start_f > fps:  # At least 1 second
                # Check face quality in this segment
                frame_qualities = []
                for frame_idx in range(start_f, end_f):
                    if frame_idx < len(face_data['frames']):
                        frame_info = face_data['frames'][frame_idx]
                        if frame_info['face_detected']:
                            frame_qualities.append(frame_info['quality_score'])
                
                if frame_qualities:
                    avg_quality = np.mean(frame_qualities)
                    if avg_quality > 25:  # Acceptable quality for speaking
                        speaking_segments.append(VideoSegment(
                            start_f, end_f, 'speaking', avg_quality
                        ))
        
        # Identify action segments (head movements, gestures)
        action_segments = []
        # Simple heuristic: look for periods with significant face movement
        for i in range(0, total_frames - int(fps * 2), int(fps)):
            end_frame = min(i + int(fps * 2), total_frames)  # 2-second segments
            
            # Calculate face movement in this segment
            face_positions = []
            for frame_idx in range(i, end_frame):
                if frame_idx < len(face_data['frames']):
                    frame_info = face_data['frames'][frame_idx]
                    if frame_info['face_detected'] and frame_info['face_bbox']:
                        bbox = frame_info['face_bbox']
                        center_x = bbox[0] + bbox[2] // 2
                        center_y = bbox[1] + bbox[3] // 2
                        face_positions.append((center_x, center_y))
            
            if len(face_positions) > fps:  # Enough data points
                # Calculate movement variance
                positions = np.array(face_positions)
                movement = np.var(positions, axis=0).sum()
                
                if movement > 100:  # Threshold for significant movement
                    avg_quality = np.mean([face_data['frames'][j]['quality_score'] 
                                         for j in range(i, end_frame) 
                                         if j < len(face_data['frames']) and 
                                         face_data['frames'][j]['face_detected']])
                    
                    if avg_quality > 25:
                        action_segments.append(VideoSegment(
                            i, end_frame, 'action', avg_quality,
                            {'movement_score': float(movement)}
                        ))
        
        # Sort all segments by quality and select the best ones
        idle_segments.sort(key=lambda x: x.quality_score, reverse=True)
        speaking_segments.sort(key=lambda x: x.quality_score, reverse=True)
        action_segments.sort(key=lambda x: x.quality_score, reverse=True)
        
        # Add to final segments list
        segments.extend(idle_segments[:self.required_videos['idle']])
        segments.extend(speaking_segments[:self.required_videos['speaking']])
        segments.extend(action_segments[:self.required_videos['action']])
        
        # Fill missing segments with fallbacks if needed
        segments = await self._ensure_all_segments(segments, total_frames, fps)
        
        return segments
    
    async def _ensure_all_segments(self, segments: List[VideoSegment], 
                                 total_frames: int, fps: float) -> List[VideoSegment]:
        """Ensure we have enough segments of each type."""
        segment_counts = {'idle': 0, 'speaking': 0, 'action': 0}
        
        for seg in segments:
            segment_counts[seg.segment_type] += 1
        
        # Generate fallback segments if needed
        for seg_type, required_count in self.required_videos.items():
            current_count = segment_counts[seg_type]
            
            if current_count < required_count:
                # Generate additional segments by reusing/modifying existing ones
                existing_segments = [s for s in segments if s.segment_type == seg_type]
                
                for i in range(required_count - current_count):
                    if existing_segments:
                        # Modify existing segment (different time range)
                        base_segment = existing_segments[i % len(existing_segments)]
                        new_start = max(0, base_segment.start_frame - int(fps))
                        new_end = min(total_frames, base_segment.end_frame + int(fps))
                        
                        fallback_segment = VideoSegment(
                            new_start, new_end, seg_type, 
                            base_segment.quality_score * 0.8,  # Lower quality score
                            {'fallback': True, 'based_on': base_segment.start_frame}
                        )
                    else:
                        # Create generic segment
                        segment_duration = int(fps * 3)  # 3 seconds
                        start_frame = (i * segment_duration) % (total_frames - segment_duration)
                        end_frame = start_frame + segment_duration
                        
                        fallback_segment = VideoSegment(
                            start_frame, end_frame, seg_type, 50.0,
                            {'fallback': True, 'generic': True}
                        )
                    
                    segments.append(fallback_segment)
        
        return segments
    
    async def _generate_avatar_videos(self, video_path: str, 
                                    segments: List[VideoSegment], 
                                    output_dir: str) -> List[str]:
        """Generate individual avatar videos from segments."""
        cap = cv2.VideoCapture(video_path)
        video_info = await self._extract_video_info(video_path)
        fps = video_info['fps']
        
        generated_videos = []
        
        # Group segments by type
        segments_by_type = {}
        for seg in segments:
            if seg.segment_type not in segments_by_type:
                segments_by_type[seg.segment_type] = []
            segments_by_type[seg.segment_type].append(seg)
        
        # Generate videos for each type
        for seg_type, type_segments in segments_by_type.items():
            for i, segment in enumerate(type_segments):
                if seg_type == 'idle':
                    video_name = f"idle_{i}.mp4"
                elif seg_type == 'speaking':
                    video_name = f"speaking_{i}.mp4"
                elif seg_type == 'action':
                    video_name = f"action_{i+1}.mp4"
                else:
                    continue
                
                output_path = os.path.join(output_dir, video_name)
                
                # Extract segment using ffmpeg for better quality
                start_time = segment.start_frame / fps
                duration = segment.duration_frames() / fps
                
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-vf', f'scale={self.target_resolution[0]}:{self.target_resolution[1]}',
                    '-r', str(self.target_fps),
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    output_path
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    generated_videos.append(video_name)
                    print(f"Generated {video_name}")
                except subprocess.CalledProcessError as e:
                    print(f"Error generating {video_name}: {e}")
        
        cap.release()
        return generated_videos
    
    def video2imgs(self, video_path: str, save_path: str, ext: str = '.png', cut_frame: int = 10000000):
        """Extract frames from video, similar to realtime_inference.py"""
        cap = cv2.VideoCapture(video_path)
        count = 0
        extracted_frames = []
        
        save_path_obj = Path(save_path)
        save_path_obj.mkdir(parents=True, exist_ok=True)
        
        while True:
            if count > cut_frame:
                break
            ret, frame = cap.read()
            if ret:
                frame_path = save_path_obj / f"{count:08d}.png"
                cv2.imwrite(str(frame_path), frame)
                extracted_frames.append(str(frame_path))
                count += 1
            else:
                break
        
        cap.release()
        return extracted_frames

    async def _prepare_avatar_musetalk_style(self, video_path: str, user_id: str, output_dir: str) -> Dict:
        """
        Prepare avatar following the exact MuseTalk realtime_inference.py workflow
        """
        try:
            
            # Change to MuseTalk root for relative paths
            original_cwd = os.getcwd()
            os.chdir(self.musetalk_root)
            
            try:
                # Create avatar directory structure like MuseTalk
                avatar_path = Path(output_dir)
                full_imgs_path = avatar_path / "full_imgs"
                mask_out_path = avatar_path / "mask"
                
                # Create directories
                for path in [avatar_path, full_imgs_path, mask_out_path]:
                    path.mkdir(parents=True, exist_ok=True)
                
                # Step 1: Extract frames from video (like video2imgs in realtime_inference.py)
                print("Extracting frames from video...")
                extracted_frames = self.video2imgs(video_path, str(full_imgs_path))
                
                if not extracted_frames:
                    raise ValueError("No frames extracted from video")
                
                # Step 2: Get landmarks and bbox (like realtime_inference.py)
                print("Extracting landmarks and bounding boxes...")
                bbox_shift = 0  # For v1.5, use 0 as default
                
                # Import get_landmark_and_bbox only when needed to avoid initialization issues
                try:
                    from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
                    coord_list, frame_list = get_landmark_and_bbox(extracted_frames, bbox_shift)
                except ImportError as e:
                    print(f"Could not import get_landmark_and_bbox: {e}")
                    raise ValueError("MuseTalk preprocessing not available")
                
                # Step 3: Process each frame to get VAE latents (like realtime_inference.py)
                print("Processing frames and extracting VAE latents...")
                input_latent_list = []
                coord_placeholder = (0.0, 0.0, 0.0, 0.0)
                
                for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
                    if bbox == coord_placeholder:
                        continue
                    
                    x1, y1, x2, y2 = bbox
                    
                    # For v1.5, add extra margin
                    extra_margin = 10
                    y2 = y2 + extra_margin
                    y2 = min(y2, frame.shape[0])
                    coord_list[idx] = [x1, y1, x2, y2]  # Update coord_list
                    
                    # Crop and resize frame
                    crop_frame = frame[y1:y2, x1:x2]
                    resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    
                    # Get VAE latents
                    latents = self.vae.get_latents_for_unet(resized_crop_frame)
                    input_latent_list.append(latents)
                
                # Step 4: Create forward+backward cycles (like realtime_inference.py)
                frame_list_cycle = frame_list + frame_list[::-1]
                coord_list_cycle = coord_list + coord_list[::-1]
                input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
                
                # Step 5: Generate masks and mask coordinates
                print("Generating face masks...")
                mask_coords_list_cycle = []
                mask_list_cycle = []
                
                for i, frame in enumerate(frame_list_cycle):
                    # Save full frame
                    cv2.imwrite(str(full_imgs_path / f"{str(i).zfill(8)}.png"), frame)
                    
                    x1, y1, x2, y2 = coord_list_cycle[i]
                    
                    # Generate mask using MuseTalk's face parsing
                    mask, crop_box = get_image_prepare_material(
                        frame, [x1, y1, x2, y2], 
                        fp=self.face_parser, 
                        mode="jaw"  # v1.5 parsing mode
                    )
                    
                    # Save mask
                    cv2.imwrite(str(mask_out_path / f"{str(i).zfill(8)}.png"), mask)
                    mask_coords_list_cycle.append(crop_box)
                    mask_list_cycle.append(mask)
                
                # Step 6: Save all data files
                coords_path = avatar_path / "coords.pkl"
                latents_path = avatar_path / "latents.pt"
                mask_coords_path = avatar_path / "mask_coords.pkl"
                avatar_info_path = avatar_path / "avator_info.json"  # MuseTalk uses "avator" typo
                
                # Save coordinates
                with open(coords_path, 'wb') as f:
                    pickle.dump(coord_list_cycle, f)
                
                # Save latents
                torch.save(input_latent_list_cycle, latents_path)
                
                # Save mask coordinates
                with open(mask_coords_path, 'wb') as f:
                    pickle.dump(mask_coords_list_cycle, f)
                
                # Save avatar info (like realtime_inference.py)
                avatar_info = {
                    "avatar_id": user_id,
                    "video_path": video_path,
                    "bbox_shift": bbox_shift,
                    "version": "v15"
                }
                with open(avatar_info_path, "w") as f:
                    json.dump(avatar_info, f)
                
                return {
                    'latents_path': str(latents_path),
                    'coords_path': str(coords_path),
                    'mask_coords_path': str(mask_coords_path),
                    'avatar_info_path': str(avatar_info_path),
                    'full_imgs_path': str(full_imgs_path),
                    'mask_out_path': str(mask_out_path),
                    'model_loaded': True,
                    'landmarks_count': len(frame_list),
                    'bbox_count': len(coord_list_cycle),
                    'latents_count': len(input_latent_list_cycle),
                    'preprocessing_success': True
                }
                
            finally:
                os.chdir(original_cwd)
            
        except Exception as e:
            print(f"FATAL: Error in MuseTalk avatar preparation: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Avatar preparation failed: {e}")

    async def _generate_model_data(self, video_path: str, face_data: Dict, 
                                 output_dir: str) -> Dict:
        """Generate MuseTalk model data (latents, coordinates, etc.)."""
        # This method is now deprecated in favor of _prepare_avatar_musetalk_style
        # but kept for backward compatibility
        return await self._prepare_avatar_musetalk_style(video_path, face_data.get('user_id', 'unknown'), output_dir)
    
    async def _generate_mock_model_data(self, output_dir: str) -> Dict:
        """Generate mock model data for testing."""
        if TORCH_AVAILABLE:
            # Create mock latents
            mock_latents = torch.randn(100, 4, 32, 32)
            latents_path = os.path.join(output_dir, 'latents.pt')
            torch.save(mock_latents, latents_path)
        else:
            # Create empty latents file
            latents_path = os.path.join(output_dir, 'latents.pt')
            with open(latents_path, 'wb') as f:
                f.write(b'mock_latents_data')
        
        # Create mock coordinates
        mock_coords = [(100, 100, 400, 400)] * 100
        coords_path = os.path.join(output_dir, 'coords.pkl')
        
        if TORCH_AVAILABLE:
            with open(coords_path, 'wb') as f:
                pickle.dump(mock_coords, f)
        else:
            # Simple JSON fallback
            with open(coords_path.replace('.pkl', '.json'), 'w') as f:
                json.dump(mock_coords, f)
            coords_path = coords_path.replace('.pkl', '.json')
        
        # Create mock mask coordinates
        mask_coords_path = os.path.join(output_dir, 'mask_coords.pkl')
        if TORCH_AVAILABLE:
            with open(mask_coords_path, 'wb') as f:
                pickle.dump(mock_coords, f)
        else:
            # Simple JSON fallback
            with open(mask_coords_path.replace('.pkl', '.json'), 'w') as f:
                json.dump(mock_coords, f)
            mask_coords_path = mask_coords_path.replace('.pkl', '.json')
        
        return {
            'latents_path': latents_path,
            'coords_path': coords_path,
            'mask_coords_path': mask_coords_path,
            'model_loaded': True,
            'preprocessing_success': False,
            'bbox_count': 100,
            'landmarks_count': 0
        }
    
    def get_processing_stats(self) -> Dict:
        """Get processing service statistics."""
        return {
            'temp_dir': str(self.temp_dir),
            'target_fps': self.target_fps,
            'target_resolution': self.target_resolution,
            'required_videos': self.required_videos,
            'musetalk_available': self.musetalk_available
        }