"""
Video generation service for MuseTalk WebSocket.
"""
import asyncio
import numpy as np
import cv2
import pickle
import glob
from typing import Optional, Tuple, Dict, Any
import sys
import os
from pathlib import Path

import torch

# Add MuseTalk to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# Import MuseTalk utilities - fail fast if not available
from musetalk.utils.utils import load_all_model, datagen
from musetalk.utils.blending import get_image_blending, get_image_prepare_material  
from musetalk.utils.face_parsing import FaceParsing

from utils.video_codec import SimpleH264Encoder
from models.messages import VideoStateType
from models.session import Session


class VideoService:
    """Service for generating lip-synced video frames."""
    
    def __init__(self,
                 model_path: str = "../models",
                 device: str = "cuda",
                 dtype = None,
                 use_float16: bool = False):
        """
        Initialize video generation service.
        
        Args:
            model_path: Path to model directory
            device: Device to run on (cuda/cpu)
            dtype: Data type for models
            use_float16: Whether to use float16 for faster inference
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if use_float16 else (dtype if dtype is not None else torch.float32)
        
        # Initialize models
        self._init_models()
        
        # H.264 encoders per session
        self.encoders = {}  # session_id -> encoder
        
        # Face parsing with correct model paths
        # Change working directory temporarily to fix relative paths
        import os
        current_dir = os.getcwd()
        try:
            # Find MuseTalk root directory (where models/ folder exists)
            musetalk_root = self.model_path.parent
            os.chdir(str(musetalk_root))
            self.face_parser = FaceParsing()
        finally:
            # Restore original working directory
            os.chdir(current_dir)
    
    def _init_models(self):
        """Initialize VAE, UNet, and PE models using MuseTalk's load_all_model."""
        try:
            # Change to MuseTalk root directory for proper model loading
            current_dir = os.getcwd()
            musetalk_root = self.model_path.parent
            os.chdir(str(musetalk_root))
            
            try:
                # Use MuseTalk's load_all_model function with relative paths (like realtime_inference.py)
                self.vae, self.unet, self.pe = load_all_model(
                    unet_model_path="models/musetalkV15/unet.pth",
                    vae_type="sd-vae",
                    unet_config="models/musetalkV15/musetalk.json",
                    device=self.device
                )
                
                # Convert to appropriate dtype (like realtime_inference.py)
                if self.dtype == torch.float16:
                    self.pe = self.pe.half().to(self.device)
                    self.vae.vae = self.vae.vae.half().to(self.device)
                    self.unet.model = self.unet.model.half().to(self.device)
                else:
                    self.pe = self.pe.to(self.device)
                    self.vae.vae = self.vae.vae.to(self.device)
                    self.unet.model = self.unet.model.to(self.device)
                
                # Fixed timestep for non-diffusion inference (like realtime_inference.py)
                self.timesteps = torch.tensor([0], device=self.device)
                
                print(f"MuseTalk models loaded successfully on {self.device}")
                
            finally:
                # Restore original working directory
                os.chdir(current_dir)
            
        except Exception as e:
            print(f"FATAL: Failed to load MuseTalk models: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Cannot initialize VideoService without MuseTalk models: {e}")
    
    def get_encoder(self, session_id: str) -> SimpleH264Encoder:
        """Get or create H.264 encoder for session."""
        if session_id not in self.encoders:
            self.encoders[session_id] = SimpleH264Encoder()
        return self.encoders[session_id]
    
    async def generate_frame(self,
                           session: Session,
                           audio_features: torch.Tensor,
                           frame_index: int = 0) -> Optional[str]:
        """
        Generate a single video frame.
        
        Args:
            session: Session information
            audio_features: Audio features from Whisper
            frame_index: Frame index for avatar
            
        Returns:
            Base64 encoded H.264 frame or None
        """
        try:
            # Get encoder
            encoder = self.get_encoder(session.session_id)
            
            if self.vae and self.unet and self.pe:
                # Real generation
                frame = await self._generate_frame_real(
                    session, audio_features, frame_index
                )
            else:
                # Mock generation for POC
                frame = await self._generate_frame_mock(session, frame_index)
            
            if frame is not None:
                # Encode to H.264
                encoded = encoder.encode_frame(frame)
                return encoded
            
            return None
            
        except Exception as e:
            print(f"Frame generation error: {e}")
            return None
    
    async def _generate_frame_real(self,
                                 session: Session,
                                 audio_features: torch.Tensor,
                                 frame_index: int) -> Optional[np.ndarray]:
        """Generate frame using real models."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_frame_sync,
            session,
            audio_features,
            frame_index
        )
    
    def _generate_frame_sync(self,
                           session: Session,
                           audio_features: torch.Tensor,
                           frame_index: int) -> Optional[np.ndarray]:
        """Synchronous frame generation using MuseTalk pipeline (like realtime_inference.py)."""
        try:
            # Load avatar data for the session
            avatar_info = session.avatar_info
            if not avatar_info or not avatar_info.latents_path:
                print("No avatar data available for frame generation")
                return None
            
            # Load avatar latents and coordinates
            latents = torch.load(avatar_info.latents_path, map_location=self.device)
            with open(avatar_info.coords_path, 'rb') as f:
                coord_list_cycle = pickle.load(f)
            
            # Load mask coordinates if available
            mask_coords_list_cycle = None
            if avatar_info.mask_coords_path and os.path.exists(avatar_info.mask_coords_path):
                with open(avatar_info.mask_coords_path, 'rb') as f:
                    mask_coords_list_cycle = pickle.load(f)
            
            # Load original frames
            full_imgs_path = os.path.join(os.path.dirname(avatar_info.latents_path), "full_imgs")
            frame_list_cycle = None
            if os.path.exists(full_imgs_path):
                import glob
                img_files = sorted(glob.glob(os.path.join(full_imgs_path, "*.png")))
                if img_files and MUSETALK_IMPORTS_AVAILABLE:
                    try:
                        # Import read_imgs only when needed to avoid initialization issues
                        current_dir = os.getcwd()
                        musetalk_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                        os.chdir(musetalk_root)
                        try:
                            from musetalk.utils.preprocessing import read_imgs
                            frame_list_cycle = read_imgs(img_files)
                        finally:
                            os.chdir(current_dir)
                    except Exception as e:
                        print(f"Could not load frames with read_imgs: {e}")
                        # Fallback: load frames manually with cv2
                        frame_list_cycle = []
                        for img_file in img_files:
                            frame = cv2.imread(img_file)
                            if frame is not None:
                                frame_list_cycle.append(frame)
            
            with torch.no_grad():
                # Get current frame index in cycle
                cycle_idx = frame_index % len(latents)
                
                # Get audio embedding (like realtime_inference.py)
                audio_emb = self.pe(audio_features.to(device=self.device, dtype=self.dtype))
                
                # Get latent for current frame
                current_latent = latents[cycle_idx]
                if not isinstance(current_latent, torch.Tensor):
                    current_latent = torch.tensor(current_latent)
                current_latent = current_latent.to(device=self.device, dtype=self.dtype)
                
                # Ensure correct batch dimension
                if len(current_latent.shape) == 3:
                    current_latent = current_latent.unsqueeze(0)
                
                # UNet inference (like realtime_inference.py)
                pred_latents = self.unet.model(
                    current_latent, 
                    self.timesteps, 
                    encoder_hidden_states=audio_emb
                ).sample
                
                # VAE decode (like realtime_inference.py)
                pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
                recon_frame = self.vae.decode_latents(pred_latents)
                
                # Get the reconstructed frame
                if isinstance(recon_frame, list):
                    recon_frame = recon_frame[0]
                
                # Convert to numpy and ensure correct format
                if isinstance(recon_frame, torch.Tensor):
                    recon_frame = recon_frame.cpu().numpy()
                
                # Ensure correct shape (H, W, C)
                if len(recon_frame.shape) == 4:
                    recon_frame = recon_frame[0]  # Remove batch
                if recon_frame.shape[0] == 3:  # CHW -> HWC
                    recon_frame = np.transpose(recon_frame, (1, 2, 0))
                
                # Convert to uint8 RGB
                recon_frame = (recon_frame * 255).clip(0, 255).astype(np.uint8)
                
                # Blend with original frame if available (like realtime_inference.py)
                if frame_list_cycle and mask_coords_list_cycle:
                    try:
                        ori_frame = frame_list_cycle[cycle_idx].copy()
                        bbox = coord_list_cycle[cycle_idx]
                        x1, y1, x2, y2 = bbox
                        
                        # Resize generated frame to match bbox
                        recon_resized = cv2.resize(recon_frame, (x2 - x1, y2 - y1))
                        
                        # Load mask for blending
                        mask_path = os.path.join(os.path.dirname(avatar_info.latents_path), "mask")
                        mask_file = os.path.join(mask_path, f"{str(cycle_idx).zfill(8)}.png")
                        
                        if os.path.exists(mask_file):
                            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                            mask_crop_box = mask_coords_list_cycle[cycle_idx]
                            
                            # Use MuseTalk's blending function
                            final_frame = get_image_blending(ori_frame, recon_resized, bbox, mask, mask_crop_box)
                        else:
                            # Simple overlay without mask
                            final_frame = ori_frame.copy()
                            final_frame[y1:y2, x1:x2] = recon_resized
                        
                        # Resize to output resolution
                        final_frame = cv2.resize(final_frame, (512, 512))
                        return final_frame
                        
                    except Exception as e:
                        print(f"Error in frame blending: {e}")
                        # Fall back to just the generated frame
                        pass
                
                # If no blending, just return the generated frame
                if recon_frame.shape[:2] != (512, 512):
                    recon_frame = cv2.resize(recon_frame, (512, 512))
                
                return recon_frame
                
        except Exception as e:
            print(f"MuseTalk frame generation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _generate_frame_mock(self,
                                 session: Session,
                                 frame_index: int) -> np.ndarray:
        """Generate mock frame for testing."""
        # Create a test frame with some variation
        frame = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Different patterns based on state
        if session.state.video_state == VideoStateType.IDLE:
            # Blue gradient for idle
            frame[:, :, 2] = np.linspace(100, 200, 512, dtype=np.uint8)
        elif session.state.video_state == VideoStateType.SPEAKING:
            # Green gradient for speaking
            frame[:, :, 1] = np.linspace(100, 200, 512, dtype=np.uint8).reshape(-1, 1)
            # Add some animation
            offset = (frame_index * 10) % 512
            frame[offset:offset+20, :, 0] = 255
        elif session.state.video_state == VideoStateType.ACTION:
            # Red pattern for action
            frame[:, :, 0] = 150
            # Animated circle
            center = (256 + int(100 * np.sin(frame_index * 0.1)), 256)
            cv2.circle(frame, center, 50, (255, 255, 0), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame {frame_index}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add session info
        cv2.putText(frame, f"State: {session.state.video_state}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        return frame
    
    async def generate_action_frame(self,
                                  session: Session,
                                  action_type: str,
                                  audio_features: torch.Tensor,
                                  progress: float) -> Tuple[Optional[str], float]:
        """
        Generate action frame.
        
        Args:
            session: Session information
            action_type: Type of action
            audio_features: Audio features
            progress: Current action progress (0.0 to 1.0)
            
        Returns:
            Tuple of (encoded frame, new progress)
        """
        # Calculate frame index based on progress
        # Assuming 10 second action at 25 fps = 250 frames
        total_frames = 250
        frame_index = int(progress * total_frames)
        
        # Generate frame
        encoded = await self.generate_frame(session, audio_features, frame_index)
        
        # Update progress (40ms per frame at 25fps)
        new_progress = min(1.0, progress + (1.0 / total_frames))
        
        return encoded, new_progress
    
    async def generate_base_frame(self, 
                                session: Session,
                                base_video: str,
                                frame_index: int) -> Optional[str]:
        """
        Generate frame from base video without audio.
        
        Args:
            session: Session information
            base_video: Base video identifier (e.g., "idle_0", "speaking_1")
            frame_index: Frame index in the video
            
        Returns:
            Base64 encoded H.264 frame or None
        """
        try:
            print(f"VideoService: generate_base_frame called for session {session.session_id}, video {base_video}, frame {frame_index}")
            
            # Get or create encoder
            if session.session_id not in self.encoders:
                self.encoders[session.session_id] = SimpleH264Encoder(
                    width=512, height=512, fps=25
                )
                print(f"VideoService: Created new encoder for session {session.session_id}")
            
            encoder = self.encoders[session.session_id]
            
            # For POC, generate mock frame
            # In production, this would load actual video frame
            frame = await self._generate_base_frame_mock(session, base_video, frame_index)
            print(f"VideoService: Mock frame generated, shape: {frame.shape if frame is not None else 'None'}")
            
            if frame is None:
                print(f"VideoService: No frame data generated")
                return None
            
            # Encode frame
            encoded = encoder.encode_frame(frame)
            print(f"VideoService: Frame encoded, length: {len(encoded) if encoded else 'None'}")
            return encoded
            
        except Exception as e:
            print(f"VideoService: Base frame generation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _generate_base_frame_mock(self,
                                      session: Session,
                                      base_video: str,
                                      frame_index: int) -> np.ndarray:
        """Generate mock base video frame for testing."""
        # Create a test frame with variation based on video type
        frame = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Different patterns based on video type
        if base_video.startswith("idle_"):
            # Blue gradient for idle
            frame[:, :, 2] = np.linspace(50, 150, 512, dtype=np.uint8)
            # Add slow animation
            offset = int(50 * np.sin(frame_index * 0.02))
            cv2.rectangle(frame, (200 + offset, 200), (312 + offset, 312), (100, 150, 255), -1)
            
        elif base_video.startswith("speaking_"):
            # Green gradient for speaking
            frame[:, :, 1] = np.linspace(50, 150, 512, dtype=np.uint8).reshape(-1, 1)
            # Add mouth movement animation
            mouth_y = 350 + int(20 * np.sin(frame_index * 0.2))
            cv2.ellipse(frame, (256, mouth_y), (60, 30), 0, 0, 180, (0, 255, 0), -1)
            
        elif base_video.startswith("action_"):
            # Red pattern for action
            frame[:, :, 0] = 100
            # Animated elements
            angle = (frame_index * 5) % 360
            cv2.ellipse(frame, (256, 256), (100, 150), angle, 0, 360, (255, 100, 100), 3)
        
        # Add video name
        cv2.putText(frame, base_video, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame {frame_index % 250}", (10, 480),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        return frame
    
    def cleanup_session(self, session_id: str):
        """Clean up resources for session."""
        if session_id in self.encoders:
            del self.encoders[session_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get video service statistics."""
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "models_loaded": self.vae is not None,
            "active_encoders": len(self.encoders),
            "encoder_sessions": list(self.encoders.keys())
        }