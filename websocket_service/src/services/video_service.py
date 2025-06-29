"""
Video generation service for MuseTalk WebSocket.
"""
import asyncio
import numpy as np
import torch
import cv2
from typing import Optional, Tuple, Dict, Any
import sys
import os
from pathlib import Path

# Add MuseTalk to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from musetalk.utils.utils import load_all_model
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing

from ..utils.video_codec import SimpleH264Encoder
from ..models.messages import VideoStateType
from ..models.session import Session


class VideoService:
    """Service for generating lip-synced video frames."""
    
    def __init__(self,
                 model_path: str = "../models",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32,
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
        self.dtype = torch.float16 if use_float16 else torch.float32
        
        # Initialize models
        self._init_models()
        
        # H.264 encoders per session
        self.encoders = {}  # session_id -> encoder
        
        # Face parsing
        self.face_parser = FaceParsing()
    
    def _init_models(self):
        """Initialize VAE, UNet, and PE models."""
        try:
            # Load models
            unet_path = self.model_path / "musetalkV15" / "unet.pth"
            unet_config = self.model_path / "musetalkV15" / "musetalk.json"
            
            self.vae, self.unet, self.pe = load_all_model(
                unet_model_path=str(unet_path),
                vae_type="sd-vae",
                unet_config=str(unet_config),
                device=self.device
            )
            
            # Convert to appropriate dtype
            if self.dtype == torch.float16:
                self.pe = self.pe.half()
                self.vae.vae = self.vae.vae.half()
                self.unet.model = self.unet.model.half()
            
            # Move to device
            self.pe = self.pe.to(self.device)
            self.vae.vae = self.vae.vae.to(self.device)
            self.unet.model = self.unet.model.to(self.device)
            
            # Fixed timestep for non-diffusion inference
            self.timesteps = torch.tensor([0], device=self.device)
            
            print(f"Video models loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Failed to load video models: {e}")
            # For POC, we'll continue without real models
            self.vae = None
            self.unet = None
            self.pe = None
    
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
        """Synchronous frame generation."""
        try:
            # This is a simplified version
            # Real implementation would:
            # 1. Load avatar latents for the frame
            # 2. Get audio embedding with PE
            # 3. Run UNet inference
            # 4. Decode with VAE
            # 5. Blend with original frame
            
            with torch.no_grad():
                # Get audio embedding
                audio_emb = self.pe(audio_features)
                
                # Mock latent (in real version, load from avatar)
                latent = torch.randn(1, 4, 32, 32, device=self.device, dtype=self.dtype)
                
                # UNet inference
                pred_latents = self.unet.model(
                    latent, 
                    self.timesteps, 
                    encoder_hidden_states=audio_emb
                ).sample
                
                # VAE decode
                frame = self.vae.decode_latents(pred_latents)
                
                # Convert to numpy
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                
                # Ensure correct shape and type
                if len(frame.shape) == 4:
                    frame = frame[0]  # Remove batch dimension
                
                # Convert to uint8 RGB
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
                
                # Resize to 512x512 if needed
                if frame.shape[:2] != (512, 512):
                    frame = cv2.resize(frame, (512, 512))
                
                return frame
                
        except Exception as e:
            print(f"Real frame generation error: {e}")
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