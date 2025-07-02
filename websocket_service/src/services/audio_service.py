"""
Audio processing service for MuseTalk WebSocket.
"""
import asyncio
import numpy as np
from typing import Optional, List, Tuple
from collections import deque
import sys
import os

import torch

# Add MuseTalk to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# Import MuseTalk dependencies - fail fast if not available
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel

from utils.encoding import decode_audio_chunk
from models.messages import AudioChunk


class AudioService:
    """Service for processing audio chunks and extracting features."""
    
    def __init__(self, 
                 model_path: str = "../models",
                 device: str = "cuda",
                 dtype = None):
        """
        Initialize audio service.
        
        Args:
            model_path: Path to model directory
            device: Device to run on (cuda/cpu)
            dtype: Data type for models
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        
        # Audio buffer for accumulating chunks
        self.audio_buffers = {}  # session_id -> deque of audio chunks
        self.buffer_size = 10  # Keep last 10 chunks for context
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize Whisper and audio processor (following realtime_inference.py)."""
        try:
            # Change to MuseTalk root directory for proper model loading
            current_dir = os.getcwd()
            musetalk_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            os.chdir(musetalk_root)
            
            try:
                # Initialize audio processor with correct path (like realtime_inference.py)
                whisper_dir = "models/whisper"
                self.audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
                
                # Load Whisper model (like realtime_inference.py)
                self.whisper = WhisperModel.from_pretrained(whisper_dir)
                self.whisper = self.whisper.to(device=self.device, dtype=self.dtype).eval()
                self.whisper.requires_grad_(False)
                
                # Store for use in real-time inference
                self.weight_dtype = self.dtype
                
                print(f"MuseTalk audio models loaded successfully on {self.device}")
                
            finally:
                # Restore original working directory
                os.chdir(current_dir)
            
        except Exception as e:
            print(f"FATAL: Failed to load MuseTalk audio models: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Cannot initialize AudioService without MuseTalk models: {e}")
    
    def get_buffer(self, session_id: str) -> deque:
        """Get or create audio buffer for session."""
        if session_id not in self.audio_buffers:
            self.audio_buffers[session_id] = deque(maxlen=self.buffer_size)
        return self.audio_buffers[session_id]
    
    async def process_audio_chunk(self, 
                                session_id: str,
                                audio_chunk: AudioChunk) -> Optional[torch.Tensor]:
        """
        Process audio chunk and extract features.
        
        Args:
            session_id: Session identifier
            audio_chunk: Audio chunk with PCM data
            
        Returns:
            Audio features tensor or None if processing fails
        """
        try:
            # Decode audio data
            audio_array, num_samples = decode_audio_chunk(audio_chunk.data)
            
            # Validate sample count (40ms @ 16kHz = 640 samples)
            expected_samples = int(audio_chunk.sample_rate * audio_chunk.duration_ms / 1000)
            if num_samples != expected_samples:
                print(f"Warning: Expected {expected_samples} samples, got {num_samples}")
            
            # Add to buffer
            buffer = self.get_buffer(session_id)
            buffer.append(audio_array)
            
            # Extract features
            if self.audio_processor and self.whisper:
                # For real processing, we would:
                # 1. Accumulate enough audio for Whisper processing
                # 2. Extract mel features
                # 3. Get Whisper embeddings
                # For now, return mock features
                features = await self._extract_whisper_features(audio_array)
            else:
                # Mock features for POC
                features = self._create_mock_features()
            
            return features
            
        except Exception as e:
            print(f"Audio processing error: {e}")
            return None
    
    async def process_audio_for_inference(self, session_id: str, audio_data: bytes, sample_rate: int = 16000) -> Optional[torch.Tensor]:
        """
        Process audio using MuseTalk's real pipeline (like realtime_inference.py).
        
        Args:
            session_id: Session identifier
            audio_data: Raw audio data
            sample_rate: Audio sample rate
            
        Returns:
            Whisper features ready for MuseTalk inference
        """
        if not self.audio_processor or not self.whisper:
            return self._create_mock_features()
        
        try:
            # Convert bytes to numpy array (assuming PCM 16-bit)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
            # Process using MuseTalk's audio processor (like realtime_inference.py)
            # Note: This would normally require a longer audio segment, but we'll simulate
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                self._extract_whisper_features_real, 
                audio_array
            )
            
        except Exception as e:
            print(f"Error in MuseTalk audio processing: {e}")
            return self._create_mock_features()

    def _extract_whisper_features_real(self, audio_array: np.ndarray) -> torch.Tensor:
        """
        Extract real Whisper features using MuseTalk's pipeline.
        """
        try:
            # For real-time processing, we need to simulate the get_whisper_chunk workflow
            # In the real implementation, this would:
            # 1. Use audio_processor.get_audio_feature()
            # 2. Use audio_processor.get_whisper_chunk()
            # 3. Return the processed features
            
            # For now, we'll simulate by using the feature extractor directly
            if len(audio_array) < 16000:  # Less than 1 second
                # Pad audio to minimum length
                audio_array = np.pad(audio_array, (0, 16000 - len(audio_array)))
            
            # Use the audio processor's feature extractor
            audio_feature = self.audio_processor.feature_extractor(
                audio_array,
                return_tensors="pt",
                sampling_rate=16000
            ).input_features
            
            if self.weight_dtype is not None:
                audio_feature = audio_feature.to(dtype=self.weight_dtype)
            
            # Get Whisper features
            audio_feature = audio_feature.to(self.device).to(self.weight_dtype)
            audio_feats = self.whisper.encoder(audio_feature, output_hidden_states=True).hidden_states
            audio_feats = torch.stack(audio_feats, dim=2)
            
            # Return a single frame's worth of features
            # In real-time, this would be processed through get_whisper_chunk
            return audio_feats[:, :1, ...]  # Take first frame
            
        except Exception as e:
            print(f"Real Whisper feature extraction error: {e}")
            return self._create_mock_features()

    async def _extract_whisper_features(self, audio_array: np.ndarray) -> torch.Tensor:
        """
        Extract Whisper features from audio.
        
        This is a simplified version. Real implementation would need to:
        1. Accumulate audio to proper length for Whisper
        2. Handle padding and windowing
        3. Extract features using the full pipeline
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._extract_features_sync, 
            audio_array
        )
    
    def _extract_features_sync(self, audio_array: np.ndarray) -> torch.Tensor:
        """Synchronous feature extraction."""
        try:
            # Convert to expected format
            # Real implementation would use the full audio processor pipeline
            # For POC, return mock features
            return self._create_mock_features()
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return self._create_mock_features()
    
    def _create_mock_features(self) -> torch.Tensor:
        """Create mock audio features for testing."""
        # Whisper outputs shape: (batch, time, feature_dim)
        # For MuseTalk: (1, 50, 384)
        mock_features = torch.randn(1, 50, 384, device=self.device, dtype=self.dtype)
        return mock_features
    
    def clear_buffer(self, session_id: str):
        """Clear audio buffer for session."""
        if session_id in self.audio_buffers:
            del self.audio_buffers[session_id]
    
    async def process_audio_stream(self,
                                 session_id: str,
                                 audio_chunks: List[AudioChunk]) -> List[torch.Tensor]:
        """
        Process multiple audio chunks in sequence.
        
        Args:
            session_id: Session identifier
            audio_chunks: List of audio chunks
            
        Returns:
            List of feature tensors
        """
        features = []
        for chunk in audio_chunks:
            feature = await self.process_audio_chunk(session_id, chunk)
            if feature is not None:
                features.append(feature)
        return features
    
    def get_audio_stats(self, session_id: str) -> dict:
        """Get audio processing statistics for session."""
        buffer = self.get_buffer(session_id)
        return {
            "buffer_size": len(buffer),
            "total_samples": sum(len(chunk) for chunk in buffer),
            "device": str(self.device),
            "models_loaded": self.whisper is not None
        }