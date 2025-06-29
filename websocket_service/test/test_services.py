#!/usr/bin/env python3
"""
Unit tests for MuseTalk WebSocket services.
"""
import pytest
import asyncio
import numpy as np
import base64
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from models.messages import AudioChunk, VideoStateType
    from models.session import Session, SessionStatus
    from services.avatar_service import AvatarService
    from services.audio_service import AudioService
    from services.video_service import VideoService
    from utils.encoding import base64_to_bytes, bytes_to_base64, pcm_to_numpy
except ImportError as e:
    print(f"Import error: {e}")
    # Skip tests if imports fail
    pytest.skip("Service imports not available", allow_module_level=True)


class TestAvatarService:
    """Test avatar service."""
    
    @pytest.fixture
    def avatar_service(self, tmp_path):
        """Create avatar service with temporary directory."""
        return AvatarService(avatars_dir=str(tmp_path))
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_avatar(self, avatar_service):
        """Test loading non-existent avatar creates mock."""
        avatar_info = await avatar_service.load_avatar("test_user")
        
        assert avatar_info is not None
        assert avatar_info.user_id == "test_user"
        assert avatar_info.model_loaded is True  # Mock returns True
        assert len(avatar_info.available_videos) > 0
    
    @pytest.mark.asyncio
    async def test_prepare_avatar(self, avatar_service):
        """Test avatar preparation."""
        success = await avatar_service.prepare_avatar("test_user", "test_video.mp4")
        assert success is True
        
        # Should be able to load it now
        avatar_info = await avatar_service.load_avatar("test_user")
        assert avatar_info is not None
    
    def test_avatar_cache(self, avatar_service):
        """Test avatar caching."""
        # Initially empty
        assert len(avatar_service.avatar_cache) == 0
        
        # Clear cache
        avatar_service.clear_cache()
        assert len(avatar_service.avatar_cache) == 0


class TestAudioService:
    """Test audio service."""
    
    @pytest.fixture
    def audio_service(self):
        """Create audio service."""
        return AudioService(device="cpu")  # Use CPU for testing
    
    def test_create_mock_features(self, audio_service):
        """Test mock feature creation."""
        features = audio_service._create_mock_features()
        
        assert features.shape == (1, 50, 384)
        assert features.device.type == "cpu"
    
    @pytest.mark.asyncio
    async def test_process_audio_chunk(self, audio_service):
        """Test audio chunk processing."""
        # Create test audio chunk
        audio_data = np.random.randn(640).astype(np.float32)  # 40ms at 16kHz
        pcm_bytes = (audio_data * 32768).astype(np.int16).tobytes()
        base64_audio = base64.b64encode(pcm_bytes).decode('utf-8')
        
        audio_chunk = AudioChunk(
            format="pcm_s16le",
            sample_rate=16000,
            channels=1,
            duration_ms=40,
            data=base64_audio
        )
        
        # Process chunk
        features = await audio_service.process_audio_chunk("test_session", audio_chunk)
        
        assert features is not None
        assert features.shape == (1, 50, 384)
    
    def test_audio_buffer(self, audio_service):
        """Test audio buffering."""
        # Get buffer
        buffer = audio_service.get_buffer("test_session")
        assert len(buffer) == 0
        
        # Add some data
        test_data = np.array([1, 2, 3])
        buffer.append(test_data)
        assert len(buffer) == 1
        
        # Clear buffer
        audio_service.clear_buffer("test_session")
        assert "test_session" not in audio_service.audio_buffers


class TestVideoService:
    """Test video service."""
    
    @pytest.fixture
    def video_service(self):
        """Create video service."""
        return VideoService(device="cpu", use_float16=False)
    
    @pytest.mark.asyncio
    async def test_generate_mock_frame(self, video_service):
        """Test mock frame generation."""
        # Create test session
        session = Session(user_id="test_user")
        session.state.video_state = VideoStateType.SPEAKING
        
        # Generate frame
        frame = await video_service._generate_frame_mock(session, 0)
        
        assert frame is not None
        assert frame.shape == (512, 512, 3)
        assert frame.dtype == np.uint8
    
    @pytest.mark.asyncio
    async def test_generate_frame(self, video_service):
        """Test full frame generation."""
        # Create test session
        session = Session(user_id="test_user")
        
        # Create mock audio features
        audio_features = video_service._create_mock_features() if hasattr(video_service, '_create_mock_features') else None
        if audio_features is None:
            import torch
            audio_features = torch.randn(1, 50, 384)
        
        # Generate frame
        encoded_frame = await video_service.generate_frame(session, audio_features, 0)
        
        assert encoded_frame is not None
        assert isinstance(encoded_frame, str)  # Base64 encoded
    
    def test_encoder_management(self, video_service):
        """Test H.264 encoder management."""
        # Get encoder
        encoder1 = video_service.get_encoder("session1")
        encoder2 = video_service.get_encoder("session1")
        
        # Should be the same instance
        assert encoder1 is encoder2
        
        # Different session should get different encoder
        encoder3 = video_service.get_encoder("session2")
        assert encoder1 is not encoder3
        
        # Cleanup
        video_service.cleanup_session("session1")
        assert "session1" not in video_service.encoders


class TestEncodingUtils:
    """Test encoding utilities."""
    
    def test_base64_roundtrip(self):
        """Test base64 encoding/decoding."""
        original_data = b"Hello, World!"
        
        # Encode
        encoded = bytes_to_base64(original_data)
        assert isinstance(encoded, str)
        
        # Decode
        decoded = base64_to_bytes(encoded)
        assert decoded == original_data
    
    def test_pcm_to_numpy(self):
        """Test PCM to numpy conversion."""
        # Create test PCM data (16-bit signed)
        test_values = [0, 16384, -16384, 32767, -32768]
        pcm_bytes = b''.join(val.to_bytes(2, 'little', signed=True) for val in test_values)
        
        # Convert to numpy
        audio_array = pcm_to_numpy(pcm_bytes)
        
        assert len(audio_array) == len(test_values)
        assert np.allclose(audio_array[0], 0.0, atol=1e-4)
        assert np.allclose(audio_array[1], 0.5, atol=1e-4)
        assert np.allclose(audio_array[2], -0.5, atol=1e-4)


class TestSession:
    """Test session model."""
    
    def test_session_creation(self):
        """Test session creation."""
        session = Session(user_id="test_user")
        
        assert session.user_id == "test_user"
        assert session.status == SessionStatus.INITIALIZING
        assert session.frames_generated == 0
        assert session.total_audio_ms == 0
    
    def test_session_state_management(self):
        """Test session state updates."""
        session = Session(user_id="test_user")
        
        # Update status
        session.set_status(SessionStatus.READY)
        assert session.status == SessionStatus.READY
        
        # Update video state
        session.set_video_state(VideoStateType.SPEAKING, "speaking_0")
        assert session.state.video_state == VideoStateType.SPEAKING
        assert session.state.current_video == "speaking_0"
        
        # Increment frames
        session.increment_frames()
        assert session.frames_generated == 1
        
        # Add audio duration
        session.add_audio_duration(40)
        assert session.total_audio_ms == 40
    
    def test_session_action_management(self):
        """Test action management."""
        session = Session(user_id="test_user")
        
        # Start action
        session.start_action("action_1")
        assert session.state.action_in_progress == "action_1"
        assert session.state.action_progress == 0.0
        
        # Update progress
        session.update_action_progress(0.5)
        assert session.state.action_progress == 0.5
        
        # Complete action
        session.update_action_progress(1.0)
        assert session.state.action_in_progress is None
        assert session.state.action_progress == 1.0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])