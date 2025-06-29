"""
Encoding/decoding utilities for audio and video data.
"""
import base64
import numpy as np
from typing import Union, Tuple
import struct


def base64_to_bytes(base64_string: str) -> bytes:
    """
    Convert base64 string to bytes.
    
    Args:
        base64_string: Base64 encoded string
        
    Returns:
        Decoded bytes
    """
    return base64.b64decode(base64_string)


def bytes_to_base64(data: bytes) -> str:
    """
    Convert bytes to base64 string.
    
    Args:
        data: Binary data
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(data).decode('utf-8')


def pcm_to_numpy(pcm_data: bytes, sample_rate: int = 16000) -> np.ndarray:
    """
    Convert PCM audio data to numpy array.
    
    Args:
        pcm_data: Raw PCM bytes (16-bit signed, little-endian)
        sample_rate: Sample rate (default 16000)
        
    Returns:
        Numpy array of audio samples normalized to [-1, 1]
    """
    # Convert bytes to int16 array
    audio_array = np.frombuffer(pcm_data, dtype=np.int16)
    
    # Normalize to [-1, 1]
    audio_array = audio_array.astype(np.float32) / 32768.0
    
    return audio_array


def numpy_to_pcm(audio_array: np.ndarray) -> bytes:
    """
    Convert numpy audio array to PCM bytes.
    
    Args:
        audio_array: Numpy array of audio samples (expected range [-1, 1])
        
    Returns:
        PCM bytes (16-bit signed, little-endian)
    """
    # Clip to valid range
    audio_array = np.clip(audio_array, -1.0, 1.0)
    
    # Convert to int16
    audio_int16 = (audio_array * 32768.0).astype(np.int16)
    
    # Convert to bytes
    return audio_int16.tobytes()


def decode_audio_chunk(base64_audio: str) -> Tuple[np.ndarray, int]:
    """
    Decode base64 audio chunk to numpy array.
    
    Args:
        base64_audio: Base64 encoded PCM audio
        
    Returns:
        Tuple of (audio_array, num_samples)
    """
    pcm_bytes = base64_to_bytes(base64_audio)
    audio_array = pcm_to_numpy(pcm_bytes)
    return audio_array, len(audio_array)


def validate_audio_chunk_size(base64_audio: str, expected_duration_ms: int = 40, 
                            sample_rate: int = 16000) -> bool:
    """
    Validate that audio chunk has expected size.
    
    Args:
        base64_audio: Base64 encoded PCM audio
        expected_duration_ms: Expected duration in milliseconds
        sample_rate: Sample rate
        
    Returns:
        True if size matches expected
    """
    try:
        pcm_bytes = base64_to_bytes(base64_audio)
        expected_samples = int(sample_rate * expected_duration_ms / 1000)
        expected_bytes = expected_samples * 2  # 16-bit = 2 bytes per sample
        return len(pcm_bytes) == expected_bytes
    except Exception:
        return False


def encode_frame_data(frame: np.ndarray) -> str:
    """
    Encode video frame to base64.
    
    This is a placeholder for H.264 encoding.
    In production, this would use PyAV for proper H.264 encoding.
    
    Args:
        frame: Video frame as numpy array (H, W, C)
        
    Returns:
        Base64 encoded frame data
    """
    # For POC, we'll just encode the raw frame
    # In production, this should use H.264 encoding
    frame_bytes = frame.tobytes()
    return bytes_to_base64(frame_bytes)


def create_mock_h264_frame() -> str:
    """
    Create a mock H.264 frame for testing.
    
    Returns:
        Base64 encoded mock H.264 data
    """
    # This is just mock data for testing
    # Real implementation would generate actual H.264 frames
    mock_data = b"MOCK_H264_FRAME_DATA"
    return bytes_to_base64(mock_data)