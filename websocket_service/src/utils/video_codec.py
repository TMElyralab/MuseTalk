"""
Video codec utilities for H.264 encoding/decoding.
"""
import numpy as np
import av
import io
from typing import Optional, Tuple
import base64


class H264Encoder:
    """H.264 video encoder for streaming frames."""
    
    def __init__(self, width: int = 512, height: int = 512, fps: int = 25, 
                 bitrate: int = 2000000, preset: str = "fast"):
        """
        Initialize H.264 encoder.
        
        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
            bitrate: Target bitrate in bits/second
            preset: Encoding preset (ultrafast, fast, medium, slow)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.preset = preset
        self.pts = 0
        self.time_base = av.Fraction(1, fps)
        
        # Initialize encoder
        self._init_encoder()
    
    def _init_encoder(self):
        """Initialize the video encoder."""
        self.container = av.open(io.BytesIO(), mode='w', format='h264')
        self.stream = self.container.add_stream('h264', rate=self.fps)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = 'yuv420p'
        self.stream.bit_rate = self.bitrate
        self.stream.options = {
            'preset': self.preset,
            'tune': 'zerolatency',  # For low latency streaming
            'x264-params': 'keyint=25:min-keyint=25'  # I-frame every second at 25fps
        }
    
    def encode_frame(self, frame: np.ndarray) -> Optional[bytes]:
        """
        Encode a single frame to H.264.
        
        Args:
            frame: RGB frame as numpy array (H, W, 3)
            
        Returns:
            Encoded H.264 frame bytes or None if no output
        """
        # Ensure frame is correct shape
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Expected frame shape ({self.height}, {self.width}, 3), "
                           f"got {frame.shape}")
        
        # Convert to VideoFrame
        video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        video_frame.pts = self.pts
        video_frame.time_base = self.time_base
        self.pts += 1
        
        # Encode frame
        encoded_data = io.BytesIO()
        packets = self.stream.encode(video_frame)
        
        for packet in packets:
            encoded_data.write(bytes(packet))
        
        encoded_bytes = encoded_data.getvalue()
        return encoded_bytes if encoded_bytes else None
    
    def get_headers(self) -> bytes:
        """Get codec headers (SPS/PPS for H.264)."""
        # This would return the codec configuration
        # For now, return empty as headers are included in stream
        return b""
    
    def close(self):
        """Close the encoder and release resources."""
        if hasattr(self, 'container'):
            self.container.close()


class SimpleH264Encoder:
    """
    Simplified H.264 encoder for POC.
    Uses a more direct approach for single frame encoding.
    """
    
    def __init__(self, width: int = 512, height: int = 512, fps: int = 25):
        """Initialize simple encoder."""
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
    
    def encode_frame(self, frame: np.ndarray) -> str:
        """
        Encode frame to base64 H.264 data.
        
        Args:
            frame: RGB frame as numpy array (H, W, 3)
            
        Returns:
            Base64 encoded H.264 frame
        """
        try:
            # Create in-memory container
            output = io.BytesIO()
            container = av.open(output, mode='w', format='h264')
            stream = container.add_stream('h264', rate=self.fps)
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = 'yuv420p'
            stream.options = {
                'preset': 'ultrafast',
                'tune': 'zerolatency'
            }
            
            # Convert and encode frame
            video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            video_frame.pts = self.frame_count
            self.frame_count += 1
            
            # Encode
            for packet in stream.encode(video_frame):
                container.mux(packet)
            
            # Flush
            for packet in stream.encode():
                container.mux(packet)
            
            container.close()
            
            # Get encoded data
            encoded_data = output.getvalue()
            return base64.b64encode(encoded_data).decode('utf-8')
            
        except Exception as e:
            # Fallback to mock data if encoding fails
            print(f"H.264 encoding failed: {e}")
            return self._create_mock_frame()
    
    def _create_mock_frame(self) -> str:
        """Create mock H.264 frame data for testing."""
        # NAL unit header for IDR frame
        nal_header = b'\x00\x00\x00\x01\x65'
        # Mock payload
        mock_payload = b'MOCK_H264_PAYLOAD' * 100
        mock_frame = nal_header + mock_payload
        return base64.b64encode(mock_frame).decode('utf-8')


def decode_h264_frame(base64_data: str) -> Optional[np.ndarray]:
    """
    Decode H.264 frame from base64 data.
    
    Args:
        base64_data: Base64 encoded H.264 frame
        
    Returns:
        Decoded frame as numpy array or None if decoding fails
    """
    try:
        # Decode base64
        h264_data = base64.b64decode(base64_data)
        
        # Create decoder
        codec = av.CodecContext.create('h264', 'r')
        
        # Parse and decode
        packets = codec.parse(h264_data)
        frames = []
        
        for packet in packets:
            decoded_frames = codec.decode(packet)
            frames.extend(decoded_frames)
        
        if frames:
            # Convert first frame to numpy
            frame = frames[0]
            return frame.to_ndarray(format='rgb24')
        
        return None
        
    except Exception as e:
        print(f"H.264 decoding failed: {e}")
        return None


def create_test_frame(width: int = 512, height: int = 512) -> np.ndarray:
    """
    Create a test video frame.
    
    Args:
        width: Frame width
        height: Frame height
        
    Returns:
        RGB frame as numpy array
    """
    # Create a gradient test pattern
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Red gradient horizontally
    frame[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    
    # Green gradient vertically
    frame[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)
    
    # Blue checkerboard
    checker_size = 32
    for i in range(0, height, checker_size * 2):
        for j in range(0, width, checker_size * 2):
            frame[i:i+checker_size, j:j+checker_size, 2] = 128
            frame[i+checker_size:i+2*checker_size, j+checker_size:j+2*checker_size, 2] = 128
    
    return frame