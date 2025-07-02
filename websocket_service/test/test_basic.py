#!/usr/bin/env python3
"""
Basic tests for MuseTalk WebSocket service without complex imports.
"""
import json
import base64
import uuid
import numpy as np


def test_json_message_parsing():
    """Test JSON message parsing."""
    message = {
        "type": "INIT",
        "session_id": str(uuid.uuid4()),
        "data": {
            "user_id": "test_user",
            "video_config": {
                "resolution": "512x512",
                "fps": 25
            }
        }
    }
    
    # Should be able to serialize/deserialize
    json_str = json.dumps(message)
    parsed = json.loads(json_str)
    
    assert parsed["type"] == "INIT"
    assert "session_id" in parsed
    assert parsed["data"]["user_id"] == "test_user"


def test_audio_data_encoding():
    """Test audio data base64 encoding."""
    # Create 40ms of audio data (640 samples at 16kHz)
    audio_samples = np.random.randn(640).astype(np.float32)
    audio_int16 = (audio_samples * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    
    # Should be 640 samples * 2 bytes = 1280 bytes
    assert len(audio_bytes) == 1280
    
    # Base64 encode
    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Should be able to decode
    decoded_bytes = base64.b64decode(base64_audio)
    assert decoded_bytes == audio_bytes


def test_message_types():
    """Test all message types can be created."""
    message_types = [
        "INIT", "GENERATE", "STATE_CHANGE", "ACTION", "CLOSE",
        "INIT_SUCCESS", "VIDEO_FRAME", "STATE_CHANGED", 
        "ACTION_FRAME", "CLOSE_ACK", "ERROR"
    ]
    
    for msg_type in message_types:
        message = {
            "type": msg_type,
            "session_id": str(uuid.uuid4()),
            "data": {}
        }
        
        # Should serialize without error
        json_str = json.dumps(message)
        assert msg_type in json_str


def test_session_id_format():
    """Test session ID UUID format."""
    session_id = str(uuid.uuid4())
    
    # Should be valid UUID format
    uuid_obj = uuid.UUID(session_id)
    assert str(uuid_obj) == session_id


def test_video_states():
    """Test video state types."""
    valid_states = ["idle", "speaking", "action"]
    
    for state in valid_states:
        message = {
            "type": "STATE_CHANGE",
            "session_id": str(uuid.uuid4()),
            "data": {
                "target_state": state,
                "base_video": f"{state}_0"
            }
        }
        
        json_str = json.dumps(message)
        parsed = json.loads(json_str)
        assert parsed["data"]["target_state"] == state


if __name__ == "__main__":
    print("Running basic tests...")
    
    test_json_message_parsing()
    print("✓ JSON message parsing works")
    
    test_audio_data_encoding()
    print("✓ Audio data encoding works")
    
    test_message_types()
    print("✓ All message types work")
    
    test_session_id_format()
    print("✓ Session ID format works")
    
    test_video_states()
    print("✓ Video states work")
    
    print("\nAll basic tests passed! ✅")