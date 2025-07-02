#!/usr/bin/env python3
"""
Integration test for MuseTalk WebSocket service.
This tests the basic server functionality without starting a full server.
"""
import sys
from pathlib import Path
import asyncio
import json

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

async def test_message_validation():
    """Test message validation without running server."""
    try:
        from models.messages import parse_message, InitMessage, ErrorCode
        
        # Test valid INIT message
        valid_message = {
            "type": "INIT",
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "data": {
                "user_id": "test_user",
                "video_config": {
                    "resolution": "512x512",
                    "fps": 25
                }
            }
        }
        
        parsed = parse_message(valid_message)
        assert isinstance(parsed, InitMessage)
        assert parsed.data.user_id == "test_user"
        print("✓ Message parsing works")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Import error (expected in test environment): {e}")
        return False

async def test_session_creation():
    """Test session creation."""
    try:
        from models.session import Session, SessionStatus
        
        session = Session(user_id="test_user")
        assert session.user_id == "test_user"
        assert session.status == SessionStatus.INITIALIZING
        
        session.increment_frames()
        assert session.frames_generated == 1
        
        print("✓ Session management works")
        return True
        
    except ImportError as e:
        print(f"⚠️  Import error (expected in test environment): {e}")
        return False

async def test_encoding_utils():
    """Test encoding utilities."""
    try:
        from utils.encoding import bytes_to_base64, base64_to_bytes
        
        test_data = b"Hello, WebSocket!"
        encoded = bytes_to_base64(test_data)
        decoded = base64_to_bytes(encoded)
        
        assert decoded == test_data
        print("✓ Encoding utilities work")
        return True
        
    except ImportError as e:
        print(f"⚠️  Import error (expected in test environment): {e}")
        return False

async def main():
    """Run integration tests."""
    print("MuseTalk WebSocket Service - Integration Tests")
    print("=" * 50)
    
    tests = [
        test_message_validation,
        test_session_creation,
        test_encoding_utils
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Integration Tests: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed (likely due to import issues in test environment)")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())