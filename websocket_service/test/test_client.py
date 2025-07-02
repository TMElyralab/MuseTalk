#!/usr/bin/env python3
"""
Test client for MuseTalk WebSocket service.

This client demonstrates how to connect to the WebSocket service
and send/receive messages.
"""
import asyncio
import json
import base64
import uuid
import numpy as np
import websockets
from typing import Optional, Dict, Any
import argparse
import sys
import time


class MuseTalkWebSocketClient:
    """WebSocket client for MuseTalk service."""
    
    def __init__(self, server_url: str, user_id: str):
        """
        Initialize client.
        
        Args:
            server_url: WebSocket server URL
            user_id: User identifier
        """
        self.server_url = server_url
        self.user_id = user_id
        self.session_id = None  # Will be set by server during INIT
        self.websocket = None
        self.available_videos = []
        self.current_state = "idle"
        self.frames_received = 0
        self.start_time = None
    
    async def connect(self):
        """Connect to WebSocket server."""
        ws_url = f"{self.server_url}/musetalk/v1/ws/{self.user_id}"
        print(f"Connecting to {ws_url}...")
        
        try:
            # Increase timeout for connection and ping
            self.websocket = await websockets.connect(
                ws_url,
                ping_timeout=60,  # 60 second ping timeout
                ping_interval=60,  # Ping every 60 seconds
                close_timeout=10   # 10 second close timeout
            )
            print("Connected successfully!")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from server."""
        if self.websocket:
            await self.send_close()
            await self.websocket.close()
            print("Disconnected")
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to server."""
        if not self.websocket:
            print("Not connected")
            return
        
        message_str = json.dumps(message)
        await self.websocket.send(message_str)
        print(f"Sent: {message['type']}")
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message from server."""
        if not self.websocket:
            return None
        
        try:
            message_str = await self.websocket.recv()
            message = json.loads(message_str)
            return message
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed by server")
            return None
        except Exception as e:
            print(f"Receive error: {e}")
            return None
    
    async def send_init(self):
        """Send INIT message."""
        # For INIT message, we need to send a temporary session_id
        # The server will return the actual session_id we should use
        temp_session_id = str(uuid.uuid4())
        message = {
            "type": "INIT",
            "session_id": temp_session_id,
            "data": {
                "user_id": self.user_id,
                "video_config": {
                    "resolution": "512x512",
                    "fps": 25
                }
            }
        }
        await self.send_message(message)
    
    async def send_generate(self, audio_data: bytes, video_state: str = "speaking", 
                          base_video: Optional[str] = None):
        """Send GENERATE message."""
        if not base_video and self.available_videos:
            # Use first available video for the state
            matching_videos = [v for v in self.available_videos if v.startswith(video_state)]
            base_video = matching_videos[0] if matching_videos else self.available_videos[0]
        
        message = {
            "type": "GENERATE",
            "session_id": self.session_id,
            "data": {
                "audio_chunk": {
                    "format": "pcm_s16le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "duration_ms": 40,
                    "data": base64.b64encode(audio_data).decode('utf-8')
                },
                "video_state": {
                    "type": video_state,
                    "base_video": base_video or "speaking_0"
                }
            }
        }
        await self.send_message(message)
    
    async def send_state_change(self, target_state: str, base_video: str):
        """Send STATE_CHANGE message."""
        message = {
            "type": "STATE_CHANGE",
            "session_id": self.session_id,
            "data": {
                "target_state": target_state,
                "base_video": base_video
            }
        }
        await self.send_message(message)
    
    async def send_action(self, action_index: int):
        """Send ACTION message (v1.1 - no audio required)."""
        message = {
            "type": "ACTION",
            "session_id": self.session_id,
            "data": {
                "action_index": action_index
            }
        }
        await self.send_message(message)
    
    async def send_close(self):
        """Send CLOSE message."""
        message = {
            "type": "CLOSE",
            "session_id": self.session_id,
            "data": {}
        }
        await self.send_message(message)
    
    def create_mock_audio(self, duration_ms: int = 40) -> bytes:
        """Create mock PCM audio data."""
        # 40ms at 16kHz = 640 samples
        num_samples = int(16000 * duration_ms / 1000)
        # Create sine wave
        t = np.linspace(0, duration_ms / 1000, num_samples)
        frequency = 440  # A4 note
        audio = np.sin(2 * np.pi * frequency * t)
        # Convert to int16 PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()
    
    async def handle_response(self, response: Dict[str, Any]):
        """Handle response from server."""
        msg_type = response.get("type")
        
        if msg_type == "INIT_SUCCESS":
            # Extract and store the session ID from the server response
            self.session_id = response.get("session_id")
            self.available_videos = response["data"]["available_videos"]
            default_video = response["data"].get("default_video", "idle_0")
            streaming_started = response["data"].get("streaming_started", False)
            print(f"Initialization successful!")
            print(f"Session ID: {self.session_id}")
            print(f"Default video: {default_video}")
            print(f"Streaming started: {streaming_started}")
            print(f"Available videos: {', '.join(self.available_videos[:5])}...")
            
        elif msg_type == "VIDEO_FRAME":
            self.frames_received += 1
            frame_size = len(response["data"]["frame_data"])
            timestamp = response["data"]["frame_timestamp"]
            
            if self.frames_received == 1:
                self.start_time = time.time()
            
            elapsed = time.time() - self.start_time if self.start_time else 0
            fps = self.frames_received / elapsed if elapsed > 0 else 0
            
            print(f"Frame {self.frames_received}: {frame_size} bytes, "
                  f"timestamp: {timestamp}ms, FPS: {fps:.2f}")
            
        elif msg_type == "STATE_CHANGED":
            self.current_state = response["data"]["current_state"]
            print(f"State changed to: {self.current_state}")
            
        elif msg_type == "ACTION_FRAME":
            progress = response["data"]["action_progress"]
            print(f"Action frame received, progress: {progress:.2%}")
            
        elif msg_type == "ACTION_TRIGGERED":
            action_index = response["data"]["action_index"]
            inserted = response["data"]["inserted"]
            print(f"Action {action_index} triggered, inserted: {inserted}")
            
        elif msg_type == "ERROR":
            error_code = response["data"]["code"]
            error_msg = response["data"]["message"]
            error_details = response["data"].get("details")
            if error_details:
                print(f"ERROR: {error_code} - {error_msg} - Details: {error_details}")
            else:
                print(f"ERROR: {error_code} - {error_msg}")
            
        elif msg_type == "CLOSE_ACK":
            reason = response["data"]["reason"]
            print(f"Connection closed: {reason}")
            
        elif msg_type == "PING":
            # Heartbeat
            pass
            
        else:
            print(f"Unknown message type: {msg_type}")
    
    async def run_test_sequence(self):
        """Run a test sequence of operations."""
        print("\n=== Starting Test Sequence ===\n")
        
        # Connect
        if not await self.connect():
            return
        
        # Create background task to continuously receive frames
        receive_task = None
        
        async def receive_frames():
            """Continuously receive video frames."""
            while True:
                try:
                    response = await self.receive_message()
                    if response:
                        await self.handle_response(response)
                except Exception as e:
                    print(f"Receive error: {e}")
                    break
        
        try:
            # Initialize
            print("\n1. Initializing session...")
            await self.send_init()
            
            # Start receiving frames in background
            receive_task = asyncio.create_task(receive_frames())
            
            # Wait for initialization and initial frames
            await asyncio.sleep(2)
            
            # Change state to speaking
            if self.available_videos:
                print("\n2. Changing state to speaking...")
                speaking_videos = [v for v in self.available_videos if v.startswith("speaking")]
                if speaking_videos:
                    await self.send_state_change("speaking", speaking_videos[0])
                    # Don't manually receive here - let the background task handle it
                    await asyncio.sleep(0.5)  # Wait for state change to process
            
            # Send some audio frames
            print("\n3. Sending audio frames...")
            for i in range(10):
                audio_data = self.create_mock_audio()
                await self.send_generate(audio_data)
                # Small delay between frames
                await asyncio.sleep(0.04)  # 40ms
            
            # Wait a bit to see lip-sync frames
            await asyncio.sleep(1)
            
            # Try an action
            print("\n4. Triggering action...")
            await self.send_action(1)  # Trigger action_1
            
            # Wait for action to complete
            await asyncio.sleep(2)
            
            # Stats
            if self.frames_received > 0 and self.start_time:
                total_time = time.time() - self.start_time
                avg_fps = self.frames_received / total_time
                print(f"\n=== Test Complete ===")
                print(f"Total frames: {self.frames_received}")
                print(f"Total time: {total_time:.2f}s")
                print(f"Average FPS: {avg_fps:.2f}")
            
        except Exception as e:
            print(f"Test error: {e}")
            
        finally:
            # Cancel receive task
            if receive_task:
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
            
            # Disconnect
            print("\n5. Closing connection...")
            await self.disconnect()


async def interactive_mode(client: MuseTalkWebSocketClient):
    """Run client in interactive mode."""
    print("\n=== Interactive Mode ===")
    print("Commands:")
    print("  init - Initialize session")
    print("  generate [n] - Send n audio chunks (default 1)")
    print("  state <state> <video> - Change state")
    print("  action <index> - Trigger action (1 or 2)")
    print("  quit - Exit")
    print()
    
    if not await client.connect():
        return
    
    # Background task to receive messages
    async def receive_loop():
        while client.websocket:
            response = await client.receive_message()
            if response:
                await client.handle_response(response)
    
    receive_task = asyncio.create_task(receive_loop())
    
    try:
        while True:
            # Get command
            command = await asyncio.get_event_loop().run_in_executor(
                None, input, ">> "
            )
            
            parts = command.strip().split()
            if not parts:
                continue
            
            cmd = parts[0].lower()
            
            if cmd == "quit":
                break
                
            elif cmd == "init":
                await client.send_init()
                
            elif cmd == "generate":
                count = int(parts[1]) if len(parts) > 1 else 1
                for i in range(count):
                    audio_data = client.create_mock_audio()
                    await client.send_generate(audio_data)
                    await asyncio.sleep(0.04)
                    
            elif cmd == "state" and len(parts) >= 3:
                await client.send_state_change(parts[1], parts[2])
                
            elif cmd == "action" and len(parts) >= 2:
                action_index = int(parts[1])
                await client.send_action(action_index)
                
            else:
                print("Unknown command")
                
    except KeyboardInterrupt:
        print("\nInterrupted")
        
    finally:
        receive_task.cancel()
        await client.disconnect()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MuseTalk WebSocket Test Client")
    parser.add_argument(
        "--server", 
        default="ws://localhost:8000",
        help="WebSocket server URL"
    )
    parser.add_argument(
        "--user-id",
        default="test_user",
        help="User ID"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Create client
    client = MuseTalkWebSocketClient(args.server, args.user_id)
    
    # Run
    if args.interactive:
        asyncio.run(interactive_mode(client))
    else:
        asyncio.run(client.run_test_sequence())


if __name__ == "__main__":
    main()