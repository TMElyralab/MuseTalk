#!/usr/bin/env python3
"""
Mock MuseTalk WebSocket Server

This server implements the MuseTalk WebSocket API specification to allow testing
of the voice agent demo without requiring the actual MuseTalk service.
"""

import asyncio
import websockets
import json
import base64
import uuid
import random
from typing import Dict, Set
import logging
from websockets.server import serve

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockMuseTalkSession:
    def __init__(self, websocket, path):
        self.websocket = websocket
        self.path = path
        self.session_id = None
        self.user_id = None
        self.initialized = False
        self.current_state = "idle"
        self.available_videos = [
            "idle_0", "idle_1", "idle_2", "idle_3", "idle_4", "idle_5", "idle_6",
            "speaking_0", "speaking_1", "speaking_2", "speaking_3", "speaking_4", "speaking_5", "speaking_6", "speaking_7",
            "action_1", "action_2"
        ]
        self.default_video = "idle_0"
        self.frame_counter = 0
        self.streaming_task = None
        self.start_time = None
        
    async def handle_message(self, message):
        try:
            data = json.loads(message)
            message_type = data.get("type")
            self.session_id = data.get("session_id")
            
            logger.info(f"Received message: {message_type} for session {self.session_id}")
            
            if message_type == "INIT":
                await self.handle_init(data)
            elif message_type == "GENERATE":
                await self.handle_generate(data)
            elif message_type == "STATE_CHANGE":
                await self.handle_state_change(data)
            elif message_type == "ACTION":
                await self.handle_action(data)
            elif message_type == "CLOSE":
                await self.handle_close(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Failed to parse message: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def handle_init(self, data):
        self.user_id = data["data"].get("user_id")
        auth_token = data["data"].get("auth_token")
        video_config = data["data"].get("video_config", {})
        
        # Simulate initialization success
        response = {
            "type": "INIT_SUCCESS",
            "session_id": self.session_id,
            "data": {
                "model_loaded": True,
                "available_videos": self.available_videos,
                "default_video": self.default_video,
                "streaming_started": True
            }
        }
        
        await self.websocket.send(json.dumps(response))
        self.initialized = True
        self.start_time = asyncio.get_event_loop().time()
        
        # Start streaming default video frames
        self.streaming_task = asyncio.create_task(self.stream_default_video())
        logger.info(f"Initialized session for user {self.user_id}")
    
    async def handle_generate(self, data):
        if not self.initialized:
            logger.warning("Received GENERATE request but session not initialized")
            return
            
        audio_chunk = data["data"].get("audio_chunk", {})
        video_state = data["data"].get("video_state", {})
        
        # Extract audio info
        duration_ms = audio_chunk.get("duration_ms", 40)
        audio_size = len(base64.b64decode(audio_chunk.get("data", ""))) if audio_chunk.get("data") else 0
        
        # Validate audio chunk size
        expected_size = int(duration_ms * 16 * 2 / 1000)  # 16kHz, 16-bit (2 bytes)
        if audio_size > 0 and audio_size != expected_size:
            logger.warning(f"Audio chunk size mismatch: expected {expected_size} bytes for {duration_ms}ms, got {audio_size} bytes")
        
        # Validate base_video if provided
        if video_state.get("base_video") and video_state["base_video"] not in self.available_videos:
            logger.warning(f"Invalid base_video in GENERATE: {video_state['base_video']}")
            error_response = {
                "type": "ERROR",
                "session_id": self.session_id,
                "data": {
                    "error": "invalid_base_video",
                    "message": f"Base video '{video_state['base_video']}' not in available videos"
                }
            }
            await self.websocket.send(json.dumps(error_response))
            return
        
        logger.info(f"Processing GENERATE request:")
        logger.info(f"- Audio chunk: {audio_size} bytes, {duration_ms}ms")
        logger.info(f"- Video state: {video_state}")
        
        # Update current state based on video_state
        if video_state.get("type"):
            old_state = self.current_state
            self.current_state = video_state["type"]
            logger.info(f"State changed from {old_state} to {self.current_state}")
        
        # Generate mock video frame
        logger.debug(f"Generating video frame for {duration_ms}ms audio")
        await self.send_video_frame(duration_ms)
    
    async def handle_state_change(self, data):
        if not self.initialized:
            return
            
        target_state = data["data"].get("target_state")
        base_video = data["data"].get("base_video")
        
        # Validate base_video if provided
        if base_video and base_video not in self.available_videos:
            logger.warning(f"Invalid base_video: {base_video}. Not in available videos.")
            error_response = {
                "type": "ERROR",
                "session_id": self.session_id,
                "data": {
                    "error": "invalid_base_video",
                    "message": f"Base video '{base_video}' not in available videos"
                }
            }
            await self.websocket.send(json.dumps(error_response))
            return
        
        self.current_state = target_state
        
        response = {
            "type": "STATE_CHANGED",
            "session_id": self.session_id,
            "data": {
                "current_state": target_state,
                "current_video": base_video
            }
        }
        
        await self.websocket.send(json.dumps(response))
        logger.info(f"State changed to {target_state}")
    
    async def handle_action(self, data):
        if not self.initialized:
            return
            
        action_index = data["data"].get("action_index")
        
        # Validate action_index
        if action_index not in [1, 2]:
            logger.warning(f"Invalid action_index: {action_index}. Must be 1 or 2.")
            error_response = {
                "type": "ERROR",
                "session_id": self.session_id,
                "data": {
                    "error": "invalid_action_index",
                    "message": f"Action index must be 1 or 2, got {action_index}"
                }
            }
            await self.websocket.send(json.dumps(error_response))
            return
        
        response = {
            "type": "ACTION_TRIGGERED",
            "session_id": self.session_id,
            "data": {
                "action_index": action_index,
                "inserted": True
            }
        }
        
        await self.websocket.send(json.dumps(response))
        logger.info(f"Action {action_index} triggered")
        
        # Send a few frames for the action
        for _ in range(5):  # 5 frames for action
            await asyncio.sleep(0.04)  # 40ms delay
            await self.send_video_frame(40)
    
    async def handle_close(self, data):
        response = {
            "type": "CLOSE_ACK",
            "session_id": self.session_id,
            "data": {
                "reason": "client_request"
            }
        }
        
        await self.websocket.send(json.dumps(response))
        
        if self.streaming_task:
            self.streaming_task.cancel()
        
        logger.info("Session closed")
    
    async def send_video_frame(self, duration_ms=40):
        """Send a mock H.264 video frame"""
        self.frame_counter += 1
        
        # Create mock H.264 frame data (just some random bytes for testing)
        # In a real implementation, this would be actual H.264 encoded data
        mock_frame_size = random.randint(1000, 5000)  # Simulate variable frame sizes
        mock_frame_data = bytes([random.randint(0, 255) for _ in range(mock_frame_size)])
        frame_base64 = base64.b64encode(mock_frame_data).decode('utf-8')
        
        # Calculate accurate timestamp based on actual elapsed time
        if self.start_time is None:
            self.start_time = asyncio.get_event_loop().time()
        
        elapsed_time = asyncio.get_event_loop().time() - self.start_time
        timestamp = int(elapsed_time * 1000)  # Convert to milliseconds
        
        response = {
            "type": "VIDEO_FRAME",
            "session_id": self.session_id,
            "data": {
                "frame_data": frame_base64,
                "frame_timestamp": timestamp
            }
        }
        
        try:
            await self.websocket.send(json.dumps(response))
            logger.debug(f"Sent video frame #{self.frame_counter}: {mock_frame_size} bytes, timestamp: {timestamp}ms")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed while sending frame")
            if self.streaming_task:
                self.streaming_task.cancel()
    
    async def stream_default_video(self):
        """Stream default video frames continuously"""
        try:
            while self.initialized:
                await self.send_video_frame()
                await asyncio.sleep(0.04)  # 25fps = 40ms per frame
        except asyncio.CancelledError:
            logger.info("Default video streaming cancelled")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed during default video streaming")
        except Exception as e:
            logger.error(f"Error in default video streaming: {e}")


class MockMuseTalkServer:
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        self.sessions: Dict[str, MockMuseTalkSession] = {}
    
    async def handle_connection(self, websocket, path):
        """Handle new WebSocket connection"""
        logger.info(f"New connection from {websocket.remote_address}")
        
        # For simplicity, generate a unique user_id for each connection
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        session_key = f"{websocket.remote_address}_{user_id}"
        session = MockMuseTalkSession(websocket, path)
        self.sessions[session_key] = session
        
        try:
            async for message in websocket:
                await session.handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed for {session_key}")
        except Exception as e:
            logger.error(f"Error handling connection {session_key}: {e}")
        finally:
            # Clean up
            if session.streaming_task:
                session.streaming_task.cancel()
                try:
                    await session.streaming_task
                except asyncio.CancelledError:
                    pass
            if session_key in self.sessions:
                del self.sessions[session_key]
            logger.info(f"Session {session_key} cleaned up")
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting Mock MuseTalk Server on {self.host}:{self.port}")
        
        # Create server using the modern websockets approach
        server = await serve(
            lambda websocket, path: self.handle_connection(websocket, path),
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info(f"Mock MuseTalk Server started!")
        logger.info(f"WebSocket endpoint: ws://{self.host}:{self.port}/musetalk/v1/ws/{{user_id}}")
        
        return server


async def main():
    server_instance = MockMuseTalkServer(host="0.0.0.0", port=8000)
    server = await server_instance.start_server()
    
    try:
        # Keep the server running
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    finally:
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())