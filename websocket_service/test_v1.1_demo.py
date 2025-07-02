#!/usr/bin/env python3
"""
Demo script for MuseTalk WebSocket v1.1 - Continuous Streaming Mode

This script demonstrates the key differences in v1.1:
1. Video streaming starts immediately after connection
2. Actions don't require audio
3. Seamless transitions between states
"""
import asyncio
import json
import websockets
import uuid
import base64
import numpy as np
import time


async def v11_demo():
    """Demonstrate v1.1 continuous streaming features."""
    server_url = "ws://localhost:8000"
    user_id = "demo_user"
    session_id = str(uuid.uuid4())
    
    # Connect to WebSocket
    ws_url = f"{server_url}/musetalk/v1/ws/{user_id}"
    print(f"Connecting to {ws_url}...")
    
    async with websockets.connect(ws_url) as websocket:
        print("Connected! Video streaming will start automatically after initialization.\n")
        
        # Frame counter
        frame_count = 0
        start_time = None
        
        # Background task to receive frames
        async def receive_frames():
            nonlocal frame_count, start_time
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if data["type"] == "INIT_SUCCESS":
                        print("=== INIT SUCCESS ===")
                        print(f"Default video: {data['data']['default_video']}")
                        print(f"Streaming started: {data['data']['streaming_started']}")
                        print(f"Available videos: {len(data['data']['available_videos'])}")
                        print("\nVideo frames will now start arriving automatically...")
                        print("=" * 50 + "\n")
                        
                    elif data["type"] == "VIDEO_FRAME":
                        frame_count += 1
                        if frame_count == 1:
                            start_time = time.time()
                            print("First frame received! Continuous streaming active.")
                        
                        if frame_count % 25 == 0:  # Log every second
                            elapsed = time.time() - start_time
                            fps = frame_count / elapsed
                            print(f"Frames: {frame_count}, FPS: {fps:.2f}")
                            
                    elif data["type"] == "ACTION_TRIGGERED":
                        print(f"\n🎬 Action {data['data']['action_index']} triggered!")
                        print(f"Inserted into stream: {data['data']['inserted']}\n")
                        
                except Exception as e:
                    print(f"Receive error: {e}")
                    break
        
        # Start receiving frames
        receive_task = asyncio.create_task(receive_frames())
        
        try:
            # Step 1: Initialize
            print("Step 1: Sending INIT message...")
            init_msg = {
                "type": "INIT",
                "session_id": session_id,
                "data": {
                    "user_id": user_id,
                    "video_config": {
                        "resolution": "512x512",
                        "fps": 25
                    }
                }
            }
            await websocket.send(json.dumps(init_msg))
            
            # Wait for initialization and observe automatic streaming
            await asyncio.sleep(3)
            
            # Step 2: Send some audio for lip-sync
            print("\nStep 2: Sending audio for lip-sync...")
            for i in range(5):
                # Create mock 40ms PCM audio
                audio_data = np.zeros(640, dtype=np.int16).tobytes()
                
                generate_msg = {
                    "type": "GENERATE",
                    "session_id": session_id,
                    "data": {
                        "audio_chunk": {
                            "format": "pcm_s16le",
                            "sample_rate": 16000,
                            "channels": 1,
                            "duration_ms": 40,
                            "data": base64.b64encode(audio_data).decode('utf-8')
                        },
                        "video_state": {
                            "type": "speaking",
                            "base_video": "speaking_1"
                        }
                    }
                }
                await websocket.send(json.dumps(generate_msg))
                await asyncio.sleep(0.04)
            
            print("Audio sent - lip-sync frames should be generated")
            await asyncio.sleep(2)
            
            # Step 3: Trigger an action (no audio needed in v1.1!)
            print("\nStep 3: Triggering action (no audio required in v1.1)...")
            action_msg = {
                "type": "ACTION",
                "session_id": session_id,
                "data": {
                    "action_index": 1  # Simple action_index instead of action_type + audio
                }
            }
            await websocket.send(json.dumps(action_msg))
            
            # Wait to see action frames
            await asyncio.sleep(3)
            
            # Final stats
            print("\n=== DEMO COMPLETE ===")
            print(f"Total frames received: {frame_count}")
            if start_time:
                total_time = time.time() - start_time
                avg_fps = frame_count / total_time
                print(f"Total time: {total_time:.2f}s")
                print(f"Average FPS: {avg_fps:.2f}")
            
            print("\nKey v1.1 Features Demonstrated:")
            print("✓ Automatic video streaming on connection")
            print("✓ Continuous frame generation at 25 FPS")
            print("✓ Action triggering without audio")
            print("✓ Seamless state transitions")
            
        finally:
            # Clean up
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
            
            # Send close
            close_msg = {
                "type": "CLOSE",
                "session_id": session_id,
                "data": {}
            }
            await websocket.send(json.dumps(close_msg))


if __name__ == "__main__":
    print("MuseTalk WebSocket v1.1 Demo - Continuous Streaming Mode")
    print("=" * 60)
    print("This demo shows the key differences in v1.1:")
    print("- Video streaming starts automatically after connection")
    print("- Actions don't require audio data")
    print("- Continuous 25 FPS video generation")
    print("=" * 60 + "\n")
    
    asyncio.run(v11_demo())