"""
WebSocket connection handler for MuseTalk service.
"""
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from ..models.session import Session, SessionStatus
from ..models.messages import ErrorCode, CloseReason
from .message_handler import MessageHandler


class ConnectionHandler:
    """Handles WebSocket connections and session management."""
    
    def __init__(self, message_handler: MessageHandler):
        """
        Initialize connection handler.
        
        Args:
            message_handler: Message handler instance
        """
        self.message_handler = message_handler
        self.sessions: Dict[str, Session] = {}
        self.websockets: Dict[str, WebSocket] = {}
        self.heartbeat_interval = 30  # seconds
        self.session_timeout = 300  # 5 minutes
    
    async def handle_connection(self, websocket: WebSocket, user_id: str):
        """
        Handle WebSocket connection lifecycle.
        
        Args:
            websocket: WebSocket connection
            user_id: User identifier
        """
        session = None
        
        try:
            # Accept connection
            await websocket.accept()
            
            # Create session
            session = Session(user_id=user_id)
            self.sessions[session.session_id] = session
            self.websockets[session.session_id] = websocket
            
            print(f"New connection: user={user_id}, session={session.session_id}")
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(session.session_id)
            )
            
            # Handle messages
            await self._message_loop(session, websocket)
            
        except WebSocketDisconnect:
            print(f"Client disconnected: session={session.session_id if session else 'unknown'}")
            
        except Exception as e:
            print(f"Connection error: {e}")
            if session:
                await self._send_error(
                    websocket, 
                    session.session_id,
                    ErrorCode.INTERNAL_ERROR,
                    str(e)
                )
        
        finally:
            # Cleanup
            if session:
                await self._cleanup_session(session.session_id)
    
    async def _message_loop(self, session: Session, websocket: WebSocket):
        """Handle incoming messages."""
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                
                # Parse JSON
                try:
                    message_data = json.loads(data)
                except json.JSONDecodeError:
                    await self._send_error(
                        websocket,
                        session.session_id,
                        ErrorCode.INVALID_MESSAGE,
                        "Invalid JSON format"
                    )
                    continue
                
                # Handle message
                response = await self.message_handler.handle_message(
                    session, 
                    message_data
                )
                
                # Send response if any
                if response:
                    await websocket.send_text(json.dumps(response))
                
                # Check if session is closing
                if session.status == SessionStatus.CLOSED:
                    break
                    
            except WebSocketDisconnect:
                break
                
            except Exception as e:
                print(f"Message loop error: {e}")
                await self._send_error(
                    websocket,
                    session.session_id,
                    ErrorCode.PROCESSING_ERROR,
                    str(e)
                )
    
    async def _heartbeat_loop(self, session_id: str):
        """Send periodic heartbeat to keep connection alive."""
        try:
            while session_id in self.sessions:
                await asyncio.sleep(self.heartbeat_interval)
                
                session = self.sessions.get(session_id)
                websocket = self.websockets.get(session_id)
                
                if not session or not websocket:
                    break
                
                # Check session timeout
                if (datetime.utcnow() - session.updated_at).total_seconds() > self.session_timeout:
                    print(f"Session timeout: {session_id}")
                    await self._close_connection(
                        session_id, 
                        CloseReason.TIMEOUT
                    )
                    break
                
                # Send ping
                try:
                    await websocket.send_json({"type": "PING"})
                except Exception:
                    # Connection likely closed
                    break
                    
        except Exception as e:
            print(f"Heartbeat error: {e}")
    
    async def _send_error(self, 
                        websocket: WebSocket,
                        session_id: str,
                        code: ErrorCode,
                        message: str):
        """Send error message to client."""
        try:
            error_response = self.message_handler._create_error_response(
                session_id, code, message
            )
            await websocket.send_text(json.dumps(error_response))
        except Exception as e:
            print(f"Failed to send error: {e}")
    
    async def _close_connection(self, session_id: str, reason: CloseReason):
        """Close WebSocket connection."""
        websocket = self.websockets.get(session_id)
        if websocket:
            try:
                # Send close message
                close_message = {
                    "type": "CLOSE_ACK",
                    "session_id": session_id,
                    "data": {"reason": reason.value}
                }
                await websocket.send_text(json.dumps(close_message))
                
                # Close WebSocket
                await websocket.close()
                
            except Exception as e:
                print(f"Error closing connection: {e}")
        
        # Cleanup
        await self._cleanup_session(session_id)
    
    async def _cleanup_session(self, session_id: str):
        """Clean up session resources."""
        print(f"Cleaning up session: {session_id}")
        
        # Update session status
        session = self.sessions.get(session_id)
        if session:
            session.set_status(SessionStatus.CLOSED)
            
            # Clean up services
            self.message_handler.audio_service.clear_buffer(session_id)
            self.message_handler.video_service.cleanup_session(session_id)
        
        # Remove from tracking
        self.sessions.pop(session_id, None)
        self.websockets.pop(session_id, None)
    
    def get_active_sessions(self) -> Dict[str, Any]:
        """Get information about active sessions."""
        return {
            session_id: {
                "user_id": session.user_id,
                "status": session.status.value,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "frames_generated": session.frames_generated,
                "total_audio_ms": session.total_audio_ms
            }
            for session_id, session in self.sessions.items()
        }
    
    async def broadcast_message(self, message: Dict[str, Any], 
                              exclude_session: Optional[str] = None):
        """Broadcast message to all connected clients."""
        disconnected = []
        
        for session_id, websocket in self.websockets.items():
            if session_id == exclude_session:
                continue
                
            try:
                await websocket.send_text(json.dumps(message))
            except Exception:
                disconnected.append(session_id)
        
        # Clean up disconnected sessions
        for session_id in disconnected:
            await self._cleanup_session(session_id)