"""
WebSocket message handler for MuseTalk service.
"""
import json
from typing import Dict, Any, Optional
from datetime import datetime

from models.messages import (
    MessageType, parse_message, ErrorCode,
    InitMessage, GenerateMessage, StateChangeMessage, 
    ActionMessage, CloseMessage,
    InitSuccessResponse, VideoFrameResponse, StateChangedResponse,
    ActionFrameResponse, ActionTriggeredResponse, CloseAckResponse, ErrorResponse,
    InitSuccessData, VideoFrameData, StateChangedData,
    ActionFrameData, ActionTriggeredData, CloseAckData, ErrorData, CloseReason
)
from models.session import Session, SessionStatus, AvatarInfo
from services.avatar_service import AvatarService
from services.audio_service import AudioService
from services.video_service import VideoService


class MessageHandler:
    """Handles WebSocket messages and coordinates services."""
    
    def __init__(self,
                 avatar_service: AvatarService,
                 audio_service: AudioService,
                 video_service: VideoService):
        """
        Initialize message handler.
        
        Args:
            avatar_service: Avatar management service
            audio_service: Audio processing service
            video_service: Video generation service
        """
        self.avatar_service = avatar_service
        self.audio_service = audio_service
        self.video_service = video_service
        
        # Message type handlers
        self.handlers = {
            MessageType.INIT: self.handle_init,
            MessageType.GENERATE: self.handle_generate,
            MessageType.STATE_CHANGE: self.handle_state_change,
            MessageType.ACTION: self.handle_action,
            MessageType.CLOSE: self.handle_close,
        }
    
    async def handle_message(self, 
                           session: Session,
                           message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle incoming WebSocket message.
        
        Args:
            session: Current session
            message_data: Parsed message dictionary
            
        Returns:
            Response message dictionary or None
        """
        try:
            # Parse message
            message = parse_message(message_data)
            
            # Validate session ID (except for INIT message)
            if message.type != MessageType.INIT and message.session_id != session.session_id:
                return self._create_error_response(
                    session.session_id,
                    ErrorCode.INVALID_SESSION,
                    f"Session ID mismatch: expected {session.session_id}"
                )
            
            # Get handler
            handler = self.handlers.get(message.type)
            if not handler:
                return self._create_error_response(
                    session.session_id,
                    ErrorCode.INVALID_MESSAGE,
                    f"Unknown message type: {message.type}"
                )
            
            # Handle message
            return await handler(session, message)
            
        except Exception as e:
            print(f"Message handling error: {e}")
            return self._create_error_response(
                session.session_id,
                ErrorCode.PROCESSING_ERROR,
                str(e)
            )
    
    async def handle_init(self, session: Session, message: InitMessage) -> Dict[str, Any]:
        """Handle INIT message."""
        try:
            # Update session
            session.user_id = message.data.user_id
            session.video_config = message.data.video_config.model_dump()
            session.set_status(SessionStatus.INITIALIZING)
            
            # Load avatar
            avatar_info = await self.avatar_service.load_avatar(message.data.user_id)
            if not avatar_info:
                return self._create_error_response(
                    session.session_id,
                    ErrorCode.MODEL_NOT_FOUND,
                    f"Avatar not found for user: {message.data.user_id}"
                )
            
            # Update session with avatar info
            session.avatar_info = avatar_info
            session.set_status(SessionStatus.READY)
            
            # Select default video (first idle video)
            default_video = "idle_0"
            if avatar_info.available_videos:
                idle_videos = [v for v in avatar_info.available_videos if v.startswith("idle_")]
                if idle_videos:
                    default_video = idle_videos[0]
            
            # Set initial video state
            session.set_video_state("idle", default_video)
            
            # Create success response
            response = InitSuccessResponse(
                session_id=session.session_id,
                data=InitSuccessData(
                    model_loaded=avatar_info.model_loaded,
                    available_videos=avatar_info.available_videos,
                    default_video=default_video,
                    streaming_started=True
                )
            )
            
            return response.model_dump()
            
        except Exception as e:
            print(f"Init error: {e}")
            return self._create_error_response(
                session.session_id,
                ErrorCode.INTERNAL_ERROR,
                f"Initialization failed: {str(e)}"
            )
    
    async def handle_generate(self, session: Session, message: GenerateMessage) -> Dict[str, Any]:
        """Handle GENERATE message."""
        try:
            # Check session is ready
            if not session.can_process():
                debug_info = {
                    "status": session.status.value,
                    "avatar_info_exists": session.avatar_info is not None,
                    "model_loaded": session.avatar_info.model_loaded if session.avatar_info else None,
                    "is_processing": session.state.is_processing
                }
                return self._create_error_response(
                    session.session_id,
                    ErrorCode.INVALID_SESSION,
                    "Session not ready for processing",
                    debug_info
                )
            
            # Process audio using MuseTalk pipeline
            from utils.encoding import decode_audio_chunk
            audio_data, _ = decode_audio_chunk(message.data.audio_chunk.data)
            audio_bytes = audio_data.tobytes()
            
            audio_features = await self.audio_service.process_audio_for_inference(
                session.session_id,
                audio_bytes,
                message.data.audio_chunk.sample_rate
            )
            
            if audio_features is None:
                return self._create_error_response(
                    session.session_id,
                    ErrorCode.PROCESSING_ERROR,
                    "Audio processing failed"
                )
            
            # Update video state if needed
            if message.data.video_state.type != session.state.video_state:
                session.set_video_state(
                    message.data.video_state.type,
                    message.data.video_state.base_video
                )
            
            # Store audio features for the streaming loop
            session._pending_audio_features = audio_features
            session.state.is_processing = True
            
            # Update session stats
            session.add_audio_duration(message.data.audio_chunk.duration_ms)
            
            # Reset processing state immediately - the streaming loop will handle the audio
            session.state.is_processing = False
            
            # In v1.1, we don't return frames here - the streaming loop handles it
            # Just acknowledge that we received the audio
            return None
            
        except Exception as e:
            session.state.is_processing = False
            print(f"Generate error: {e}")
            return self._create_error_response(
                session.session_id,
                ErrorCode.INTERNAL_ERROR,
                f"Generation failed: {str(e)}"
            )
    
    async def handle_state_change(self, session: Session, 
                                message: StateChangeMessage) -> Dict[str, Any]:
        """Handle STATE_CHANGE message."""
        try:
            # Validate base video
            if (session.avatar_info and 
                message.data.base_video not in session.avatar_info.available_videos):
                return self._create_error_response(
                    session.session_id,
                    ErrorCode.INVALID_MESSAGE,
                    f"Invalid base video: {message.data.base_video}"
                )
            
            # Update state
            session.set_video_state(
                message.data.target_state,
                message.data.base_video
            )
            
            # Create response
            response = StateChangedResponse(
                session_id=session.session_id,
                data=StateChangedData(
                    current_state=session.state.video_state,
                    current_video=session.state.current_video
                )
            )
            
            return response.model_dump()
            
        except Exception as e:
            print(f"State change error: {e}")
            return self._create_error_response(
                session.session_id,
                ErrorCode.INTERNAL_ERROR,
                f"State change failed: {str(e)}"
            )
    
    async def handle_action(self, session: Session, message: ActionMessage) -> Dict[str, Any]:
        """Handle ACTION message."""
        try:
            # Check if action already in progress
            if session.state.action_in_progress:
                return self._create_error_response(
                    session.session_id,
                    ErrorCode.INVALID_MESSAGE,
                    "Action already in progress"
                )
            
            # Start action based on action_index
            action_name = f"action_{message.data.action_index}"
            session.start_action(action_name)
            
            # Mark action as inserted into stream
            # The actual action video will be handled by the continuous streaming logic
            
            # Create response
            response = ActionTriggeredResponse(
                session_id=session.session_id,
                data=ActionTriggeredData(
                    action_index=message.data.action_index,
                    inserted=True
                )
            )
            
            return response.model_dump()
            
        except Exception as e:
            session.state.action_in_progress = None
            print(f"Action error: {e}")
            return self._create_error_response(
                session.session_id,
                ErrorCode.INTERNAL_ERROR,
                f"Action failed: {str(e)}"
            )
    
    async def handle_close(self, session: Session, message: CloseMessage) -> Dict[str, Any]:
        """Handle CLOSE message."""
        try:
            # Update session status
            session.set_status(SessionStatus.CLOSING)
            
            # Clean up resources
            self.audio_service.clear_buffer(session.session_id)
            self.video_service.cleanup_session(session.session_id)
            
            # Create response
            response = CloseAckResponse(
                session_id=session.session_id,
                data=CloseAckData(reason=CloseReason.CLIENT_REQUEST)
            )
            
            session.set_status(SessionStatus.CLOSED)
            
            return response.model_dump()
            
        except Exception as e:
            print(f"Close error: {e}")
            return self._create_error_response(
                session.session_id,
                ErrorCode.INTERNAL_ERROR,
                f"Close failed: {str(e)}"
            )
    
    def _create_error_response(self, 
                             session_id: str,
                             code: ErrorCode, 
                             message: str,
                             details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create error response."""
        response = ErrorResponse(
            session_id=session_id,
            data=ErrorData(
                code=code,
                message=message,
                details=details
            )
        )
        return response.model_dump()