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
    ActionFrameResponse, CloseAckResponse, ErrorResponse,
    InitSuccessData, VideoFrameData, StateChangedData,
    ActionFrameData, CloseAckData, ErrorData, CloseReason
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
            
            # Validate session ID
            if message.session_id != session.session_id:
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
            
            # Create success response
            response = InitSuccessResponse(
                session_id=session.session_id,
                data=InitSuccessData(
                    model_loaded=avatar_info.model_loaded,
                    available_videos=avatar_info.available_videos
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
                return self._create_error_response(
                    session.session_id,
                    ErrorCode.INVALID_SESSION,
                    "Session not ready for processing"
                )
            
            # Mark as processing
            session.state.is_processing = True
            
            # Process audio
            audio_features = await self.audio_service.process_audio_chunk(
                session.session_id,
                message.data.audio_chunk
            )
            
            if audio_features is None:
                session.state.is_processing = False
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
            
            # Generate video frame
            frame_data = await self.video_service.generate_frame(
                session,
                audio_features,
                session.frames_generated
            )
            
            if frame_data is None:
                session.state.is_processing = False
                return self._create_error_response(
                    session.session_id,
                    ErrorCode.PROCESSING_ERROR,
                    "Video generation failed"
                )
            
            # Update session stats
            session.increment_frames()
            session.add_audio_duration(message.data.audio_chunk.duration_ms)
            session.state.last_frame_timestamp += message.data.audio_chunk.duration_ms
            session.state.is_processing = False
            
            # Create response
            response = VideoFrameResponse(
                session_id=session.session_id,
                data=VideoFrameData(
                    frame_data=frame_data,
                    frame_timestamp=session.state.last_frame_timestamp
                )
            )
            
            return response.model_dump()
            
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
            
            # Start action
            session.start_action(message.data.action_type.value)
            
            # Process audio
            audio_features = await self.audio_service.process_audio_chunk(
                session.session_id,
                message.data.audio_chunk
            )
            
            if audio_features is None:
                session.state.action_in_progress = None
                return self._create_error_response(
                    session.session_id,
                    ErrorCode.PROCESSING_ERROR,
                    "Audio processing failed"
                )
            
            # Generate action frame
            frame_data, new_progress = await self.video_service.generate_action_frame(
                session,
                message.data.action_type.value,
                audio_features,
                session.state.action_progress
            )
            
            if frame_data is None:
                session.state.action_in_progress = None
                return self._create_error_response(
                    session.session_id,
                    ErrorCode.PROCESSING_ERROR,
                    "Action frame generation failed"
                )
            
            # Update progress
            session.update_action_progress(new_progress)
            session.state.last_frame_timestamp += message.data.audio_chunk.duration_ms
            
            # Create response
            response = ActionFrameResponse(
                session_id=session.session_id,
                data=ActionFrameData(
                    frame_data=frame_data,
                    frame_timestamp=session.state.last_frame_timestamp,
                    action_progress=session.state.action_progress
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