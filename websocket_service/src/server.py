"""
MuseTalk WebSocket Server

Real-time lip-sync generation service using WebSocket.
"""
import os
import sys
from pathlib import Path
from typing import Optional
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from handlers.connection_handler import ConnectionHandler
from handlers.message_handler import MessageHandler
from services.avatar_service import AvatarService
from services.audio_service import AudioService
from services.video_service import VideoService


# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "../models"
    avatars_dir: str = "../results/v15/avatars"
    device: str = "cuda"
    use_float16: bool = True
    cors_origins: list[str] = ["*"]
    
    class Config:
        env_file = ".env"


# Global settings
settings = Settings()

# Global services (initialized in lifespan)
avatar_service: Optional[AvatarService] = None
audio_service: Optional[AudioService] = None
video_service: Optional[VideoService] = None
message_handler: Optional[MessageHandler] = None
connection_handler: Optional[ConnectionHandler] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global avatar_service, audio_service, video_service, message_handler, connection_handler
    
    print("Starting MuseTalk WebSocket Server...")
    
    # Initialize services
    avatar_service = AvatarService(avatars_dir=settings.avatars_dir)
    audio_service = AudioService(
        model_path=settings.model_path,
        device=settings.device
    )
    video_service = VideoService(
        model_path=settings.model_path,
        device=settings.device,
        use_float16=settings.use_float16
    )
    
    # Initialize handlers
    message_handler = MessageHandler(
        avatar_service=avatar_service,
        audio_service=audio_service,
        video_service=video_service
    )
    connection_handler = ConnectionHandler(message_handler)
    
    print("Services initialized successfully")
    
    yield
    
    # Cleanup
    print("Shutting down services...")


# Create FastAPI app
app = FastAPI(
    title="MuseTalk WebSocket Service",
    description="Real-time lip-sync generation service",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation."""
    return """
    <html>
        <head>
            <title>MuseTalk WebSocket Service</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                .endpoint { background: #f4f4f4; padding: 10px; margin: 10px 0; }
                code { background: #e0e0e0; padding: 2px 4px; }
            </style>
        </head>
        <body>
            <h1>MuseTalk WebSocket Service</h1>
            <p>Real-time lip-sync generation using WebSocket</p>
            
            <h2>Endpoints</h2>
            <div class="endpoint">
                <strong>🔌 WebSocket (Main Service):</strong> <code>ws://localhost:8000/musetalk/v1/ws/{user_id}</code><br>
                <small>Real-time lip-sync generation via WebSocket. Supports INIT, GENERATE, STATE_CHANGE, ACTION, and CLOSE messages.</small>
            </div>
            <div class="endpoint">
                <strong>🏥 Health Check:</strong> <code>GET /health</code><br>
                <small>Check service status and model loading state.</small>
            </div>
            <div class="endpoint">
                <strong>📊 Sessions Info:</strong> <code>GET /sessions</code><br>
                <small>Get active WebSocket sessions and service statistics.</small>
            </div>
            <div class="endpoint">
                <strong>👤 Prepare Avatar:</strong> <code>POST /prepare_avatar/{user_id}?video_path=...</code><br>
                <small>Pre-process avatar video for a user (testing endpoint).</small>
            </div>
            <div class="endpoint">
                <strong>📚 API Documentation:</strong> <code>GET /docs</code><br>
                <small>Swagger UI for REST endpoints (WebSocket not included - see below).</small>
            </div>
            
            <h2>WebSocket Message Types</h2>
            <div class="endpoint">
                <strong>INIT:</strong> Initialize session with user_id and video config<br>
                <strong>GENERATE:</strong> Send audio chunk for lip-sync generation<br>
                <strong>STATE_CHANGE:</strong> Change video state (speaking/idle)<br>
                <strong>ACTION:</strong> Trigger action video playback<br>
                <strong>CLOSE:</strong> Close session gracefully
            </div>
            
            <h2>Quick Test</h2>
            <p><strong>Test Client:</strong> <code>python test/test_client.py --help</code></p>
            <p><strong>Health Check:</strong> <code>curl http://localhost:8000/health</code></p>
            <p><strong>OpenAPI Spec:</strong> <a href="/openapi.yaml">openapi.yaml</a> | <a href="/openapi.json">openapi.json</a></p>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "avatar": avatar_service is not None,
            "audio": audio_service is not None,
            "video": video_service is not None
        }
    }


@app.get("/openapi.yaml")
async def get_openapi_yaml():
    """Get the complete OpenAPI specification including WebSocket documentation."""
    from pathlib import Path
    openapi_path = Path(__file__).parent.parent / "openapi.yaml"
    if openapi_path.exists():
        return open(openapi_path).read()
    else:
        return {"error": "OpenAPI YAML file not found"}


@app.get("/sessions")
async def get_sessions():
    """Get active sessions information."""
    if not connection_handler:
        return {"error": "Service not initialized"}
    
    return {
        "active_sessions": connection_handler.get_active_sessions(),
        "stats": {
            "avatar": avatar_service.get_avatar_stats() if avatar_service else {},
            "audio": audio_service.get_audio_stats("global") if audio_service else {},
            "video": video_service.get_stats() if video_service else {}
        }
    }


@app.websocket("/musetalk/v1/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time lip-sync generation.
    
    Args:
        websocket: WebSocket connection
        user_id: User identifier
    """
    if not connection_handler:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return
    
    # Validate user_id format
    if not user_id or not user_id.replace("-", "").replace("_", "").isalnum():
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        return
    
    # Handle connection
    await connection_handler.handle_connection(websocket, user_id)


@app.post("/prepare_avatar/{user_id}")
async def prepare_avatar(user_id: str, video_path: str):
    """
    Prepare avatar for user (endpoint for testing).
    
    This would normally be part of a separate avatar preparation service.
    """
    if not avatar_service:
        return JSONResponse(
            status_code=503,
            content={"error": "Service not initialized"}
        )
    
    success = await avatar_service.prepare_avatar(user_id, video_path)
    
    if success:
        return {"status": "success", "user_id": user_id}
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Avatar preparation failed"}
        )


def main():
    """Run the WebSocket server."""
    print(f"Starting server on {settings.host}:{settings.port}")
    print(f"WebSocket endpoint: ws://{settings.host}:{settings.port}/musetalk/v1/ws/{{user_id}}")
    print(f"API documentation: http://{settings.host}:{settings.port}/docs")
    
    uvicorn.run(
        "server:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()