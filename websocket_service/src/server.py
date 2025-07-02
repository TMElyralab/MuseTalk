"""
MuseTalk WebSocket Server

Real-time lip-sync generation service using WebSocket.
"""
import os
import sys
import json
from pathlib import Path
from typing import Optional
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from handlers.connection_handler import ConnectionHandler
from handlers.message_handler import MessageHandler
from services.avatar_service import AvatarService
from services.audio_service import AudioService
from services.video_service import VideoService


# Load environment variables
load_dotenv()


def load_config():
    """Load configuration from YAML file with environment overrides."""
    config_path = Path(__file__).parent.parent / "configs" / "musetalk.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration if file doesn't exist
        config = {
            "model": {"device": "cuda", "use_float16": True},
            "server": {"host": "0.0.0.0", "port": 8000, "avatars_dir": "../results/v15/avatars", "cors_origins": ["*"]},
            "avatar": {"bbox_shift": 0, "fps": 25},
            "audio": {"sample_rate": 16000, "batch_size": 20}
        }
    
    return config

class Settings(BaseSettings):
    """Application settings with YAML configuration support."""
    host: str = "0.0.0.0"
    port: int = 8000
    models_dir: str = "../models"
    avatars_dir: str = "../results/v15/avatars"
    device: str = "cuda"
    use_float16: bool = True
    cors_origins: list[str] = ["*"]
    
    def __init__(self, **kwargs):
        # Load YAML config first
        config = load_config()
        
        # Override with YAML values
        server_config = config.get("server", {})
        model_config = config.get("model", {})
        
        # Set defaults from YAML
        kwargs.setdefault("host", server_config.get("host", "0.0.0.0"))
        kwargs.setdefault("port", server_config.get("port", 8000))
        kwargs.setdefault("avatars_dir", server_config.get("avatars_dir", "../results/v15/avatars"))
        kwargs.setdefault("device", model_config.get("device", "cuda"))
        kwargs.setdefault("use_float16", model_config.get("use_float16", True))
        kwargs.setdefault("cors_origins", server_config.get("cors_origins", ["*"]))
        
        super().__init__(**kwargs)
    
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
    
    # Initialize lightweight services first (no model loading)
    avatar_service = AvatarService(avatars_dir=settings.avatars_dir)
    
    # Initialize services with model loading in background
    print("Initializing services in background...")
    
    # Create placeholder services that will be replaced once models are loaded
    audio_service = None
    video_service = None
    message_handler = None
    connection_handler = None
    
    # Start background model loading
    asyncio.create_task(load_models_background())
    
    print("Server ready - models loading in background")
    
    yield
    
    # Cleanup
    print("Shutting down services...")

async def load_models_background():
    """Load models in background to avoid blocking server startup."""
    global audio_service, video_service, message_handler, connection_handler
    
    try:
        print("Loading MuseTalk models in background...")
        
        # Initialize services with model loading
        audio_service = AudioService(
            model_path=settings.models_dir,
            device=settings.device
        )
        print("Audio service initialized")
        
        video_service = VideoService(
            model_path=settings.models_dir,
            device=settings.device,
            use_float16=settings.use_float16
        )
        print("Video service initialized")
        
        
        # Initialize handlers
        message_handler = MessageHandler(
            avatar_service=avatar_service,
            audio_service=audio_service,
            video_service=video_service
        )
        connection_handler = ConnectionHandler(message_handler)
        
        print("✅ All MuseTalk models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Failed to load MuseTalk models: {e}")
        import traceback
        traceback.print_exc()


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

# Mount static files for frontend
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve frontend app."""
    try:
        frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
        if not frontend_path.exists():
            return {"error": f"Frontend file not found: {frontend_path}"}
        return FileResponse(str(frontend_path))
    except Exception as e:
        return {"error": f"Failed to serve frontend: {e}"}


@app.get("/test")
async def test_endpoint():
    """Simple test endpoint."""
    return {"message": "Server is responding", "timestamp": str(datetime.now())}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    services_ready = {
        "avatar": avatar_service is not None,
        "audio": audio_service is not None,
        "video": video_service is not None,
        "message_handler": message_handler is not None,
        "connection_handler": connection_handler is not None
    }
    
    all_ready = all(services_ready.values())
    
    return {
        "status": "ready" if all_ready else "loading",
        "message": "All services ready" if all_ready else "Models loading in background...",
        "services": services_ready,
        "models_loaded": all_ready
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
    # Check if services are ready
    if not connection_handler or not message_handler or not video_service or not audio_service:
        await websocket.accept()
        await websocket.send_text(json.dumps({
            "type": "ERROR",
            "data": {
                "code": "SERVICE_NOT_READY",
                "message": "Server is still loading models. Please wait and try again.",
                "details": "Models are loading in background"
            }
        }))
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return
    
    # Validate user_id format
    if not user_id or not user_id.replace("-", "").replace("_", "").isalnum():
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        return
    
    # Handle connection
    await connection_handler.handle_connection(websocket, user_id)










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