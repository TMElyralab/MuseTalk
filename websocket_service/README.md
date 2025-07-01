# MuseTalk WebSocket Service

A real-time WebSocket service for MuseTalk that enables streaming audio-to-video lip synchronization using FastAPI and WebSocket.

**Version**: 1.1 (Continuous Streaming Mode)

## Overview

This service provides a comprehensive WebSocket API for real-time lip-sync generation with continuous video streaming. Upon connection, the service immediately begins streaming base video frames. When it receives 40ms PCM audio chunks, it seamlessly switches to lip-synchronized video generation. The service supports multiple avatar states, action insertion, session management, and includes comprehensive REST endpoints for monitoring and testing.

## Features

- **Continuous Video Streaming**: Automatic video output starts immediately upon connection
- **Real-time Processing**: 40ms PCM audio chunks (16kHz, mono)
- **H.264 Video Streaming**: Base64-encoded video frames at 25 FPS
- **Avatar State Management**: Idle, speaking, and action states with seamless transitions
- **Action Insertion**: Insert predefined actions into the video stream
- **Session-based Processing**: UUID-based session tracking
- **FastAPI Integration**: Modern async web framework
- **Comprehensive REST API**: Health checks, session monitoring, avatar preparation
- **OpenAPI Documentation**: Complete API specification with Swagger UI
- **Test Client**: Full-featured test client for development
- **Configurable**: Environment-based configuration

## Quick Start

### Installation

```bash
cd websocket_service
pip install -r requirements.txt  # Note: requirements.txt may need to be created
```

### Running the Server

```bash
# Option 1: Direct server execution
python src/server.py

# Option 2: Using the run script
python run_server.py
```

The server will start on `http://localhost:8000` with WebSocket endpoint at `ws://localhost:8000`

### WebSocket Endpoint

```
ws://localhost:8000/musetalk/v1/ws/{user_id}
```

### Web Interface

Visit `http://localhost:8000` for interactive API documentation and endpoint overview.

## API Documentation

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|--------------|
| `/` | GET | Web interface with API documentation |
| `/health` | GET | Service health check and model status |
| `/sessions` | GET | Active sessions and service statistics |
| `/prepare_avatar/{user_id}` | POST | Prepare avatar for user (testing) |
| `/docs` | GET | Swagger UI documentation |
| `/openapi.yaml` | GET | Complete OpenAPI specification |

### WebSocket Message Flow

1. **Connect** to WebSocket endpoint with user_id
2. **Send INIT** message with session_id and user configuration
3. **Send GENERATE** messages with audio chunks and video state
4. **Receive VIDEO_FRAME** responses with synchronized frames
5. **Send STATE_CHANGE** to modify avatar behavior (optional)
6. **Send ACTION** to trigger predefined actions (optional)
7. **Send CLOSE** to end session gracefully

### WebSocket Message Types

#### Client → Server Messages
- **`INIT`**: Initialize session with user_id, auth_token, and video_config
- **`GENERATE`**: Process audio chunk with video state information
- **`STATE_CHANGE`**: Change avatar state (idle/speaking/action) and base video
- **`ACTION`**: Trigger predefined actions (action_1, action_2) with audio
- **`CLOSE`**: Close connection and cleanup resources

#### Server → Client Messages
- **`INIT_SUCCESS`**: Confirms initialization with model status and available videos
- **`VIDEO_FRAME`**: Generated H.264 frame with timestamp
- **`STATE_CHANGED`**: Confirms state change with current state and video
- **`ACTION_FRAME`**: Action-specific frame with progress indicator
- **`CLOSE_ACK`**: Acknowledges close request with reason
- **`ERROR`**: Error information with code, message, and details

## Architecture

```
websocket_service/
├── src/
│   ├── server.py              # FastAPI WebSocket server with CORS
│   ├── handlers/
│   │   ├── connection_handler.py  # WebSocket connection management
│   │   └── message_handler.py     # Message routing and processing
│   ├── models/
│   │   ├── messages.py        # Pydantic message models with validation
│   │   └── session.py         # Session state management
│   ├── services/
│   │   ├── avatar_service.py  # Avatar loading and management
│   │   ├── audio_service.py   # Audio processing (Whisper-based)
│   │   └── video_service.py   # Video generation (MuseTalk UNet)
│   └── utils/
│       ├── encoding.py        # Base64 encoding/decoding utilities
│       └── video_codec.py     # H.264 video encoding
├── test/
│   ├── test_client.py         # Full-featured WebSocket test client
│   ├── test_basic.py          # Unit tests
│   ├── test_services.py       # Service integration tests
│   └── fixtures/              # Test audio and configuration files
├── fixtures/                  # Additional test data
├── run_server.py              # Simple server launcher
├── openapi.yaml               # Complete API specification
├── musetalk_websocket_spec.md # Detailed WebSocket protocol documentation
└── pytest.ini                # Test configuration
```

## Development

### Running Tests

```bash
# Run all tests
pytest test/

# Run specific test categories
pytest test/test_basic.py          # Unit tests
pytest test/test_services.py       # Service tests
pytest test_integration.py         # Integration tests

# Run with coverage
pytest --cov=src test/
```

### Test Client Usage

The included test client provides a comprehensive example:

```bash
# Basic connection test
python test/test_client.py --user-id test_user_123

# Send test audio file
python test/test_client.py --user-id test_user_123 --audio-file fixtures/speech_like_2s.wav

# Test with specific server
python test/test_client.py --server ws://localhost:8000 --user-id test_user_123

# Help and options
python test/test_client.py --help
```

### Example Integration

See `test/test_client.py` for a complete example of:
- WebSocket connection management
- Session initialization
- Audio chunk streaming
- Video frame processing
- Error handling
- Graceful disconnection

## Configuration

The service supports configuration through environment variables and `.env` files:

### Core Settings
- **`HOST`**: Server host (default: `0.0.0.0`)
- **`PORT`**: Server port (default: `8000`)
- **`MODEL_PATH`**: Path to MuseTalk models (default: `../models`)
- **`AVATARS_DIR`**: Avatar storage directory (default: `../results/v15/avatars`)
- **`DEVICE`**: Computation device (default: `cuda`)
- **`USE_FLOAT16`**: Enable FP16 optimization (default: `true`)
- **`CORS_ORIGINS`**: CORS allowed origins (default: `["*"]`)

### Example `.env` file
```bash
HOST=localhost
PORT=8000
MODEL_PATH=/path/to/musetalk/models
AVATARS_DIR=/path/to/avatars
DEVICE=cuda
USE_FLOAT16=true
CORS_ORIGINS=["http://localhost:3000", "https://app.example.com"]
```

## Message Protocol

### Audio Format Requirements
- **Format**: PCM signed 16-bit little-endian
- **Sample Rate**: 16,000 Hz
- **Channels**: Mono (1 channel)
- **Chunk Duration**: 40ms (640 bytes per chunk)
- **Encoding**: Base64 for transmission

### Video Output Format
- **Resolution**: 512x512 pixels (face region)
- **Format**: H.264 encoded frames
- **FPS**: 25 frames per second
- **Encoding**: Base64 for transmission

### Session Management
- **Session ID**: UUID v4 format
- **User ID**: Alphanumeric with hyphens/underscores, 1-64 characters
- **State Tracking**: Comprehensive session state with processing flags
- **Resource Cleanup**: Automatic cleanup on disconnect

## Error Handling

The service provides detailed error responses with specific error codes:

- **`INVALID_MESSAGE`**: Malformed or unsupported message
- **`INVALID_SESSION`**: Session ID mismatch or invalid state
- **`MODEL_NOT_FOUND`**: Avatar or model not available
- **`PROCESSING_ERROR`**: Audio/video processing failure
- **`RATE_LIMIT_EXCEEDED`**: Too many requests
- **`INTERNAL_ERROR`**: Server-side error

## Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Session Statistics
```bash
curl http://localhost:8000/sessions
```

### Service Metrics
The service tracks:
- Active WebSocket connections
- Audio processing statistics
- Video generation performance
- Avatar usage statistics
- Error rates and types

## Future Enhancements

The architecture is designed to support:
- **Authentication**: JWT token validation
- **Authorization**: Role-based access control
- **Monitoring**: Prometheus metrics and health checks
- **Scaling**: Horizontal scaling with load balancing
- **Persistence**: Message queue integration for reliability
- **Caching**: Avatar and model caching layer
- **Rate Limiting**: Per-user request throttling
- **Logging**: Structured logging with correlation IDs