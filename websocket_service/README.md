# MuseTalk WebSocket Service

A real-time WebSocket service for MuseTalk that enables streaming audio-to-video lip synchronization.

## Overview

This service provides a WebSocket API for real-time lip-sync generation. It accepts audio chunks and generates synchronized video frames using the MuseTalk model.

## Features

- Real-time audio processing (40ms PCM chunks)
- H.264 video frame streaming
- Multiple avatar state management (idle, speaking, action)
- Session-based processing
- Scalable architecture for future enhancements

## Quick Start

### Installation

```bash
cd websocket_service
pip install -r requirements.txt
```

### Running the Server

```bash
python src/server.py
```

The server will start on `ws://localhost:8000`

### WebSocket Endpoint

```
ws://localhost:8000/musetalk/v1/ws/{user_id}
```

## API Documentation

See `openapi.yaml` for the complete API specification.

### Message Flow

1. **Connect** to WebSocket endpoint
2. **Send INIT** message to initialize session
3. **Send GENERATE** messages with audio chunks
4. **Receive VIDEO_FRAME** responses
5. **Send CLOSE** to end session

### Message Types

- `INIT`: Initialize session and load user avatar
- `GENERATE`: Process audio chunk and generate video frame
- `STATE_CHANGE`: Change avatar state (idle/speaking/action)
- `ACTION`: Trigger predefined actions
- `CLOSE`: Close connection

## Architecture

```
websocket_service/
├── src/
│   ├── server.py              # FastAPI WebSocket server
│   ├── handlers/              # Message handlers
│   ├── models/                # Pydantic models
│   ├── services/              # Business logic
│   └── utils/                 # Utilities
└── test/                      # Tests and examples
```

## Development

### Running Tests

```bash
pytest test/
```

### Example Client

See `test/test_client.py` for a complete example of connecting to the WebSocket service.

## Configuration

The service can be configured through environment variables:

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `MODEL_PATH`: Path to MuseTalk models (default: ../models)

## Future Enhancements

The architecture is designed to support:
- Authentication and authorization
- Monitoring and metrics
- Horizontal scaling
- Message queue integration
- Caching layer