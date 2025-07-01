# MuseTalk WebSocket服务接口规范

## 1. 概述

MuseTalk WebSocket服务负责实时接收音频数据，基于用户个性化模型进行唇形同步推理，并流式输出同步后的视频帧。

**连接地址**: `wss://server_ip/musetalk/v1/ws/{user_id}`

## 2. 通用报文格式

### 2.1 请求报文结构
```json
{
  "type": "MESSAGE_TYPE",
  "session_id": "uuid-string",
  "data": {
    // 具体数据内容
  }
}
```

### 2.2 响应报文结构
```json
{
  "type": "RESPONSE_TYPE", 
  "session_id": "uuid-string",
  "data": {
    // 返回数据内容
  }
}
```

## 3. 接口定义

### 3.1 连接初始化

**请求**: 建立WebSocket连接后立即发送
```json
{
  "type": "INIT",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "data": {
    "user_id": "user123",
    "auth_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "video_config": {
      "resolution": "512x512",
      "fps": 25
    }
  }
}
```

**响应**: 连接和模型加载状态确认
```json
{
  "type": "INIT_SUCCESS",
  "session_id": "550e8400-e29b-41d4-a716-446655440000", 
  "data": {
    "model_loaded": true,
    "available_videos": [
      "idle_0", "idle_1", "idle_2", "idle_3", "idle_4", "idle_5", "idle_6",
      "speaking_0", "speaking_1", "speaking_2", "speaking_3", "speaking_4", "speaking_5", "speaking_6", "speaking_7",
      "action_1", "action_2"
    ]
  }
}
```

**说明**: 
- `user_id`: 用户唯一标识，用于加载对应的个性化模型
- `auth_token`: JWT认证令牌
- `video_config`: 输出视频配置，分辨率固定512x512，帧率25fps
- `available_videos`: 返回该用户可用的基础视频列表

### 3.2 视频生成

**请求**: 发送音频数据进行唇形同步处理
```json
{
  "type": "GENERATE",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "data": {
    "audio_chunk": {
      "format": "pcm_s16le",
      "sample_rate": 16000,
      "channels": 1,
      "duration_ms": 40,
      "data": "base64-encoded-pcm-data"
    },
    "video_state": {
      "type": "speaking",
      "base_video": "speaking_3"
    }
  }
}
```

**响应**: 返回同步后的视频帧
```json
{
  "type": "VIDEO_FRAME",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "data": {
    "frame_data": "base64-encoded-h264-frame",
    "frame_timestamp": 40
  }
}
```

**说明**:
- `audio_chunk`: 音频数据块，40ms时长的PCM格式音频
- `format`: 固定为pcm_s16le (16位小端序PCM)
- `sample_rate`: 固定16000Hz
- `duration_ms`: 音频时长，通常为40ms
- `video_state.type`: 视频状态类型，支持"idle"、"speaking"、"action"
- `base_video`: 指定使用的基础视频，必须在available_videos列表中
- `frame_data`: H.264编码的视频帧数据
- `frame_timestamp`: 帧时间戳，与输入音频时间对应

### 3.3 状态切换

**请求**: 切换视频状态
```json
{
  "type": "STATE_CHANGE", 
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "data": {
    "target_state": "speaking",
    "base_video": "speaking_1"
  }
}
```

**响应**: 状态切换确认
```json
{
  "type": "STATE_CHANGED",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "data": {
    "current_state": "speaking",
    "current_video": "speaking_1"
  }
}
```

**说明**:
- `target_state`: 目标状态，"idle"/"speaking"/"action"
- `base_video`: 对应状态的基础视频
- 状态切换会影响后续GENERATE请求的默认base_video选择

### 3.4 动作触发

**请求**: 触发特定动作
```json
{
  "type": "ACTION",
  "session_id": "550e8400-e29b-41d4-a716-446655440000", 
  "data": {
    "action_type": "action_1",
    "audio_chunk": {
      "format": "pcm_s16le",
      "sample_rate": 16000,
      "channels": 1,
      "duration_ms": 40,
      "data": "base64-encoded-pcm-data"
    }
  }
}
```

**响应**: 动作视频帧
```json
{
  "type": "ACTION_FRAME",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "data": {
    "frame_data": "base64-encoded-h264-frame",
    "frame_timestamp": 40,
    "action_progress": 0.1
  }
}
```

**说明**:
- `action_type`: 动作类型，"action_1"或"action_2"
- `action_progress`: 动作播放进度，0.0-1.0
- 动作视频通常持续10秒，会覆盖正常的speaking状态

### 3.5 连接关闭

**请求**: 主动关闭连接
```json
{
  "type": "CLOSE",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "data": {}
}
```

**响应**: 关闭确认（可选）
```json
{
  "type": "CLOSE_ACK", 
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "data": {
    "reason": "client_request"
  }
}
```

## 4. 数据格式说明

### 4.1 音频数据格式
- **编码**: PCM 16位小端序
- **采样率**: 16000 Hz
- **声道**: 单声道
- **帧长**: 40ms (640字节)
- **传输**: Base64编码

### 4.2 视频数据格式
- **编码**: H.264
- **分辨率**: 512x512像素
- **帧率**: 25fps
- **关键帧**: 每1秒1个I帧
- **传输**: Base64编码

### 4.3 Base64编码示例
```javascript
// 音频PCM数据编码
const audioBuffer = new ArrayBuffer(640); // 40ms@16kHz
const base64Audio = btoa(String.fromCharCode(...new Uint8Array(audioBuffer)));

// 解码视频帧数据
const frameData = atob(base64VideoFrame);
const frameBuffer = new Uint8Array(frameData.length);
for (let i = 0; i < frameData.length; i++) {
    frameBuffer[i] = frameData.charCodeAt(i);
}
```

## 5. 使用流程

### 5.1 标准对话流程
```
1. 建立WebSocket连接 → wss://api.domain.com/musetalk/v1/ws/user123
2. 发送INIT消息 → 加载用户模型
3. 接收INIT_SUCCESS → 确认模型就绪
4. 发送STATE_CHANGE → 切换到speaking状态
5. 循环发送GENERATE → 处理音频流
6. 接收VIDEO_FRAME → 获取同步视频帧
7. 发送CLOSE → 结束会话
```

### 5.2 动作插入流程
```
1. 正常speaking状态处理音频
2. 检测到动作信号 → 发送ACTION消息
3. 接收ACTION_FRAME → 播放动作视频
4. 动作完成 → 自动返回speaking状态
```

## 6. 实现示例

### 6.1 Python客户端示例
```python
import websocket
import json
import base64
import uuid
import threading
import time


class MuseTalkClient:
    def __init__(self, user_id, auth_token):
        self.user_id = user_id
        self.auth_token = auth_token
        self.session_id = self.generate_uuid()
        self.ws = None
        self.available_videos = None
    
    def connect(self):
        websocket_url = f"wss://api.domain.com/musetalk/v1/ws/{self.user_id}"
        self.ws = websocket.WebSocketApp(
            websocket_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # 在新线程中运行WebSocket连接
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
    
    def on_open(self, ws):
        print("WebSocket连接已建立")
        self.init()
    
    def on_message(self, ws, message):
        message_data = json.loads(message)
        self.handle_message(message_data)
    
    def on_error(self, ws, error):
        print(f"WebSocket错误: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket连接已关闭")
    
    def init(self):
        init_message = {
            "type": "INIT",
            "session_id": self.session_id,
            "data": {
                "user_id": self.user_id,
                "auth_token": self.auth_token,
                "video_config": {
                    "resolution": "512x512",
                    "fps": 25
                }
            }
        }
        self.ws.send(json.dumps(init_message))
    
    def generate_video(self, audio_buffer, video_state):
        message = {
            "type": "GENERATE",
            "session_id": self.session_id,
            "data": {
                "audio_chunk": {
                    "format": "pcm_s16le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "duration_ms": 40,
                    "data": self.array_buffer_to_base64(audio_buffer)
                },
                "video_state": video_state
            }
        }
        self.ws.send(json.dumps(message))
    
    def change_state(self, target_state, base_video):
        message = {
            "type": "STATE_CHANGE",
            "session_id": self.session_id,
            "data": {
                "target_state": target_state,
                "base_video": base_video
            }
        }
        self.ws.send(json.dumps(message))
    
    def trigger_action(self, action_type, audio_buffer):
        message = {
            "type": "ACTION",
            "session_id": self.session_id,
            "data": {
                "action_type": action_type,
                "audio_chunk": {
                    "format": "pcm_s16le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "duration_ms": 40,
                    "data": self.array_buffer_to_base64(audio_buffer)
                }
            }
        }
        self.ws.send(json.dumps(message))
    
    def handle_message(self, message):
        message_type = message.get("type")
        
        if message_type == "INIT_SUCCESS":
            print("MuseTalk模型加载成功")
            self.available_videos = message["data"]["available_videos"]
        
        elif message_type == "VIDEO_FRAME":
            frame_data = message["data"]["frame_data"]
            self.display_frame(frame_data)
        
        elif message_type == "STATE_CHANGED":
            current_state = message["data"]["current_state"]
            print(f"状态切换完成: {current_state}")
        
        elif message_type == "ACTION_FRAME":
            action_frame = message["data"]["frame_data"]
            self.display_frame(action_frame)
    
    def display_frame(self, frame_data):
        # 处理帧数据的方法，根据实际需求实现
        print(f"收到视频帧数据: {len(frame_data)} bytes")
    
    def array_buffer_to_base64(self, buffer):
        # 将字节数据转换为base64字符串
        if isinstance(buffer, bytes):
            return base64.b64encode(buffer).decode('utf-8')
        elif isinstance(buffer, bytearray):
            return base64.b64encode(bytes(buffer)).decode('utf-8')
        else:
            # 如果是其他类型，尝试转换为bytes
            return base64.b64encode(bytes(buffer)).decode('utf-8')
    
    def generate_uuid(self):
        return str(uuid.uuid4())


# 使用示例
if __name__ == "__main__":
    client = MuseTalkClient("user123", "jwt-token")
    client.connect()
    
    # 等待连接建立
    time.sleep(1)
    
    # 40ms后开始发送音频
    def delayed_action():
        time.sleep(1)  # 等待1秒
        client.change_state("speaking", "speaking_1")
        
        # 模拟640字节的40ms音频数据
        audio_buffer = bytes(640)  # 创建640字节的空数据
        client.generate_video(audio_buffer, {
            "type": "speaking",
            "base_video": "speaking_1"
        })
    
    # 在新线程中执行延时操作
    action_thread = threading.Thread(target=delayed_action)
    action_thread.daemon = True
    action_thread.start()
    
    # 保持主程序运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("程序退出")
        if client.ws:
            client.ws.close()
```
