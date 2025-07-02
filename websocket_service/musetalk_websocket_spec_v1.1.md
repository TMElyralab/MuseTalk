# MuseTalk WebSocket服务接口规范

## 1. 概述

MuseTalk WebSocket服务负责实时流式输出唇形同步视频。连接建立后，服务立即开始持续输出基础视频流；当接收到音频数据时，自动切换为基于用户个性化模型的唇形同步视频流；当收到动作触发请求时，在当前视频流中插入指定的动作视频片段。

**核心工作模式**：
- **默认模式**：连接后立即输出基础视频流（如idle状态视频）
- **音频同步模式**：接收音频流时输出lip-sync合成的视频流
- **动作插入模式**：收到ACTION请求时在视频流中插入动作片段

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

**响应**: 连接和模型加载状态确认，同时开始输出默认视频流
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
    ],
    "default_video": "idle_0",
    "streaming_started": true
  }
}
```

**说明**: 
- `user_id`: 用户唯一标识，用于加载对应的个性化模型
- `auth_token`: JWT认证令牌
- `video_config`: 输出视频配置，分辨率固定512x512，帧率25fps
- `available_videos`: 返回该用户可用的基础视频列表
- `default_video`: 连接后立即开始输出的默认视频
- `streaming_started`: 表示视频流已开始输出，客户端可立即接收VIDEO_FRAME消息

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
    "action_index": 1
  }
}
```

**响应**: 动作触发确认
```json
{
  "type": "ACTION_TRIGGERED",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "data": {
    "action_index": 1,
    "inserted": true
  }
}
```

**说明**:
- `action_index`: 动作序号，1对应"action_1"，2对应"action_2"
- `inserted`: 表示动作已插入到当前视频流中
- 动作视频会插入到当前视频流中，播放完成后自动恢复之前的状态
- ACTION请求和音频流请求互斥，不会同时发生

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

### 5.1 基础连接流程
```
1. 建立WebSocket连接 → wss://api.domain.com/musetalk/v1/ws/user123
2. 发送INIT消息 → 加载用户模型
3. 接收INIT_SUCCESS → 确认模型就绪，立即开始接收VIDEO_FRAME
4. 持续接收VIDEO_FRAME → 获取默认基础视频流（如idle状态）
```

### 5.2 音频同步流程
```
1. 基础视频流持续输出中
2. 发送GENERATE消息 → 传入音频数据
3. 视频流自动切换为lip-sync同步视频
4. 持续发送GENERATE → 处理音频流，接收同步视频帧
5. 停止发送音频 → 自动恢复基础视频流
```

### 5.3 动作插入流程
```
1. 当前视频流正常输出中（基础流或音频同步流）
2. 发送ACTION消息 → 触发指定动作
3. 接收ACTION_TRIGGERED → 确认动作已插入
4. 视频流中插入动作片段 → 持续接收VIDEO_FRAME
5. 动作播放完成 → 自动恢复之前的视频流状态
```

### 5.4 完整工作流程
```
连接建立 → 默认视频流 ⇄ 音频同步流
              ↓         ↓
           动作插入 → 恢复原状态
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
    
    def trigger_action(self, action_index):
        message = {
            "type": "ACTION",
            "session_id": self.session_id,
            "data": {
                "action_index": action_index
            }
        }
        self.ws.send(json.dumps(message))
    
    def handle_message(self, message):
        message_type = message.get("type")
        
        if message_type == "INIT_SUCCESS":
            print("MuseTalk模型加载成功，视频流已开始输出")
            self.available_videos = message["data"]["available_videos"]
            default_video = message["data"]["default_video"]
            print(f"默认视频: {default_video}")
        
        elif message_type == "VIDEO_FRAME":
            frame_data = message["data"]["frame_data"]
            frame_timestamp = message["data"]["frame_timestamp"]
            self.display_frame(frame_data, frame_timestamp)
        
        elif message_type == "STATE_CHANGED":
            current_state = message["data"]["current_state"]
            print(f"状态切换完成: {current_state}")
        
        elif message_type == "ACTION_TRIGGERED":
            action_index = message["data"]["action_index"]
            inserted = message["data"]["inserted"]
            print(f"动作已触发: action_{action_index}，插入状态: {inserted}")
    
    def display_frame(self, frame_data, frame_timestamp=None):
        # 处理帧数据的方法，根据实际需求实现
        timestamp_info = f", 时间戳: {frame_timestamp}ms" if frame_timestamp else ""
        print(f"收到视频帧数据: {len(frame_data)} bytes{timestamp_info}")
    
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
    
    # 等待连接建立，连接成功后会立即开始接收基础视频流
    time.sleep(2)
    
    def demo_workflow():
        time.sleep(3)  # 等待3秒，观察基础视频流
        
        print("开始发送音频，切换到lip-sync模式")
        # 模拟640字节的40ms音频数据
        audio_buffer = bytes(640)
        client.generate_video(audio_buffer, {
            "type": "speaking",
            "base_video": "speaking_1"
        })
        
        time.sleep(2)  # 模拟音频同步2秒
        
        print("触发动作1")
        client.trigger_action(1)  # 触发action_1
        
        time.sleep(5)  # 等待动作播放
        
        print("继续发送音频")
        client.generate_video(audio_buffer, {
            "type": "speaking", 
            "base_video": "speaking_2"
        })
    
    # 在新线程中执行演示流程
    demo_thread = threading.Thread(target=demo_workflow)
    demo_thread.daemon = True
    demo_thread.start()
    
    # 保持主程序运行，持续接收视频流
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("程序退出")
        if client.ws:
            client.ws.close()
```
