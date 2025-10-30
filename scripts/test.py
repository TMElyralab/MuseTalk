import ffmpeg
import numpy as np
import pyaudio
import time

def load_audio_ffmpeg(audio_path, sample_rate=48000):
    """ 用 FFmpeg 读取音频并转换为 int16 PCM """
    out, _ = (
        ffmpeg.input(audio_path)
        .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=sample_rate)
        .run(capture_stdout=True)
    )
    audio_data = np.frombuffer(out, dtype=np.int16)
    return audio_data, sample_rate

def split_audio(audio_data, num_chunks):
    """ 将音频数据均分为 num_chunks 份 """
    chunk_size = len(audio_data) // num_chunks
    audio_chunks = [audio_data[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
    return audio_chunks

def play_audio_chunk(audio_chunk, sample_rate=48000):
    """ 播放单个音频片段 """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True)
    
    print(f"🎧 播放音频片段: {len(audio_chunk)} samples")
    stream.write(audio_chunk.tobytes())  # 确保是 int16 PCM
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("✅ 音频播放完成")

def main():
    audio_path = "input.wav"  # 你的音频文件路径
    num_chunks = 10  # 切分成 10 段

    # 读取音频
    print("🎵 读取音频中...")
    audio_data, sample_rate = load_audio_ffmpeg("/home/fan370/Documents/MuseTalk/data/audio/sun.wav")
    print(f"📊 音频数据大小: {audio_data.shape}, 采样率: {sample_rate} Hz")

    # 切割音频
    print("✂️ 正在切割音频...")
    audio_chunks = split_audio(audio_data, num_chunks)
    
    # 播放音频
    for i, chunk in enumerate(audio_chunks):
        print(f"🎧 播放第 {i+1}/{len(audio_chunks)} 个音频片段")
        play_audio_chunk(chunk, sample_rate)
        time.sleep(0.1)  # 避免过快播放

if __name__ == "__main__":
    main()
