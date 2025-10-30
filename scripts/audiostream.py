import gi
import numpy as np
import subprocess
import time
import ffmpeg

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject


class GStreamerAudio:
    """ 使用 GStreamer 进行音频推流 """

    def __init__(self):
        Gst.init(None)

        # 创建 GStreamer 管道
        self.pipeline = Gst.parse_launch(
            "appsrc name=audio_source format=time is-live=true "
            "caps=audio/x-raw,format=S16LE,channels=2,rate=48000,layout=interleaved ! "
            "queue ! audioconvert ! queue ! audioresample ! "
            "queue ! opusenc ! queue ! rtpopuspay ! "
            "udpsink host=127.0.0.1 port=5001 sync=false"
        )

        self.appsrc = self.pipeline.get_by_name("audio_source")
        self.appsrc.set_property("blocksize", 65536)  # 增大 blocksize 避免数据阻塞
        self.appsrc.set_property("format", Gst.Format.TIME)

        self.pipeline.set_state(Gst.State.PLAYING)

    def send_audio(self, audio_data):
        """ 发送 PCM 音频数据到 GStreamer """
        buffer = Gst.Buffer.new_allocate(None, len(audio_data.tobytes()), None)
        buffer.fill(0, audio_data.tobytes())
        self.appsrc.emit("push-buffer", buffer)
        print(f"✅ 推送音频: {len(audio_data)} samples")

    def stop(self):
        """ 关闭音频推流 """
        self.pipeline.set_state(Gst.State.NULL)
        print("✅ GStreamer 音频推流已关闭")


class FFmpegAudioReader:
    """ 使用 FFmpeg 读取整个音频文件，并转换为 PCM """

    def __init__(self, audio_file):
        self.audio_file = audio_file
        probe = ffmpeg.probe(audio_file)
        self.sample_rate = int(probe['streams'][0]['sample_rate'])
        self.channels = int(probe['streams'][0]['channels'])

    def read_full_audio(self):
        """ 使用 FFmpeg 读取整个音频文件 """
        process = subprocess.Popen(
            ["ffmpeg", "-i", self.audio_file, "-f", "s16le", "-ac", "2", "-ar", "48000", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        raw_data = process.stdout.read()
        process.stdout.close()
        process.wait()

        if not raw_data:
            print("❌ 读取音频文件失败！")
            return None

        audio_data = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, 2)
        print(f"✅ 读取完整音频，共 {len(audio_data)} samples")
        return audio_data


def split_audio(audio_data, num_chunks):
    """ 将音频数据分割成 num_chunks 份 """
    if num_chunks <= 1:
        return [audio_data]

    chunk_size = len(audio_data) // num_chunks
    chunks = [audio_data[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]

    # 如果有剩余样本，加到最后一个 chunk
    remainder = len(audio_data) % num_chunks
    if remainder > 0:
        chunks[-1] = np.vstack((chunks[-1], audio_data[-remainder:]))

    print(f"✅ 音频已分割为 {num_chunks} 份，每份约 {chunk_size} samples")
    return chunks


def main(audio_file, split_chunks=1):
    """ 读取完整音频文件，并用 GStreamer 推流 """
    audio_reader = FFmpegAudioReader(audio_file)
    audio_data = audio_reader.read_full_audio()

    if audio_data is None:
        return

    # 分割音频
    audio_chunks = split_audio(audio_data, split_chunks)

    gst_audio = GStreamerAudio()
    try:
        for i, chunk in enumerate(audio_chunks):
            gst_audio.send_audio(chunk)
            print(f"📡 推送第 {i+1}/{split_chunks} 份音频...")
            time.sleep(0.1)  # 控制发送速率，避免堵塞
    except KeyboardInterrupt:
        print("\n⏹️ 手动停止音频流")
    finally:
        gst_audio.stop()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("❌ 用法: python audiostream.py <音频文件路径> [分块数量]")
        sys.exit(1)

    audio_file = sys.argv[1]
    split_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # 默认为 1，不分割

    main(audio_file, split_chunks)
