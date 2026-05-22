import argparse
import os
import sys
import yaml
import subprocess

def validate_inputs(face_path, audio_path):
    # Check if face_path exists and is non-empty
    if not os.path.exists(face_path):
        print(f"❌ 错误: 输入的 Face 文件/目录不存在: {face_path}")
        return False
    if not os.path.isdir(face_path) and os.path.getsize(face_path) == 0:
        print(f"❌ 错误: 输入的 Face 文件为空 (0 字节): {face_path}")
        return False
        
    # Check if audio_path exists and is non-empty
    if not os.path.exists(audio_path):
        print(f"❌ 错误: 输入的 Audio 文件不存在: {audio_path}")
        return False
    if os.path.getsize(audio_path) == 0:
        print(f"❌ 错误: 输入的 Audio 文件为空 (0 字节): {audio_path}")
        return False

    # Check face path validity (video or image)
    _, ext = os.path.splitext(face_path)
    ext_lower = ext.lower()
    if ext_lower in ['.avi', '.mp4', '.mov', '.flv', '.mkv']:
        # Try to open with OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(face_path)
            if not cap.isOpened():
                print(f"❌ 错误: 无法打开视频文件 {face_path}，文件可能损坏或写入不完整（如 moov atom 缺失）。")
                cap.release()
                return False
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps <= 0:
                print(f"❌ 错误: 视频文件 {face_path} 的帧率无效（{fps}），文件可能损坏或写入不完整。")
                return False
        except ImportError:
            # If cv2 is not installed (though it should be), we fall back to a basic check
            pass
    elif ext_lower in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        try:
            import cv2
            img = cv2.imread(face_path)
            if img is None or img.size == 0:
                print(f"❌ 错误: 无法读取图像文件 {face_path}，文件可能损坏或不合法。")
                return False
        except ImportError:
            pass
    elif os.path.isdir(face_path):
        # Check if directory contains images
        import glob
        img_list = glob.glob(os.path.join(face_path, '*.[jpJP][pnPN]*[gG]'))
        if not img_list:
            print(f"❌ 错误: 输入目录中未找到任何图片: {face_path}")
            return False
    else:
        print(f"❌ 错误: 不支持的 Face 输入类型或后缀 {face_path}。")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description="MuseTalk CLI Wrapper")
    parser.add_argument("--face", type=str, required=True, help="Input face video path")
    parser.add_argument("--audio", type=str, required=True, help="Input audio path")
    parser.add_argument("--outfile", type=str, required=True, help="Output video path (must be a specific .mp4 file)")
    parser.add_argument("--version", type=str, default="v1.5", choices=["v1.0", "v1.5"])
    
    args = parser.parse_args()

    # 将输入的路径转化为绝对路径，确保 MuseTalk 底层能够正确识别
    face_path = os.path.abspath(args.face)
    audio_path = os.path.abspath(args.audio)
    outfile_path = os.path.abspath(args.outfile)

    # 验证输入文件的有效性
    if not validate_inputs(face_path, audio_path):
        sys.exit(1)

    # MuseTalk 默认是通过 yaml 配置文件来执行批量任务的，为了支持单次 CLI 调用，这里动态生成一个临时的 yaml
    config_dict = {
        "task_0": {
            "video_path": face_path,
            "audio_path": audio_path
        }
    }
    
    temp_yaml = os.path.abspath("configs/inference/temp_cli_task.yaml")
    os.makedirs(os.path.dirname(temp_yaml), exist_ok=True)
    with open(temp_yaml, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)
        
    # 提取输出目录
    out_dir = os.path.dirname(outfile_path)
    os.makedirs(out_dir, exist_ok=True)

    # 根据选定的版本，自动组装底层的模型路径
    if args.version == "v1.0":
        unet_model_path = "./models/musetalk/pytorch_model.bin"
        unet_config = "./models/musetalk/musetalk.json"
        version_arg = "v1"
    else:
        unet_model_path = "./models/musetalkV15/unet.pth"
        unet_config = "./models/musetalkV15/musetalk.json"
        version_arg = "v15"

    # 组装最终的底层调用命令
    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--inference_config", temp_yaml,
        "--result_dir", out_dir,
        "--output_vid_name", outfile_path,  # 传入绝对路径，底层代码 os.path.join 遇到绝对路径会自动使用绝对路径
        "--unet_model_path", unet_model_path,
        "--unet_config", unet_config,
        "--version", version_arg
    ]
    
    print(f"🚀 开始执行数字人合成任务...\n底层调用命令: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ 任务圆满完成！\n视频已保存至: {outfile_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 推理过程中发生错误。底层推理命令执行失败，请检查上方日志。")
        sys.exit(1)
    finally:
        # 清理临时的 yaml 配置文件
        if os.path.exists(temp_yaml):
            os.remove(temp_yaml)

if __name__ == "__main__":
    main()
