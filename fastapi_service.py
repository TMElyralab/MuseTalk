import os
import cv2
import math
import copy
import torch
import glob
import shutil
import pickle
import numpy as np
import subprocess
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from omegaconf import OmegaConf
from transformers import WhisperModel

app = FastAPI(title="MuseTalk FastAPI Service")


def fast_check_ffmpeg(ffmpeg_path=None):
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        if ffmpeg_path:
            os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"
            try:
                subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
                return True
            except:
                return False
        return False


class InferenceConfig(BaseModel):
    ffmpeg_path: str = "./ffmpeg-4.4-amd64-static/"
    gpu_id: int = 0
    vae_type: str = "sd-vae"
    unet_config: str = "./models/musetalkV15/musetalk.json"
    unet_model_path: str = "./models/musetalkV15/unet.pth"
    whisper_dir: str = "./models/whisper"
    inference_config: str = "configs/inference/test.yaml"
    bbox_shift: int = 0
    result_dir: str = './results'
    extra_margin: int = 10
    fps: int = 25
    audio_padding_length_left: int = 2
    audio_padding_length_right: int = 2
    batch_size: int = 8
    output_vid_name: Optional[str] = None
    use_saved_coord: bool = False
    saved_coord: bool = False
    use_float16: bool = False
    parsing_mode: str = 'jaw'
    left_cheek_width: int = 90
    right_cheek_width: int = 90
    version: str = "v15"
    video_path: Optional[str] = None      # <-- add this!
    audio_path: Optional[str] = None


# @app.post("/inference")
# async def run_inference(
#     video: UploadFile = File(...),
#     audio: UploadFile = File(...),
#     inference_params: InferenceConfig = Form(...),
# ):
    
@app.post("/inference")
async def run_inference(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    inference_params: str = Form("{}")
):
    import json
    params_dict = json.loads(inference_params)
    args = InferenceConfig(**params_dict)  # <-- make sure InferenceConfig is imported
    
    # Make temp paths for uploads
    video_path = f"temp_{video.filename}"
    audio_path = f"temp_{audio.filename}"
    with open(video_path, "wb") as f:
        f.write(await video.read())
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    
    # Attach the file paths to the config object
    args.video_path = video_path
    args.audio_path = audio_path
    
    # Proceed with your logic using args
    result_path, msg = await process_inference(args)
    # Clean up temp files
    os.remove(video_path)
    os.remove(audio_path)
    if result_path:
        return FileResponse(path=result_path, filename=os.path.basename(result_path))
    else:
        return JSONResponse({"error": msg or "Failed to process."}, status_code=500)


async def process_inference(args):
    try:
        print("Entering process_inference...")
        if not fast_check_ffmpeg(args.ffmpeg_path):
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")
            return None, "ffmpeg missing"
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        vae, unet, pe = load_all_model(
            unet_model_path=args.unet_model_path, 
            vae_type=args.vae_type,
            unet_config=args.unet_config,
            device=device
        )
        timesteps = torch.tensor([0], device=device)
        if args.use_float16:
            pe = pe.half()
            vae.vae = vae.vae.half()
            unet.model = unet.model.half()
        pe = pe.to(device)
        vae.vae = vae.vae.to(device)
        unet.model = unet.model.to(device)
        audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
        weight_dtype = unet.model.dtype
        whisper = WhisperModel.from_pretrained(args.whisper_dir)
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)
        # Face parser
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        ) if args.version == "v15" else FaceParsing()

        inference_config = OmegaConf.load(args.inference_config)
        out_paths = []
        print("Loaded inference config:", inference_config)

        for task_id in inference_config:
            try:
                print(f"Processing task: {task_id}")
                # 1. Get config for this task
                task = inference_config[task_id]
                video_path = args.video_path
                audio_path = args.audio_path
                output_vid_name = task.get("result_name", None)

                # 2. Set other params
                bbox_shift = 0 if args.version == "v15" else task.get("bbox_shift", args.bbox_shift)
                input_basename = os.path.basename(video_path).split('.')[0]
                audio_basename = os.path.basename(audio_path).split('.')[0]
                output_basename = f"{input_basename}_{audio_basename}"
                temp_dir = os.path.join(args.result_dir, f"{args.version}")
                os.makedirs(temp_dir, exist_ok=True)
                result_img_save_path = os.path.join(temp_dir, output_basename)
                crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename + ".pkl")
                os.makedirs(result_img_save_path, exist_ok=True)
                save_dir_full = os.path.join(temp_dir, input_basename)

                # 3. Extract frames from video
                if get_file_type(video_path) == "video":
                    os.makedirs(save_dir_full, exist_ok=True)
                    cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
                    os.system(cmd)
                    input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
                    fps = get_video_fps(video_path)
                elif get_file_type(video_path) == "image":
                    input_img_list = [video_path]
                    fps = args.fps
                elif os.path.isdir(video_path):
                    input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
                    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    fps = args.fps
                else:
                    raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")

                # 4. Extract audio features
                whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
                whisper_chunks = audio_processor.get_whisper_chunk(
                    whisper_input_features, device, weight_dtype, whisper, librosa_length,
                    fps=fps,
                    audio_padding_length_left=args.audio_padding_length_left,
                    audio_padding_length_right=args.audio_padding_length_right,
                )

                # 5. Preprocess input images
                if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
                    print("Using saved coordinates")
                    with open(crop_coord_save_path, 'rb') as f:
                        coord_list = pickle.load(f)
                    frame_list = read_imgs(input_img_list)
                else:
                    print("Extracting landmarks... time-consuming operation")
                    coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
                    with open(crop_coord_save_path, 'wb') as f:
                        pickle.dump(coord_list, f)

                print(f"Number of frames: {len(frame_list)}")

                input_latent_list = []
                for bbox, frame in zip(coord_list, frame_list):
                    if bbox == coord_placeholder:
                        continue
                    x1, y1, x2, y2 = bbox
                    if args.version == "v15":
                        y2 = y2 + args.extra_margin
                        y2 = min(y2, frame.shape[0])
                    crop_frame = frame[y1:y2, x1:x2]
                    crop_frame = cv2.resize(crop_frame, (256,256), interpolation=cv2.INTER_LANCZOS4)
                    latents = vae.get_latents_for_unet(crop_frame)
                    input_latent_list.append(latents)

                frame_list_cycle = frame_list + frame_list[::-1]
                coord_list_cycle = coord_list + coord_list[::-1]
                input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

                # 6. Batch inference
                print("Starting inference")
                video_num = len(whisper_chunks)
                batch_size = args.batch_size
                gen = datagen(
                    whisper_chunks=whisper_chunks,
                    vae_encode_latents=input_latent_list_cycle,
                    batch_size=batch_size,
                    delay_frame=0,
                    device=device,
                )
                res_frame_list = []
                total = int(np.ceil(float(video_num) / batch_size))
                from tqdm import tqdm
                for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
                    audio_feature_batch = pe(whisper_batch)
                    latent_batch = latent_batch.to(dtype=unet.model.dtype)
                    pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                    recon = vae.decode_latents(pred_latents)
                    for res_frame in recon:
                        res_frame_list.append(res_frame)

                # Pad generated images
                print("Padding generated images to original video size")
                for i, res_frame in enumerate(tqdm(res_frame_list)):
                    bbox = coord_list_cycle[i%(len(coord_list_cycle))]
                    ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
                    x1, y1, x2, y2 = bbox
                    if args.version == "v15":
                        y2 = y2 + args.extra_margin
                        y2 = min(y2, frame.shape[0])
                    try:
                        res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                    except:
                        continue
                    if args.version == "v15":
                        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
                    else:
                        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=fp)
                    cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)

                # 7. Save prediction results
                temp_vid_path = f"{temp_dir}/temp_{input_basename}_{audio_basename}.mp4"
                cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid_path}"
                print("Video generation command:", cmd_img2video)
                os.system(cmd_img2video)

                if output_vid_name is None:
                    output_vid_name_full = os.path.join(temp_dir, output_basename + ".mp4")
                else:
                    output_vid_name_full = os.path.join(temp_dir, output_vid_name)
                cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} {output_vid_name_full}"
                print("Audio combination command:", cmd_combine_audio)
                os.system(cmd_combine_audio)

                # Clean up temporary files for this task
                shutil.rmtree(result_img_save_path)
                os.remove(temp_vid_path)
                shutil.rmtree(save_dir_full)
                if not args.saved_coord:
                    os.remove(crop_coord_save_path)

                print(f"Results saved to {output_vid_name_full}")

                # APPEND OUTPUT
                out_paths.append(output_vid_name_full)
            except Exception as e_task:
                print(f"Error in task {task_id}:", str(e_task))
                import traceback; traceback.print_exc()

        # Return result (first output video path)
        return out_paths[0] if out_paths else None, "No output video produced!"
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Exception:", str(e))
        return None, str(e)
