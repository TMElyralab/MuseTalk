import os
import cv2
import copy
import glob
import torch
import shutil
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import WhisperModel

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder



@torch.no_grad()
def main(args):
    # Configure ffmpeg path
    if args.ffmpeg_path not in os.getenv('PATH'):
        print("Adding ffmpeg to PATH")
        os.environ["PATH"] = f"{args.ffmpeg_path}:{os.environ['PATH']}"
        
    # Set computing device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path, 
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)
    
    
    if args.use_float16 is True:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
        
    # Initialize audio processor and Whisper model  
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    
    # Initialize face parser
    fp = FaceParsing()
    
    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)
    for task_id in inference_config:
        video_path = inference_config[task_id]["video_path"]
        audio_path = inference_config[task_id]["audio_path"]
        bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)

        input_basename = os.path.basename(video_path).split('.')[0]
        audio_basename  = os.path.basename(audio_path).split('.')[0]
        output_basename = f"{input_basename}_{audio_basename}"
        result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
        crop_coord_save_path = os.path.join(args.result_dir, input_basename+".pkl") # only related to video input
        os.makedirs(result_img_save_path,exist_ok =True)
        
        if args.output_vid_name is None:
            output_vid_name = os.path.join(args.result_dir, output_basename+".mp4")
        else:
            output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
        ############################################## extract frames from source video ##############################################
        if get_file_type(video_path)=="video":
            save_dir_full = os.path.join(args.result_dir, input_basename)
            os.makedirs(save_dir_full,exist_ok = True)
            cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
            os.system(cmd)
            input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
            fps = get_video_fps(video_path)
        elif get_file_type(video_path)=="image":
            input_img_list = [video_path, ]
            fps = args.fps
        elif os.path.isdir(video_path):  # input img folder
            input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            fps = args.fps
        else:
            raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")

        ############################################## extract audio feature ##############################################
        # Extract audio features
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features, 
            device, 
            weight_dtype, 
            whisper, 
            librosa_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
        
        ############################################## preprocess input image  ##############################################
        if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
            print("using extracted coordinates")
            with open(crop_coord_save_path,'rb') as f:
                saved_lists = pickle.load(f)
                coord_list = saved_lists['coord']
                input_latent_list = saved_lists['latent']
            frame_list = read_imgs(input_img_list)
        else:
            print("extracting landmarks...time consuming")
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
            input_latent_list = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
                latents = vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)
            with open(crop_coord_save_path, 'wb') as f:
                pickle.dump({'coord': coord_list, 'latent': input_latent_list}, f)
                

    
        # to smooth the first and the last frame
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        ############################################## inference batch by batch ##############################################
        print("start inference")
        video_num = len(whisper_chunks)
        batch_size = args.batch_size
        gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
        res_frame_list = []
        for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
            audio_feature_batch = pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)
                
        ############################################## pad to full image ##############################################
        print("pad talking image to original video")
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            bbox = coord_list_cycle[i%(len(coord_list_cycle))]
            ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
            except:
                continue
            
            # Merge results
            combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=fp)      
            cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)

        cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
        print(cmd_img2video)
        os.system(cmd_img2video)
        
        cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 {output_vid_name}"
        print(cmd_combine_audio)
        os.system(cmd_combine_audio)
        
        os.remove("temp.mp4")
        shutil.rmtree(result_img_save_path)
        print(f"result is save to {output_vid_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml")
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_vid_name", type=str, default=None)
    parser.add_argument("--use_saved_coord",
                        action="store_true",
                        help='use saved coordinate to save time')
    parser.add_argument("--use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalk/pytorch_model.bin", help="Path to UNet model weights")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/config.json", help="Path to UNet configuration file")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    args = parser.parse_args()
    main(args)
