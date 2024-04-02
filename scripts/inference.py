import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy

from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model

# load model weights
audio_processor,vae,unet,pe  = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)

@torch.no_grad()
def main(args):
    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)
    for task_id in inference_config:
        video_path = inference_config[task_id]["video_path"]
        audio_path = inference_config[task_id]["audio_path"]

        input_basename = os.path.basename(video_path).split('.')[0]
        audio_basename  = os.path.basename(audio_path).split('.')[0]
        output_basename = f"{input_basename}_{audio_basename}"
        crop_coord_save_path = os.path.join(args.result_dir, input_basename+".pkl") # only related to video input
        result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
        os.makedirs(result_img_save_path,exist_ok =True)
        
        if args.output_vid_name=="":
            output_vid_name = os.path.join(args.result_dir, output_basename+".mp4")
        else:
            output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
        ############################################## extract frames from source video ##############################################
        if get_file_type(video_path)=="video":
            save_dir_full = os.path.join(args.result_dir, input_basename)
            os.makedirs(save_dir_full,exist_ok = True)
            cmd = f"ffmpeg -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
            os.system(cmd)
            input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
            fps = get_video_fps(video_path)
        else: # input img folder
            input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            fps = args.fps
        #print(input_img_list)
        ############################################## extract audio feature ##############################################
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        ############################################## preprocess input image  ##############################################
        if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
            print("using extracted coordinates")
            with open(crop_coord_save_path,'rb') as f:
                coord_list = pickle.load(f)
            frame_list = read_imgs(input_img_list)
        else:
            print("extracting landmarks...time consuming")
            coord_list, frame_list = get_landmark_and_bbox(input_img_list,args.bbox_shift)
            with open(crop_coord_save_path, 'wb') as f:
                pickle.dump(coord_list, f)
                
        i = 0
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
    
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
            
            tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
            audio_feature_batch = torch.stack(tensor_list).to(unet.device) # torch, B, 5*N,384
            audio_feature_batch = pe(audio_feature_batch)
            
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
#                 print(bbox)
                continue
            
            combine_frame = get_image(ori_frame,res_frame,bbox)
            cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
            
        cmd_img2video = f"ffmpeg -y -v fatal -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
        print(cmd_img2video)
        os.system(cmd_img2video)
        
        cmd_combine_audio = f"ffmpeg -i {audio_path} -i temp.mp4 {output_vid_name} -y"
        print(cmd_combine_audio)
        os.system(cmd_combine_audio)
        
        os.system("rm temp.mp4")
        os.system(f"rm -r {result_img_save_path}")
        print(f"result is save to {output_vid_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config",type=str, default="configs/inference/test_img.yaml")
    parser.add_argument("--bbox_shift",type=int, default=0)
    parser.add_argument("--result_dir", default='./results', help="path to output")

    parser.add_argument("--fps",type=int, default=25)
    parser.add_argument("--batch_size",type=int, default=8)
    parser.add_argument("--output_vid_name",type=str,default='')
    parser.add_argument("--use_saved_coord",action="store_true", help='use saved coordinate to save time')


    args = parser.parse_args()
    main(args)
    