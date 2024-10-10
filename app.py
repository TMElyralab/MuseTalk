import os
import time
import pdb
import re

import gradio as gr
import spaces
import numpy as np
import sys
import subprocess

from huggingface_hub import snapshot_download
import requests

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
from argparse import Namespace
import shutil
import gdown
import imageio
import ffmpeg
from moviepy.editor import *


ProjectDir = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir = os.path.join(ProjectDir, "models")

def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            print(child_path)

def download_model():
    if not os.path.exists(CheckpointsDir):
        os.makedirs(CheckpointsDir)
        print("Checkpoint Not Downloaded, start downloading...")
        tic = time.time()
        snapshot_download(
            repo_id="TMElyralab/MuseTalk",
            local_dir=CheckpointsDir,
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True, resume_download=False
        )
        # weight
        os.makedirs(f"{CheckpointsDir}/sd-vae-ft-mse/")
        snapshot_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=CheckpointsDir+'/sd-vae-ft-mse',
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True, resume_download=False
        )
        #dwpose
        os.makedirs(f"{CheckpointsDir}/dwpose/")
        snapshot_download(
            repo_id="yzd-v/DWPose",
            local_dir=CheckpointsDir+'/dwpose',
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True, resume_download=False
        )
        #vae
        url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
        response = requests.get(url)
        # 确保请求成功
        if response.status_code == 200:
            # 指定文件保存的位置
            file_path = f"{CheckpointsDir}/whisper/tiny.pt"
            os.makedirs(f"{CheckpointsDir}/whisper/")
            # 将文件内容写入指定位置
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"请求失败，状态码：{response.status_code}")
        #gdown face parse
        url = "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"
        os.makedirs(f"{CheckpointsDir}/face-parse-bisent/")
        file_path = f"{CheckpointsDir}/face-parse-bisent/79999_iter.pth"
        gdown.download(url, file_path, quiet=False)
        #resnet
        url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
        response = requests.get(url)
        # 确保请求成功
        if response.status_code == 200:
            # 指定文件保存的位置
            file_path = f"{CheckpointsDir}/face-parse-bisent/resnet18-5c106cde.pth"
            # 将文件内容写入指定位置
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"请求失败，状态码：{response.status_code}")


        toc = time.time()

        print(f"download cost {toc-tic} seconds")
        print_directory_contents(CheckpointsDir)

    else:
        print("Already download the model.")





download_model()  # for huggingface deployment.


from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder,get_bbox_range
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model






@spaces.GPU(duration=600)
@torch.no_grad()
def inference(audio_path,video_path,bbox_shift,progress=gr.Progress(track_tqdm=True)):
    args_dict={"result_dir":'./results/output', "fps":25, "batch_size":8, "output_vid_name":'', "use_saved_coord":False}#same with inferenece script
    args = Namespace(**args_dict)

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename  = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
    crop_coord_save_path = os.path.join(result_img_save_path, input_basename+".pkl") # only related to video input
    os.makedirs(result_img_save_path,exist_ok =True)

    if args.output_vid_name=="":
        output_vid_name = os.path.join(args.result_dir, output_basename+".mp4")
    else:
        output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
    ############################################## extract frames from source video ##############################################
    if get_file_type(video_path)=="video":
        save_dir_full = os.path.join(args.result_dir, input_basename)
        os.makedirs(save_dir_full,exist_ok = True)
        # cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
        # os.system(cmd)
        # 读取视频
        reader = imageio.get_reader(video_path)

        # 保存图片
        for i, im in enumerate(reader):
            imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
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
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
    bbox_shift_text=get_bbox_range(input_img_list, bbox_shift)
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
        
    # cmd_img2video = f"ffmpeg -y -v fatal -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p temp.mp4"
    # print(cmd_img2video)
    # os.system(cmd_img2video)
    # 帧率
    fps = 25
    # 图片路径
    # 输出视频路径
    output_video = 'temp.mp4'

    # 读取图片
    def is_valid_image(file):
        pattern = re.compile(r'\d{8}\.png')
        return pattern.match(file)

    images = []
    files = [file for file in os.listdir(result_img_save_path) if is_valid_image(file)]
    files.sort(key=lambda x: int(x.split('.')[0]))

    for file in files:
        filename = os.path.join(result_img_save_path, file)
        images.append(imageio.imread(filename))
        

    # 保存视频
    imageio.mimwrite(output_video, images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')

    # cmd_combine_audio = f"ffmpeg -y -v fatal -i {audio_path} -i temp.mp4 {output_vid_name}"
    # print(cmd_combine_audio)
    # os.system(cmd_combine_audio)

    input_video = './temp.mp4'
    # Check if the input_video and audio_path exist
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # 读取视频
    reader = imageio.get_reader(input_video)
    fps = reader.get_meta_data()['fps']  # 获取原视频的帧率
    reader.close() # 否则在win11上会报错：PermissionError: [WinError 32] 另一个程序正在使用此文件，进程无法访问。: 'temp.mp4'
    # 将帧存储在列表中
    frames = images

    # 保存视频并添加音频
    # imageio.mimwrite(output_vid_name, frames, 'FFMPEG', fps=fps, codec='libx264', audio_codec='aac', input_params=['-i', audio_path])
    
    # input_video = ffmpeg.input(input_video)
    
    # input_audio = ffmpeg.input(audio_path)
    
    print(len(frames))

    # imageio.mimwrite(
    #     output_video,
    #     frames,
    #     'FFMPEG',
    #     fps=25,
    #     codec='libx264',
    #     audio_codec='aac',
    #     input_params=['-i', audio_path],
    #     output_params=['-y'],  # Add the '-y' flag to overwrite the output file if it exists
    # )
    # writer = imageio.get_writer(output_vid_name, fps = 25, codec='libx264', quality=10, pixelformat='yuvj444p')
    # for im in frames:
    #     writer.append_data(im)
    # writer.close()




    # Load the video
    video_clip = VideoFileClip(input_video)

    # Load the audio
    audio_clip = AudioFileClip(audio_path)

    # Set the audio to the video
    video_clip = video_clip.set_audio(audio_clip)

    # Write the output video
    video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac',fps=25)

    os.remove("temp.mp4")
    #shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}")
    return output_vid_name,bbox_shift_text



# load model weights
audio_processor,vae,unet,pe  = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)




def check_video(video):
    if not isinstance(video, str):
        return video # in case of none type
    # Define the output video file name
    dir_path, file_name = os.path.split(video)
    if file_name.startswith("outputxxx_"):
        return video
    # Add the output prefix to the file name
    output_file_name = "outputxxx_" + file_name

    os.makedirs('./results',exist_ok=True)
    os.makedirs('./results/output',exist_ok=True)
    os.makedirs('./results/input',exist_ok=True)

    # Combine the directory path and the new file name
    output_video = os.path.join('./results/input', output_file_name)


    # # Run the ffmpeg command to change the frame rate to 25fps
    # command = f"ffmpeg -i {video} -r 25 -vcodec libx264 -vtag hvc1 -pix_fmt yuv420p crf 18   {output_video}  -y"

    # read video
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']  # get fps from original video

    # conver fps to 25
    frames = [im for im in reader]
    target_fps = 25
    
    L = len(frames)
    L_target = int(L / fps * target_fps)
    original_t = [x / fps for x in range(1, L+1)]
    t_idx = 0
    target_frames = []
    for target_t in range(1, L_target+1):
        while target_t / target_fps > original_t[t_idx]:
            t_idx += 1      # find the first t_idx so that target_t / target_fps <= original_t[t_idx]
            if t_idx >= L:
                break
        target_frames.append(frames[t_idx])

    # save video
    imageio.mimwrite(output_video, target_frames, 'FFMPEG', fps=25, codec='libx264', quality=9, pixelformat='yuv420p')
    return output_video




css = """#input_img {max-width: 1024px !important} #output_vid {max-width: 1024px; max-height: 576px}"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        "<div align='center'> <h1>MuseTalk: Real-Time High Quality Lip Synchronization with Latent Space Inpainting </span> </h1> \
                    <h2 style='font-weight: 450; font-size: 1rem; margin: 0rem'>\
                    </br>\
                    Yue Zhang <sup>\*</sup>,\
                    Minhao Liu<sup>\*</sup>,\
                    Zhaokang Chen,\
                    Bin Wu<sup>†</sup>,\
                    Yingjie He,\
                    Chao Zhan,\
                    Wenjiang Zhou\
                    (<sup>*</sup>Equal Contribution, <sup>†</sup>Corresponding Author, benbinwu@tencent.com)\
                    Lyra Lab, Tencent Music Entertainment\
                </h2> \
                <a style='font-size:18px;color: #000000' href='https://github.com/TMElyralab/MuseTalk'>[Github Repo]</a>\
                <a style='font-size:18px;color: #000000' href='https://github.com/TMElyralab/MuseTalk'>[Huggingface]</a>\
                <a style='font-size:18px;color: #000000' href=''> [Technical report(Coming Soon)] </a>\
                <a style='font-size:18px;color: #000000' href=''> [Project Page(Coming Soon)] </a>  </div>"
    )

    with gr.Row():
        with gr.Column():
            audio = gr.Audio(label="Driven Audio",type="filepath")
            video = gr.Video(label="Reference Video",sources=['upload'])
            bbox_shift = gr.Number(label="BBox_shift value, px", value=0)
            bbox_shift_scale = gr.Textbox(label="BBox_shift recommend value lower bound,The corresponding bbox range is generated after the initial result is generated. \n If the result is not good, it can be adjusted according to this reference value", value="",interactive=False)

            btn = gr.Button("Generate")
        out1 = gr.Video()
    
    video.change(
        fn=check_video, inputs=[video], outputs=[video]
    )
    btn.click(
        fn=inference,
        inputs=[
            audio,
            video,
            bbox_shift,
        ],
        outputs=[out1,bbox_shift_scale]
    )

# Set the IP and port
ip_address = "0.0.0.0"  # Replace with your desired IP address
port_number = 7860  # Replace with your desired port number


demo.queue().launch(
    share=False , debug=True, server_name=ip_address, server_port=port_number
)
