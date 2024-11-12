import os, random, cv2, argparse
import torch
from torch.utils import data as data_utils
from os.path import dirname, join, basename, isfile
import numpy as np
from glob import glob
from utils.utils import prepare_mask_and_masked_image
import torchvision.utils as vutils
import torchvision.transforms as transforms
import shutil
from tqdm import tqdm
import ast
import json
import re
import heapq

syncnet_T = 1
RESIZED_IMG = 256

connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),(7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),(13,14),(14,15),(15,16),  # 下颌线
                       (17, 18), (18, 19), (19, 20), (20, 21), #左眉毛
                       (22, 23), (23, 24), (24, 25), (25, 26), #右眉毛
                       (27, 28),(28,29),(29,30),# 鼻梁
                       (31,32),(32,33),(33,34),(34,35), #鼻子
                       (36,37),(37,38),(38, 39), (39, 40), (40, 41), (41, 36), # 左眼
                       (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42), # 右眼
                       (48, 49),(49, 50), (50, 51),(51, 52),(52, 53), (53, 54), # 上嘴唇 外延
                       (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),  # 下嘴唇 外延
                       (60, 61), (61, 62), (62, 63), (63, 64), (64, 65),  (65, 66), (66, 67), (67, 60) #嘴唇内圈
              ]  


def get_image_list(data_root, split):
    filelist = []
    imgNumList = []
    with open('filelists/{}.txt'.format(split)) as f:
        for line in f:
            line = line.strip()
            if ' ' in line:
                filename = line.split()[0]
                imgNum = int(line.split()[1])
                filelist.append(os.path.join(data_root, filename))
            imgNumList.append(imgNum)
    return filelist, imgNumList



class Dataset(object):
    def __init__(self, 
                 data_root, 
                 json_path, 
                 use_audio_length_left=1,
                 use_audio_length_right=1,
                 whisper_model_type = "tiny"
                 ):
        # self.all_videos, self.all_imgNum = get_image_list(data_root, split)
        self.audio_feature = [use_audio_length_left,use_audio_length_right]
        self.all_img_names = []
        # self.split = split
        self.img_names_path = '../data'
        self.whisper_model_type = whisper_model_type
        self.use_audio_length_left = use_audio_length_left
        self.use_audio_length_right = use_audio_length_right

        if self.whisper_model_type =="tiny":
            self.whisper_path = '../data/audios'
            self.whisper_feature_W = 5
            self.whisper_feature_H = 384
        elif self.whisper_model_type =="largeV2":
            self.whisper_path = '...'
            self.whisper_feature_W = 33
            self.whisper_feature_H = 1280
        self.whisper_feature_concateW = self.whisper_feature_W*2*(self.use_audio_length_left+self.use_audio_length_right+1) #5*2*（2+2+1）= 50
        with open(json_path, 'r') as file:
            self.all_videos = json.load(file)

        for vidname in tqdm(self.all_videos, desc="Preparing dataset"):
            json_path_names = f"{self.img_names_path}/{vidname.split('/')[-1].split('.')[0]}.json"
            if not os.path.exists(json_path_names):
                img_names = glob(join(vidname, '*.png'))
                img_names.sort(key=lambda x:int(x.split("/")[-1].split('.')[0]))
                with open(json_path_names, "w") as f:
                    json.dump(img_names,f)
            else:
                with open(json_path_names, "r") as f:
                    img_names = json.load(f)
            self.all_img_names.append(img_names)
            
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.png'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames
    
    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (RESIZED_IMG, RESIZED_IMG))
            except Exception as e:
                print("read_window has error fname not exist:",fname)
                return None

            window.append(img)

        return window
    
    def prepare_window(self, window):
        #  1 x H x W x 3
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            #随机选择某个video里
            vidname = self.all_videos[idx].split('/')[-1]
            video_imgs = self.all_img_names[idx]
            if len(video_imgs) <= 11:
                continue
            img_name = random.choice(video_imgs)
            img_idx = int(basename(img_name).split(".")[0])
            random_element = random.randint(0,len(video_imgs)-1)
            while abs(random_element - img_idx) <= 5:
                random_element = random.randint(0,len(video_imgs)-1)
            img_dir = os.path.dirname(img_name)
            ref_image = os.path.join(img_dir, f"{str(random_element)}.png")
            target_window_fnames = self.get_window(img_name)
            ref_window_fnames = self.get_window(ref_image)
           
            if target_window_fnames is None or ref_window_fnames is None:
                print("No such img",img_name, ref_image)
                continue
                 
            try:
                #构建目标img数据
                target_window = self.read_window(target_window_fnames)
                if target_window is None :
                    print("No such target window,",target_window_fnames)
                    continue
                #构建参考img数据
                ref_window = self.read_window(ref_window_fnames)
    
                if ref_window is None:
                    print("No such target ref window,",ref_window)
                    continue
            except Exception as e:
                print(f"发生未知错误：{e}")
                continue
          
            #构建target输入
            target_window = self.prepare_window(target_window)
            image = gt = target_window.copy().squeeze()
            target_window[:, :, target_window.shape[2]//2:] = 0.                   # upper half face, mask掉下半部分        V1：输入       
            ref_image = self.prepare_window(ref_window).squeeze()   
       
            

            mask = torch.zeros((ref_image.shape[1], ref_image.shape[2]))
            mask[:ref_image.shape[2]//2,:] = 1
            image = torch.FloatTensor(image)
            mask, masked_image = prepare_mask_and_masked_image(image,mask)
           
            
            
            #音频特征
            window_index = self.get_frame_id(img_name)
            sub_folder_name = vidname.split('/')[-1]
            
            ## 根据window_index加载相邻的音频
            audio_feature_all = []
            is_index_out_of_range = False
            if os.path.isdir(os.path.join(self.whisper_path, sub_folder_name)):
                for feat_idx in range(window_index-self.use_audio_length_left,window_index+self.use_audio_length_right+1):
                    # 判定是否越界
                    audio_feat_path = os.path.join(self.whisper_path, sub_folder_name, str(feat_idx) + ".npy")
                    if not os.path.exists(audio_feat_path):
                        is_index_out_of_range = True
                        break

                    try:
                        audio_feature_all.append(np.load(audio_feat_path))
                    except Exception as e:
                        print(f"发生未知错误：{e}")
                        print(f"npy load error {audio_feat_path}")
                if is_index_out_of_range:
                    continue
                audio_feature = np.concatenate(audio_feature_all, axis=0)
            else:
                continue

            audio_feature = audio_feature.reshape(1, -1, self.whisper_feature_H) #1， -1， 384
            if audio_feature.shape != (1,self.whisper_feature_concateW, self.whisper_feature_H):  #1 50 384
                print(f"shape error!! {vidname} {window_index}, audio_feature.shape: {audio_feature.shape}")
                continue
            audio_feature = torch.squeeze(torch.FloatTensor(audio_feature))
            return ref_image, image, masked_image, mask, audio_feature
         
    
    
if __name__ == "__main__":
    data_root = '...'
    val_data = Dataset(data_root, 
                          'val', 
                          use_audio_length_left = 2,
                          use_audio_length_right = 2,
                          whisper_model_type = "tiny"
                          )
    val_data_loader = data_utils.DataLoader(
        val_data, batch_size=4, shuffle=True,
        num_workers=1)

    for i, data in enumerate(val_data_loader):
        ref_image, image, masked_image, mask, audio_feature = data

 
