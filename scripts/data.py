import cv2
import os
# import dlib
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
import uuid

from musetalk.utils.utils import get_file_type,get_video_fps
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model
import shutil
import gc

# load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)

def get_largest_integer_filename(folder_path):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        return -1

    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Check if the folder is empty
    if not files:
        return -1

    # Extract the integer part of filenames and find the largest
    largest_integer = -1
    for file in files:
        try:
            # Get the integer part of the filename
            file_int = int(os.path.splitext(file)[0])
            if file_int > largest_integer:
                largest_integer = file_int
        except ValueError:
            # Skip files that don't have an integer filename
            continue

    return largest_integer

def datagen(whisper_chunks,
            crop_images,
            batch_size=8,
            delay_frame=0):
    whisper_batch, crop_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(crop_images)
        crop_image = crop_images[idx]
        whisper_batch.append(w)
        crop_batch.append(crop_image)

        if len(crop_batch) >= batch_size:
            whisper_batch = np.stack(whisper_batch)
            # latent_batch = torch.cat(latent_batch, dim=0)
            yield whisper_batch, crop_batch
            whisper_batch, crop_batch = [], []

    # the last batch may smaller than batch size
    if len(crop_batch) > 0:
        whisper_batch = np.stack(whisper_batch)
        # latent_batch = torch.cat(latent_batch, dim=0)

        yield whisper_batch, crop_batch

@torch.no_grad()
def main(args):
    global pe
    if args.use_float16 is True:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
    
    inference_config = OmegaConf.load(args.inference_config)
    total_audio_index=get_largest_integer_filename(f"data/audios/{args.folder_name}")
    total_image_index=get_largest_integer_filename(f"data/images/{args.folder_name}")
    temp_audio_index=total_audio_index
    temp_image_index=total_image_index
    for task_id in inference_config:
        video_path = inference_config[task_id]["video_path"]
        audio_path = inference_config[task_id]["audio_path"]
        bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)
        folder_name = args.folder_name
        if not os.path.exists(f"data/images/{folder_name}/"):
            os.makedirs(f"data/images/{folder_name}")
        if not os.path.exists(f"data/audios/{folder_name}/"):
            os.makedirs(f"data/audios/{folder_name}")
        input_basename = os.path.basename(video_path).split('.')[0]
        audio_basename  = os.path.basename(audio_path).split('.')[0]
        output_basename = f"{input_basename}_{audio_basename}"
        result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
        crop_coord_save_path = os.path.join(result_img_save_path, input_basename+".pkl") # only related to video input
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
        whisper_feature = audio_processor.audio2feat(audio_path)
        for __ in range(0, len(whisper_feature) - 1, 2):  # -1 to avoid index error if the list has an odd number of elements
            # Combine two consecutive chunks
            # pair_of_chunks = np.array([whisper_feature[__], whisper_feature[__+1]])
            concatenated_chunks = np.concatenate([whisper_feature[__], whisper_feature[__+1]], axis=0)
            # Save the pair to a .npy file
            np.save(f'data/audios/{folder_name}/{total_audio_index+(__//2)+1}.npy', concatenated_chunks)
            temp_audio_index=(__//2)+total_audio_index+1
        total_audio_index=temp_audio_index
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)

        ############################################## preprocess input image  ##############################################
        gc.collect()
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

                
        i = 0
        input_latent_list = []
        crop_i=0
        crop_data=[]
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox

            x1=max(0,x1)
            y1=max(0,y1)
            x2=max(0,x2)
            y2=max(0,y2)

            if ((y2-y1)<=0) or ((x2-x1)<=0):
                continue
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(crop_frame)
            crop_data.append(crop_frame)
            input_latent_list.append(latents)
            crop_i+=1
    
        # to smooth the first and the last frame
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        crop_data = crop_data + crop_data[::-1]
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)
        batch_size = args.batch_size
        gen = datagen(whisper_chunks,crop_data,batch_size)
        crop_index = 0
        for i, (whisper_batch,crop_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
            for image,audio in zip(crop_batch,whisper_batch):
                cv2.imwrite(f"data/images/{folder_name}/{str(crop_index+total_image_index+1)}.png",image)
                temp_image_index = crop_index + total_image_index + 1
                crop_index += 1
                
                # np.save(f'data/audios/{folder_name}/{str(i+crop_index)}.npy', audio)
        total_image_index=temp_image_index
        gc.collect()

           

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml")
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--folder_name", default=f'{uuid.uuid4()}', help="path to output")

    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_vid_name", type=str, default=None)
    parser.add_argument("--use_saved_coord",
                        action="store_true",
                        help='use saved coordinate to save time')
    parser.add_argument("--use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )

    args = parser.parse_args()
    main(args)


def process_audio(audio_path):
    whisper_feature = audio_processor.audio2feat(audio_path)
    np.save('audio/your_filename.npy', whisper_feature)

def mask_face(image):
    # Load dlib's face detector and the landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor_path = "/content/shape_predictor_68_face_landmarks.dat"  # Set path to your downloaded predictor file
    predictor = dlib.shape_predictor(predictor_path)

    # Load your input image
    # image_path = "/content/ori_frame_00000077.png"  # Replace with the path to your input image
    # image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    # Process each detected face
    for face in faces:
        # Predict landmarks
        landmarks = predictor(gray, face)

        # The indices of nose landmarks are 27 to 35
        nose_tip = landmarks.part(33).y

        # Blacken the region below the nose tip
        blacken_area = image[nose_tip:, :]
        blacken_area[:] = (0, 0, 0)

    # Save the final image or display it
    # cv2.imwrite("output_image.jpg", image)
    return image
