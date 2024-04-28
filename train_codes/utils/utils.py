import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange

import torch
import torchvision.transforms as transforms

from diffusers import AutoencoderKL
import matplotlib.pyplot as plt
import PIL
import os
import cv2
from glob import glob


def preprocess_img_tensor(image_tensor):
    # 假设输入是一个形状为 (N, C, H, W) 的 PyTorch 张量
    N, C, H, W = image_tensor.shape
    # 计算新的宽度和高度，使其为 32 的整数倍
    new_w = W - W % 32
    new_h = H - H % 32
    # 使用 torchvision.transforms 库中的方法进行缩放和重采样
    transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # 对每个图像应用变换，并将结果存储在一个新的张量中
    preprocessed_images = torch.empty((N, C, new_h, new_w), dtype=torch.float32)
    for i in range(N):
        # 使用 F.interpolate 替换 transforms.Resize
        resized_image = F.interpolate(image_tensor[i].unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
        preprocessed_images[i] = transform(resized_image.squeeze(0))

    return preprocessed_images


def prepare_mask_and_masked_image(image_tensor, mask_tensor):
    # 假设输入 image_tensor 的形状为 [C, H, W]，输入 mask_tensor 的形状为 [H, W]
#     # 对图像张量进行归一化
    image_tensor_ori = (image_tensor.to(dtype=torch.float32) / 127.5) - 1.0
#     # 对遮罩张量进行归一化和二值化
#     mask_tensor = (mask_tensor.to(dtype=torch.float32) / 255.0).unsqueeze(0)
    mask_tensor[mask_tensor < 0.5] = 0
    mask_tensor[mask_tensor >= 0.5] = 1
    # 创建遮罩后的图像
    masked_image_tensor = image_tensor * (mask_tensor > 0.5)

    return mask_tensor, masked_image_tensor


def encode_latents(vae, image):
#     init_image = preprocess_image(image) 
    init_latent_dist = vae.encode(image.to(vae.dtype)).latent_dist
    init_latents = 0.18215 * init_latent_dist.sample()
    return init_latents

def decode_latents(vae, latents, ref_images=None):
    latents = (1/  0.18215) * latents
    image = vae.decode(latents.to(vae.dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    if ref_images is not None:
        ref_images = ref_images.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        ref_images = (ref_images * 255).round().astype("uint8")
        h = image.shape[1]
        image[:, :h//2] = ref_images[:, :h//2]
    image = [Image.fromarray(im) for im in image]

    return image[0]

