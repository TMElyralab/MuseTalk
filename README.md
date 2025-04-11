# MuseTalk

<strong>MuseTalk: Real-Time High-Fidelity Video Dubbing via Spatio-Temporal Sampling</strong>

Yue Zhang<sup>\*</sup>,
Zhizhou Zhong<sup>\*</sup>,
Minhao Liu<sup>\*</sup>,
Zhaokang Chen,
Bin Wu<sup>†</sup>,
Yubin Zeng, 
Chao Zhan,
Junxin Huang,
Yingjie He,
Wenjiang Zhou
(<sup>*</sup>Equal Contribution, <sup>†</sup>Corresponding Author, benbinwu@tencent.com)

Lyra Lab, Tencent Music Entertainment

**[github](https://github.com/TMElyralab/MuseTalk)**    **[huggingface](https://huggingface.co/TMElyralab/MuseTalk)**    **[space](https://huggingface.co/spaces/TMElyralab/MuseTalk)**    **[Technical report](https://arxiv.org/abs/2410.10122)**

We introduce `MuseTalk`, a **real-time high quality** lip-syncing model (30fps+ on an NVIDIA Tesla V100). MuseTalk can be applied with input videos, e.g., generated by [MuseV](https://github.com/TMElyralab/MuseV), as a complete virtual human solution.

## 🔥 Updates
We're excited to unveil MuseTalk 1.5. 
This version **(1)** integrates training with perceptual loss, GAN loss, and sync loss, significantly boosting its overall performance. **(2)** We've implemented a two-stage training strategy and a spatio-temporal data sampling approach to strike a balance between visual quality and lip-sync accuracy. 
Learn more details [here](https://arxiv.org/abs/2410.10122).
**The inference codes, training codes and model weights of MuseTalk 1.5 are all available now!** 🚀

# Overview
`MuseTalk` is a real-time high quality audio-driven lip-syncing model trained in the latent space of `ft-mse-vae`, which

1. modifies an unseen face according to the input audio, with a size of face region of `256 x 256`.
1. supports audio in various languages, such as Chinese, English, and Japanese.
1. supports real-time inference with 30fps+ on an NVIDIA Tesla V100.
1. supports modification of the center point of the face region proposes, which **SIGNIFICANTLY** affects generation results. 
1. checkpoint available trained on the HDTF and private dataset.

# News
- [04/05/2025] :mega: We are excited to announce that the training code is now open-sourced! You can now train your own MuseTalk model using our provided training scripts and configurations.
- [03/28/2025] We are thrilled to announce the release of our 1.5 version. This version is a significant improvement over the 1.0 version, with enhanced clarity, identity consistency, and precise lip-speech synchronization. We update the [technical report](https://arxiv.org/abs/2410.10122) with more details.
- [10/18/2024] We release the [technical report](https://arxiv.org/abs/2410.10122v2). Our report details a superior model to the open-source L1 loss version. It includes GAN and perceptual losses for improved clarity, and sync loss for enhanced performance.
- [04/17/2024] We release a pipeline that utilizes MuseTalk for real-time inference.
- [04/16/2024] Release Gradio [demo](https://huggingface.co/spaces/TMElyralab/MuseTalk) on HuggingFace Spaces (thanks to HF team for their community grant)
- [04/02/2024] Release MuseTalk project and pretrained models.


## Model
![Model Structure](https://github.com/user-attachments/assets/02f4a214-1bdd-4326-983c-e70b478accba)
MuseTalk was trained in latent spaces, where the images were encoded by a freezed VAE. The audio was encoded by a freezed `whisper-tiny` model. The architecture of the generation network was borrowed from the UNet of the `stable-diffusion-v1-4`, where the audio embeddings were fused to the image embeddings by cross-attention. 

Note that although we use a very similar architecture as Stable Diffusion, MuseTalk is distinct in that it is **NOT** a diffusion model. Instead, MuseTalk operates by inpainting in the latent space with a single step.

## Cases

<table>
<tr>
<td width="33%">

### Input Video
---
https://github.com/TMElyralab/MuseTalk/assets/163980830/37a3a666-7b90-4244-8d3a-058cb0e44107

---
https://github.com/user-attachments/assets/1ce3e850-90ac-4a31-a45f-8dfa4f2960ac

---
https://github.com/user-attachments/assets/fa3b13a1-ae26-4d1d-899e-87435f8d22b3

---
https://github.com/user-attachments/assets/15800692-39d1-4f4c-99f2-aef044dc3251

---
https://github.com/user-attachments/assets/a843f9c9-136d-4ed4-9303-4a7269787a60

---
https://github.com/user-attachments/assets/6eb4e70e-9e19-48e9-85a9-bbfa589c5fcb

</td>
<td width="33%">

### MuseTalk 1.0
---
https://github.com/user-attachments/assets/c04f3cd5-9f77-40e9-aafd-61978380d0ef

---
https://github.com/user-attachments/assets/2051a388-1cef-4c1d-b2a2-3c1ceee5dc99

---
https://github.com/user-attachments/assets/b5f56f71-5cdc-4e2e-a519-454242000d32

---
https://github.com/user-attachments/assets/a5843835-04ab-4c31-989f-0995cfc22f34

---
https://github.com/user-attachments/assets/3dc7f1d7-8747-4733-bbdd-97874af0c028

---
https://github.com/user-attachments/assets/3c78064e-faad-4637-83ae-28452a22b09a

</td>
<td width="33%">

### MuseTalk 1.5
---
https://github.com/user-attachments/assets/999a6f5b-61dd-48e1-b902-bb3f9cbc7247

---
https://github.com/user-attachments/assets/d26a5c9a-003c-489d-a043-c9a331456e75

---
https://github.com/user-attachments/assets/471290d7-b157-4cf6-8a6d-7e899afa302c

---
https://github.com/user-attachments/assets/1ee77c4c-8c70-4add-b6db-583a12faa7dc

---
https://github.com/user-attachments/assets/370510ea-624c-43b7-bbb0-ab5333e0fcc4

---
https://github.com/user-attachments/assets/b011ece9-a332-4bc1-b8b7-ef6e383d7bde

</td>
</tr>
</table>


# TODO:
- [x] trained models and inference codes.
- [x] Huggingface Gradio [demo](https://huggingface.co/spaces/TMElyralab/MuseTalk).
- [x] codes for real-time inference.
- [x] [technical report](https://arxiv.org/abs/2410.10122v2).
- [x] a better model with updated [technical report](https://arxiv.org/abs/2410.10122).
- [x] realtime inference code for 1.5 version.
- [x] training and data preprocessing codes. 
- [ ] **always** welcome to submit issues and PRs to improve this repository! 😊


# Getting Started
We provide a detailed tutorial about the installation and the basic usage of MuseTalk for new users:

## Third party integration
Thanks for the third-party integration, which makes installation and use more convenient for everyone.
We also hope you note that we have not verified, maintained, or updated third-party. Please refer to this project for specific results.

### [ComfyUI](https://github.com/chaojie/ComfyUI-MuseTalk)

## Installation
To prepare the Python environment and install additional packages such as opencv, diffusers, mmcv, etc., please follow the steps below:
### Build environment

We recommend a python version >=3.10 and cuda version =11.7. Then build environment as follows:

```shell
pip install -r requirements.txt
```

### mmlab packages
```bash
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 
```

### Download ffmpeg-static
Download the ffmpeg-static and
```
export FFMPEG_PATH=/path/to/ffmpeg
```
for example:
```
export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static
```
### Download weights
You can download weights manually as follows:

1. Download our trained [weights](https://huggingface.co/TMElyralab/MuseTalk).
```bash
# !pip install -U "huggingface_hub[cli]" 
export HF_ENDPOINT=https://hf-mirror.com 
huggingface-cli download TMElyralab/MuseTalk --local-dir models/
```

2. Download the weights of other components:
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   - [whisper](https://huggingface.co/openai/whisper-tiny/tree/main)
   - [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
   - [face-parse-bisent](https://github.com/zllrunning/face-parsing.PyTorch)
   - [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)
   - [syncnet](https://huggingface.co/ByteDance/LatentSync/tree/main)


Finally, these weights should be organized in `models` as follows:
```
./models/
├── musetalk
│   └── musetalk.json
│   └── pytorch_model.bin
├── musetalkV15
│   └── musetalk.json
│   └── unet.pth
├── syncnet
│   └── latentsync_syncnet.pt
├── dwpose
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── sd-vae-ft-mse
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── whisper
    ├── config.json
    ├── pytorch_model.bin
    └── preprocessor_config.json
    
```
## Quickstart

### Inference
We provide inference scripts for both versions of MuseTalk:

#### MuseTalk 1.5 (Recommended)
```bash
# Run MuseTalk 1.5 inference
sh inference.sh v1.5 normal
```

#### MuseTalk 1.0
```bash
# Run MuseTalk 1.0 inference
sh inference.sh v1.0 normal
```

The inference script supports both MuseTalk 1.5 and 1.0 models:
- For MuseTalk 1.5: Use the command above with the V1.5 model path
- For MuseTalk 1.0: Use the same script but point to the V1.0 model path

The configuration file `configs/inference/test.yaml` contains the inference settings, including:
- `video_path`: Path to the input video, image file, or directory of images
- `audio_path`: Path to the input audio file

Note: For optimal results, we recommend using input videos with 25fps, which is the same fps used during model training. If your video has a lower frame rate, you can use frame interpolation or convert it to 25fps using ffmpeg.

#### Real-time Inference
For real-time inference, use the following command:
```bash
# Run real-time inference
sh inference.sh v1.5 realtime  # For MuseTalk 1.5
# or
sh inference.sh v1.0 realtime  # For MuseTalk 1.0
```

The real-time inference configuration is in `configs/inference/realtime.yaml`, which includes:
- `preparation`: Set to `True` for new avatar preparation
- `video_path`: Path to the input video
- `bbox_shift`: Adjustable parameter for mouth region control
- `audio_clips`: List of audio clips for generation

Important notes for real-time inference:
1. Set `preparation` to `True` when processing a new avatar
2. After preparation, the avatar will generate videos using audio clips from `audio_clips`
3. The generation process can achieve 30fps+ on an NVIDIA Tesla V100
4. Set `preparation` to `False` for generating more videos with the same avatar

For faster generation without saving images, you can use:
```bash
python -m scripts.realtime_inference --inference_config configs/inference/realtime.yaml --skip_save_images
```

## Training

### Data Preparation
To train MuseTalk, you need to prepare your dataset following these steps:

1. **Place your source videos** 

   For example, if you're using the HDTF dataset, place all your video files in `./dataset/HDTF/source`.

2. **Run the preprocessing script**
   ```bash
   python -m scripts.preprocess --config ./configs/training/preprocess.yaml
   ```
   This script will:
   - Extract frames from videos
   - Detect and align faces
   - Generate audio features
   - Create the necessary data structure for training

### Training Process
After data preprocessing, you can start the training process:

1. **First Stage**
   ```bash
   sh train.sh stage1
   ```

2. **Second Stage**
   ```bash
   sh train.sh stage2
   ```

### Configuration Adjustment
Before starting the training, you should adjust the configuration files according to your hardware and requirements:

1. **GPU Configuration** (`configs/training/gpu.yaml`):
   - `gpu_ids`: Specify the GPU IDs you want to use (e.g., "0,1,2,3")
   - `num_processes`: Set this to match the number of GPUs you're using

2. **Stage 1 Configuration** (`configs/training/stage1.yaml`):
   - `data.train_bs`: Adjust batch size based on your GPU memory (default: 32)
   - `data.n_sample_frames`: Number of sampled frames per video (default: 1)

3. **Stage 2 Configuration** (`configs/training/stage2.yaml`):
   - `random_init_unet`: Must be set to `False` to use the model from stage 1
   - `data.train_bs`: Smaller batch size due to high GPU memory cost (default: 2)
   - `data.n_sample_frames`: Higher value for temporal consistency (default: 16)
   - `solver.gradient_accumulation_steps`: Increase to simulate larger batch sizes (default: 8)
  

### GPU Memory Requirements
Based on our testing on a machine with 8 NVIDIA H20 GPUs:

#### Stage 1 Memory Usage
| Batch Size | Gradient Accumulation | Memory per GPU | Recommendation |
|:----------:|:----------------------:|:--------------:|:--------------:|
| 8          | 1                      | ~32GB          |                |
| 16         | 1                      | ~45GB          |                |
| 32         | 1                      | ~74GB          | ✓              |

#### Stage 2 Memory Usage
| Batch Size | Gradient Accumulation | Memory per GPU | Recommendation |
|:----------:|:----------------------:|:--------------:|:--------------:|
| 1          | 8                      | ~54GB          |                |
| 2          | 2                      | ~80GB          |                |
| 2          | 8                      | ~85GB          | ✓              |

<details close>
## TestCases For 1.0
<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="33%">Image</td>
        <td width="33%">MuseV</td>
        <td width="33%">+MuseTalk</td>
  </tr>
  <tr>
    <td>
      <img src=assets/demo/musk/musk.png width="95%">
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/4a4bb2d1-9d14-4ca9-85c8-7f19c39f712e controls preload></video>
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/b2a879c2-e23a-4d39-911d-51f0343218e4 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=assets/demo/yongen/yongen.jpeg width="95%">
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/57ef9dee-a9fd-4dc8-839b-3fbbbf0ff3f4 controls preload></video>
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/94d8dcba-1bcd-4b54-9d1d-8b6fc53228f0 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=assets/demo/sit/sit.jpeg width="95%">
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/5fbab81b-d3f2-4c75-abb5-14c76e51769e controls preload></video>
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/f8100f4a-3df8-4151-8de2-291b09269f66 controls preload></video>
    </td>
  </tr>
   <tr>
    <td>
      <img src=assets/demo/man/man.png width="95%">
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/a6e7d431-5643-4745-9868-8b423a454153 controls preload></video>
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/6ccf7bc7-cb48-42de-85bd-076d5ee8a623 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=assets/demo/monalisa/monalisa.png width="95%">
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/1568f604-a34f-4526-a13a-7d282aa2e773 controls preload></video>
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/a40784fc-a885-4c1f-9b7e-8f87b7caf4e0 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=assets/demo/sun1/sun.png width="95%">
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/37a3a666-7b90-4244-8d3a-058cb0e44107 controls preload></video>
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/172f4ff1-d432-45bd-a5a7-a07dec33a26b controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <img src=assets/demo/sun2/sun.png width="95%">
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/37a3a666-7b90-4244-8d3a-058cb0e44107 controls preload></video>
    </td>
    <td >
      <video src=https://github.com/TMElyralab/MuseTalk/assets/163980830/85a6873d-a028-4cce-af2b-6c59a1f2971d controls preload></video>
    </td>
  </tr>
</table >

#### Use of bbox_shift to have adjustable results(For 1.0)
:mag_right: We have found that upper-bound of the mask has an important impact on mouth openness. Thus, to control the mask region, we suggest using the `bbox_shift` parameter. Positive values (moving towards the lower half) increase mouth openness, while negative values (moving towards the upper half) decrease mouth openness.

You can start by running with the default configuration to obtain the adjustable value range, and then re-run the script within this range. 

For example, in the case of `Xinying Sun`, after running the default configuration, it shows that the adjustable value rage is [-9, 9]. Then, to decrease the mouth openness, we set the value to be `-7`. 
```
python -m scripts.inference --inference_config configs/inference/test.yaml --bbox_shift -7 
```
:pushpin: More technical details can be found in [bbox_shift](assets/BBOX_SHIFT.md).


#### Combining MuseV and MuseTalk

As a complete solution to virtual human generation, you are suggested to first apply [MuseV](https://github.com/TMElyralab/MuseV) to generate a video (text-to-video, image-to-video or pose-to-video) by referring [this](https://github.com/TMElyralab/MuseV?tab=readme-ov-file#text2video). Frame interpolation is suggested to increase frame rate. Then, you can use `MuseTalk` to generate a lip-sync video by referring [this](https://github.com/TMElyralab/MuseTalk?tab=readme-ov-file#inference).

# Acknowledgement
1. We thank open-source components like [whisper](https://github.com/openai/whisper), [dwpose](https://github.com/IDEA-Research/DWPose), [face-alignment](https://github.com/1adrianb/face-alignment), [face-parsing](https://github.com/zllrunning/face-parsing.PyTorch), [S3FD](https://github.com/yxlijun/S3FD.pytorch) and [LatentSync](https://huggingface.co/ByteDance/LatentSync/tree/main). 
1. MuseTalk has referred much to [diffusers](https://github.com/huggingface/diffusers) and [isaacOnline/whisper](https://github.com/isaacOnline/whisper/tree/extract-embeddings).
1. MuseTalk has been built on [HDTF](https://github.com/MRzzm/HDTF) datasets.

Thanks for open-sourcing!

# Limitations
- Resolution: Though MuseTalk uses a face region size of 256 x 256, which make it better than other open-source methods, it has not yet reached the theoretical resolution bound. We will continue to deal with this problem.  
If you need higher resolution, you could apply super resolution models such as [GFPGAN](https://github.com/TencentARC/GFPGAN) in combination with MuseTalk.

- Identity preservation: Some details of the original face are not well preserved, such as mustache, lip shape and color.

- Jitter: There exists some jitter as the current pipeline adopts single-frame generation.

# Citation
```bib
@article{musetalk,
  title={MuseTalk: Real-Time High-Fidelity Video Dubbing via Spatio-Temporal Sampling},
  author={Zhang, Yue and Zhong, Zhizhou and Liu, Minhao and Chen, Zhaokang and Wu, Bin and Zeng, Yubin and Zhan, Chao and He, Yingjie and Huang, Junxin and Zhou, Wenjiang},
  journal={arxiv},
  year={2025}
}
```
# Disclaimer/License
1. `code`: The code of MuseTalk is released under the MIT License. There is no limitation for both academic and commercial usage.
1. `model`: The trained model are available for any purpose, even commercially.
1. `other opensource model`: Other open-source models used must comply with their license, such as `whisper`, `ft-mse-vae`, `dwpose`, `S3FD`, etc..
1. The testdata are collected from internet, which are available for non-commercial research purposes only.
1. `AIGC`: This project strives to impact the domain of AI-driven video generation positively. Users are granted the freedom to create videos using this tool, but they are expected to comply with local laws and utilize it responsibly. The developers do not assume any responsibility for potential misuse by users.
