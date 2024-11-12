import argparse
import itertools
import math
import os
import random
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

import sys
sys.path.append("./")

from DataLoader import Dataset
from utils.utils import preprocess_img_tensor
from torch.utils import data as data_utils
from utils.model_utils import validation,PositionalEncoding
import time
import pandas as pd
from PIL import Image


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--unet_config_file",
        type=str,
        default=None,
        required=True,
        help="the configuration of unet file.",
    )
    parser.add_argument(
        "--reconstruction",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
   
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )


    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--testing_speed", action="store_true", help="Whether to caculate the running time")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--train_json", type=str, default="train.json", help="The json file containing train image folders")
    parser.add_argument("--val_json", type=str, default="test.json", help="The json file containing validation image folders")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint and are suitable for resuming training"
            " using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Conduct validation every X updates."
        ),
    )
    parser.add_argument(
        "--val_out_dir",
        type=str,
        default = '',
        help=(
            "Conduct validation every X updates."
        ),
    )
    
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--use_audio_length_left",
        type=int,
        default=1,
        help="number of audio length (left).",
    )
    parser.add_argument(
        "--use_audio_length_right",
        type=int,
        default=1,
        help="number of audio length (right)",
    )
    parser.add_argument(
        "--whisper_model_type",
        type=str,
        default="landmark_nearest",
        choices=["tiny","largeV2"],
        help="Determine whisper feature type",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    return args

def print_model_dtypes(model, model_name):
    for name, param in model.named_parameters():
        if(param.dtype!=torch.float32):
            print(f"{name}: {param.dtype}")


def main():
    args = parse_args()
    args.output_dir = f"output/{args.output_dir}"
    args.val_out_dir = f"val/{args.val_out_dir}"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.val_out_dir, exist_ok=True)
    
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae')
    
    
    logging_dir = Path(args.output_dir, args.logging_dir)

    project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
#         set_seed(args.seed)
        set_seed(seed + accelerator.process_index)



    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id


    # Load models and create wrapper for stable diffusion
    with open(args.unet_config_file, 'r') as f:
        unet_config = json.load(f)
        
    #text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    
    # Todo:
    print("Loading AutoencoderKL")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae')
    vae_fp32 = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    print("Loading UNet2DConditionModel")
    unet = UNet2DConditionModel(**unet_config)
    
    if args.whisper_model_type == "tiny":
        pe = PositionalEncoding(d_model=384)
    elif args.whisper_model_type == "largeV2":
        pe = PositionalEncoding(d_model=1280)
    else:
        print(f"not support whisper_model_type {args.whisper_model_type}")
        
    print("Loading models done...")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters()))
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    print("loading train_dataset ...")
    train_dataset = Dataset(args.data_root, 
                            args.train_json, 
                            use_audio_length_left=args.use_audio_length_left,
                            use_audio_length_right=args.use_audio_length_right,
                            whisper_model_type=args.whisper_model_type
                            )
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=8)
    print("loading val_dataset ...")
    val_dataset = Dataset(args.data_root, 
                          args.val_json, 
                          use_audio_length_left=args.use_audio_length_left,
                          use_audio_length_right=args.use_audio_length_right,
                          whisper_model_type=args.whisper_model_type
                         )
    val_data_loader = data_utils.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=8)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_data_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

  
    
    unet, optimizer, train_data_loader, val_data_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_data_loader, val_data_loader,lr_scheduler
    )
    
    vae.requires_grad_(False)
    vae_fp32.requires_grad_(False)

    weight_dtype = torch.float32
    # weight_dtype = torch.float16
    vae_fp32.to(accelerator.device, dtype=weight_dtype)
    vae_fp32.encoder = None
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16    
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.decoder = None
    pe.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_data_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print(f"  Num batches each epoch = {len(train_data_loader)}")

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_data_loader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        # path="../models/pytorch_model.bin"
        #TODO change path
        # path=None
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
            
        
            
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    # caluate the elapsed time
    elapsed_time = []
    start = time.time()


   
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, (ref_image, image, masked_image, masks, audio_feature) in enumerate(train_data_loader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            dataloader_time = time.time() - start
            start = time.time()    
            masks = masks.unsqueeze(1).unsqueeze(1).to(vae.device)
            # """
            # print("=============epoch:{0}=step:{1}=====".format(epoch,step))
            # print("ref_image: ",ref_image.shape)
            # print("masks: ", masks.shape)
            # print("masked_image: ", masked_image.shape)
            # print("audio feature: ", audio_feature.shape)
            # print("image: ", image.shape)
            # """
            ref_image = preprocess_img_tensor(ref_image).to(vae.device)
            image = preprocess_img_tensor(image).to(vae.device)
            masked_image = preprocess_img_tensor(masked_image).to(vae.device)
            
            img_process_time = time.time() - start
            start = time.time()
            with accelerator.accumulate(unet):
                vae = vae.half()
                # Convert images to latent space 
                latents = vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample() # init image
                latents = latents * vae.config.scaling_factor

                # Convert masked images to latent space
                masked_latents = vae.encode(
                    masked_image.reshape(image.shape).to(dtype=weight_dtype)  # masked image
                ).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor
                
                # Convert ref images to latent space
                ref_latents = vae.encode(
                    ref_image.reshape(image.shape).to(dtype=weight_dtype)  # ref image
                ).latent_dist.sample()
                ref_latents = ref_latents * vae.config.scaling_factor
                
                vae_time = time.time() - start
                start = time.time()

                mask = torch.stack(
                    [
                        torch.nn.functional.interpolate(mask, size=(mask.shape[-1] // 8, mask.shape[-1] // 8))
                        for mask in masks
                    ]
                )
                mask = mask.reshape(-1, 1, mask.shape[-1], mask.shape[-1])


                bsz = latents.shape[0]
                # fix timestep for each image
                timesteps = torch.tensor([0], device=latents.device)
                # concatenate the latents with the mask and the masked latents
                """
                print("=============vae latents=====".format(epoch,step))
                print("ref_latents: ",ref_latents.shape)
                print("mask: ", mask.shape)
                print("masked_latents: ", masked_latents.shape)
                """

                if unet_config['in_channels'] == 9:
                    latent_model_input = torch.cat([mask, masked_latents, ref_latents], dim=1)
                else:
                    latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)

                audio_feature = audio_feature.to(dtype=weight_dtype)
                audio_feature = pe(audio_feature)
                
                # Predict the noise residual
                image_pred = unet(latent_model_input, 
                                  timesteps, 
                                  encoder_hidden_states=audio_feature).sample

                if args.reconstruction: # decode the image from the predicted latents
                    image_pred_img = (1 / vae_fp32.config.scaling_factor) * image_pred
                    image_pred_img = vae_fp32.decode(image_pred_img).sample

                    # Mask the top half of the image and calculate the loss only for the lower half of the image.
                    image_pred_img = image_pred_img[:, :, image_pred_img.shape[2]//2:, :]
                    image = image[:, :, image.shape[2]//2:, :]    
                    loss_lip = F.l1_loss(image_pred_img.float(), image.float(), reduction="mean") # the loss of the decoded images
                    loss_latents = F.l1_loss(image_pred.float(), latents.float(), reduction="mean") # the loss of the latents
               
                    loss = 2.0*loss_lip + loss_latents # add some weight to balance the loss
                
                else:
                    loss = F.mse_loss(image_pred.float(), latents.float(), reduction="mean")
#                   

                unet_elapsed_time = time.time() - start
                start = time.time()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm) 
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                backward_elapsed_time = time.time() - start
                start = time.time()
                
                if args.testing_speed is True and accelerator.is_main_process:
                    elapsed_time.append(
                        [dataloader_time, unet_elapsed_time, vae_time, backward_elapsed_time,img_process_time]
                    )
                

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        

                if global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        logger.info(
                            f"Running validation... epoch={epoch}, global_step={global_step}"
                        )
                        print("===========start validation==========")
                        # Use the helper function to check the data types for each model
                        vae_new = vae.float()
                        print_model_dtypes(accelerator.unwrap_model(vae_new), "VAE")
                        print_model_dtypes(accelerator.unwrap_model(vae_fp32), "VAE_FP32")
                        print_model_dtypes(accelerator.unwrap_model(unet), "UNET")

                        print(f"weight_dtype: {weight_dtype}")
                        print(f"epoch type: {type(epoch)}")
                        print(f"global_step type: {type(global_step)}")
                        validation(
                            # vae=accelerator.unwrap_model(vae),
                            vae=accelerator.unwrap_model(vae_new),
                            vae_fp32=accelerator.unwrap_model(vae_fp32),
                            unet=accelerator.unwrap_model(unet),
                            unet_config=unet_config,
                            # weight_dtype=weight_dtype,
                            weight_dtype=torch.float32,
                            epoch=epoch,
                            global_step=global_step,
                            val_data_loader=val_data_loader,
                            output_dir = args.val_out_dir,
                            whisper_model_type = args.whisper_model_type
                        )
                        logger.info(f"Saved samples to images/val")
                    start = time.time()                    

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0],
                   "unet": unet_elapsed_time,
                    "backward": backward_elapsed_time,
                    "data": dataloader_time,
                    "img_process":img_process_time,
                    "vae":vae_time
                   }
            progress_bar.set_postfix(**logs)
#             accelerator.log(logs, step=global_step)

            accelerator.log(
                {
                    "loss/step_loss": logs["loss"],
                    "parameter/lr": logs["lr"],
                    "time/unet_forward_time": unet_elapsed_time,
                    "time/unet_backward_time": backward_elapsed_time,
                    "time/data_time": dataloader_time,
                    "time/img_process_time":img_process_time,
                    "time/vae_time": vae_time
                },
                step=global_step,
            )

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
    accelerator.end_training()



if __name__ == "__main__":
    main()
