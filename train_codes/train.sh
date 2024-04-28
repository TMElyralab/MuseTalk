export VAE_MODEL="./sd-vae-ft-mse/"
export DATASET="..."
export UNET_CONFIG="./musetalk.json"

accelerate launch --multi_gpu train.py \
--mixed_precision="fp16" \
--unet_config_file=$UNET_CONFIG \
--pretrained_model_name_or_path=$VAE_MODEL \
--data_root=$DATASET \
--train_batch_size=8 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--max_train_steps=200000 \
--learning_rate=5e-05 \
--max_grad_norm=1 \
--lr_scheduler="cosine" \
--lr_warmup_steps=0 \
--output_dir="..." \
--val_out_dir='...' \
--testing_speed \
--checkpointing_steps=1000 \
--validation_steps=1000 \
--reconstruction \
--resume_from_checkpoint="latest" \
--use_audio_length_left=2 \
--use_audio_length_right=2 \
--whisper_model_type="tiny" \
