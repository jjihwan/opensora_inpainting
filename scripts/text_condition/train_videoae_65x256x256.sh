export WANDB_KEY="f2b81eab0686ae96a304a95211b4afa7d5722925"
export ENTITY=""
export PROJECT="t2v-f65-256-img4-videovae488-bf16-ckpt-xformers-bs4-lr2e-5-t5"
accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_t2v.py \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "./pretrained/Open-Sora-Plan-v1.0.0/vae" \
    --data_path "./dataset/mixkit/sample.json" \
    --video_folder "./dataset/mixkit/sky" \
    --sample_rate 1 \
    --num_frames 65 \
    --max_image_size 256 \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size=1 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=100 \
    --max_train_steps=1000000 \
    --learning_rate=2e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=500 \
    --output_dir="t2v-f65-256-img4-videovae488-bf16-ckpt-xformers-bs4-lr2e-5-t5" \
    --allow_tf32 \
    --pretrained "./pretrained/Open-Sora-Plan-v1.0.0/17x256x256/diffusion_pytorch_model.safetensors" \
    --use_deepspeed \
    --model_max_length 300 \
    --use_image_num 4 \
    --use_img_from_vid
