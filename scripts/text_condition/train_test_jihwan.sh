export WANDB_KEY="f2b81eab0686ae96a304a95211b4afa7d5722925"
export ENTITY=""
export PROJECT="inpainting-finetune-test"
accelerate launch \
    --config_file scripts/accelerate_configs/default_config_jihwan.yaml \
    opensora/train/train_inpainting_val.py \
    --model LatteT2VInpainting-XL/122 \
    --version 17x256x256 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "./pretrained/Open-Sora-Plan-v1.0.0/vae" \
    --data_path "./dataset/mixkit/sample.json" \
    --video_folder "./dataset/mixkit/sky" \
    --sample_rate 1 \
    --num_frames 17 \
    --max_image_size 256 \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size 4 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps 64 \
    --max_train_steps 100000 \
    --learning_rate 2e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps 0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps 1000 \
    --output_dir="lattet2vinpainting_with_val" \
    --allow_tf32 \
    --pretrained "./pretrained/Open-Sora-Plan-v1.0.0/17x256x256/diffusion_pytorch_model.safetensors" \
    --use_deepspeed \
    --model_max_length 300 \
    --use_image_num 4 \
    --use_img_from_vid \
    --validation_sample_method 'PNDM' \
    --enable_tracker \
    --cache_dir "./cache_dir"