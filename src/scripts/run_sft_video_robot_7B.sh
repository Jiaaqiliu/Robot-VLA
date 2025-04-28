#!/bin/bash

# Change to the script directory
# cd "$(dirname "$0")"
# cd ../r1-v
cd src/r1-v

# Source the local configuration file
if [ -f "../scripts/config.local.sh" ]; then
    source "../scripts/config.local.sh"
else
    echo "Error: config.local.sh not found. Please copy config.template.sh to config.local.sh and modify it according to your environment."
    exit 1
fi

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
# Add base path for video data
export VIDEO_BASE_PATH="${VIDEO_BASE_PATH}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/sft_video.py \
    --output_dir "./log/Qwen2.5-VL-7B-Video-7B-cot-sft" \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "${DATASET_PATH}" \
    --deepspeed local_scripts/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-7B-Video-7B-cot-sft \
    --save_steps 1000 \
    --max_grad_norm 5 \
    --save_only_model true \