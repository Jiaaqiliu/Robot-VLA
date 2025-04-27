#!/bin/bash

# Change to the script directory
cd src/r1-v

# Source the local configuration file
if [ -f "../scripts/config.local.sh" ]; then
    source "../scripts/config.local.sh"
else
    echo "Error: config.local.sh not found. Please copy config.template.sh to config.local.sh and modify it according to your environment."
    exit 1
fi

# Set wandb environment variables
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_ENTITY="${WANDB_ENTITY}"
export WANDB_PROJECT="${WANDB_PROJECT}"

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
# Add base path for video data
export VIDEO_BASE_PATH="${VIDEO_BASE_PATH}"
# Dataset path
export DATASET_PATH="${DATASET_PATH}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    src/open_r1/grpo.py \
    --output_dir "./log/Qwen2.5-VL-3B-GRPO" \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "${DATASET_PATH}" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --temporal true \
    --len_control true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-3B-GRPO-Robot \
    --save_steps 100 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8 \
    --report_to wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" 