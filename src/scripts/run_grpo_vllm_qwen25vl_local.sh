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

export DEBUG_MODE="true"
export LOG_PATH="./vllm_run.txt"
# Add base path for video data
export VIDEO_BASE_PATH="${VIDEO_BASE_PATH}"

# Set output directory
OUTPUT_DIR="./log/Qwen2.5-VL-7B-Video-GRPO"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Set run name and deepspeed config
RUN_NAME="Qwen2.5-VL-7B-Video-GRPO"
DS_CONFIG="local_scripts/zero3.json"

# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and --nproc_per_node="4"
CUDA_VISIBLE_DEVICES="0,1,2,3,4" torchrun \
    --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --use_vllm true \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "${DATASET_PATH}" \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --min_pixels 3136 \
    --max_pixels 501760 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --save_only_model false \
    --temporal true \
    --len_control true \
    --report_to wandb \
    --beta 0.04 \
    --max_grad_norm 5 \
    --temperature 1.0 \
    --num_generations 2 \
    --vllm_device "cuda:4" \
    --vllm_gpu_memory_utilization 0.7 \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt" 