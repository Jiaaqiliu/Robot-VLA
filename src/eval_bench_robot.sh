#!/bin/bash
# run_models.sh

model_paths=(
    "/home/jqliu/Myprojects/Video-R1/src/r1-v/log/Qwen2.5-VL-3B-Video-3B-cot-sft"
)

file_names=(
    "Qwen2.5-VL-3B-Video-3B-cot-sft"
)

export DECORD_EOF_RETRY_MAX=20480
export VIDEO_BASE_PATH="/home/jqliu/Myprojects/RoboBrain/ShareRobot/planning/Video_data/planning_task"


for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    file_name="${file_names[$i]}"
    CUDA_VISIBLE_DEVICES=0 python ./src/eval_bench_robot.py --model_path "$model" --file_name "$file_name"
done
