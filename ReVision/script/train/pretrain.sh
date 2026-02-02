#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0

PYTHON_CMD="/mnt/public/data/h200/.miniconda/envs/bunny/bin/python"
MODEL_TYPE=llama3-8b
Data_TYPE=bunny-2M
OUTPUT_DIR=bunny-$MODEL_TYPE-pretrain

mkdir -p /mnt/public/data/h200/victor/xiaomin/pretrain9999/checkpoints-pretrain-$Data_TYPE/$OUTPUT_DIR

$PYTHON_CMD -m deepspeed.launcher.runner bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero2.json \
    --model_name_or_path /mnt/public/data/h200/victor/xiaomin/model/llama3 \
    --model_type $MODEL_TYPE \
    --version plain \
    --data_path /mnt/public/data/h200/victor/xiaomin/datasets/bunny_data/filtered_output.json \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --image_aspect_ratio square \
    --bf16 True \
    --output_dir /mnt/public/data/h200/victor/xiaomin/pretrain9999/checkpoints-pretrain-$Data_TYPE/$OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
