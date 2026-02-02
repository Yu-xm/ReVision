#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 网络接口配置 (保持与 Pretrain 一致)
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0

if [ -z "$1" ]; then
    echo "错误: 请提供 Data_TYPE 参数。"
    echo "用法: bash $0 <Data_TYPE>"
    echo "示例: bash $0 densefusion"
    exit 1
fi

# Python 解释器路径
PYTHON_CMD="/mnt/public/data/h200/.miniconda/envs/bunny/bin/python"

MODEL_TYPE=llama3-8b
Data_TYPE=$1

# 定义 Pretrain 文件夹名 (用于加载 adapter) 和 SFT 输出文件夹名
PRETRAIN_DIR=bunny-$MODEL_TYPE-pretrain
OUTPUT_DIR=bunny-$MODEL_TYPE-sft

# 创建 SFT 输出目录
mkdir -p /mnt/public/data/h200/victor/xiaomin/model/finetune_417/checkpoints-sft-$Data_TYPE-417K/$OUTPUT_DIR
# 启动命令
$PYTHON_CMD -m deepspeed.launcher.runner bunny/train/train.py \
    --deepspeed ./script/deepspeed/zero2.json \
    --model_name_or_path /mnt/public/data/h200/victor/xiaomin/model/llama3 \
    --model_type $MODEL_TYPE \
    --version llama \
    --data_path /mnt/public/data/h200/victor/xiaomin/datasets/internvl_dataset/sft_435K.json \
    --pretrain_mm_mlp_adapter /mnt/public/data/h200/victor/xiaomin/model/pretrained/checkpoints-pretrain-$Data_TYPE/$PRETRAIN_DIR/checkpoint-3906/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir /mnt/public/data/h200/victor/xiaomin/model/finetune_417/checkpoints-sft-$Data_TYPE-417K/$OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
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
