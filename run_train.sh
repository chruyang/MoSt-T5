#!/bin/bash

# ================== 1. 环境配置 ==================
export CUDA_VISIBLE_DEVICES=0

# ================== 2. 路径配置 ==================
DATA_DIR="dataset"
TRAIN_FILE="${DATA_DIR}/3d-pubchem.lmdb"
VOCAB_PATH="asset/mol_vocabs/frag_merged.txt"
OUTPUT_DIR="output/mol2text_a100_v1"
MODEL_NAME="google/t5-v1_1-base"

# ================== 3. A100 性能调优 ==================
# --- 方案 A: A100 40GB ---
# BATCH_SIZE=32
# GRAD_ACCUM=1   # 等效 Batch Size = 32

# --- 方案 B: A100 80GB (推荐) ---
BATCH_SIZE=64
GRAD_ACCUM=1     # 等效 Batch Size = 64

# 如果显存溢出 (OOM)，请回退到方案 A，或者开启梯度检查点 (Gradient Checkpointing)
# --gradient_checkpointing True

LEARNING_RATE=5e-4
EPOCHS=10
NUM_WORKERS=8    # A100 需要更快的数据喂入

echo "🚀 Starting MoSt-T5 Training on A100..."
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"

# ================== 4. 启动命令 ==================
python train.py \
    --model_name_or_path "$MODEL_NAME" \
    --train_file "$TRAIN_FILE" \
    --vocab_path "$VOCAB_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --do_train \
    --overwrite_output_dir \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$EPOCHS" \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 2 \
    --dataloader_num_workers "$NUM_WORKERS" \
    --bf16 True \
    --tf32 True \
    --report_to none \
    --remove_unused_columns False \
    --task_type mol2text \
    --e3fp_num_levels 4 \
    --e3fp_vocab_size 4096 \
    --fusion_type residual \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine"
    # --gradient_checkpointing True # 如果 OOM，取消注释此行