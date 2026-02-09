#!/bin/bash

# ================== 1. 环境配置 ==================
export CUDA_VISIBLE_DEVICES=0
# 减少显存碎片的魔法配置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ================== 2. 路径配置 ==================
DATA_DIR="dataset"
TRAIN_FILE="${DATA_DIR}/3d-pubchem-train.lmdb"
VALID_FILE="${DATA_DIR}/3d-pubchem-valid.lmdb"
TEST_FILE="${DATA_DIR}/3d-pubchem-test.lmdb"
VOCAB_PATH="asset/mol_vocabs/frag_merged.txt"

OUTPUT_DIR="output/mol2text_3090_final"
MODEL_NAME="google/t5-v1_1-base"

# ================== 3. 训练参数 (RTX 3090 24GB 安全版) ==================
# ⚠️ 针对超大词表(160k+)的显存优化
BATCH_SIZE=4        # 从 16 降到 4
GRAD_ACCUM=16       # 从 4 升到 16，保持总 Batch=64 不变
LEARNING_RATE=5e-4
EPOCHS=10

echo "🚀 Starting MoSt-T5 Training (Safe Mode)..."

# ================== 4. 训练命令 ==================
CMD=(
    python train.py
    --model_name_or_path "$MODEL_NAME"
    --output_dir "$OUTPUT_DIR"

    # --- 核心策略 ---
    --do_train
    --do_eval
    --evaluation_strategy steps
    --save_strategy steps
    --load_best_model_at_end True
    --metric_for_best_model bleu
    --greater_is_better True

    # --- 数据路径 ---
    --train_file "$TRAIN_FILE"
    --validation_file "$VALID_FILE"
    --vocab_path "$VOCAB_PATH"

    # --- 步数与保存 ---
    --eval_steps 500
    --save_steps 500
    --logging_steps 50
    --save_total_limit 2
    --overwrite_output_dir

    # --- 显存优化参数 ---
    --per_device_train_batch_size "$BATCH_SIZE"
    --per_device_eval_batch_size 8  # 验证集也稍微调小一点，防止验证时OOM
    --gradient_accumulation_steps "$GRAD_ACCUM"

    # --- 学习率 ---
    --learning_rate "$LEARNING_RATE"
    --num_train_epochs "$EPOCHS"
    --warmup_ratio 0.05
    --lr_scheduler_type linear

    # --- 硬件加速 ---
    --dataloader_num_workers 4
    --bf16 True
    --tf32 True
    --report_to none
    --remove_unused_columns False

    # --- 模型特定 ---
    --task_type mol2text
    --e3fp_num_levels 4
    --predict_with_generate True
    --generation_max_length 512
    --generation_num_beams 4
)

# 执行训练
"${CMD[@]}"

if [ $? -ne 0 ]; then
    echo "❌ Training Failed! Please check the error log."
    exit 1
fi

# ================== 5. 测试评估 ==================
echo "✅ Training Success! Starting Testing..."

# 检查 Test 文件是否存在
if [ -f "$TEST_FILE" ]; then
    python train.py \
        --model_name_or_path "$OUTPUT_DIR" \
        --train_file "$TRAIN_FILE" \
        --validation_file "$TEST_FILE" \
        --vocab_path "$VOCAB_PATH" \
        --output_dir "${OUTPUT_DIR}/test_results" \
        --do_eval \
        --per_device_eval_batch_size 16 \
        --dataloader_num_workers 4 \
        --bf16 True \
        --task_type mol2text \
        --e3fp_num_levels 4 \
        --predict_with_generate True \
        --generation_max_length 512 \
        --generation_num_beams 4
else
    echo "⚠️ Test file not found at $TEST_FILE, skipping testing."
fi