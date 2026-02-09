#!/bin/bash

# ================== 1. 环境配置 ==================
export CUDA_VISIBLE_DEVICES=0

# ================== 2. 路径配置 ==================
# ⚠️ 请确保以下路径真实存在
TRAIN_FILE="dataset/3d-pubchem.lmdb"
# 如果您没有准备好 valid.lmdb，请暂时注释掉 VALID_FILE 相关行，并去掉 --validation_file 参数
VALID_FILE="dataset/3d-pubchem-valid.lmdb"
VOCAB_PATH="asset/mol_vocabs/frag_merged.txt"
OUTPUT_DIR="output/mol2text_a100_final"
MODEL_NAME="google/t5-v1_1-base"

# ================== 3. 训练参数 (A100 80G配置) ==================
# 如果是 A100 40G，建议 BATCH_SIZE=32
BATCH_SIZE=64
GRAD_ACCUM=1
LEARNING_RATE=5e-4
EPOCHS=10

echo "🚀 Starting MoSt-T5 Training..."

python train.py \
    --model_name_or_path "$MODEL_NAME" \
    --train_file "$TRAIN_FILE" \
    --validation_file "$VALID_FILE" \
    --vocab_path "$VOCAB_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 20 \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model bleu \
    --greater_is_better True \
    --overwrite_output_dir \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$EPOCHS" \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "linear" \
    --dataloader_num_workers 8 \
    --bf16 True \
    --tf32 True \
    --report_to none \
    --remove_unused_columns False \
    --task_type mol2text \
    --e3fp_num_levels 4 \
    --predict_with_generate True \
    --generation_max_length 512 \
    --generation_num_beams 4

# ================== 6. 测试评估 (Test Evaluation) ==================
TEST_FILE="dataset/3d-pubchem-test.lmdb"

if [ -f "$TEST_FILE" ]; then
    echo "🧪 Starting Testing on held-out Test Set..."
    python train.py \
        --model_name_or_path "$OUTPUT_DIR" \
        --train_file "$TRAIN_FILE" \
        --validation_file "$TEST_FILE" \  # 这里把 Test 文件传给 validation_file 参数来进行推理
        --vocab_path "$VOCAB_PATH" \
        --output_dir "${OUTPUT_DIR}/test_results" \
        --do_eval \
        --per_device_eval_batch_size 16 \
        --dataloader_num_workers 4 \
        --bf16 True \
        --report_to none \
        --task_type mol2text \
        --e3fp_num_levels 4 \
        --predict_with_generate True \
        --generation_max_length 512 \
        --generation_num_beams 4
fi