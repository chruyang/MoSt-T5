#!/bin/bash

# 🚀 修复后的单卡调试版脚本
NUM_GPUS=8
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NCCL_P2P_DISABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1

echo "🧪 正在启动 MoSt-T5 多卡训练..."

# 重点：确保每一行末尾的 \ 后面没有任何字符（包括空格）
torchrun --nproc_per_node=$NUM_GPUS train2.py \
    --model_name_or_path "./MoSt-T5-Phase1-Final/checkpoint-100000" \
    --tokenizer_name "google/t5-v1_1-base" \
    --vocab_file "asset/mol_vocabs/vocab_phase2_25k.txt" \
    --train_file "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_pubchem_final.lmdb" \
    --c4_file "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/c4_pretrain.lmdb" \
    --text_weight_path "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_text_weights.json" \
    --output_dir "checkpoints/MoSt-T5-Phase2-Final" \
    --do_train \
    --max_steps 30000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --weight_decay 0.05 \
    --warmup_steps 3000 \
    --lr_scheduler_type cosine \
    --logging_strategy steps \
    --logging_steps 50 \
    --save_strategy steps \
    --save_steps 3000 \
    --save_total_limit 3 \
    --max_seq_length 768 \
    --e3fp_num_levels 4 \
    --e3fp_vocab_size 4096 \
    --bf16 True \
    --tf32 True \
    --dataloader_num_workers 16 \
    --dataloader_prefetch_factor 2 \
    --dataloader_pin_memory True \
    --ddp_find_unused_parameters True \
    --optim adamw_torch_fused \
    --save_safetensors False \
    --report_to tensorboard \
    --optim adamw_torch_fused \
