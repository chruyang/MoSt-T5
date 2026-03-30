#!/bin/bash

export OMP_NUM_THREADS=1
export NCCL_IB_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0

NUM_GPUS=$(nvidia-smi -L | wc -l)

# 🚀 Phase 1: MoSt-T5 稳定版预训练脚本
torchrun --nproc_per_node=$NUM_GPUS train1.py \
    --model_name_or_path google/t5-v1_1-base \
    --tokenizer_name google/t5-v1_1-base \
    --vocab_file asset/mol_vocabs/vocab_20k.txt \
    --train_file /root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchemqc/pubchemqc_final.lmdb \
    --output_dir checkpoints/MoSt-T5-Phase1-Final \
    --do_train \
    --max_steps 100000 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --weight_decay 0.0 \
    --warmup_steps 10000 \
    --lr_scheduler_type cosine \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 5 \
    --max_seq_length 512 \
    --e3fp_num_levels 4 \
    --e3fp_vocab_size 4096 \
    --bf16 True \
    --tf32 True \
    --dataloader_num_workers 16 \
    --dataloader_prefetch_factor 16 \
    --dataloader_pin_memory True \
    --ddp_find_unused_parameters True \
    --save_safetensors False \
    --report_to tensorboard