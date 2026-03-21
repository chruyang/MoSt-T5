import logging
import os
import sys
import torch
import numpy as np
import lmdb
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoConfig
)

from tokenization.text_tokenizer import TextTokenizer
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
# 🚀 引入 Phase 2 专属的 Dataset 和 Collator (我们在 dataset.py 中新增的)
from dataset.dataset import MoStPhase2Dataset, GSMATPhase2Collator
from model.configuration import MoStT5Config
from model.modeling import MoStT5ForConditionalGeneration
from arguments import ModelArguments, DataArguments

logger = logging.getLogger(__name__)


def main():
    # 1. 注册模型并解析参数
    AutoConfig.register("most-t5", MoStT5Config)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ==========================================
    # 🚨 极其关键：禁止 Trainer 自动删除 3D 特征列
    # ==========================================
    training_args.remove_unused_columns = False

    # 配置标准日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    set_seed(training_args.seed)

    # ==========================================
    # 🚀 2. 加载 Phase 1 预训练权重
    # ==========================================
    logger.info(f"🚀 Loading Phase 1 Checkpoint from: {model_args.model_name_or_path}")
    model = MoStT5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

    # ==========================================
    # 📦 3. 初始化三大分词器
    # ==========================================
    logger.info("Initializing tokenizers...")
    motif_tokenizer = MotifTokenizer(vocab_file="asset/mol_vocabs/vocab_20k.txt")
    text_tokenizer = TextTokenizer(model_name="google/t5-v1_1-base")
    e3fp_tokenizer = E3FPTokenizer(fp_level=4, fp_bits=4096)

    # ==========================================
    # 🌍 4. 构建 Phase 2 混合多任务数据集和 Collator
    # ==========================================
    logger.info("📦 Building Phase 2 Multi-modal Dataset...")

    # 加载 LMDB 数据集
    try:
        # data_args.data_path 应该是包含图文对的 LMDB 文件夹路径
        lmdb_env = lmdb.open(data_args.data_path, readonly=True, lock=False)
        logger.info(f"✅ LMDB Environment loaded successfully from {data_args.data_path}")
    except Exception as e:
        logger.warning(f"⚠️ LMDB open failed at {data_args.data_path}. Error: {e}")
        lmdb_env = None

    # 调用我们在 dataset.py 中配置的 60/25/15 黄金配比 Dataset
    max_len = data_args.max_seq_length if hasattr(data_args, 'max_seq_length') else 1024
    train_dataset = MoStPhase2Dataset(
        lmdb_env=lmdb_env,
        motif_tokenizer=motif_tokenizer,
        text_tokenizer=text_tokenizer,
        e3fp_tokenizer=e3fp_tokenizer,
        max_seq_len=max_len
    )

    data_collator = GSMATPhase2Collator(
        motif_pad_id=motif_tokenizer.pad_id,
        text_pad_id=text_tokenizer.pad_id,
        e3fp_pad_id=-1
    )

    # 5. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # ==========================================
    # 🔍 6. 权重健康检查 (照妖镜逻辑，防 NaN 崩溃)
    # ==========================================
    logger.info("=" * 40)
    logger.info("🔍 WEIGHT SANITY CHECK (Before Phase 2 Training)")
    if hasattr(model, "shared"):
        std = model.shared.weight.std().item()
        logger.info(f"  -> Shared Embeddings STD: {std:.6f} (Target: ~0.002 or match Base Model)")
        if std > 10.0:
            logger.warning("⚠️ WARNING: Shared Embeddings are exceptionally large! Might cause NaN in Text MLM.")
    if hasattr(model, "lm_head"):
        std = model.lm_head.weight.std().item()
        logger.info(f"  -> LM Head Weights STD:   {std:.6f}")
    if hasattr(model.encoder, "gsm_embeddings"):
        std = model.encoder.gsm_embeddings.word_embeddings.weight.std().item()
        logger.info(f"  -> Encoder Embeddings STD: {std:.6f}")
    logger.info("=" * 40)

    # ==========================================
    # 🔥 7. 启动 Phase 2 训练
    # ==========================================
    logger.info("🔥 Starting Phase 2: Generative Modality Alignment...")
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # 保存最终的 Phase 2 模型大一统权重


if __name__ == "__main__":
    # 保持纯净的通信环境，防止双卡 3090 NCCL 死锁
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    main()