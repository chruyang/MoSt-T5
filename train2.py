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
# 🚀 引入大一统 Dataset 和 Phase 2 专属 Collator
from dataset.dataset import GSMATDataset, GSMATPhase2Collator
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

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    set_seed(training_args.seed)

    # ==========================================
    # 📦 2. 初始化三大分词器 (强制使用 Phase 2 专属词表)
    # ==========================================
    logger.info("Initializing tokenizers...")

    # 🚨 动态读取我们刚刚生成的 25k 词表
    vocab_file_path = "asset/mol_vocabs/vocab_phase2_25k.txt"
    if not os.path.exists(vocab_file_path):
        raise FileNotFoundError(f"找不到 Phase 2 词表: {vocab_file_path}")

    motif_tokenizer = MotifTokenizer(vocab_file=vocab_file_path)
    text_tokenizer = TextTokenizer(model_name="google/t5-v1_1-base")
    e3fp_tokenizer = E3FPTokenizer(fp_level=4, fp_bits=4096)

    # 获取实际新词表的大小
    new_vocab_size = len(motif_tokenizer.tokenizer)
    logger.info(f"✅ Motif Tokenizer 成功加载，当前词表大小: {new_vocab_size}")

    # ==========================================
    # 🚀 3. 加载 Phase 1 权重并扩充张量 (极其重要)
    # ==========================================
    logger.info(f"🚀 Loading Phase 1 Checkpoint from: {model_args.model_name_or_path}")
    model = MoStT5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

    # 🚨 扩充模型的 Embedding 矩阵以兼容 25k 词表
    logger.info(f"🔄 正在根据新词表 Resize Token Embeddings -> {new_vocab_size}...")
    # 注意：T5 底层结构复杂，调用专门的 resize_token_embeddings
    model.resize_token_embeddings(new_vocab_size)

    # ==========================================
    # 🌍 4. 构建 Phase 2 混合多任务数据集和 Collator
    # ==========================================
    logger.info("📦 Building Phase 2 Multi-modal Dataset...")

    # 获取传参，如果没有指定则回退到我们生成的终极数据库
    data_path = data_args.train_file if data_args.train_file else "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_pubchem_final.lmdb"
    text_weight_path = data_args.text_weight_path if data_args.text_weight_path else "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_text_weights.json"

    # 调用大一统 Dataset
    max_len = data_args.max_len if hasattr(data_args, 'max_len') else 512
    train_dataset = GSMATDataset(
        lmdb_path=data_path,
        motif_tokenizer=motif_tokenizer,
        text_tokenizer=text_tokenizer,
        e3fp_tokenizer=e3fp_tokenizer
    )

    # 🚨 修复 Collator 初始化，传入真正的 Tokenizer 实例和权重路径
    data_collator = GSMATPhase2Collator(
        motif_tokenizer=motif_tokenizer,
        text_tokenizer=text_tokenizer,
        text_weight_path=text_weight_path,
        e3fp_pad_id=-1,
        mask_ratio=0.15
    )

    # 5. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # ==========================================
    # 🔍 6. 权重健康检查 (防 NaN 崩溃)
    # ==========================================
    logger.info("=" * 40)
    logger.info("🔍 WEIGHT SANITY CHECK (Before Phase 2 Training)")
    if hasattr(model, "shared"):
        std = model.shared.weight.std().item()
        logger.info(f"  -> Shared Embeddings STD: {std:.6f} (Target: ~0.002 or match Base Model)")
        if std > 10.0:
            logger.warning("⚠️ WARNING: Shared Embeddings are exceptionally large! Might cause NaN in Text MLM.")
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
        trainer.save_model()


if __name__ == "__main__":
    # 保持纯净的通信环境，防止双卡 3090 NCCL 死锁
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    main()