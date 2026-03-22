import logging
import os
import sys
import torch
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
from dataset.dataset import GSMATDataset, GSMATPhase2Collator
from model.configuration import MoStT5Config
from model.modeling import MoStT5ForConditionalGeneration
from arguments import ModelArguments, DataArguments

logger = logging.getLogger(__name__)


def main():
    # 1. 注册自定义配置 & 解析参数
    AutoConfig.register("most-t5", MoStT5Config)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ==========================================
    # 🚨 致命防御：禁止 HuggingFace Trainer 自动剔除"未在forward中显式声明"的列
    # 这保证了 e3fp_ids, unmasked_e3fp_ids 和 mask_positions 能安全送达模型！
    # ==========================================
    training_args.remove_unused_columns = False

    # 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    set_seed(training_args.seed)

    # ==========================================
    # 📦 2. 初始化分词器 (加载 Phase 2 扩充词表)
    # ==========================================
    logger.info("Initializing tokenizers...")

    # 指向我们在 Phase 2 生成的 25k 新词表 (请确保路径与您的实际生成路径一致)
    vocab_file_path = getattr(data_args, 'vocab_file', "asset/mol_vocabs/vocab_phase2_25k.txt")
    if not os.path.exists(vocab_file_path):
        logger.warning(f"⚠️ 未找到 Phase 2 词表 {vocab_file_path}，将尝试回退旧词表！")
        vocab_file_path = "asset/mol_vocabs/vocab_20k.txt"

    motif_tokenizer = MotifTokenizer(vocab_file=vocab_file_path)
    text_tokenizer = TextTokenizer(model_name=model_args.model_name_or_path)

    # 🚨 fp_level=3 对应 4 层 E3FP 特征 (0, 1, 2, 3)，精确匹配
    e3fp_tokenizer = E3FPTokenizer(fp_level=3, fp_bits=4096)

    new_vocab_size = len(motif_tokenizer.tokenizer)
    logger.info(f"✅ Tokenizers loaded. Motif Vocab Size: {new_vocab_size}")

    # ==========================================
    # 🚀 3. 加载基座权重 & 词表扩容
    # ==========================================
    logger.info(f"🚀 Loading Base Model from: {model_args.model_name_or_path}")
    model = MoStT5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

    logger.info(f"🔄 Resizing Token Embeddings to {new_vocab_size}...")
    model.resize_token_embeddings(new_vocab_size)

    # ==========================================
    # 🌍 4. 构建 Phase 2 混合多任务数据集 & Collator
    # ==========================================
    logger.info("📦 Building Multi-modal Dataset with C4 Replay...")

    data_path = getattr(data_args, 'train_file',
                        "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_pubchem_final.lmdb")
    c4_path = getattr(data_args, 'c4_file', "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/c4_pretrain.lmdb")
    text_weight_path = getattr(data_args, 'text_weight_path',
                               "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_text_weights.json")

    train_dataset = GSMATDataset(
        lmdb_path=data_path,
        motif_tokenizer=motif_tokenizer,
        text_tokenizer=text_tokenizer,
        e3fp_tokenizer=e3fp_tokenizer,
        c4_lmdb_path=c4_path  # 训练集挂载 C4 弹药库
    )

    # 🚀 新增：构建纯净的验证集 (屏蔽 C4 防遗忘任务)
    eval_path = getattr(data_args, 'eval_file',
                        "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/valid/phase2_pubchem_final.lmdb")
    eval_dataset = None
    if os.path.exists(eval_path):
        logger.info(f"📊 Building Evaluation Dataset from {eval_path}...")
        eval_dataset = GSMATDataset(
            lmdb_path=eval_path,
            motif_tokenizer=motif_tokenizer,
            text_tokenizer=text_tokenizer,
            e3fp_tokenizer=e3fp_tokenizer,
            c4_lmdb_path=""  # 👈 传空字符串，强制验证集不做 C4 降噪，保证 Metric 纯洁性
        )

    data_collator = GSMATPhase2Collator(
        motif_tokenizer=motif_tokenizer,
        text_tokenizer=text_tokenizer,
        text_weight_path=text_weight_path,
        e3fp_pad_id=-1,
        mask_ratio=0.15
    )

    # ==========================================
    # 🔥 5. 启动训练引擎 (加入 Eval)
    # ==========================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # 👈 挂载验证集
        data_collator=data_collator,
    )

    logger.info("=" * 60)
    logger.info("🔥 ALL SYSTEMS GO. Starting Phase 2: MoSt-T5 Cross-Modal Alignment...")
    logger.info("=" * 60)

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()


if __name__ == "__main__":
    # 屏蔽 WANDB 提示，并关闭可能导致 3090 死锁的 NCCL 通信参数
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    main()