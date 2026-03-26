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
    # ==========================================
    # 1. 注册自定义配置 & 解析参数
    # ==========================================
    AutoConfig.register("most-t5", MoStT5Config)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 🚨 致命防御：禁止 HuggingFace Trainer 自动剔除"未在forward中显式声明"的列
    training_args.remove_unused_columns = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.info(f"Model Args: {model_args}")
    logger.info(f"Data Args: {data_args}")
    logger.info(f"Training Args: {training_args}")

    set_seed(training_args.seed)

    # ==========================================
    # 🛠️ 2. 初始化 Tokenizers (核心：串联词表共享机制)
    # ==========================================
    # 步骤 1：先初始化 TextTokenizer（它会加载 T5 基础词汇并注入 [MMM]: 等任务特殊符）
    text_tokenizer = TextTokenizer(model_args.model_name_or_path, max_len=768)

    # 步骤 2：将 text_tokenizer 的底层 C++ tokenizer 实例传递给 MotifTokenizer
    # 这样 Motif 会在 Text 扩充过的 ID 之后继续追加 25000 个化学基团，绝不碰撞！
    motif_tokenizer = MotifTokenizer(
        vocab_file=data_args.vocab_path,
        model_name=model_args.model_name_or_path,
        max_len=768,
        base_tokenizer=text_tokenizer.tokenizer  # 🚀 极其重要的词表联结操作
    )

    e3fp_tokenizer = E3FPTokenizer(fp_level=3, fp_bits=4096)

    # ==========================================
    # 🧬 3. 加载模型结构
    # ==========================================
    config = MoStT5Config.from_pretrained(
        model_args.model_name_or_path,
        e3fp_num_levels=4,  # FP层数硬编码锁定
        e3fp_vocab_size=4096  # FP词表大小硬编码锁定
    )

    model = MoStT5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True
    )

    # 🚀 致命修复：由于我们在同一个 Tokenizer 实例里塞入了文本特殊符和 25000 个 Motif
    # 必须通知模型扩充 Embedding 矩阵以匹配最新的词表总长度 (约 32128 + 25000 + x)
    model.resize_token_embeddings(len(motif_tokenizer.tokenizer))
    logger.info(f"✅ 模型 Embedding 层已成功重置！当前总词汇表大小为: {len(motif_tokenizer.tokenizer)}")

    # ==========================================
    # 📦 4. 准备数据集和数据处理流
    # ==========================================
    train_path = os.path.join(data_args.data_dir, "train/phase2_dataset.lmdb")
    eval_path = os.path.join(data_args.data_dir, "valid/phase2_dataset.lmdb")
    text_weight_path = os.path.join(data_args.data_dir, "train/text_weights.json")

    max_seq_length = data_args.max_seq_length

    train_dataset = GSMATDataset(
        lmdb_path=train_path,
        motif_tokenizer=motif_tokenizer,
        text_tokenizer=text_tokenizer,
        e3fp_tokenizer=e3fp_tokenizer,
        c4_lmdb_path="",
        whitelist_path="",
        max_seq_length=max_seq_length
    )

    eval_dataset = None
    if training_args.do_eval:
        eval_dataset = GSMATDataset(
            lmdb_path=eval_path,
            motif_tokenizer=motif_tokenizer,
            text_tokenizer=text_tokenizer,
            e3fp_tokenizer=e3fp_tokenizer,
            c4_lmdb_path="",
            whitelist_path="",
            max_seq_length=max_seq_length
        )

    data_collator = GSMATPhase2Collator(
        motif_tokenizer=motif_tokenizer,
        text_tokenizer=text_tokenizer,
        text_weight_path=text_weight_path,
        e3fp_pad_id=-1,
        mask_ratio=0.15
    )

    # ==========================================
    # 🔥 5. 启动训练引擎
    # ==========================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("=" * 60)
    logger.info("🔥 ALL SYSTEMS GO. Starting Phase 2: MoSt-T5 Cross-Modal Alignment...")
    logger.info("=" * 60)

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()