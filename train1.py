import logging
import os
import sys
import torch
import numpy as np
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)

from tokenization.text_tokenizer import TextTokenizer
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
from dataset.dataset import GSMATDataset, GSMATPretrainingCollator
from model.configuration import MoStT5Config
from model.modeling import MoStT5ForConditionalGeneration
from arguments import ModelArguments, DataArguments

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    set_seed(training_args.seed)

    # ==========================================
    # 1. 复合 Tokenizer 构建
    # ==========================================
    text_tokenizer = TextTokenizer(model_args.tokenizer_name, max_len=data_args.max_seq_length)
    motif_tokenizer = MotifTokenizer(
        vocab_file=data_args.vocab_file,
        base_tokenizer=text_tokenizer.tokenizer,
        max_len=data_args.max_seq_length
    )
    e3fp_tokenizer = E3FPTokenizer(fp_level=model_args.e3fp_num_levels - 1, fp_bits=model_args.e3fp_vocab_size)

    final_vocab_size = len(motif_tokenizer.tokenizer)

    # ==========================================
    # 2. Config & 模型初始化 (读取原生 32128 权重)
    # ==========================================
    config = MoStT5Config.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        e3fp_vocab_size=model_args.e3fp_vocab_size,
        e3fp_num_levels=model_args.e3fp_num_levels,
    )
    config.lambda_3d = 1.0

    model = MoStT5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        ignore_mismatched_sizes=True
    )

    old_vocab_size = model.config.vocab_size

    # 🚀 扩容词表
    model.resize_token_embeddings(final_vocab_size)

    # =========================================================================
    # 3. 🚀 双矩阵方差对齐 (Variance Alignment)
    # =========================================================================
    if final_vocab_size > old_vocab_size:
        logger.info(f"⚖️ 执行词表方差强制对齐: {old_vocab_size} -> {final_vocab_size}")
        with torch.no_grad():
            # 修复 Shared Embeddings (~10.26)
            old_embeddings = model.shared.weight[:old_vocab_size]
            new_embeddings = model.shared.weight[old_vocab_size:]
            new_embeddings.normal_(mean=old_embeddings.mean().item(), std=old_embeddings.std().item())

            # 修复独立 LM Head (~0.55)
            lm_head = model.get_output_embeddings()
            if lm_head is not None and lm_head.weight is not model.shared.weight:
                old_lm_head = lm_head.weight[:old_vocab_size]
                new_lm_head = lm_head.weight[old_vocab_size:]
                new_lm_head.normal_(mean=old_lm_head.mean().item(), std=old_lm_head.std().item())

    # =========================================================================
    # 4. 数据集与训练器
    # =========================================================================
    train_dataset = GSMATDataset(
        lmdb_path=data_args.train_file,
        text_tokenizer=text_tokenizer,
        motif_tokenizer=motif_tokenizer,
        e3fp_tokenizer=e3fp_tokenizer,
        c4_lmdb_path="",
        whitelist_path="",
        max_seq_length=data_args.max_seq_length,
        task_probs={"mmm": 1.0}  # 🚀 已修复语法错误
    )

    eval_dataset = None
    if data_args.validation_file:
        eval_dataset = GSMATDataset(
            lmdb_path=data_args.validation_file,
            text_tokenizer=text_tokenizer,
            motif_tokenizer=motif_tokenizer,
            e3fp_tokenizer=e3fp_tokenizer,
            c4_lmdb_path="",
            whitelist_path="",
            max_seq_length=data_args.max_seq_length,
            task_probs={"mmm": 1.0}  # 🚀 已修复语法错误
        )

    data_collator = GSMATPretrainingCollator(
        motif_tokenizer=motif_tokenizer,
        text_tokenizer=text_tokenizer,
        text_weight_path=data_args.text_weight_path,
        e3fp_pad_id=-1,
        mask_ratio=0.15,
        is_train=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    logger.info("=" * 40)
    logger.info("🔍 WEIGHT SANITY CHECK (Before Training)")
    std_shared = model.shared.weight.std().item()
    logger.info(f"  -> Final Shared Embeddings STD: {std_shared:.6f} (Target: ~10.26)")

    lm_head = model.get_output_embeddings()
    if lm_head is not None:
        std_lm = lm_head.weight.std().item()
        logger.info(f"  -> Final LM Head STD: {std_lm:.6f} (Target: ~0.55)")
    logger.info("=" * 40)

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()