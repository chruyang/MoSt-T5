import logging
import os
import sys
import torch
import numpy as np
from transformers import (
    HfArgumentParser,
    Trainer,  # 🚀 修改 1：预训练不需要 Seq2SeqTrainer，标准 Trainer 即可
    TrainingArguments,  # 🚀 修改 2：使用标准 TrainingArguments
    set_seed,
    EarlyStoppingCallback
)

from tokenization.text_tokenizer import TextTokenizer
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
from dataset.dataset import GSMATDataset, GSMATPretrainingCollator  # 🚀 修改 3：导入我们新写的预训练 Collator
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

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    set_seed(training_args.seed)

    # --- Tokenizers ---
    logger.info("Loading Tokenizers...")
    text_tokenizer = TextTokenizer(model_name=model_args.model_name_or_path, max_len=data_args.max_len)
    motif_tokenizer = MotifTokenizer(vocab_file=model_args.vocab_path, model_name=model_args.model_name_or_path)
    e3fp_tokenizer = E3FPTokenizer(padding_idx=-1)

    # --- Datasets ---
    logger.info("Loading Datasets...")
    train_dataset = GSMATDataset(
        lmdb_path=data_args.train_file,
        text_tokenizer=text_tokenizer,
        motif_tokenizer=motif_tokenizer,
        e3fp_tokenizer=e3fp_tokenizer
    )

    eval_dataset = None
    if data_args.validation_file:
        logger.info(f"Loading Validation Dataset...")
        eval_dataset = GSMATDataset(
            lmdb_path=data_args.validation_file,
            text_tokenizer=text_tokenizer,
            motif_tokenizer=motif_tokenizer,
            e3fp_tokenizer=e3fp_tokenizer
        )

    if data_args.max_eval_samples is not None and eval_dataset is not None:
        num_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = torch.utils.data.Subset(eval_dataset, range(num_samples))
        logger.info(f"🔧 Debug Mode: Truncated eval_dataset to {num_samples} samples.")

    # --- Model Config & Init ---
    logger.info("Initializing MoSt-T5 Model for Pre-training...")
    config = MoStT5Config.from_pretrained(model_args.model_name_or_path)
    config.update({
        'e3fp_num_levels': model_args.e3fp_num_levels,
        'e3fp_vocab_size': model_args.e3fp_vocab_size,
        'vocab_size': motif_tokenizer.vocab_size,
        'fusion_type': model_args.fusion_type,
        'dropout_rate': model_args.dropout_rate,
        'lambda_3d': 0.1  # 🚀 修改 4：动态注入 3D 几何 MSE Loss 的权重！
    })

    model = MoStT5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True
    )

    # 强制补全所有必要的 Token ID
    pad_token_id = text_tokenizer.tokenizer.pad_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = pad_token_id
    if model.config.bos_token_id is None:
        model.config.bos_token_id = pad_token_id

    # 🚀 修改 5：彻底删除 evaluate, nltk 以及 compute_metrics 函数！
    # 预训练只需要监控 Trainer 自动计算的 Loss。

    # --- Trainer ---
    # 🚀 修改 6：使用我们全新打造的预训练专属 Collator
    data_collator = GSMATPretrainingCollator(
        motif_tokenizer=motif_tokenizer,
        e3fp_pad_id=-1,
        mask_ratio=0.15  # 经典 15% 掩码率
    )

    # 🚀 修改 7：换用标准 Trainer，关闭所有生成相关评估
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # compute_metrics=None, # 不再需要
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] if eval_dataset else None
    )

    # 保持原有的“照妖镜”权重检测逻辑
    logger.info("=" * 40)
    logger.info("🔍 WEIGHT SANITY CHECK (Before Training)")
    if hasattr(model, "shared"):
        std = model.shared.weight.std().item()
        logger.info(f"  -> Shared Embeddings STD: {std:.6f} (Target: ~0.002)")
        if std > 0.01:
            logger.error("❌❌ DANGER: Shared Embeddings are too large! post_init() reset them!")
    if hasattr(model, "lm_head"):
        std = model.lm_head.weight.std().item()
        logger.info(f"  -> LM Head Weights STD:   {std:.6f} (Target: ~0.002)")
    if hasattr(model.encoder, "gsm_embeddings"):
        std = model.encoder.gsm_embeddings.word_embeddings.weight.std().item()
        logger.info(f"  -> Encoder Embeddings STD: {std:.6f} (Target: ~0.002)")
    logger.info("=" * 40)

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

    if training_args.do_eval and eval_dataset:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()