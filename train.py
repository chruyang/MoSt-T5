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

    # 🚨 必须保留：禁止 Trainer 删除我们的 3D 多模态数据列
    training_args.remove_unused_columns = False

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
        'fusion_type': model_args.fusion_type,
        'dropout_rate': model_args.dropout_rate,
        'lambda_3d': 0.1  # 动态注入 3D 几何 MSE Loss 的权重
    })

    model = MoStT5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True
    )

    # 💡 读取原始词表大小 (通常是 32128)
    old_vocab_size = model.shared.weight.shape[0]

    # 🚀 核心对齐：严谨地将模型 Embedding 矩阵大小与补全后的 Tokenizer 词表大小对齐
    model.resize_token_embeddings(motif_tokenizer.vocab_size)

    # 🚀 终极 SOTA 初始化：动态分布匹配 (超越 CAMT5 的核心细节)
    with torch.no_grad():
        if hasattr(model, "shared") and model.shared.weight.shape[0] > old_vocab_size:
            # 1. 精准计算 T5 原始词表的统计分布
            old_weight = model.shared.weight[:old_vocab_size]
            old_mean = old_weight.mean().item()
            old_std = old_weight.std().item()

            # 2. 将新增的 Motif 权重严格限制在这个原生分布内
            model.shared.weight[old_vocab_size:].normal_(mean=old_mean, std=old_std)

            # 3. 同步处理 LM Head (如果存在且不与 shared 绑定)
            if hasattr(model, "lm_head") and model.lm_head.weight.shape[0] > old_vocab_size:
                model.lm_head.weight[old_vocab_size:].normal_(mean=old_mean, std=old_std)

            logger.info(f"✨ 动态分布匹配完成: 新增 Token 权重已对齐原始分布 (Mean: {old_mean:.4f}, STD: {old_std:.4f})")

    # 强制补全所有必要的 Token ID
    pad_token_id = text_tokenizer.tokenizer.pad_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = pad_token_id
    if model.config.bos_token_id is None:
        model.config.bos_token_id = pad_token_id

    # --- Trainer ---
    data_collator = GSMATPretrainingCollator(
        motif_tokenizer=motif_tokenizer,
        e3fp_pad_id=-1,
        mask_ratio=0.15  # 经典 15% 掩码率
    )

    # 🚀 移除了 callbacks=[EarlyStoppingCallback...]，预训练不需要早停
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    # 保持原有的“照妖镜”权重检测逻辑
    logger.info("=" * 40)
    logger.info("🔍 WEIGHT SANITY CHECK (Before Training)")
    if hasattr(model, "shared"):
        std = model.shared.weight.std().item()
        logger.info(f"  -> Shared Embeddings STD: {std:.6f} (Target: ~0.002 or match Base Model)")
        if std > 10.0: # 稍微调高了报警阈值，以免误伤 t5-v1_1-base 的正常大方差
            logger.warning("⚠️ WARNING: Shared Embeddings are exceptionally large!")
    if hasattr(model, "lm_head"):
        std = model.lm_head.weight.std().item()
        logger.info(f"  -> LM Head Weights STD:   {std:.6f}")
    if hasattr(model.encoder, "gsm_embeddings"):
        std = model.encoder.gsm_embeddings.word_embeddings.weight.std().item()
        logger.info(f"  -> Encoder Embeddings STD: {std:.6f}")
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