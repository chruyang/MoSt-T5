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


# ==========================================
# 🚀 自定义 Trainer：负责在 TensorBoard 中显示子 Loss
# ==========================================
class MoStTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss

        # 仅在训练阶段且 outputs 包含子 Loss 时记录
        if self.model.training and hasattr(outputs, "main_lm_loss"):
            self.log({
                "loss/main_lm": outputs.main_lm_loss.item(),
                "loss/geom_3d": outputs.geom_3d_loss.item()
            })

        return (loss, outputs) if return_outputs else loss


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 🚀 Dataloader 线程数与预取支持
    training_args.dataloader_num_workers = data_args.num_workers
    training_args.remove_unused_columns = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    set_seed(training_args.seed)

    # ==========================================
    # 1. 复合 Tokenizer 构建 (保留原样)
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
    # 2. Config & 模型初始化 (保留原样)
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

    # 🚀 扩容词表与定点覆盖初始化 (完整保留你的初始化逻辑)
    model.resize_token_embeddings(final_vocab_size)

    # 针对新增 3D 层的覆盖式特殊处理
    for module in model.geometric_head:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=0.1)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    # 手动触发 3D 层的权重初始化（补救 resize 带来的潜在不确定性）
    model.encoder.fusion_layer.apply(model._init_weights)
    model.encoder.gsm_embeddings.e3fp_embeddings.apply(model._init_weights)

    # =========================================================================
    # 3. 🚀 双矩阵方差对齐 (Variance Alignment - 完整保留)
    # =========================================================================
    if final_vocab_size > old_vocab_size:
        logger.info(f"⚖️ 执行词表方差强制对齐: {old_vocab_size} -> {final_vocab_size}")
        with torch.no_grad():
            old_embeddings = model.shared.weight[:old_vocab_size]
            new_embeddings = model.shared.weight[old_vocab_size:]
            new_embeddings.normal_(mean=old_embeddings.mean().item(), std=old_embeddings.std().item())

            lm_head = model.get_output_embeddings()
            if lm_head is not None and lm_head.weight is not model.shared.weight:
                old_lm_head = lm_head.weight[:old_vocab_size]
                new_lm_head = lm_head.weight[old_vocab_size:]
                new_lm_head.normal_(mean=old_lm_head.mean().item(), std=old_lm_head.std().item())

    # =========================================================================
    # 4. 数据集与训练器 (保留所有路径与白名单参数)
    # =========================================================================
    train_dataset = GSMATDataset(
        lmdb_path=data_args.train_file,
        text_tokenizer=text_tokenizer,
        motif_tokenizer=motif_tokenizer,
        e3fp_tokenizer=e3fp_tokenizer,
        c4_lmdb_path="",  # 依照原样
        whitelist_path="",  # 依照原样
        max_seq_length=data_args.max_seq_length,
        task_probs={"mmm": 1.0}
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
            task_probs={"mmm": 1.0}
        )

    data_collator = GSMATPretrainingCollator(
        motif_tokenizer=motif_tokenizer,
        text_tokenizer=text_tokenizer,
        text_weight_path=data_args.text_weight_path,
        e3fp_pad_id=-1,
        mask_ratio=0.15,
        is_train=True
    )

    # 🚀 使用 MoStTrainer 替换原生 Trainer
    trainer = MoStTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    # 🚀 完整保留 Weight Sanity Check
    logger.info("=" * 40)
    logger.info("🔍 WEIGHT SANITY CHECK (Before Training)")
    std_shared = model.shared.weight.std().item()
    logger.info(f"  -> Final Shared Embeddings STD: {std_shared:.6f} (Target: ~10.26)")

    lm_head = model.get_output_embeddings()
    if lm_head is not None:
        std_lm = lm_head.weight.std().item()
        logger.info(f"  -> Final LM Head STD: {std_lm:.6f} (Target: ~0.55)")
    logger.info("=" * 40)

    # 开始训练与指标保存
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()