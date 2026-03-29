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
# 🚀 自定义 Trainer：负责在 TensorBoard 中聚合并显示子 Loss
# ==========================================
class MoStTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 🚀 缓冲区：用于累加一个 logging_steps 周期内的 Loss 平均值
        self._sub_loss_buffer = {"lm": 0.0, "geom": 0.0, "steps": 0}

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss

        # 🚀 在训练阶段，将子项 Loss 记录到缓冲区
        if self.model.training:
            # 稳健提取，支持 DDP 返回的对象
            lm_loss = getattr(outputs, "main_lm_loss", None)
            gm_loss = getattr(outputs, "geom_3d_loss", None)

            if lm_loss is not None and gm_loss is not None:
                # 使用 .item() 提取标量，避免显存泄漏
                self._sub_loss_buffer["lm"] += lm_loss.item()
                self._sub_loss_buffer["geom"] += gm_loss.item()
                self._sub_loss_buffer["steps"] += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict) -> None:
        """
        重写 log 方法，在达到 logging_steps 触发日志记录时注入分项平均值
        """
        if self._sub_loss_buffer["steps"] > 0:
            avg_lm = self._sub_loss_buffer["lm"] / self._sub_loss_buffer["steps"]
            avg_geom = self._sub_loss_buffer["geom"] / self._sub_loss_buffer["steps"]

            # 使用 train/ 前缀，TensorBoard 会自动将其与总 loss 归类到一组
            logs["train/loss_main_lm"] = avg_lm
            logs["train/loss_geom_3d"] = avg_geom

            # 记录完成后重置缓冲区
            self._sub_loss_buffer = {"lm": 0.0, "geom": 0.0, "steps": 0}

        super().log(logs)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 🚀 Dataloader 线程数打通
    training_args.dataloader_num_workers = data_args.num_workers
    training_args.remove_unused_columns = False

    # 🚀 关键修复：由于模型存在嵌套共享权重，强制回退到传统的 .bin 格式保存
    # 这样可以避开 safetensors 对于 "shared tensors mismatch" 的严格路径校验
    training_args.save_safetensors = False

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

    # 🚀 扩容词表与定点覆盖初始化 (保留业务逻辑)
    model.resize_token_embeddings(final_vocab_size)

    # 针对 3D 几何回归头的正交初始化
    for module in model.geometric_head:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=0.1)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    # 对新增 3D 层应用自定义初始化逻辑
    model.encoder.fusion_layer.apply(model._init_weights)
    model.encoder.gsm_embeddings.e3fp_embeddings.apply(model._init_weights)

    # =========================================================================
    # 3. 🚀 双矩阵方差对齐 (Variance Alignment - 保留原样)
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
    # 4. 数据集与训练器 (保留原样逻辑)
    # =========================================================================
    train_dataset = GSMATDataset(
        lmdb_path=data_args.train_file,
        text_tokenizer=text_tokenizer,
        motif_tokenizer=motif_tokenizer,
        e3fp_tokenizer=e3fp_tokenizer,
        c4_lmdb_path="",
        whitelist_path="",
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

    # 🚀 关键修正：确保使用自定义的子类 MoStTrainer 而非原生类
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
    logger.info(f"  -> Final Shared Embeddings STD: {std_shared:.6f}")

    lm_head = model.get_output_embeddings()
    if lm_head is not None:
        std_lm = lm_head.weight.std().item()
        logger.info(f"  -> Final LM Head STD: {std_lm:.6f}")
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