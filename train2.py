import logging
import os
import sys
import re
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from torch.nn import CrossEntropyLoss  # 🚀 必须导入，用于多任务 Loss 监控

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoConfig
)

# 🚀 引入 RDKit，用于智能初始化
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# =========================================================================
# 🚀 路径自适应补丁：确保分布式启动时能正确加载本地模块
# =========================================================================
CURRENT_ROOT = os.path.dirname(os.path.abspath(__file__))
if CURRENT_ROOT not in sys.path:
    sys.path.append(CURRENT_ROOT)

from tokenization.text_tokenizer import TextTokenizer
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
from dataset.dataset2 import GSMATDataset, GSMATPhase2Collator
from model.configuration import MoStT5Config
from model.modeling import MoStT5ForConditionalGeneration
from arguments import ModelArguments, DataArguments

logger = logging.getLogger(__name__)


# =========================================================================
# 🛠️ 核心外挂 1：多任务 Loss 监控 Trainer (正式版)
# =========================================================================
class Phase2MultiTaskTrainer(Trainer):
    def __init__(self, *args, motif_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.motif_tokenizer = motif_tokenizer

        # 提取四大任务的前缀 ID
        tok = self.motif_tokenizer.tokenizer
        self.task_ids = {
            "mmm": tok.convert_tokens_to_ids("[MMM]:"),
            "caption": tok.convert_tokens_to_ids("[Caption]:"),
            "text2mol": tok.convert_tokens_to_ids("[Text2Mol]:"),
            "denoise": tok.convert_tokens_to_ids("[Denoise]:")
        }

        # 任务 Loss 累加器
        self.task_loss_acc = {k: 0.0 for k in self.task_ids.keys()}
        self.task_step_cnt = {k: 0 for k in self.task_ids.keys()}
        self.accum_loss_3d = 0.0
        self.step_loss_3d = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        global_loss = outputs.loss

        # 🚀 [无梯度监控] 计算细粒度任务 Loss
        if not return_outputs:
            with torch.no_grad():
                logits = outputs.logits
                labels = inputs["labels"]

                loss_fct = CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)).view(labels.size(0), -1)

                valid_mask = (labels != -100).float()
                sample_losses = (token_losses * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-6)

                first_tokens = inputs["input_ids"][:, 0]
                for name, tid in self.task_ids.items():
                    mask = (first_tokens == tid)
                    if mask.any():
                        self.task_loss_acc[name] += sample_losses[mask].mean().item()
                        self.task_step_cnt[name] += 1

                if hasattr(outputs, "loss_3d") and outputs.loss_3d is not None:
                    self.accum_loss_3d += outputs.loss_3d.item()
                    self.step_loss_3d += 1

        return (global_loss, outputs) if return_outputs else global_loss

    def log(self, logs: Dict[str, float]) -> None:
        """注入 TensorBoard 指标"""
        for name in self.task_ids.keys():
            if self.task_step_cnt[name] > 0:
                logs[f"loss_{name}"] = round(self.task_loss_acc[name] / self.task_step_cnt[name], 4)
                self.task_loss_acc[name], self.task_step_cnt[name] = 0.0, 0

        if self.step_loss_3d > 0:
            logs["loss_pure_3d"] = round(self.accum_loss_3d / self.step_loss_3d, 4)
            self.accum_loss_3d, self.step_loss_3d = 0.0, 0

        super().log(logs)


# =========================================================================
# 🛠️ 核心外挂 2：鲁棒清洗器与 RDKit 状态同步
# =========================================================================
def robust_motif_to_mol(motif_str):
    if motif_str in ['<unk>', '<pad>', '<bom>', '<eom>', '[.]']: return "SPECIAL"
    if re.match(r'^<\d+\*>$', motif_str): return "ANCHOR"
    if re.match(r'^\[.*?\]:$', motif_str): return "TASK_PROMPT"

    clean_smiles = re.sub(r'\[.*?\]:', '', motif_str)
    clean_smiles = re.sub(r'<\d+\*>', '', clean_smiles).replace("()", "").strip()
    if not clean_smiles: return None

    def finalize_mol(m):
        if m is None: return None
        try:
            m.UpdatePropertyCache(strict=False)  # 🚀 解决 RDKit 隐式氢报错的核心
            Chem.FastFindRings(m)
            return m
        except:
            return None

    mol = Chem.MolFromSmiles(clean_smiles)
    if mol is not None: return finalize_mol(mol)

    # 尝试剥壳版
    stripped = clean_smiles[1:-1] if clean_smiles.startswith('[') and clean_smiles.endswith(']') else clean_smiles
    if stripped != clean_smiles:
        mol = Chem.MolFromSmiles(stripped)
        if mol is not None: return finalize_mol(mol)

    # SMARTS 兜底
    mol = Chem.MolFromSmarts(clean_smiles)
    if mol is not None: return finalize_mol(mol)
    return None


def smart_initialize_new_embeddings(model, motif_tokenizer, old_vocab_size):
    logger.info("=" * 60)
    logger.info(f"🧠 启动基于 Morgan 指纹的 3D 权重智能继承 (旧词表大小={old_vocab_size})")

    embeddings = model.get_input_embeddings().weight.data
    new_vocab_size = len(motif_tokenizer.tokenizer)

    if new_vocab_size <= old_vocab_size:
        logger.info("⏭️ 词表未发生扩容，跳过智能初始化。")
        return

    old_fps, valid_old_indices = [], []
    id_to_token = {v: k for k, v in motif_tokenizer.tokenizer.get_vocab().items()}

    for idx in range(old_vocab_size):
        mol = robust_motif_to_mol(id_to_token.get(idx, ""))
        if mol is not None and not isinstance(mol, str):
            try:
                mol.UpdatePropertyCache(strict=False)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                old_fps.append(fp)
                valid_old_indices.append(idx)
            except:
                continue

    logger.info(f"✅ 成功提取了 {len(valid_old_indices)} 个合法的化学指纹锚点。")

    success_count = 0
    success_examples = []

    for new_idx in range(old_vocab_size, new_vocab_size):
        mol = robust_motif_to_mol(id_to_token.get(new_idx, ""))
        if mol is not None and not isinstance(mol, str):
            try:
                mol.UpdatePropertyCache(strict=False)
                new_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                similarities = DataStructs.BulkTanimotoSimilarity(new_fp, old_fps)
                best_match_idx = similarities.index(max(similarities))
                actual_old_idx = valid_old_indices[best_match_idx]

                embeddings[new_idx] = embeddings[actual_old_idx].clone() + torch.randn_like(embeddings[new_idx]) * 0.01
                success_count += 1
                if len(success_examples) < 10:
                    success_examples.append((id_to_token.get(new_idx, ""), id_to_token.get(actual_old_idx, "")))
            except:
                continue

    logger.info(f"✅ 智能初始化完成！成功注入 {success_count} 个基团。")
    if success_examples:
        for nt, ot in success_examples: logger.info(f"   新: {nt:<15} -> 继承自: {ot}")
    logger.info("=" * 60)


# =========================================================================
# 🚀 主训练流程
# =========================================================================
def main():
    AutoConfig.register("most-t5", MoStT5Config)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. 核心优化项 (解决 GPU 闲置与同步报错)
    training_args.remove_unused_columns = False  # 必须！确保自定义字段能传给 Collator
    training_args.save_safetensors = False  # 保持与脚本一致

    # ⚡ GPU 利用率优化：保持 Worker 常驻，消除 Epoch 切换时的“掉速”
    training_args.dataloader_persistent_workers = True

    # ⚡ 显存优化：开启梯度检查点，用计算换空间
    # 开启后，你可以将 bash 里的 per_device_train_batch_size 从 4 改为 8 甚至 12
    training_args.gradient_checkpointing = True

    # 3. 容错优化
    # 确保分布式环境下不会因为某些任务没用到 3D Head 而锁死
    training_args.ddp_find_unused_parameters = True
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    text_tokenizer = TextTokenizer(model_args.tokenizer_name, max_len=data_args.max_seq_length)
    motif_tokenizer = MotifTokenizer(
        vocab_file=data_args.vocab_file,
        base_tokenizer=text_tokenizer.tokenizer,
        max_len=data_args.max_seq_length
    )
    e3fp_tokenizer = E3FPTokenizer(fp_level=model_args.e3fp_num_levels - 1, fp_bits=model_args.e3fp_vocab_size)

    config = MoStT5Config.from_pretrained(
        model_args.model_name_or_path,
        vocab_size=len(motif_tokenizer.tokenizer),
        e3fp_vocab_size=model_args.e3fp_vocab_size,
        e3fp_num_levels=model_args.e3fp_num_levels,
    )
    # 🚀 Phase 2 黄金平衡：降级 3D 损失，防止干扰图文对齐
    config.lambda_3d = 1.0

    model = MoStT5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True
    )

    # 🚀 自动校准：Phase 1 检查点实际词表大小为 52306
    REAL_OLD_VOCAB_SIZE = 52306
    model.resize_token_embeddings(len(motif_tokenizer.tokenizer))
    smart_initialize_new_embeddings(model, motif_tokenizer, old_vocab_size=REAL_OLD_VOCAB_SIZE)

    # 🚀 均等任务路由：25% x 4
    phase2_task_probs = {"mmm": 0.25, "caption": 0.25, "text2mol": 0.25, "denoise": 0.25}

    train_dataset = GSMATDataset(
        lmdb_path=data_args.train_file,
        text_tokenizer=text_tokenizer,
        motif_tokenizer=motif_tokenizer,
        e3fp_tokenizer=e3fp_tokenizer,
        c4_lmdb_path=data_args.c4_file,  # 🚀 统一参数名
        max_seq_length=data_args.max_seq_length,
        task_probs=phase2_task_probs
    )

    trainer = Phase2MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=GSMATPhase2Collator(motif_tokenizer, text_tokenizer, data_args.text_weight_path, is_train=True),
        motif_tokenizer=motif_tokenizer
    )

    logger.info("🔥 ALL SYSTEMS GO. Starting Phase 2 Cross-Modal Alignment...")

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()


if __name__ == "__main__":
    main()