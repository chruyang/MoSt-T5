import torch
import lmdb
import pickle
import logging
import numpy as np
from typing import List, Dict, Any
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# 引入 Tokenizers
try:
    from tokenization.text_tokenizer import TextTokenizer
    from tokenization.motif_tokenizer import MotifTokenizer
    from tokenization.e3fp_tokenizer import E3FPTokenizer
except ImportError:
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tokenization.text_tokenizer import TextTokenizer
    from tokenization.motif_tokenizer import MotifTokenizer
    from tokenization.e3fp_tokenizer import E3FPTokenizer

logger = logging.getLogger(__name__)


class GSMATDataset(Dataset):
    """
    MoSt-T5 核心数据集类 (Geo-Semantic Aligned Token Dataset)

    [功能亮点]
    1. 动态 E3FP 维度适配 (handle_dimension_mismatch)
    2. 实时 E3FP 生成 Fallback
    3. 严谨的 Atom-to-Motif Mapping 构建
    """

    def __init__(self, lmdb_path: str,
                 text_tokenizer: TextTokenizer,
                 motif_tokenizer: MotifTokenizer,
                 e3fp_tokenizer: E3FPTokenizer):

        self.lmdb_path = lmdb_path
        self.text_tokenizer = text_tokenizer
        self.motif_tokenizer = motif_tokenizer
        self.e3fp_tokenizer = e3fp_tokenizer

        # 动态获取期望的 E3FP 维度 (Level + 1)
        self.e3fp_width = self.e3fp_tokenizer.fp_level + 1

        logger.info(f"Initializing GSMATDataset from {lmdb_path}")
        logger.info(f"Expected E3FP Width: {self.e3fp_width} (Level {self.e3fp_tokenizer.fp_level})")

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            try:
                self.length = int(txn.get(b'__len__'))
            except (TypeError, ValueError):
                self.length = txn.stat()['entries']
                logger.warning(f"LMDB missing '__len__', using stat entries: {self.length}")

    def __len__(self):
        return self.length

    def handle_dimension_mismatch(self, e3fp_ids: torch.Tensor) -> torch.Tensor:
        """
        处理 E3FP 维度不匹配的情况 (鲁棒性增强)
        """
        current_width = e3fp_ids.shape[1]

        if current_width == self.e3fp_width:
            return e3fp_ids

        logger.debug(f"E3FP dim mismatch: got {current_width}, expected {self.e3fp_width}. Adjusting...")

        if current_width < self.e3fp_width:
            # 维度不足：补充 Padding 值 (使用 tokenizer 的 padding_idx)
            pad_tensor = torch.full((e3fp_ids.shape[0], self.e3fp_width - current_width),
                                    self.e3fp_tokenizer.padding_idx, dtype=torch.long)
            return torch.cat([e3fp_ids, pad_tensor], dim=1)

        elif current_width > self.e3fp_width:
            # 维度过大：截断多余层级
            return e3fp_ids[:, :self.e3fp_width]

        return e3fp_ids

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = txn.get(str(idx).encode())
            if data is None:
                # 容错：尝试下一个索引
                return self.__getitem__((idx + 1) % self.length)
            entry = pickle.loads(data)

        # 1. 基础数据获取
        # 优先使用 Kekule SMILES 以保证与 Atom Mapping 的对齐
        smiles = entry.get('smiles_kekule') or entry.get('smiles', '')
        text = entry.get('description', '') or entry.get('text', '')
        atom_mapping = entry.get('atom_mapping', [])
        e3fp_numpy = entry.get('e3fp')

        # 2. Text 处理
        text_enc = self.text_tokenizer(text, padding=False, truncation=True)
        text_ids = text_enc['input_ids'].squeeze(0)

        # 3. Motif 处理 (动态 Padding: padding=False)
        motif_ids = self.motif_tokenizer.encode(smiles, return_tensors='pt', padding=False)
        if motif_ids.dim() > 1: motif_ids = motif_ids.squeeze(0)

        # 4. E3FP 处理 (Fallback + 维度适配)
        if e3fp_numpy is not None:
            e3fp_ids = torch.tensor(e3fp_numpy, dtype=torch.long)
        else:
            # Fallback: 实时生成
            # 注意：实时生成可能较慢，建议在预处理阶段完成
            logger.debug(f"Generating E3FP on-the-fly for idx {idx}")
            e3fp_ids = self.e3fp_tokenizer.from_smiles(smiles)

        # 维度检查与修复
        e3fp_ids = self.handle_dimension_mismatch(e3fp_ids)

        # 5. Atom-to-Motif Mapping (核心架构)
        num_atoms = e3fp_ids.shape[0]
        # 初始化为 0 (配合 Mask 使用，0 通常指向 <bom> 或特殊位)
        atom_to_motif_map = torch.zeros(num_atoms, dtype=torch.long)

        for motif_idx, atom_indices in enumerate(atom_mapping):
            token_idx = motif_idx + 1  # +1 跳过 <bom>

            # 防御性检查
            if token_idx >= len(motif_ids):
                logger.debug(f"Atom mapping index {token_idx} out of bounds for idx {idx}")
                break

            for atom_idx in atom_indices:
                if atom_idx < num_atoms:
                    atom_to_motif_map[atom_idx] = token_idx

        return {
            "motif_input_ids": motif_ids,
            "e3fp_input_ids": e3fp_ids,
            "text_input_ids": text_ids,
            "atom_to_motif_map": atom_to_motif_map,
        }


class GSMATCollator:
    """
    数据整理器 (Collator) - 增强版
    自动处理 Mol2Text 任务的 Labels
    """

    def __init__(self,
                 motif_pad_id: int,
                 text_pad_id: int,
                 e3fp_pad_id: int = -1,
                 ignore_index: int = -100):
        self.motif_pad_id = motif_pad_id
        self.text_pad_id = text_pad_id
        self.e3fp_pad_id = e3fp_pad_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        motif_ids = [item['motif_input_ids'] for item in batch]
        e3fp_ids = [item['e3fp_input_ids'] for item in batch]
        text_ids = [item['text_input_ids'] for item in batch]
        atom_maps = [item['atom_to_motif_map'] for item in batch]

        # 1. Padding Inputs (batch_first=True)
        batch_motif = pad_sequence(motif_ids, batch_first=True, padding_value=self.motif_pad_id)
        batch_e3fp = pad_sequence(e3fp_ids, batch_first=True, padding_value=self.e3fp_pad_id)
        # Map Padding 使用 0 (配合 Mask 使用)
        batch_map = pad_sequence(atom_maps, batch_first=True, padding_value=0)

        # 2. Padding Labels (关键步骤)
        # 对于 Mol2Text 任务，目标是文本
        # 使用 ignore_index (-100) 填充，使模型计算 Loss 时忽略 Pad 部分
        batch_labels = pad_sequence(text_ids, batch_first=True, padding_value=self.ignore_index)

        # 3. 生成 Masks
        motif_mask = (batch_motif != self.motif_pad_id).long()
        # E3FP Mask: 只要第一列不为 Pad 即为有效
        atom_mask = (batch_e3fp[:, :, 0] != self.e3fp_pad_id).long()

        return {
            "motif_ids": batch_motif,
            "motif_attention_mask": motif_mask,

            "e3fp_ids": batch_e3fp,
            "atom_attention_mask": atom_mask,
            "atom_to_motif_map": batch_map,

            # 这里的 labels 字段会被 T5 自动识别并计算 Loss
            "labels": batch_labels
        }