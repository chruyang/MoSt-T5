import torch
import lmdb
import pickle
import logging
import numpy as np
import os
from typing import List, Dict, Any
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# 路径回退策略
try:
    from tokenization.text_tokenizer import TextTokenizer
    from tokenization.motif_tokenizer import MotifTokenizer
    from tokenization.e3fp_tokenizer import E3FPTokenizer
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tokenization.text_tokenizer import TextTokenizer
    from tokenization.motif_tokenizer import MotifTokenizer
    from tokenization.e3fp_tokenizer import E3FPTokenizer

logger = logging.getLogger(__name__)


class GSMATDataset(Dataset):
    def __init__(self, lmdb_path: str,
                 text_tokenizer: TextTokenizer,
                 motif_tokenizer: MotifTokenizer,
                 e3fp_tokenizer: E3FPTokenizer):

        self.lmdb_path = lmdb_path
        self.text_tokenizer = text_tokenizer
        self.motif_tokenizer = motif_tokenizer
        self.e3fp_tokenizer = e3fp_tokenizer

        # 期望的 E3FP 维度
        self.e3fp_width = self.e3fp_tokenizer.fp_level + 1

        # 智能判断是文件还是目录
        is_subdir = os.path.isdir(lmdb_path)
        if not is_subdir and not os.path.isfile(lmdb_path):
            raise FileNotFoundError(f"LMDB path does not exist: {lmdb_path}")

        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=is_subdir
        )

        with self.env.begin() as txn:
            try:
                self.length = int(txn.get(b'__len__'))
            except (TypeError, ValueError):
                self.length = txn.stat()['entries']
                logger.warning(f"LMDB missing '__len__', using stat entries: {self.length}")

    def __len__(self):
        return self.length

    def handle_dimension_mismatch(self, e3fp_ids: torch.Tensor) -> torch.Tensor:
        """维度修复逻辑"""
        current_width = e3fp_ids.shape[1]
        if current_width == self.e3fp_width:
            return e3fp_ids

        if current_width < self.e3fp_width:
            # 补齐
            pad_tensor = torch.full((e3fp_ids.shape[0], self.e3fp_width - current_width),
                                    self.e3fp_tokenizer.padding_idx, dtype=torch.long)
            return torch.cat([e3fp_ids, pad_tensor], dim=1)
        else:
            # 截断
            return e3fp_ids[:, :self.e3fp_width]

    def __getitem__(self, idx):
        try:
            with self.env.begin() as txn:
                data = txn.get(str(idx).encode())
                if data is None:
                    raise ValueError(f"Data not found for key {idx}")
                entry = pickle.loads(data)

            # 1. 基础数据获取
            smiles = entry.get('smiles_kekule') or entry.get('smiles', '')
            text = entry.get('enriched_description', '')
            if not text:  # 如果为空，回退到 description
                text = entry.get('description', '')
            if not text:  # 如果还是空，回退到 text (兼容 split 脚本生成的字段)
                text = entry.get('text', '')
            atom_mapping = entry.get('atom_mapping', [])
            e3fp_numpy = entry.get('e3fp')

            # 2. Text 处理
            text_enc = self.text_tokenizer(text, padding=False, truncation=True)
            text_ids = text_enc['input_ids'].squeeze(0)

            # 3. Motif 处理
            motif_ids = self.motif_tokenizer.encode(smiles, return_tensors='pt', padding=False)
            if motif_ids.dim() > 1: motif_ids = motif_ids.squeeze(0)

            # 4. E3FP 处理 (带 Fallback 和 修复)
            if e3fp_numpy is not None:
                e3fp_ids = torch.tensor(e3fp_numpy, dtype=torch.long)
            else:
                e3fp_ids = self.e3fp_tokenizer.from_smiles(smiles)

            e3fp_ids = self.handle_dimension_mismatch(e3fp_ids)

            # 5. Atom Mapping
            num_atoms = e3fp_ids.shape[0]
            atom_to_motif_map = torch.zeros(num_atoms, dtype=torch.long)

            for motif_idx, atom_indices in enumerate(atom_mapping):
                token_idx = motif_idx + 1
                if token_idx >= len(motif_ids): break
                for atom_idx in atom_indices:
                    if atom_idx < num_atoms:
                        atom_to_motif_map[atom_idx] = token_idx

            return {
                "motif_input_ids": motif_ids,
                "e3fp_input_ids": e3fp_ids,
                "text_input_ids": text_ids,
                "atom_to_motif_map": atom_to_motif_map,
            }

        except Exception as e:
            # 坏样本跳过机制
            # logger.warning(f"Error loading sample {idx}: {e}. Skipping...")
            return self.__getitem__((idx + 1) % self.length)


class GSMATCollator:
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

        # Padding Inputs
        batch_motif = pad_sequence(motif_ids, batch_first=True, padding_value=self.motif_pad_id)
        batch_e3fp = pad_sequence(e3fp_ids, batch_first=True, padding_value=self.e3fp_pad_id)
        batch_map = pad_sequence(atom_maps, batch_first=True, padding_value=0)

        # Padding Labels (Mol2Text Target)
        batch_labels = pad_sequence(text_ids, batch_first=True, padding_value=self.ignore_index)

        # Masks
        motif_mask = (batch_motif != self.motif_pad_id).long()
        atom_mask = (batch_e3fp[:, :, 0] != self.e3fp_pad_id).long()

        return {
            "motif_ids": batch_motif,
            "motif_attention_mask": motif_mask,
            "e3fp_ids": batch_e3fp,
            "atom_attention_mask": atom_mask,
            "atom_to_motif_map": batch_map,
            "labels": batch_labels
        }