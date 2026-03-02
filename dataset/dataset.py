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
            "input_ids": batch_motif,  # 👈 修改：从 "motif_ids" 改为 "input_ids"
            "attention_mask": motif_mask,  # 👈 修改：从 "motif_attention_mask" 改为 "attention_mask"
            "e3fp_ids": batch_e3fp,
            "atom_attention_mask": atom_mask,
            "atom_to_motif_map": batch_map,
            "labels": batch_labels
        }


class GSMATPretrainingCollator:
    def __init__(self,
                 motif_tokenizer: MotifTokenizer,
                 e3fp_pad_id: int = -1,
                 mask_ratio: float = 0.15):
        self.motif_tokenizer = motif_tokenizer
        self.pad_id = motif_tokenizer.pad_id
        self.e3fp_pad_id = e3fp_pad_id
        self.mask_ratio = mask_ratio

        # 🚀 1. 引入 CAMT5 的底层 Frag 引擎
        try:
            from model.CAMT5.representation import Frag
            self.frag_processor = Frag()
        except ImportError:
            raise ImportError("无法导入 Frag，请确保路径正确")

        # 🚀 2. 构建 O(1) 极速权重查表 Tensor (初始化时只执行一次)
        vocab_size = motif_tokenizer.vocab_size
        self.weight_lookup = torch.zeros(vocab_size, dtype=torch.float32)

        vocab_dict = motif_tokenizer.tokenizer.get_vocab()
        for token_str, token_id in vocab_dict.items():
            # 特殊字符 (包括 <unk>, <pad> 等) 权重设为 0，绝不掩码！
            if token_str in motif_tokenizer.tokenizer.all_special_tokens:
                self.weight_lookup[token_id] = 0.0
            else:
                try:
                    # 获取真实的重原子数量作为重要性得分
                    atom_count = self.frag_processor.get_size(token_str)
                    self.weight_lookup[token_id] = float(atom_count)
                except Exception:
                    self.weight_lookup[token_id] = 0.01

        # 对权重进行平滑 (对数平滑)，防止大小基团概率悬殊过大
        self.weight_lookup = torch.log1p(self.weight_lookup)

        # 获取 T5 的掩码替换符 (我们统一使用 <extra_id_0> 作为占位符，保持长度不变以保全 3D 映射)
        self.mask_token_id = self.motif_tokenizer.tokenizer.convert_tokens_to_ids("<extra_id_0>")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        motif_ids = [item['motif_input_ids'] for item in batch]
        e3fp_ids = [item['e3fp_input_ids'] for item in batch]
        atom_maps = [item['atom_to_motif_map'] for item in batch]

        # 1. 基础 Padding
        batch_motif = pad_sequence(motif_ids, batch_first=True, padding_value=self.pad_id)
        batch_e3fp = pad_sequence(e3fp_ids, batch_first=True, padding_value=self.e3fp_pad_id)
        batch_map = pad_sequence(atom_maps, batch_first=True, padding_value=0)

        batch_size, seq_len = batch_motif.shape

        # 2. 极速查表获取权重
        weights = self.weight_lookup[batch_motif]  # Shape: [batch_size, seq_len]
        weights[batch_motif == self.pad_id] = 0.0  # Padding 绝对不掩码

        # 3. 动态掩码采样 (Multinomial)
        masked_motif_ids = batch_motif.clone()
        masked_e3fp_ids = batch_e3fp.clone()
        labels = batch_motif.clone()  # Decoder 的目标是还原真实的 2D Motif

        # 记录哪些位置被 Mask 了 (用于计算 3D Loss)
        mask_positions = torch.zeros_like(batch_motif, dtype=torch.bool)

        for i in range(batch_size):
            row_weight = weights[i]
            # 如果全是 0 (比如极短序列)，跳过
            if row_weight.sum() <= 0:
                continue

            num_to_mask = max(1, int((batch_motif[i] != self.pad_id).sum() * self.mask_ratio))
            # 依据化学重要性抽样
            mask_idx = torch.multinomial(row_weight + 1e-6, num_samples=num_to_mask, replacement=False)

            mask_positions[i, mask_idx] = True

            # (1) 2D 文本视角的掩码：替换为 <extra_id_0>
            masked_motif_ids[i, mask_idx] = self.mask_token_id

            # (2) 3D 几何视角的熔断防作弊：将被 Mask 的 Motif 对应的底层原子 E3FP 置零
            # 通过 atom_to_motif_map 反查
            for m_idx in mask_idx:
                # 找到属于这个 Motif (Token 索引为 m_idx) 的所有底层原子
                atom_indices = (batch_map[i] == m_idx).nonzero(as_tuple=True)[0]
                if len(atom_indices) > 0:
                    masked_e3fp_ids[i, atom_indices] = self.e3fp_pad_id

        # 把没有被 Mask 的位置 Label 设为 -100，T5 计算交叉熵时会自动忽略
        labels[~mask_positions] = -100

        motif_mask = (masked_motif_ids != self.pad_id).long()
        atom_mask = (masked_e3fp_ids[:, :, 0] != self.e3fp_pad_id).long()

        return {
            "input_ids": masked_motif_ids,  # 被 Mask 破坏的 2D 序列 (Encoder 的真实输入)
            "attention_mask": motif_mask,  # 2D 序列的 Padding 掩码
            "e3fp_ids": masked_e3fp_ids,  # 被同步熔断/置零的 3D 序列 (防止模型偷看答案)
            "atom_attention_mask": atom_mask,  # 3D 原子的 Padding 掩码
            "atom_to_motif_map": batch_map,  # 2D-3D 的映射桥梁
            "labels": labels,  # Decoder 的目标 (被 Mask 的真实 2D Motif ID)
            "unmasked_e3fp_ids": batch_e3fp,  # 完好无损的原始 3D 序列 (供模型内部瞬间生成 3D MSE Loss 的 Target)
            "mask_positions": mask_positions  # 布尔矩阵，告诉模型哪些位置需要算 3D MSE Loss
        }