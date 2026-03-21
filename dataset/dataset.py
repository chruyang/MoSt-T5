import torch
import lmdb
import pickle
import logging
import numpy as np
import os
import json
from typing import List, Dict, Any
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# 路径回退策略：确保在不同运行环境下都能正确导入 Tokenizer
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


# ==========================================
# 1. 核心数据集类 (加入延迟初始化与防越界截断)
# ==========================================
class GSMATDataset(Dataset):
    def __init__(self, lmdb_path: str,
                 text_tokenizer: TextTokenizer,
                 motif_tokenizer: MotifTokenizer,
                 e3fp_tokenizer: E3FPTokenizer):

        self.lmdb_path = lmdb_path
        self.text_tokenizer = text_tokenizer
        self.motif_tokenizer = motif_tokenizer
        self.e3fp_tokenizer = e3fp_tokenizer

        self.e3fp_width = self.e3fp_tokenizer.fp_level + 1
        self.error_count = 0
        self.max_log_errors = 50
        self.max_seq_len = 512  # 🚀 定义最大序列长度，防止超大分子爆显存

        # 🚀 延迟初始化改造 1：仅临时打开获取长度，随后立刻关闭
        is_subdir = os.path.isdir(lmdb_path)
        temp_env = lmdb.open(
            lmdb_path, readonly=True, lock=False, readahead=False,
            meminit=False, subdir=is_subdir
        )

        with temp_env.begin() as txn:
            try:
                self.length = int(txn.get(b'__len__'))
            except (TypeError, ValueError):
                self.length = txn.stat()['entries']
                logger.warning(f"LMDB missing '__len__', using stat entries: {self.length}")

        temp_env.close()  # 拿完长度立刻断开连接
        self.env = None  # 🚀 切断主进程占用，交由 Worker 子进程独立打开

    def __len__(self):
        return self.length

    # 🚀 延迟初始化改造 2：供每个 DataLoader Worker 独立调用的初始化方法
    def _init_db(self):
        is_subdir = os.path.isdir(self.lmdb_path)
        self.env = lmdb.open(
            self.lmdb_path, readonly=True, lock=False, readahead=False,
            meminit=False, subdir=is_subdir
        )

    def handle_dimension_mismatch(self, e3fp_ids: torch.Tensor) -> torch.Tensor:
        current_width = e3fp_ids.shape[1]
        if current_width == self.e3fp_width:
            return e3fp_ids
        if current_width < self.e3fp_width:
            pad_tensor = torch.full((e3fp_ids.shape[0], self.e3fp_width - current_width),
                                    self.e3fp_tokenizer.padding_idx, dtype=torch.long)
            return torch.cat([e3fp_ids, pad_tensor], dim=1)
        else:
            return e3fp_ids[:, :self.e3fp_width]

    def __getitem__(self, idx):
        # 🚀 延迟初始化改造 3：Worker 首次取数据时打开 LMDB 指针
        if self.env is None:
            self._init_db()

        # 💓 降频心跳日志：确认多进程正常运行 (每 10000 条打印一次)
        if idx % 10000 == 0:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else "Main"
            logger.info(f"🔍 [Worker-{worker_id}] 正常读取数据中，当前进度索引: {idx}")

        try:
            with self.env.begin() as txn:
                data = txn.get(str(idx).encode())
                if data is None:
                    return self.__getitem__((idx + 1) % self.length)
                entry = pickle.loads(data)

            smiles = entry.get('smiles_kekule') or entry.get('smiles', '')
            text = entry.get('enriched_description', '')
            if not text: text = entry.get('description', '')
            if not text: text = entry.get('text', '')

            atom_mapping = entry.get('atom_mapping', [])
            e3fp_numpy = entry.get('e3fp')

            # Motif 编码
            motif_result = self.motif_tokenizer.encode(smiles, return_tensors='pt', padding=False, return_mapping=True)

            if isinstance(motif_result, tuple):
                motif_ids, motif_mapping = motif_result
            else:
                motif_ids = motif_result
                motif_mapping = []

            if motif_ids.dim() > 1: motif_ids = motif_ids.squeeze(0)

            # 🚀 截断保护机制：强制切断超长分子
            if motif_ids.shape[0] > self.max_seq_len:
                motif_ids = motif_ids[:self.max_seq_len]

            # E3FP 处理
            if e3fp_numpy is not None:
                e3fp_ids = torch.tensor(e3fp_numpy, dtype=torch.long)
            else:
                e3fp_ids = self.e3fp_tokenizer.from_smiles(smiles)

            e3fp_ids = self.handle_dimension_mismatch(e3fp_ids)

            # 3D 绝对坐标映射
            num_atoms = e3fp_ids.shape[0]
            atom_to_motif_map = torch.full((num_atoms,), -1, dtype=torch.long)

            for motif_idx, atom_indices in enumerate(atom_mapping):
                if motif_idx >= len(motif_mapping):
                    break
                real_token_idx = motif_mapping[motif_idx]

                # 🚀 截断协同：过滤越界索引
                if real_token_idx < self.max_seq_len:
                    for atom_idx in atom_indices:
                        if atom_idx < num_atoms:
                            atom_to_motif_map[atom_idx] = real_token_idx

            text_enc = self.text_tokenizer(text, padding=False, truncation=True)
            text_ids = text_enc['input_ids'].squeeze(0)

            return {
                "motif_input_ids": motif_ids,
                "e3fp_input_ids": e3fp_ids,
                "text_input_ids": text_ids,
                "atom_to_motif_map": atom_to_motif_map,
            }

        except Exception as e:
            if self.error_count < self.max_log_errors:
                logger.warning(f"⚠️ [Data Error] 样本 {idx} 解析失败: {e}. 正在重采样...")
                self.error_count += 1
            return self.__getitem__((idx + 1) % self.length)


# ==========================================
# 2. 基础 Collator (保留，用于未来的微调或评估)
# ==========================================
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

        batch_motif = pad_sequence(motif_ids, batch_first=True, padding_value=self.motif_pad_id)
        batch_e3fp = pad_sequence(e3fp_ids, batch_first=True, padding_value=self.e3fp_pad_id)
        batch_map = pad_sequence(atom_maps, batch_first=True, padding_value=-1)
        batch_labels = pad_sequence(text_ids, batch_first=True, padding_value=self.ignore_index)

        motif_mask = (batch_motif != self.motif_pad_id).long()
        atom_mask = (batch_e3fp[:, :, 0] != self.e3fp_pad_id).long()

        return {
            "input_ids": batch_motif,
            "attention_mask": motif_mask,
            "e3fp_ids": batch_e3fp,
            "atom_attention_mask": atom_mask,
            "atom_to_motif_map": batch_map,
            "labels": batch_labels
        }


# ==========================================
# 3. 预训练 Phase 1 Collator (纯分子，双轨重要性掩码)
# ==========================================
class GSMATPretrainingCollator:
    def __init__(self,
                 motif_tokenizer: MotifTokenizer,
                 e3fp_pad_id: int = -1,
                 mask_ratio: float = 0.15,
                 task_b_ratio: float = 0.15):
        self.motif_tokenizer = motif_tokenizer
        self.pad_id = motif_tokenizer.pad_id
        self.e3fp_pad_id = e3fp_pad_id
        self.mask_ratio = mask_ratio
        self.task_b_ratio = task_b_ratio

        try:
            from model.CAMT5.representation import Frag
            self.frag_processor = Frag()
        except ImportError:
            raise ImportError("无法导入 Frag，请检查路径")

        vocab_size = len(motif_tokenizer.tokenizer)
        self.weight_lookup = torch.zeros(vocab_size, dtype=torch.float32)
        vocab_dict = motif_tokenizer.tokenizer.get_vocab()

        for token_str, token_id in vocab_dict.items():
            if token_str in motif_tokenizer.tokenizer.all_special_tokens:
                self.weight_lookup[token_id] = 0.0
            else:
                try:
                    self.weight_lookup[token_id] = float(self.frag_processor.get_size(token_str))
                except Exception:
                    self.weight_lookup[token_id] = 0.01

        self.weight_lookup = torch.log1p(self.weight_lookup)
        self.mask_token_id = self.motif_tokenizer.tokenizer.convert_tokens_to_ids("<extra_id_0>")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        motif_ids = [item['motif_input_ids'] for item in batch]
        e3fp_ids = [item['e3fp_input_ids'] for item in batch]
        atom_maps = [item['atom_to_motif_map'] for item in batch]

        batch_motif = pad_sequence(motif_ids, batch_first=True, padding_value=self.pad_id)
        batch_e3fp = pad_sequence(e3fp_ids, batch_first=True, padding_value=self.e3fp_pad_id)
        batch_map = pad_sequence(atom_maps, batch_first=True, padding_value=-1)

        batch_size, seq_len = batch_motif.shape
        weights = self.weight_lookup[batch_motif]
        weights[batch_motif == self.pad_id] = 0.0

        masked_motif_ids = batch_motif.clone()
        masked_e3fp_ids = batch_e3fp.clone()
        labels = batch_motif.clone()

        mlm_mask_positions = torch.zeros_like(batch_motif, dtype=torch.bool)
        geometric_mask_positions = torch.zeros_like(batch_motif, dtype=torch.bool)

        for i in range(batch_size):
            row_weight = weights[i]
            if row_weight.sum() <= 0: continue

            valid_len = (batch_motif[i] != self.pad_id).sum().item()
            num_to_mask_A = max(1, int(valid_len * self.mask_ratio))
            num_to_mask_B = max(1, int(valid_len * self.task_b_ratio))

            # 任务 A
            mask_idx_A = torch.multinomial(row_weight + 1e-6, num_samples=num_to_mask_A, replacement=False)
            mlm_mask_positions[i, mask_idx_A] = True
            geometric_mask_positions[i, mask_idx_A] = True
            masked_motif_ids[i, mask_idx_A] = self.mask_token_id

            # 任务 B
            row_weight_B = row_weight.clone()
            row_weight_B[mask_idx_A] = 0.0

            if row_weight_B.sum() > 0:
                actual_num_B = min(num_to_mask_B, (row_weight_B > 0).sum().item())
                if actual_num_B > 0:
                    mask_idx_B = torch.multinomial(row_weight_B + 1e-6, num_samples=actual_num_B, replacement=False)
                    geometric_mask_positions[i, mask_idx_B] = True
                else:
                    mask_idx_B = torch.tensor([], dtype=torch.long, device=mask_idx_A.device)
            else:
                mask_idx_B = torch.tensor([], dtype=torch.long, device=mask_idx_A.device)

            # 统一执行 3D 物理熔断
            all_mask_idx = torch.cat([mask_idx_A, mask_idx_B])
            for m_idx in all_mask_idx:
                atom_indices = (batch_map[i] == m_idx).nonzero(as_tuple=True)[0]
                if len(atom_indices) > 0:
                    masked_e3fp_ids[i, atom_indices] = self.e3fp_pad_id

        labels[~mlm_mask_positions] = -100
        motif_mask = (masked_motif_ids != self.pad_id).long()
        atom_mask = (masked_e3fp_ids[:, :, 0] != self.e3fp_pad_id).long()

        return {
            "input_ids": masked_motif_ids,
            "attention_mask": motif_mask,
            "e3fp_ids": masked_e3fp_ids,
            "atom_attention_mask": atom_mask,
            "atom_to_motif_map": batch_map,
            "labels": labels,
            "unmasked_e3fp_ids": batch_e3fp,
            "mask_positions": geometric_mask_positions
        }


# ==========================================
# 4. 预训练 Phase 2 Collator (文本与分子交叉对齐)
# ==========================================
class GSMATPhase2Collator(GSMATPretrainingCollator):
    def __init__(self, motif_tokenizer, text_tokenizer, text_weight_path, e3fp_pad_id=-1, mask_ratio=0.15):
        super().__init__(motif_tokenizer, e3fp_pad_id, mask_ratio)
        self.text_pad_id = text_tokenizer.tokenizer.pad_token_id
        self.mask_token_id = motif_tokenizer.tokenizer.convert_tokens_to_ids("<extra_id_0>")

        actual_vocab_size = text_tokenizer.tokenizer.vocab_size

        if os.path.exists(text_weight_path):
            with open(text_weight_path, 'r') as f:
                text_weights = json.load(f)
            max_json_id = max([int(k) for k in text_weights.keys()]) if text_weights else 0
            safe_size = max(actual_vocab_size, max_json_id + 1)
            self.text_weight_lookup = torch.zeros(safe_size, dtype=torch.float32)
            for k, v in text_weights.items():
                curr_id = int(k)
                if curr_id < safe_size:
                    self.text_weight_lookup[curr_id] = float(v)
        else:
            self.text_weight_lookup = torch.zeros(actual_vocab_size, dtype=torch.float32) + 1.0

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        motif_batch = super().__call__(batch)

        text_ids = [item['text_input_ids'] for item in batch]
        batch_text = pad_sequence(text_ids, batch_first=True, padding_value=self.text_pad_id)
        batch_size, text_len = batch_text.shape

        text_weights = self.text_weight_lookup[batch_text]
        text_weights[batch_text == self.text_pad_id] = 0.0

        masked_text_ids = batch_text.clone()
        text_labels = batch_text.clone()
        text_mask_positions = torch.zeros_like(batch_text, dtype=torch.bool)

        for i in range(batch_size):
            row_w = text_weights[i]
            if row_w.sum() > 0:
                num_to_mask = max(1, int((batch_text[i] != self.text_pad_id).sum() * self.mask_ratio))
                mask_idx = torch.multinomial(row_w + 1e-6, num_samples=num_to_mask, replacement=False)
                text_mask_positions[i, mask_idx] = True
                masked_text_ids[i, mask_idx] = self.mask_token_id
        text_labels[~text_mask_positions] = -100

        concat_input_ids = torch.cat([masked_text_ids, motif_batch["input_ids"]], dim=1)
        text_att_mask = (masked_text_ids != self.text_pad_id).long()
        concat_attention_mask = torch.cat([text_att_mask, motif_batch["attention_mask"]], dim=1)
        concat_labels = torch.cat([text_labels, motif_batch["labels"]], dim=1)

        fp_levels, fp_dim = motif_batch["e3fp_ids"].shape[1:]
        dummy_e3fp = torch.full((batch_size, text_len, fp_dim), self.e3fp_pad_id,
                                dtype=motif_batch["e3fp_ids"].dtype, device=motif_batch["e3fp_ids"].device)
        concat_e3fp_ids = torch.cat([dummy_e3fp, motif_batch["e3fp_ids"]], dim=1)
        concat_unmasked_e3fp_ids = torch.cat([dummy_e3fp, motif_batch["unmasked_e3fp_ids"]], dim=1)

        dummy_atom_mask = torch.zeros((batch_size, text_len), dtype=torch.long,
                                      device=motif_batch["atom_attention_mask"].device)
        concat_atom_mask = torch.cat([dummy_atom_mask, motif_batch["atom_attention_mask"]], dim=1)

        concat_mask_positions = torch.cat([torch.zeros_like(text_mask_positions), motif_batch["mask_positions"]], dim=1)

        shifted_map = motif_batch["atom_to_motif_map"].clone()
        valid_map_mask = shifted_map != -1
        shifted_map[valid_map_mask] += text_len

        dummy_map = torch.full((batch_size, text_len), -1,
                               dtype=shifted_map.dtype, device=shifted_map.device)
        concat_map = torch.cat([dummy_map, shifted_map], dim=1)

        return {
            "input_ids": concat_input_ids,
            "attention_mask": concat_attention_mask,
            "labels": concat_labels,
            "e3fp_ids": concat_e3fp_ids,
            "unmasked_e3fp_ids": concat_unmasked_e3fp_ids,
            "atom_attention_mask": concat_atom_mask,
            "atom_to_motif_map": concat_map,
            "mask_positions": concat_mask_positions
        }