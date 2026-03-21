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
# 1. 核心数据集类 (加入 Phase 2 多任务路由分配)
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
        self.max_seq_len = 512

        # 🚀 Phase 2: 黄金混合比例 (MMM 60%, Caption 15%, Text2Mol 10%, Denoise 15%)
        self.task_probs = {
            "mmm": 0.60,
            "caption": 0.15,
            "text2mol": 0.10,
            "denoise": 0.15
        }

        # 延迟初始化
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

        temp_env.close()
        self.env = None

    def __len__(self):
        return self.length

    def _init_db(self):
        is_subdir = os.path.isdir(self.lmdb_path)
        self.env = lmdb.open(
            self.lmdb_path, readonly=True, lock=False, readahead=False,
            meminit=False, subdir=is_subdir
        )

    def handle_dimension_mismatch(self, e3fp_ids: torch.Tensor) -> torch.Tensor:
        current_width = e3fp_ids.shape[1]
        if current_width == self.e3fp_width: return e3fp_ids
        if current_width < self.e3fp_width:
            pad_tensor = torch.full((e3fp_ids.shape[0], self.e3fp_width - current_width),
                                    self.e3fp_tokenizer.padding_idx, dtype=torch.long)
            return torch.cat([e3fp_ids, pad_tensor], dim=1)
        return e3fp_ids[:, :self.e3fp_width]

    def __getitem__(self, idx):
        if self.env is None: self._init_db()

        if idx % 10000 == 0:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else "Main"
            logger.info(f"🔍 [Worker-{worker_id}] 正常读取数据中，当前进度索引: {idx}")

        try:
            with self.env.begin() as txn:
                data = txn.get(str(idx).encode())
                if data is None: return self.__getitem__((idx + 1) % self.length)
                entry = pickle.loads(data)

            smiles = entry.get('smiles_kekule') or entry.get('smiles', '')
            text = entry.get('enriched_description', '')
            if not text: text = entry.get('description', '')
            if not text: text = entry.get('text', '')

            # ==========================================
            # 🌟 Phase 2: 任务路由与前缀 (Task Routing & Prefixes)
            # ==========================================
            roll = np.random.rand()
            if roll < self.task_probs["mmm"]:
                task = "mmm"
                prompt_text = f"[MMM]: {text}"
            elif roll < self.task_probs["mmm"] + self.task_probs["caption"]:
                task = "caption"
                prompt_text = f"[Caption]: Generate description for the molecule:"
            elif roll < sum(self.task_probs.values()) - self.task_probs["denoise"]:
                task = "text2mol"
                prompt_text = f"[Text2Mol]: {text}"
            else:
                task = "denoise"
                prompt_text = f"[Denoise]: {text}"

            # ==========================================
            # 🌟 Phase 2: 按需解析特征 (节约算力)
            # ==========================================
            # 1. 文本处理 (所有任务都需要输入 Text/Prompt)
            text_enc = self.text_tokenizer(prompt_text, padding=False, truncation=True)
            text_ids = text_enc['input_ids'].squeeze(0)

            # Caption 任务还需要 Target Text 作为 Label
            target_text_ids = torch.tensor([], dtype=torch.long)
            if task == "caption":
                target_enc = self.text_tokenizer(text, padding=False, truncation=True)
                target_text_ids = target_enc['input_ids'].squeeze(0)

            # 2. 分子 Motif 处理 (纯文本降噪任务不需要分子)
            if task != "denoise" and smiles:
                motif_result = self.motif_tokenizer.encode(smiles, return_tensors='pt', padding=False,
                                                           return_mapping=True)
                motif_ids, motif_mapping = motif_result if isinstance(motif_result, tuple) else (motif_result, [])
                if motif_ids.dim() > 1: motif_ids = motif_ids.squeeze(0)
                if motif_ids.shape[0] > self.max_seq_len: motif_ids = motif_ids[:self.max_seq_len]
            else:
                motif_ids = torch.tensor([], dtype=torch.long)
                motif_mapping = []

            # 3. 3D E3FP 处理 (只有 mmm 和 caption 两个需要理解结构的任务才解析 E3FP)
            if task in ["mmm", "caption"] and smiles:
                e3fp_numpy = entry.get('e3fp')
                e3fp_ids = torch.tensor(e3fp_numpy,
                                        dtype=torch.long) if e3fp_numpy is not None else self.e3fp_tokenizer.from_smiles(
                    smiles)
                e3fp_ids = self.handle_dimension_mismatch(e3fp_ids)

                num_atoms = e3fp_ids.shape[0]
                atom_to_motif_map = torch.full((num_atoms,), -1, dtype=torch.long)
                atom_mapping = entry.get('atom_mapping', [])

                for motif_idx, atom_indices in enumerate(atom_mapping):
                    if motif_idx >= len(motif_mapping): break
                    real_token_idx = motif_mapping[motif_idx]
                    if real_token_idx < self.max_seq_len:
                        for atom_idx in atom_indices:
                            if atom_idx < num_atoms: atom_to_motif_map[atom_idx] = real_token_idx
            else:
                e3fp_ids = torch.empty((0, self.e3fp_width), dtype=torch.long)
                atom_to_motif_map = torch.empty((0,), dtype=torch.long)

            return {
                "task": task,
                "text_input_ids": text_ids,
                "target_text_ids": target_text_ids,
                "motif_input_ids": motif_ids,
                "e3fp_input_ids": e3fp_ids,
                "atom_to_motif_map": atom_to_motif_map,
            }

        except Exception as e:
            if self.error_count < self.max_log_errors:
                logger.warning(f"⚠️ [Data Error] 样本 {idx} 解析失败: {e}. 正在重采样...")
                self.error_count += 1
            return self.__getitem__((idx + 1) % self.length)


# ==========================================
# 2. 预训练 Phase 1 Collator (原封不动保留，安全隔离)
# ==========================================
class GSMATPretrainingCollator:
    # ...(此处应保持您原本 GSMATPretrainingCollator 的全部代码，为节省空间未全部展开，请勿删除)...
    pass


# ==========================================
# 3. 🌟 Phase 2 全新大一统 Collator
# ==========================================
class GSMATPhase2Collator:
    def __init__(self, motif_tokenizer, text_tokenizer, text_weight_path, e3fp_pad_id=-1, mask_ratio=0.15):
        self.motif_pad_id = motif_tokenizer.pad_id
        self.text_pad_id = text_tokenizer.tokenizer.pad_token_id
        self.e3fp_pad_id = e3fp_pad_id
        self.mask_ratio = mask_ratio

        # T5 专用的填空占位符 <extra_id_0>
        self.mask_token_id = motif_tokenizer.tokenizer.convert_tokens_to_ids("<extra_id_0>")

        # === 初始化 Motif 权重 (锚点免疫机制) ===
        try:
            from model.CAMT5.representation import Frag
            self.frag_processor = Frag()
        except ImportError:
            raise ImportError("无法导入 Frag，请检查路径")

        motif_vocab_size = len(motif_tokenizer.tokenizer)
        self.motif_weight_lookup = torch.zeros(motif_vocab_size, dtype=torch.float32)
        for token_str, token_id in motif_tokenizer.tokenizer.get_vocab().items():
            if token_str in motif_tokenizer.tokenizer.all_special_tokens:
                self.motif_weight_lookup[token_id] = 0.0  # 锚点与特殊符号权重为 0
            else:
                try:
                    self.motif_weight_lookup[token_id] = float(self.frag_processor.get_size(token_str))
                except:
                    self.motif_weight_lookup[token_id] = 0.01
        self.motif_weight_lookup = torch.log1p(self.motif_weight_lookup)

        # === 初始化 Text 权重 ===
        text_vocab_size = text_tokenizer.tokenizer.vocab_size
        self.text_weight_lookup = torch.zeros(text_vocab_size + 1000, dtype=torch.float32) + 1.0  # Buffer 保护越界
        if os.path.exists(text_weight_path):
            with open(text_weight_path, 'r') as f:
                text_weights = json.load(f)
            for k, v in text_weights.items():
                curr_id = int(k)
                if curr_id < len(self.text_weight_lookup):
                    self.text_weight_lookup[curr_id] = float(v)

    def _mask_sequence(self, input_ids: torch.Tensor, weight_lookup: torch.Tensor, pad_id: int):
        """通用序列掩码器 (适用于 Text 和 Denoise)"""
        seq_len = input_ids.shape[0]
        if seq_len == 0: return input_ids, input_ids.clone()

        # 安全读取权重，防止由于分词器新增 token 导致的索引越界
        safe_ids = torch.clamp(input_ids, max=weight_lookup.shape[0] - 1)
        weights = weight_lookup[safe_ids].clone()
        weights[input_ids == pad_id] = 0.0

        masked_ids = input_ids.clone()
        labels = input_ids.clone()
        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)

        if weights.sum() > 0:
            num_to_mask = max(1, int((input_ids != pad_id).sum() * self.mask_ratio))
            num_to_mask = min(num_to_mask, (weights > 0).sum().item())
            if num_to_mask > 0:
                mask_idx = torch.multinomial(weights + 1e-6, num_samples=num_to_mask, replacement=False)
                mask_positions[mask_idx] = True
                masked_ids[mask_idx] = self.mask_token_id

        labels[~mask_positions] = -100
        return masked_ids, labels

    def _mask_motif_and_e3fp(self, motif_ids: torch.Tensor, e3fp_ids: torch.Tensor, atom_map: torch.Tensor):
        """双轨分子掩码器 (继承 Phase 1 逻辑)"""
        seq_len = motif_ids.shape[0]
        if seq_len == 0: return motif_ids, motif_ids.clone(), e3fp_ids

        weights = self.motif_weight_lookup[torch.clamp(motif_ids, max=self.motif_weight_lookup.shape[0] - 1)].clone()
        weights[motif_ids == self.motif_pad_id] = 0.0

        masked_motif_ids = motif_ids.clone()
        masked_e3fp_ids = e3fp_ids.clone()
        labels = motif_ids.clone()
        mlm_mask_pos = torch.zeros_like(motif_ids, dtype=torch.bool)

        if weights.sum() > 0:
            valid_len = (motif_ids != self.motif_pad_id).sum().item()
            num_to_mask_A = max(1, int(valid_len * self.mask_ratio))
            num_to_mask_B = max(1, int(valid_len * 0.15))  # Task B 熔断率

            # Task A (Mask 1D + 3D)
            actual_A = min(num_to_mask_A, (weights > 0).sum().item())
            if actual_A > 0:
                mask_idx_A = torch.multinomial(weights + 1e-6, num_samples=actual_A, replacement=False)
                mlm_mask_pos[mask_idx_A] = True
                masked_motif_ids[mask_idx_A] = self.mask_token_id
            else:
                mask_idx_A = torch.tensor([], dtype=torch.long, device=weights.device)

            # Task B (仅熔断 3D)
            weights_B = weights.clone()
            weights_B[mask_idx_A] = 0.0
            actual_B = min(num_to_mask_B, (weights_B > 0).sum().item())
            if actual_B > 0:
                mask_idx_B = torch.multinomial(weights_B + 1e-6, num_samples=actual_B, replacement=False)
            else:
                mask_idx_B = torch.tensor([], dtype=torch.long, device=weights.device)

            # 3D 同步熔断
            all_mask_idx = torch.cat([mask_idx_A, mask_idx_B])
            for m_idx in all_mask_idx:
                atom_indices = (atom_map == m_idx).nonzero(as_tuple=True)[0]
                if len(atom_indices) > 0:
                    masked_e3fp_ids[atom_indices] = self.e3fp_pad_id

        labels[~mlm_mask_pos] = -100
        return masked_motif_ids, labels, masked_e3fp_ids

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids, batch_labels = [], []
        batch_e3fp_ids, batch_atom_maps, batch_atom_masks = [], [], []

        for item in batch:
            task = item["task"]
            text_ids = item["text_input_ids"]
            motif_ids = item["motif_input_ids"]
            e3fp_ids = item["e3fp_input_ids"]
            atom_map = item["atom_to_motif_map"]

            fp_dim = e3fp_ids.shape[1] if e3fp_ids.shape[0] > 0 else 5  # 动态获取维度 (通常是5)

            if task == "mmm":
                # 1. 独立掩码两端
                masked_text_ids, text_labels = self._mask_sequence(text_ids, self.text_weight_lookup, self.text_pad_id)
                masked_motif_ids, motif_labels, masked_e3fp_ids = self._mask_motif_and_e3fp(motif_ids, e3fp_ids,
                                                                                            atom_map)

                # 2. 拼接序列 (拼接在前面)
                input_ids = torch.cat([masked_text_ids, masked_motif_ids])
                labels = torch.cat([text_labels, motif_labels])

                # 3. 核心平移逻辑: E3FP 的索引要加上文本的长度
                text_len = len(text_ids)
                shifted_map = atom_map.clone()
                shifted_map[shifted_map != -1] += text_len

                # 4. 文本占位空 3D 特征
                dummy_e3fp = torch.full((text_len, fp_dim), self.e3fp_pad_id, dtype=torch.long)
                final_e3fp = torch.cat([dummy_e3fp, masked_e3fp_ids])
                dummy_map = torch.full((text_len,), -1, dtype=torch.long)
                final_map = torch.cat([dummy_map, shifted_map])

            elif task == "caption":
                # 翻译：看结构 -> 写描述
                input_ids = torch.cat([text_ids, motif_ids])
                labels = item["target_text_ids"]  # Teacher Forcing 的目标

                text_len = len(text_ids)
                shifted_map = atom_map.clone()
                shifted_map[shifted_map != -1] += text_len

                dummy_e3fp = torch.full((text_len, fp_dim), self.e3fp_pad_id, dtype=torch.long)
                final_e3fp = torch.cat([dummy_e3fp, e3fp_ids])
                dummy_map = torch.full((text_len,), -1, dtype=torch.long)
                final_map = torch.cat([dummy_map, shifted_map])

            elif task == "text2mol":
                # 翻译：看文本 -> 生成结构
                input_ids = text_ids
                labels = motif_ids

                final_e3fp = torch.empty((0, fp_dim), dtype=torch.long)
                final_map = torch.empty((0,), dtype=torch.long)

            elif task == "denoise":
                # 纯文本填空
                masked_text_ids, text_labels = self._mask_sequence(text_ids, self.text_weight_lookup, self.text_pad_id)
                input_ids = masked_text_ids
                labels = text_labels

                final_e3fp = torch.empty((0, fp_dim), dtype=torch.long)
                final_map = torch.empty((0,), dtype=torch.long)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)

            # 防御：确保 0 原子的分子 e3fp 至少有 [0, 5] 维度
            if final_e3fp.dim() == 1: final_e3fp = final_e3fp.unsqueeze(0)
            batch_e3fp_ids.append(final_e3fp)
            batch_atom_maps.append(final_map)

            atom_mask = (final_e3fp[:, 0] != self.e3fp_pad_id).long() if final_e3fp.shape[0] > 0 else torch.empty((0,),
                                                                                                                  dtype=torch.long)
            batch_atom_masks.append(atom_mask)

        # 全局动态 Pad (以 T5 文本 pad id 为主，因为序列基本都是文本开头)
        input_ids_padded = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.text_pad_id)
        attention_mask = (input_ids_padded != self.text_pad_id).long()
        labels_padded = pad_sequence(batch_labels, batch_first=True, padding_value=-100)

        e3fp_padded = pad_sequence(batch_e3fp_ids, batch_first=True, padding_value=self.e3fp_pad_id)
        map_padded = pad_sequence(batch_atom_maps, batch_first=True, padding_value=-1)
        atom_mask_padded = pad_sequence(batch_atom_masks, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "e3fp_ids": e3fp_padded,
            "atom_attention_mask": atom_mask_padded,
            "atom_to_motif_map": map_padded,
        }