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
# 1. 核心数据集类 (支持多任务调度与动态截断)
# ==========================================
class GSMATDataset(Dataset):
    def __init__(self, lmdb_path: str,
                 text_tokenizer: TextTokenizer,
                 motif_tokenizer: MotifTokenizer,
                 e3fp_tokenizer: E3FPTokenizer,
                 c4_lmdb_path: str = "",
                 whitelist_path: str = None,
                 max_seq_length: int = 768,
                 task_probs: Dict[str, float] = None):

        self.lmdb_path = lmdb_path
        self.c4_lmdb_path = c4_lmdb_path
        self.text_tokenizer = text_tokenizer
        self.motif_tokenizer = motif_tokenizer
        self.e3fp_tokenizer = e3fp_tokenizer

        self.e3fp_width = self.e3fp_tokenizer.fp_level + 1
        self.error_count = 0
        self.max_log_errors = 50
        self.max_seq_len = max_seq_length

        # 支持外部传入特定的任务概率 (Phase 1 纯算 MMM，Phase 2 混合)
        if task_probs:
            self.task_probs = task_probs
        else:
            self.task_probs = {
                "mmm": 0.60,
                "caption": 0.15,
                "text2mol": 0.10,
                "denoise": 0.15
            }

        self.whitelist = set()
        if whitelist_path and os.path.exists(whitelist_path):
            with open(whitelist_path, 'r', encoding='utf-8') as f:
                self.whitelist = set(json.load(f))
            logger.info(f"🛡️ 开启绝对白名单保护：仅允许 {len(self.whitelist):,} 个分子参与预训练。")
        else:
            logger.info("🔓 未配置白名单，允许全量数据参与训练。")

        is_subdir = os.path.isdir(lmdb_path)
        temp_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=is_subdir)
        with temp_env.begin() as txn:
            logger.info("📋 正在扫描主数据库中的真实 Key 映射表...")
            self.keys = [k for k in txn.cursor().iternext(keys=True, values=False) if k != b'__len__']
            self.length = len(self.keys)
        temp_env.close()
        self.env = None
        logger.info(f"✅ 主数据库 Key 映射表建立完成，共 {self.length:,} 条有效数据。")

        self.c4_length = 0
        self.c4_env = None
        if self.c4_lmdb_path and os.path.exists(self.c4_lmdb_path):
            temp_c4_env = lmdb.open(self.c4_lmdb_path, readonly=True, lock=False, subdir=False)
            with temp_c4_env.begin() as txn:
                try:
                    self.c4_length = int(txn.get(b'__len__'))
                except:
                    self.c4_length = txn.stat()['entries']
            temp_c4_env.close()
            logger.info(f"📚 成功挂载 C4 弹药库，可用数据量: {self.c4_length:,}")
        else:
            logger.warning(f"⚠️ 未找到 C4 库，降噪任务将回退使用自然语言。")

    def __len__(self):
        return self.length

    def _init_db(self):
        is_subdir = os.path.isdir(self.lmdb_path)
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False,
                             subdir=is_subdir)
        if self.c4_length > 0:
            self.c4_env = lmdb.open(self.c4_lmdb_path, readonly=True, lock=False, subdir=False)

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
        last_error = None

        for attempt in range(10):
            try:
                roll = np.random.rand()

                # 任务分配路由
                if roll >= sum(self.task_probs.values()) - self.task_probs.get("denoise", 0) and getattr(self,
                                                                                                         'c4_length',
                                                                                                         0) > 0:
                    task = "denoise"
                    c4_idx = np.random.randint(0, self.c4_length)
                    with self.c4_env.begin() as txn:
                        c4_data = pickle.loads(txn.get(str(c4_idx).encode()))
                        text = c4_data.get('text', '')
                    prompt_text = f"[Denoise]: {text}"
                    smiles = ""
                else:
                    with self.env.begin() as txn:
                        real_key = self.keys[idx]
                        data = txn.get(real_key)

                        if data is None:
                            idx = (idx + 1) % self.length
                            continue
                        entry = pickle.loads(data)

                    if getattr(self, 'whitelist', None):
                        cid = str(entry.get('cid', entry.get('index', entry.get('id', entry.get('input', ''))))).strip()
                        if cid and cid not in self.whitelist:
                            idx = (idx + 1) % self.length
                            continue

                    smiles = entry.get('smiles_kekule') or entry.get('smiles', '')
                    text = entry.get('enriched_description', '') or entry.get('description', '') or entry.get('text',
                                                                                                              '')

                    if roll < self.task_probs.get("mmm", 1.0):
                        task = "mmm"
                        prompt_text = f"[MMM]: {text}" if text else "[MMM]:"
                    elif roll < self.task_probs.get("mmm", 0) + self.task_probs.get("caption", 0):
                        task = "caption"
                        prompt_text = f"[Caption]: Generate description for the molecule:"
                    elif roll < sum(self.task_probs.values()) - self.task_probs.get("denoise", 0):
                        task = "text2mol"
                        prompt_text = f"[Text2Mol]: {text}"
                    else:
                        task = "denoise"
                        prompt_text = f"[Denoise]: {text}"
                        smiles = ""

                # 文本 Tokenizer
                text_enc = self.text_tokenizer(prompt_text, padding=False, truncation=True, max_length=self.max_seq_len,
                                               return_tensors="pt")
                text_ids = text_enc['input_ids'].squeeze(0)
                len_t = text_ids.shape[0]

                target_text_ids = torch.tensor([], dtype=torch.long)
                if task == "caption":
                    target_enc = self.text_tokenizer(text, padding=False, truncation=True, max_length=self.max_seq_len,
                                                     return_tensors="pt")
                    target_text_ids = target_enc['input_ids'].squeeze(0)

                # Motif Tokenizer
                if task != "denoise" and smiles:
                    motif_result = self.motif_tokenizer.encode(smiles, return_tensors='pt', padding=False,
                                                               return_mapping=True)
                    motif_ids, motif_mapping = motif_result if isinstance(motif_result, tuple) else (motif_result, [])
                    if motif_ids.dim() > 1: motif_ids = motif_ids.squeeze(0)
                    len_m = motif_ids.shape[0]
                else:
                    motif_ids = torch.tensor([], dtype=torch.long)
                    motif_mapping = []
                    len_m = 0

                # ⚡ 触发动态注水分配 (极速张量切片)
                if len_t + len_m > self.max_seq_len:
                    half_quota = self.max_seq_len // 2
                    if len_t < half_quota:
                        allow_m = self.max_seq_len - len_t
                        motif_ids = motif_ids[:allow_m]
                        if len(motif_ids) > 0: motif_ids[-1] = self.motif_tokenizer.eom_id
                    elif len_m < half_quota:
                        allow_t = self.max_seq_len - len_m
                        text_ids = text_ids[:allow_t]
                        if task == "caption":
                            target_text_ids = target_text_ids[:allow_t]
                    else:
                        allow_t = half_quota
                        allow_m = self.max_seq_len - allow_t
                        text_ids = text_ids[:allow_t]
                        if task == "caption":
                            target_text_ids = target_text_ids[:allow_t]
                        motif_ids = motif_ids[:allow_m]
                        if len(motif_ids) > 0: motif_ids[-1] = self.motif_tokenizer.eom_id

                    len_t = text_ids.shape[0]
                    len_m = motif_ids.shape[0]

                # 3D E3FP 处理
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

                        if real_token_idx < len_m - 1:
                            if isinstance(atom_indices, int): atom_indices = [atom_indices]
                            for atom_idx in atom_indices:
                                if atom_idx < num_atoms:
                                    atom_to_motif_map[atom_idx] = real_token_idx
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
                last_error = e
                if getattr(self, 'error_count', 0) < getattr(self, 'max_log_errors', 50):
                    logger.warning(f"⚠️ [Data Error] 样本 {idx} 解析失败: {e}. 正在重试下一个...")
                    self.error_count = getattr(self, 'error_count', 0) + 1
                idx = (idx + 1) % self.length

        raise RuntimeError(f"数据集连续读取失败超过 10 次！致命 Bug 或 LMDB 已损坏: {last_error}")


# ==========================================
# 2. 基础 Collator 与非坍缩掩码核心逻辑
# ==========================================
class BaseGSMATCollator:
    """提供通用的重要性计算与 T5 原生非坍缩掩码方法"""

    def __init__(self, motif_tokenizer, text_tokenizer, text_weight_path=None, e3fp_pad_id=-1, mask_ratio=0.15,
                 task_b_ratio=0.15, is_train=True):
        self.motif_tokenizer = motif_tokenizer
        self.text_tokenizer = text_tokenizer
        self.motif_pad_id = motif_tokenizer.pad_id
        self.text_pad_id = text_tokenizer.tokenizer.pad_token_id
        self.e3fp_pad_id = e3fp_pad_id
        self.mask_ratio = mask_ratio
        self.task_b_ratio = task_b_ratio
        self.is_train = is_train

        # 初始化 Motif 权重
        try:
            from model.CAMT5.representation import Frag
            self.frag_processor = Frag()
        except ImportError:
            self.frag_processor = None

        motif_vocab_size = len(motif_tokenizer.tokenizer)
        self.motif_weight_lookup = torch.zeros(motif_vocab_size, dtype=torch.float32)
        for token_str, token_id in motif_tokenizer.tokenizer.get_vocab().items():
            # 保护所有特殊符号（包括锚点 <1*> 等）
            if token_str in motif_tokenizer.tokenizer.all_special_tokens:
                self.motif_weight_lookup[token_id] = 0.0
            else:
                try:
                    self.motif_weight_lookup[token_id] = float(
                        self.frag_processor.get_size(token_str)) if self.frag_processor else 1.0
                except:
                    self.motif_weight_lookup[token_id] = 0.01
        self.motif_weight_lookup = torch.log1p(self.motif_weight_lookup)

        # 初始化文本权重
        text_vocab_size = text_tokenizer.tokenizer.vocab_size
        self.text_weight_lookup = torch.zeros(text_vocab_size + 1000, dtype=torch.float32) + 1.0
        if text_weight_path and os.path.exists(text_weight_path):
            with open(text_weight_path, 'r') as f:
                text_weights = json.load(f)
            for k, v in text_weights.items():
                curr_id = int(k)
                if curr_id < len(self.text_weight_lookup):
                    self.text_weight_lookup[curr_id] = float(v)

    def _apply_non_collapsing_t5_mask(self, input_ids: torch.Tensor, weight_lookup: torch.Tensor, pad_id: int,
                                      tokenizer, starting_counter=0):
        """
        🚀 SOTA 核心：非坍缩独立哨兵掩码 (Non-Collapsing Sentinel Masking)
        保证输入序列长度绝对不变（稳固 3D 锚点映射），同时输出标准的 T5 Decoder 序列格式。
        """
        seq_len = input_ids.shape[0]
        if seq_len == 0:
            return input_ids, [], torch.zeros(0, dtype=torch.bool), starting_counter

        safe_ids = torch.clamp(input_ids, max=weight_lookup.shape[0] - 1)
        weights = weight_lookup[safe_ids].clone()
        weights[input_ids == pad_id] = 0.0

        for sp_id in tokenizer.all_special_ids:
            weights[input_ids == sp_id] = 0.0

        mlm_mask_pos = torch.zeros(seq_len, dtype=torch.bool)
        geo_mask_pos = torch.zeros(seq_len, dtype=torch.bool)

        valid_len = (input_ids != pad_id).sum().item()

        # 1. 抽取 MLM 掩码 (同时影响 1D 和 3D)
        weights_A = weights.clone()
        if weights_A.sum() > 0:
            num_A = max(1, int(valid_len * self.mask_ratio))
            num_A = min(num_A, (weights_A > 0).sum().item())
            if num_A > 0:
                idx_A = torch.multinomial(weights_A + 1e-6, num_samples=num_A, replacement=False)
                mlm_mask_pos[idx_A] = True
                geo_mask_pos[idx_A] = True

        # 2. 抽取纯 3D 几何掩码 (仅清空 3D E3FP，逼迫模型从 1D 预测 3D)
        weights_B = weights.clone()
        weights_B[mlm_mask_pos] = 0.0
        if weights_B.sum() > 0:
            num_B = max(1, int(valid_len * self.task_b_ratio))
            num_B = min(num_B, (weights_B > 0).sum().item())
            if num_B > 0:
                idx_B = torch.multinomial(weights_B + 1e-6, num_samples=num_B, replacement=False)
                geo_mask_pos[idx_B] = True

        # 3. 构建符合 T5 Decoder 的输出 Labels，但不缩减 Input 长度
        masked_input_ids = input_ids.clone()
        labels = []

        base_sentinel_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
        if base_sentinel_id is None or base_sentinel_id == tokenizer.unk_token_id:
            base_sentinel_id = 32099

        counter = starting_counter
        for i in range(seq_len):
            if mlm_mask_pos[i]:
                current_sentinel = base_sentinel_id - counter
                masked_input_ids[i] = current_sentinel
                labels.extend([current_sentinel, input_ids[i].item()])
                counter += 1

        return masked_input_ids, labels, geo_mask_pos, counter


# ==========================================
# 3. Phase 1 专属 Collator (纯粹的 1D-3D 几何对齐)
# ==========================================
class GSMATPretrainingCollator(BaseGSMATCollator):
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch: return {}

        batch_input_ids, batch_labels = [], []
        batch_e3fp_ids, batch_unmasked_e3fp = [], []
        batch_atom_maps, batch_atom_masks, batch_mask_positions = [], [], []

        base_sentinel_id = self.motif_tokenizer.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        if base_sentinel_id is None: base_sentinel_id = 32099

        for item in valid_batch:
            motif_ids = item['motif_input_ids']
            e3fp_ids = item['e3fp_input_ids']
            atom_map = item['atom_to_motif_map']

            masked_motif_ids, labels_list, geo_mask_pos, counter = self._apply_non_collapsing_t5_mask(
                motif_ids, self.motif_weight_lookup, self.motif_pad_id, self.motif_tokenizer.tokenizer, 0
            )

            if counter > 0:
                labels_list.append(base_sentinel_id - counter)
            labels_tensor = torch.tensor(labels_list, dtype=torch.long)

            masked_e3fp_ids = e3fp_ids.clone()
            mask_indices = geo_mask_pos.nonzero(as_tuple=True)[0]
            for m_idx in mask_indices:
                atom_indices = (atom_map == m_idx).nonzero(as_tuple=True)[0]
                if len(atom_indices) > 0:
                    masked_e3fp_ids[atom_indices] = self.e3fp_pad_id

            batch_input_ids.append(masked_motif_ids)
            batch_labels.append(labels_tensor)
            batch_e3fp_ids.append(masked_e3fp_ids)
            batch_unmasked_e3fp.append(e3fp_ids.clone())
            batch_atom_maps.append(atom_map)
            batch_mask_positions.append(geo_mask_pos)

            atom_mask = (masked_e3fp_ids[:, 0] != self.e3fp_pad_id).long() if masked_e3fp_ids.shape[
                                                                                  0] > 0 else torch.zeros(0,
                                                                                                          dtype=torch.long)
            batch_atom_masks.append(atom_mask)

        input_ids_padded = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.motif_pad_id)
        attention_mask = (input_ids_padded != self.motif_pad_id).long()
        labels_padded = pad_sequence(batch_labels, batch_first=True, padding_value=-100)

        e3fp_padded = pad_sequence(batch_e3fp_ids, batch_first=True, padding_value=self.e3fp_pad_id)
        unmasked_e3fp_padded = pad_sequence(batch_unmasked_e3fp, batch_first=True, padding_value=self.e3fp_pad_id)
        map_padded = pad_sequence(batch_atom_maps, batch_first=True, padding_value=-1)
        atom_mask_padded = pad_sequence(batch_atom_masks, batch_first=True, padding_value=0)
        mask_positions_padded = pad_sequence(batch_mask_positions, batch_first=True, padding_value=False)

        safe_map = map_padded.clone()
        safe_map[safe_map >= input_ids_padded.shape[1]] = -1

        # 🚀 E3FP 壳层随机失活
        if self.is_train:
            batch_size = e3fp_padded.shape[0]
            rand_probs = torch.rand(batch_size)
            mask_l3 = rand_probs < 0.15
            if mask_l3.any(): e3fp_padded[mask_l3, :, 3] = self.e3fp_pad_id
            mask_l2 = rand_probs < 0.05
            if mask_l2.any(): e3fp_padded[mask_l2, :, 2:] = self.e3fp_pad_id

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "e3fp_ids": e3fp_padded,
            "unmasked_e3fp_ids": unmasked_e3fp_padded,
            "atom_attention_mask": atom_mask_padded,
            "atom_to_motif_map": safe_map,
            "mask_positions": mask_positions_padded
        }


# ==========================================
# 4. Phase 2 跨模态大一统 Collator (包含四大任务)
# ==========================================
class GSMATPhase2Collator(BaseGSMATCollator):
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch: return {}

        batch_input_ids, batch_labels = [], []
        batch_e3fp_ids, batch_atom_maps, batch_atom_masks = [], [], []
        batch_mask_positions, batch_unmasked_e3fp = [], []

        base_sentinel_id = 32099

        for item in valid_batch:
            task = item["task"]
            text_ids = item["text_input_ids"]
            motif_ids = item["motif_input_ids"]
            e3fp_ids = item["e3fp_input_ids"]
            atom_map = item["atom_to_motif_map"]

            fp_dim = e3fp_ids.shape[1] if e3fp_ids.shape[0] > 0 else 4

            if task == "mmm":
                masked_text_ids, text_labels, _, counter = self._apply_non_collapsing_t5_mask(
                    text_ids, self.text_weight_lookup, self.text_pad_id, self.text_tokenizer.tokenizer, 0
                )
                masked_motif_ids, motif_labels, geo_mask_pos, counter = self._apply_non_collapsing_t5_mask(
                    motif_ids, self.motif_weight_lookup, self.motif_pad_id, self.motif_tokenizer.tokenizer, counter
                )

                input_ids = torch.cat([masked_text_ids, masked_motif_ids])

                all_labels = text_labels + motif_labels
                if counter > 0: all_labels.append(base_sentinel_id - counter)
                labels = torch.tensor(all_labels, dtype=torch.long)

                text_len = len(text_ids)
                shifted_map = atom_map.clone()
                shifted_map[shifted_map != -1] += text_len

                masked_e3fp_ids = e3fp_ids.clone()
                mask_indices = geo_mask_pos.nonzero(as_tuple=True)[0]
                for m_idx in mask_indices:
                    atom_indices = (atom_map == m_idx).nonzero(as_tuple=True)[0]
                    if len(atom_indices) > 0:
                        masked_e3fp_ids[atom_indices] = self.e3fp_pad_id

                dummy_e3fp = torch.full((text_len, fp_dim), self.e3fp_pad_id, dtype=torch.long)
                final_e3fp = torch.cat([dummy_e3fp, masked_e3fp_ids])
                final_unmasked_e3fp = torch.cat([dummy_e3fp, e3fp_ids])

                dummy_map = torch.full((text_len,), -1, dtype=torch.long)
                final_map = torch.cat([dummy_map, shifted_map])

                item_mask_pos = torch.cat([torch.zeros(text_len, dtype=torch.bool), geo_mask_pos])

            elif task == "caption":
                input_ids = torch.cat([text_ids, motif_ids])
                labels = item["target_text_ids"]
                text_len = len(text_ids)
                shifted_map = atom_map.clone()
                shifted_map[shifted_map != -1] += text_len

                dummy_e3fp = torch.full((text_len, fp_dim), self.e3fp_pad_id, dtype=torch.long)
                final_e3fp = torch.cat([dummy_e3fp, e3fp_ids])
                final_unmasked_e3fp = final_e3fp.clone()

                dummy_map = torch.full((text_len,), -1, dtype=torch.long)
                final_map = torch.cat([dummy_map, shifted_map])
                item_mask_pos = torch.zeros(len(input_ids), dtype=torch.bool)

            elif task == "text2mol":
                input_ids = text_ids
                labels = motif_ids
                final_e3fp = torch.empty((0, fp_dim), dtype=torch.long)
                final_unmasked_e3fp = final_e3fp.clone()
                final_map = torch.empty((0,), dtype=torch.long)
                item_mask_pos = torch.zeros(len(input_ids), dtype=torch.bool)

            elif task == "denoise":
                masked_text_ids, text_labels, _, counter = self._apply_non_collapsing_t5_mask(
                    text_ids, self.text_weight_lookup, self.text_pad_id, self.text_tokenizer.tokenizer, 0
                )
                input_ids = masked_text_ids
                if counter > 0: text_labels.append(base_sentinel_id - counter)
                labels = torch.tensor(text_labels, dtype=torch.long)

                final_e3fp = torch.empty((0, fp_dim), dtype=torch.long)
                final_unmasked_e3fp = final_e3fp.clone()
                final_map = torch.empty((0,), dtype=torch.long)
                item_mask_pos = torch.zeros(len(input_ids), dtype=torch.bool)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_e3fp_ids.append(final_e3fp if final_e3fp.dim() == 2 else final_e3fp.unsqueeze(0))
            batch_unmasked_e3fp.append(
                final_unmasked_e3fp if final_unmasked_e3fp.dim() == 2 else final_unmasked_e3fp.unsqueeze(0))
            batch_atom_maps.append(final_map)
            batch_mask_positions.append(item_mask_pos)

            atom_mask = (final_e3fp[:, 0] != self.e3fp_pad_id).long() if final_e3fp.shape[0] > 0 else torch.empty((0,),
                                                                                                                  dtype=torch.long)
            batch_atom_masks.append(atom_mask)

        input_ids_padded = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.text_pad_id)
        attention_mask = (input_ids_padded != self.text_pad_id).long()
        labels_padded = pad_sequence(batch_labels, batch_first=True, padding_value=-100)

        e3fp_padded = pad_sequence(batch_e3fp_ids, batch_first=True, padding_value=self.e3fp_pad_id)
        unmasked_e3fp_padded = pad_sequence(batch_unmasked_e3fp, batch_first=True, padding_value=self.e3fp_pad_id)
        map_padded = pad_sequence(batch_atom_maps, batch_first=True, padding_value=-1)
        atom_mask_padded = pad_sequence(batch_atom_masks, batch_first=True, padding_value=0)
        mask_positions_padded = pad_sequence(batch_mask_positions, batch_first=True, padding_value=False)

        safe_map = map_padded.clone()
        safe_map[safe_map >= input_ids_padded.shape[1]] = -1

        # 🚀 E3FP 壳层随机失活
        if self.is_train:
            batch_size = e3fp_padded.shape[0]
            rand_probs = torch.rand(batch_size)
            mask_l3 = rand_probs < 0.15
            if mask_l3.any(): e3fp_padded[mask_l3, :, 3] = self.e3fp_pad_id
            mask_l2 = rand_probs < 0.05
            if mask_l2.any(): e3fp_padded[mask_l2, :, 2:] = self.e3fp_pad_id

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "e3fp_ids": e3fp_padded,
            "unmasked_e3fp_ids": unmasked_e3fp_padded,
            "atom_attention_mask": atom_mask_padded,
            "atom_to_motif_map": safe_map,
            "mask_positions": mask_positions_padded,
        }