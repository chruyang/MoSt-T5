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
# 1. 核心数据集类 (满血版：修复 Tokenizer 包装限制)
# ==========================================
class GSMATDataset(Dataset):
    def __init__(self, lmdb_path: str,
                 text_tokenizer: TextTokenizer,
                 motif_tokenizer: MotifTokenizer,
                 e3fp_tokenizer: E3FPTokenizer,
                 c4_lmdb_path: str = "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/c4_pretrain.lmdb"):

        self.lmdb_path = lmdb_path
        self.c4_lmdb_path = c4_lmdb_path

        self.text_tokenizer = text_tokenizer
        self.motif_tokenizer = motif_tokenizer
        self.e3fp_tokenizer = e3fp_tokenizer

        self.e3fp_width = self.e3fp_tokenizer.fp_level + 1
        self.error_count = 0
        self.max_log_errors = 50
        self.max_seq_len = 512

        self.task_probs = {
            "mmm": 0.60,
            "caption": 0.15,
            "text2mol": 0.10,
            "denoise": 0.15
        }

        self.whitelist = set()
        whitelist_path = os.path.expanduser("~/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchemqc/pretrain_whitelist.json")
        if os.path.exists(whitelist_path):
            with open(whitelist_path, 'r', encoding='utf-8') as f:
                self.whitelist = set(json.load(f))
            logger.info(f"🛡️ 开启绝对白名单保护：仅允许 {len(self.whitelist):,} 个分子参与预训练。")

        is_subdir = os.path.isdir(lmdb_path)
        temp_env = lmdb.open(
            lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=is_subdir
        )
        with temp_env.begin() as txn:
            try:
                self.length = int(txn.get(b'__len__'))
            except (TypeError, ValueError):
                self.length = txn.stat()['entries']
        temp_env.close()
        self.env = None

        self.c4_length = 0
        self.c4_env = None
        if os.path.exists(self.c4_lmdb_path):
            temp_c4_env = lmdb.open(self.c4_lmdb_path, readonly=True, lock=False, subdir=False)
            with temp_c4_env.begin() as txn:
                try:
                    self.c4_length = int(txn.get(b'__len__'))
                except:
                    self.c4_length = txn.stat()['entries']
            temp_c4_env.close()
            logger.info(f"📚 成功挂载 C4 弹药库，可用数据量: {self.c4_length:,}")
        else:
            logger.warning(f"⚠️ 未找到 C4 库 ({self.c4_lmdb_path})，降噪任务将回退使用 PubChem。")

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

        try:
            roll = np.random.rand()

            # 🚀 分流：前往 C4 库抽取纯英文文本
            if roll >= sum(self.task_probs.values()) - self.task_probs["denoise"] and self.c4_length > 0:
                task = "denoise"
                c4_idx = np.random.randint(0, self.c4_length)
                with self.c4_env.begin() as txn:
                    c4_data = pickle.loads(txn.get(str(c4_idx).encode()))
                    text = c4_data.get('text', '')
                prompt_text = f"[Denoise]: {text}"
                smiles = ""

            # 🧪 常规：抽取 PubChem 分子
            else:
                with self.env.begin() as txn:
                    data = txn.get(str(idx).encode())
                    if data is None: return self.__getitem__((idx + 1) % self.length)
                    entry = pickle.loads(data)

                if self.whitelist:
                    cid = str(entry.get('cid', entry.get('index', entry.get('id', entry.get('input', ''))))).strip()
                    if cid and cid not in self.whitelist:
                        return self.__getitem__((idx + 1) % self.length)

                smiles = entry.get('smiles_kekule') or entry.get('smiles', '')
                text = entry.get('enriched_description', '') or entry.get('description', '') or entry.get('text', '')

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
                    smiles = ""

            # 🧠 动态注水截断 (Water-filling)
            max_total_len = self.max_seq_len

            # 🌟 修复关键点：直接调用底层的 `.tokenizer` 避开自建包装类的严格参数检查
            text_enc = self.text_tokenizer.tokenizer(prompt_text, padding=False, truncation=True,
                                                     max_length=max_total_len, return_tensors="pt")
            text_ids = text_enc['input_ids'].squeeze(0)
            len_t = text_ids.shape[0]

            target_text_ids = torch.tensor([], dtype=torch.long)
            if task == "caption":
                target_enc = self.text_tokenizer.tokenizer(text, padding=False, truncation=True,
                                                           max_length=max_total_len, return_tensors="pt")
                target_text_ids = target_enc['input_ids'].squeeze(0)

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

            # 触发动态注水分配
            if len_t + len_m > max_total_len:
                half_quota = max_total_len // 2
                if len_t < half_quota:
                    allow_m = max_total_len - len_t
                    motif_ids = motif_ids[:allow_m]
                elif len_m < half_quota:
                    allow_t = max_total_len - len_m
                    text_enc = self.text_tokenizer.tokenizer(prompt_text, padding=False, truncation=True,
                                                             max_length=allow_t, return_tensors="pt")
                    text_ids = text_enc['input_ids'].squeeze(0)
                    if task == "caption":
                        target_enc = self.text_tokenizer.tokenizer(text, padding=False, truncation=True,
                                                                   max_length=allow_t, return_tensors="pt")
                        target_text_ids = target_enc['input_ids'].squeeze(0)
                else:
                    allow_t = half_quota
                    allow_m = half_quota
                    text_enc = self.text_tokenizer.tokenizer(prompt_text, padding=False, truncation=True,
                                                             max_length=allow_t, return_tensors="pt")
                    text_ids = text_enc['input_ids'].squeeze(0)
                    if task == "caption":
                        target_enc = self.text_tokenizer.tokenizer(text, padding=False, truncation=True,
                                                                   max_length=allow_t, return_tensors="pt")
                        target_text_ids = target_enc['input_ids'].squeeze(0)
                    motif_ids = motif_ids[:allow_m]

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
                    if real_token_idx < motif_ids.shape[0]:
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
            # 引入保险丝：打印错误，避免无尽沉默地抓取
            if self.error_count < self.max_log_errors:
                logger.warning(f"⚠️ [Data Error] 样本 {idx} 解析失败: {e}. 正在重采样...")
                self.error_count += 1
            return self.__getitem__((idx + 1) % self.length)


# ==========================================
# 2. 预训练 Phase 1 Collator
# ==========================================
class GSMATPretrainingCollator:
    def __init__(self, motif_tokenizer: MotifTokenizer, e3fp_pad_id: int = -1, mask_ratio: float = 0.15,
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
                except:
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

            mask_idx_A = torch.multinomial(row_weight + 1e-6, num_samples=num_to_mask_A, replacement=False)
            mlm_mask_positions[i, mask_idx_A] = True
            geometric_mask_positions[i, mask_idx_A] = True
            masked_motif_ids[i, mask_idx_A] = self.mask_token_id

            row_weight_B = row_weight.clone()
            row_weight_B[mask_idx_A] = 0.0

            if row_weight_B.sum() > 0:
                actual_num_B = min(num_to_mask_B, (row_weight_B > 0).sum().item())
                if actual_num_B > 0:
                    mask_idx_B = torch.multinomial(row_weight_B + 1e-6, num_samples=actual_num_B, replacement=False)
                    geometric_mask_positions[i, mask_idx_B] = True

            all_mask_idx = torch.cat([mask_idx_A, mask_idx_B] if row_weight_B.sum() > 0 else [mask_idx_A])
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
# 3. 🌟 Phase 2 全新大一统多模态 Collator
# ==========================================
class GSMATPhase2Collator:
    def __init__(self, motif_tokenizer, text_tokenizer, text_weight_path, e3fp_pad_id=-1, mask_ratio=0.15):
        self.motif_pad_id = motif_tokenizer.pad_id
        self.text_pad_id = text_tokenizer.tokenizer.pad_token_id
        self.e3fp_pad_id = e3fp_pad_id
        self.mask_ratio = mask_ratio

        self.mask_token_id = motif_tokenizer.tokenizer.convert_tokens_to_ids("<extra_id_0>")

        try:
            from model.CAMT5.representation import Frag
            self.frag_processor = Frag()
        except ImportError:
            self.frag_processor = None

        motif_vocab_size = len(motif_tokenizer.tokenizer)
        self.motif_weight_lookup = torch.zeros(motif_vocab_size, dtype=torch.float32)
        for token_str, token_id in motif_tokenizer.tokenizer.get_vocab().items():
            if token_str in motif_tokenizer.tokenizer.all_special_tokens:
                self.motif_weight_lookup[token_id] = 0.0
            else:
                try:
                    self.motif_weight_lookup[token_id] = float(
                        self.frag_processor.get_size(token_str)) if self.frag_processor else 1.0
                except:
                    self.motif_weight_lookup[token_id] = 0.01
        self.motif_weight_lookup = torch.log1p(self.motif_weight_lookup)

        text_vocab_size = text_tokenizer.tokenizer.vocab_size
        self.text_weight_lookup = torch.zeros(text_vocab_size + 1000, dtype=torch.float32) + 1.0
        if os.path.exists(text_weight_path):
            with open(text_weight_path, 'r') as f:
                text_weights = json.load(f)
            for k, v in text_weights.items():
                curr_id = int(k)
                if curr_id < len(self.text_weight_lookup):
                    self.text_weight_lookup[curr_id] = float(v)

    def _mask_sequence(self, input_ids: torch.Tensor, weight_lookup: torch.Tensor, pad_id: int):
        seq_len = input_ids.shape[0]
        if seq_len == 0: return input_ids, input_ids.clone()

        safe_ids = torch.clamp(input_ids, max=weight_lookup.shape[0] - 1)
        weights = weight_lookup[safe_ids].clone()
        weights[input_ids == pad_id] = 0.0

        masked_ids = input_ids.clone()
        labels = input_ids.clone()
        labels[labels == pad_id] = -100

        if weights.sum() > 0:
            num_to_mask = max(1, int((input_ids != pad_id).sum() * self.mask_ratio))
            num_to_mask = min(num_to_mask, (weights > 0).sum().item())
            if num_to_mask > 0:
                mask_idx = torch.multinomial(weights + 1e-6, num_samples=num_to_mask, replacement=False)
                masked_ids[mask_idx] = self.mask_token_id

        return masked_ids, labels

    def _mask_motif_and_e3fp(self, motif_ids: torch.Tensor, e3fp_ids: torch.Tensor, atom_map: torch.Tensor):
        seq_len = motif_ids.shape[0]
        if seq_len == 0:
            return motif_ids, motif_ids.clone(), e3fp_ids, torch.zeros(0, dtype=torch.bool)

        weights = self.motif_weight_lookup[torch.clamp(motif_ids, max=self.motif_weight_lookup.shape[0] - 1)].clone()
        weights[motif_ids == self.motif_pad_id] = 0.0

        masked_motif_ids = motif_ids.clone()
        masked_e3fp_ids = e3fp_ids.clone()

        labels = motif_ids.clone()
        labels[labels == self.motif_pad_id] = -100

        motif_3d_mask_pos = torch.zeros_like(motif_ids, dtype=torch.bool)

        if weights.sum() > 0:
            valid_len = (motif_ids != self.motif_pad_id).sum().item()
            num_to_mask_A = max(1, int(valid_len * self.mask_ratio))
            num_to_mask_B = max(1, int(valid_len * 0.15))

            actual_A = min(num_to_mask_A, (weights > 0).sum().item())
            if actual_A > 0:
                mask_idx_A = torch.multinomial(weights + 1e-6, num_samples=actual_A, replacement=False)
                masked_motif_ids[mask_idx_A] = self.mask_token_id
            else:
                mask_idx_A = torch.tensor([], dtype=torch.long, device=weights.device)

            weights_B = weights.clone()
            weights_B[mask_idx_A] = 0.0
            actual_B = min(num_to_mask_B, (weights_B > 0).sum().item())
            if actual_B > 0:
                mask_idx_B = torch.multinomial(weights_B + 1e-6, num_samples=actual_B, replacement=False)
            else:
                mask_idx_B = torch.tensor([], dtype=torch.long, device=weights.device)

            all_mask_idx = torch.cat([mask_idx_A, mask_idx_B])
            valid_3d_mask_idx = []

            for m_idx in all_mask_idx:
                atom_indices = (atom_map == m_idx).nonzero(as_tuple=True)[0]
                if len(atom_indices) > 0:
                    masked_e3fp_ids[atom_indices] = self.e3fp_pad_id
                    valid_3d_mask_idx.append(m_idx)

            if valid_3d_mask_idx:
                motif_3d_mask_pos[torch.tensor(valid_3d_mask_idx, dtype=torch.long, device=weights.device)] = True

        return masked_motif_ids, labels, masked_e3fp_ids, motif_3d_mask_pos

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids, batch_labels = [], []
        batch_e3fp_ids, batch_atom_maps, batch_atom_masks = [], [], []
        batch_mask_positions, batch_unmasked_e3fp = [], []

        for item in batch:
            task = item["task"]
            text_ids = item["text_input_ids"]
            motif_ids = item["motif_input_ids"]
            e3fp_ids = item["e3fp_input_ids"]
            atom_map = item["atom_to_motif_map"]

            # ✅ 动态维度获取：永不报错
            fp_dim = e3fp_ids.shape[1] if e3fp_ids.shape[0] > 0 else 4

            if task == "mmm":
                masked_text_ids, text_labels = self._mask_sequence(text_ids, self.text_weight_lookup, self.text_pad_id)
                masked_motif_ids, motif_labels, masked_e3fp_ids, motif_3d_mask_pos = self._mask_motif_and_e3fp(
                    motif_ids, e3fp_ids, atom_map)

                input_ids = torch.cat([masked_text_ids, masked_motif_ids])
                labels = torch.cat([text_labels, motif_labels])

                text_len = len(text_ids)
                shifted_map = atom_map.clone()
                shifted_map[shifted_map != -1] += text_len

                dummy_e3fp = torch.full((text_len, fp_dim), self.e3fp_pad_id, dtype=torch.long)
                final_e3fp = torch.cat([dummy_e3fp, masked_e3fp_ids])
                final_unmasked_e3fp = torch.cat([dummy_e3fp, e3fp_ids])

                dummy_map = torch.full((text_len,), -1, dtype=torch.long)
                final_map = torch.cat([dummy_map, shifted_map])

                text_mask_pos = torch.zeros(text_len, dtype=torch.bool)
                item_mask_pos = torch.cat([text_mask_pos, motif_3d_mask_pos])

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
                masked_text_ids, text_labels = self._mask_sequence(text_ids, self.text_weight_lookup, self.text_pad_id)
                input_ids = masked_text_ids
                labels = text_labels

                final_e3fp = torch.empty((0, fp_dim), dtype=torch.long)
                final_unmasked_e3fp = final_e3fp.clone()
                final_map = torch.empty((0,), dtype=torch.long)
                item_mask_pos = torch.zeros(len(input_ids), dtype=torch.bool)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)

            if final_e3fp.dim() == 1:
                final_e3fp = final_e3fp.unsqueeze(0)
                final_unmasked_e3fp = final_unmasked_e3fp.unsqueeze(0)

            batch_e3fp_ids.append(final_e3fp)
            batch_unmasked_e3fp.append(final_unmasked_e3fp)
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

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "e3fp_ids": e3fp_padded,
            "unmasked_e3fp_ids": unmasked_e3fp_padded,
            "atom_attention_mask": atom_mask_padded,
            "atom_to_motif_map": map_padded,
            "mask_positions": mask_positions_padded,
        }