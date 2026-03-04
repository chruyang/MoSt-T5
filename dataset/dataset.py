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

        # 🚀 错误限流机制：防止死循环和海量报错刷屏
        self.error_count = 0
        self.max_log_errors = 50

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
            atom_mapping = entry.get('atom_to_motif_map', [])
            e3fp_numpy = entry.get('e3fp')

            # 2. Text 处理
            text_enc = self.text_tokenizer(text, padding=False, truncation=True)
            text_ids = text_enc['input_ids'].squeeze(0)

            # 3. Motif 处理
            motif_ids = self.motif_tokenizer.encode(smiles, return_tensors='pt', padding=False)
            if motif_ids.dim() > 1: motif_ids = motif_ids.squeeze(0)

            # 🚀 修复 Bug 3: OOV 灾难熔断机制
            unk_token_id = self.motif_tokenizer.tokenizer.unk_token_id
            if unk_token_id is not None and unk_token_id in motif_ids:
                raise ValueError(f"OOV token <unk> detected in 2D sequence! Skipping bad sample.")

            # 4. E3FP 处理 (带 Fallback 和 修复)
            if e3fp_numpy is not None:
                e3fp_ids = torch.tensor(e3fp_numpy, dtype=torch.long)
            else:
                e3fp_ids = self.e3fp_tokenizer.from_smiles(smiles)

            e3fp_ids = self.handle_dimension_mismatch(e3fp_ids)

            # 5. Atom Mapping
            num_atoms = e3fp_ids.shape[0]

            # 🚀 必须使用 -1 作为无归属原子的默认值！
            atom_to_motif_map = torch.full((num_atoms,), -1, dtype=torch.long)

            for motif_idx, atom_indices in enumerate(atom_mapping):
                token_idx = motif_idx + 1  # +1 完美跳过开头的 <bom>
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
            # 🚀 优雅解法：限流日志，既暴露异常又防止刷屏
            if self.error_count < self.max_log_errors:
                logger.warning(f"⚠️ [Data Error] 样本 {idx} 解析失败: {e}. 正在重采样...")
                self.error_count += 1
            elif self.error_count == self.max_log_errors:
                logger.warning(f"🚫 [Data Error] 错误次数已达上限 {self.max_log_errors} 次，后续数据错误将静默跳过...")
                self.error_count += 1

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

        # 🚀 修复 Bug 2：Padding 占位符必须为 -1，绝不能指向 <bom>
        batch_map = pad_sequence(atom_maps, batch_first=True, padding_value=-1)

        # Padding Labels (Mol2Text Target)
        batch_labels = pad_sequence(text_ids, batch_first=True, padding_value=self.ignore_index)

        # Masks
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


class GSMATPretrainingCollator:
    def __init__(self,
                 motif_tokenizer: MotifTokenizer,
                 e3fp_pad_id: int = -1,
                 mask_ratio: float = 0.15):
        self.motif_tokenizer = motif_tokenizer
        self.pad_id = motif_tokenizer.pad_id
        self.e3fp_pad_id = e3fp_pad_id
        self.mask_ratio = mask_ratio

        # 1. 引入 CAMT5 的底层 Frag 引擎
        try:
            from model.CAMT5.representation import Frag
            self.frag_processor = Frag()
        except ImportError:
            raise ImportError("无法导入 Frag，请确保路径正确")

        # 2. 构建 O(1) 极速权重查表 Tensor (初始化时只执行一次)
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
                except Exception as e:
                    # 🚀 优雅解法：显式警告基团解析失败
                    logger.warning(f"🪲 Motif 解析失败降级 -> Token: '{token_str}', ID: {token_id}, Reason: {e}")
                    self.weight_lookup[token_id] = 0.01

        # 对权重进行平滑 (对数平滑)，防止大小基团概率悬殊过大
        self.weight_lookup = torch.log1p(self.weight_lookup)

        # 获取 T5 的掩码替换符
        self.mask_token_id = self.motif_tokenizer.tokenizer.convert_tokens_to_ids("<extra_id_0>")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        motif_ids = [item['motif_input_ids'] for item in batch]
        e3fp_ids = [item['e3fp_input_ids'] for item in batch]
        atom_maps = [item['atom_to_motif_map'] for item in batch]

        # 1. 基础 Padding
        batch_motif = pad_sequence(motif_ids, batch_first=True, padding_value=self.pad_id)
        batch_e3fp = pad_sequence(e3fp_ids, batch_first=True, padding_value=self.e3fp_pad_id)

        # 🚀 修复 Bug 2：Padding 占位符必须为 -1，绝不能指向 <bom>
        batch_map = pad_sequence(atom_maps, batch_first=True, padding_value=-1)

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


class GSMATPhase2Collator(GSMATPretrainingCollator):
    """
    专门为 Phase 2 (文本-分子对齐) 设计的双视角掩码 Collator。
    采用“序列拼接 + 掩码隔离”的 SOTA 工程架构。
    """

    def __init__(self,
                 motif_tokenizer: MotifTokenizer,
                 text_tokenizer: TextTokenizer,
                 text_weight_path: str,
                 e3fp_pad_id: int = -1,
                 mask_ratio: float = 0.15):
        # 初始化一阶段的分子掩码引擎
        super().__init__(motif_tokenizer, e3fp_pad_id, mask_ratio)

        self.text_pad_id = text_tokenizer.tokenizer.pad_token_id
        # 使用统一的 extra_id_0 掩盖文本，因为是物理拼接，T5 会自动处理同一个掩码符的预测
        self.text_mask_token_id = text_tokenizer.tokenizer.convert_tokens_to_ids("<extra_id_0>")

        # 加载文本端静态权重字典
        vocab_size = text_tokenizer.tokenizer.vocab_size
        self.text_weight_lookup = torch.zeros(vocab_size, dtype=torch.float32)

        if os.path.exists(text_weight_path):
            with open(text_weight_path, 'r') as f:
                text_weights = json.load(f)
                for token_id_str, weight in text_weights.items():
                    self.text_weight_lookup[int(token_id_str)] = float(weight)
        else:
            logger.warning(f"TF-IDF weights not found at {text_weight_path}. Using random masking for text.")
            self.text_weight_lookup += 1.0  # 退化为纯随机掩码

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. 运行父类逻辑，获取完美掩码过的 Motif 和 3D 字典
        motif_batch = super().__call__(batch)

        # 2. 独立处理文本端的掩码
        text_ids = [item['text_input_ids'] for item in batch]
        batch_text = pad_sequence(text_ids, batch_first=True, padding_value=self.text_pad_id)
        batch_size, text_len = batch_text.shape

        # 获取文本掩码权重并进行多项式采样
        text_weights = self.text_weight_lookup[batch_text]
        text_weights[batch_text == self.text_pad_id] = 0.0  # padding 绝不掩码

        masked_text_ids = batch_text.clone()
        text_labels = batch_text.clone()
        text_mask_positions = torch.zeros_like(batch_text, dtype=torch.bool)

        for i in range(batch_size):
            row_weight = text_weights[i]
            if row_weight.sum() <= 0:
                continue

            num_to_mask = max(1, int((batch_text[i] != self.text_pad_id).sum() * self.mask_ratio))
            mask_idx = torch.multinomial(row_weight + 1e-6, num_samples=num_to_mask, replacement=False)

            text_mask_positions[i, mask_idx] = True
            masked_text_ids[i, mask_idx] = self.text_mask_token_id

        text_labels[~text_mask_positions] = -100
        text_attention_mask = (masked_text_ids != self.text_pad_id).long()

        # ==========================================================
        # 🚀 3. 核心大招：序列级物理拼接 & 掩码隔离
        # ==========================================================

        # A. 拼接 input_ids: [Text] + [Motif]
        concat_input_ids = torch.cat([masked_text_ids, motif_batch["input_ids"]], dim=1)

        # B. 拼接 attention_mask
        concat_attention_mask = torch.cat([text_attention_mask, motif_batch["attention_mask"]], dim=1)

        # C. 拼接 labels: Decoder算交叉熵使用，包含文本和分子两端的掩码目标
        concat_labels = torch.cat([text_labels, motif_batch["labels"]], dim=1)

        # D. 3D 特征扩充：文本部分没有 3D，全部填充 e3fp_pad_id (-1)
        fp_levels, fp_dim = motif_batch["e3fp_ids"].shape[1:]
        dummy_e3fp = torch.full((batch_size, text_len, fp_dim), self.e3fp_pad_id,
                                dtype=motif_batch["e3fp_ids"].dtype, device=motif_batch["e3fp_ids"].device)

        concat_e3fp_ids = torch.cat([dummy_e3fp, motif_batch["e3fp_ids"]], dim=1)
        concat_unmasked_e3fp_ids = torch.cat([dummy_e3fp, motif_batch["unmasked_e3fp_ids"]], dim=1)

        # E. 3D 原子 Padding 掩码拼接
        dummy_atom_mask = torch.zeros((batch_size, text_len), dtype=torch.long,
                                      device=motif_batch["atom_attention_mask"].device)
        concat_atom_mask = torch.cat([dummy_atom_mask, motif_batch["atom_attention_mask"]], dim=1)

        # F. 掩码矩阵隔离 (Geometric Head 专属): 文本处强制为 False，绝不回归 3D 坐标！
        concat_mask_positions = torch.cat([torch.zeros_like(text_mask_positions), motif_batch["mask_positions"]], dim=1)

        # G. 🚀 致命修复：偏移 atom_to_motif_map
        # 由于前面插入了 text_len 长度的文本，原先指向 Motif 的索引必须全部向右平移 text_len！
        shifted_map = motif_batch["atom_to_motif_map"].clone()
        valid_map_mask = shifted_map != -1
        shifted_map[valid_map_mask] += text_len

        return {
            "input_ids": concat_input_ids,
            "attention_mask": concat_attention_mask,
            "labels": concat_labels,
            "e3fp_ids": concat_e3fp_ids,
            "unmasked_e3fp_ids": concat_unmasked_e3fp_ids,
            "atom_attention_mask": concat_atom_mask,
            "atom_to_motif_map": shifted_map,  # 修正后的 3D 挂载地图
            "mask_positions": concat_mask_positions  # 绝对纯净的 3D 回归目标掩码
        }