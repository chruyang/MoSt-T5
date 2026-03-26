import os
import sys
import torch
import logging
import re
from typing import List, Union
from transformers import T5Tokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from model.CAMT5.representation import Frag
except ImportError:
    try:
        from most_t5.model.CAMT5.representation import Frag
    except ImportError:
        Frag = None

logger = logging.getLogger(__name__)


class MotifTokenizer:
    """
    基团分词器 (Motif Modality) - 终极解耦与共享词表版
    - 拓扑与语义解耦：动态剥离内嵌锚点，将其作为独立的 Token 放入序列。
    - 完美防御：采用 Direct ID Mapping 彻底绕过 T5 的空格切碎 Bug。
    - 严密 OOV 拦截：强制截断非 20k 词表内的长尾 Motif 为 <unk>。
    - 🚀 词表共享机制：接收外部 base_tokenizer，避免与文本任务标签发生 ID 碰撞！
    """

    def __init__(self,
                 vocab_file: str = "asset/mol_vocabs/vocab_20k.txt",
                 model_name: str = "google/t5-v1_1-base",
                 max_len: int = 256,
                 base_tokenizer=None):  # 🚀 新增参数：接收已有的分词器

        if Frag is None:
            raise ImportError("❌ 无法导入 Frag 模块！请检查 model.CAMT5.representation 是否存在。")

        # 🚀 核心修复：如果传了 base_tokenizer，就直接共享实例，否则才重新加载
        if base_tokenizer is not None:
            self.tokenizer = base_tokenizer
            logger.info("✅ MotifTokenizer 成功继承共享的底层 TextTokenizer 实例，实现 ID 空间隔离！")
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            logger.warning("⚠️ MotifTokenizer 正在独立加载模型词表。如果在多模态任务中，建议使用共享的 base_tokenizer。")

        self.max_len = max_len

        # 保留原生特殊符号的原始 ID，作为 fallback
        self.pad_id = self.tokenizer.pad_token_id or 0
        self.unk_id = self.tokenizer.unk_token_id or 2

        # 强行获取或追加多模态边界符
        self._ensure_special_tokens(["<bom>", "<eom>"])
        self.bom_id = self.tokenizer.convert_tokens_to_ids("<bom>")
        self.eom_id = self.tokenizer.convert_tokens_to_ids("<eom>")

        # ---------------- 增量加载自定义 20k/25k Motif 词表 ----------------
        self.vocab = self._load_vocab(vocab_file)
        self.motif_to_id = {}

        # 将 Motif 作为 additional_special_tokens 加入到 tokenizer 中
        # 这保证了它们不会被 T5 自带的 SentencePiece 规则切碎
        num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': self.vocab})
        logger.info(f"Successfully added {num_added} motif tokens to tokenizer.")

        for motif in self.vocab:
            self.motif_to_id[motif] = self.tokenizer.convert_tokens_to_ids(motif)

        logger.info(f"Loaded {len(self.vocab)} motifs. Special IDs -> BOM: {self.bom_id}, EOM: {self.eom_id}")

    def _ensure_special_tokens(self, tokens: List[str]):
        """确保必须存在的特殊 token 已被添加"""
        missing = [t for t in tokens if t not in self.tokenizer.get_vocab()]
        if missing:
            self.tokenizer.add_special_tokens({'additional_special_tokens': missing})

    def _load_vocab(self, vocab_file: str) -> List[str]:
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocab file not found at {vocab_file}")

        with open(vocab_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        vocab = []
        for line in lines:
            token = line.strip().split('\t')[0]
            if token:
                vocab.append(token)
        return vocab

    def encode(self, smiles: str, return_tensors: str = None, padding: bool = False, return_mapping: bool = False) -> \
    Union[List[int], torch.Tensor, tuple]:
        """
        将 SMILES 编码为纯净的 Motif ID 序列，并返回 orig_to_new_map 供 3D 锚点对齐。
        """
        orig_to_new_map = []
        final_tokens = []

        try:
            frag = Frag(smiles)
            # 添加分子起始符
            final_tokens.append('<bom>')

            for f in frag.frags:
                raw_motif = f.get('smiles', '')
                clean_motif = re.sub(r'\[\*:\d+\]', '[*]', raw_motif)

                if clean_motif in self.motif_to_id:
                    final_tokens.append(clean_motif)
                else:
                    final_tokens.append('<unk>')

                orig_to_new_map.append(len(final_tokens) - 1)

            final_tokens.append('<eom>')
            input_ids = self.tokenizer.convert_tokens_to_ids(final_tokens)

        except Exception as e:
            logger.debug(f"Frag parsing failed for SMILES: {smiles}. Error: {e}")
            input_ids = [self.bom_id, self.unk_id, self.eom_id]
            orig_to_new_map = [1]

            # Padding 与截断逻辑
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len - 1] + [self.eom_id]
            orig_to_new_map = [idx for idx in orig_to_new_map if idx < self.max_len - 1]
        elif padding and len(input_ids) < self.max_len:
            input_ids += [self.pad_id] * (self.max_len - len(input_ids))

        if return_tensors == "pt":
            input_ids_tensor = torch.tensor([input_ids])
            if return_mapping: return input_ids_tensor, orig_to_new_map
            return input_ids_tensor

        if return_mapping: return input_ids, orig_to_new_map
        return input_ids