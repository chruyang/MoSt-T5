import os
import sys
import torch
import logging
import re
from typing import List, Union
# 🚀 引入 AddedToken 赋予特殊字符免疫权
from transformers import T5Tokenizer, AddedToken

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
    基团分词器 (Motif Modality) - 模态隔离重构版
    """

    def __init__(self,
                 vocab_file: str = "asset/mol_vocabs/vocab_phase2_25k.txt",  # 确保指向您新生成的干净词表
                 model_name: str = "google/t5-v1_1-base",
                 max_len: int = 768):

        # 必须关闭 use_fast，否则无法精细控制底层 BPE
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=False)
        self.max_len = max_len
        self.vocab = self.load_vocab(vocab_file)

        # ====================================================================
        # 🚀 核心修复：使用 AddedToken 严格隔离特殊字符
        # lstrip=False, rstrip=False 确保它们拼接时不会吞掉相邻的化学键
        # ====================================================================
        special_control_tokens = [
            AddedToken("<bom>", lstrip=False, rstrip=False, normalized=False, special=True),
            AddedToken("<eom>", lstrip=False, rstrip=False, normalized=False, special=True),
            AddedToken("<s>", lstrip=False, rstrip=False, normalized=False, special=True)
        ]

        # 1. 注册特殊控制字符
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_control_tokens})

        # 2. 将化学骨架作为普通词汇加入
        self.tokenizer.add_tokens(self.vocab, special_tokens=False)

        self.token2id = self.tokenizer.get_vocab()

        # ====================================================================
        # 🚀 核心修复：安全提取 ID，绝不重复制造 pad 和 unk
        # ====================================================================
        self.bom_id = self.token2id.get("<bom>")
        self.eom_id = self.token2id.get("<eom>")
        self.bos_id = self.token2id.get("<s>")

        # 直接使用 T5 原生的机制，防止 Mask 混乱
        self.pad_id = self.tokenizer.pad_token_id
        self.unk_id = self.tokenizer.unk_token_id

        if None in [self.bom_id, self.eom_id, self.bos_id, self.pad_id, self.unk_id]:
            raise ValueError("🚨 Tokenizer 初始化失败：特殊字符未能成功挂载到词表！")

    def load_vocab(self, filepath: str) -> List[str]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocab file not found: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def encode(self, smiles: str, return_tensors: str = 'pt', padding: bool = False, return_mapping: bool = False):
        try:
            frag = Frag(smiles)
            final_tokens = ['<s>', '<bom>']  # 🚀 规范化起始符的拼接
            orig_to_new_map = []

            for f in frag.frags:
                raw_motif = f.get('smiles', '')
                clean_motif = re.sub(r'\[\*:\d+\]', '[*]', raw_motif)

                if clean_motif in self.token2id:
                    final_tokens.append(clean_motif)
                else:
                    final_tokens.append('<unk>')

                # 记录映射位置，注意减去开头的 <s> 和 <bom> 的偏移量
                orig_to_new_map.append(len(final_tokens) - 1)

            final_tokens.append('<eom>')
            input_ids = self.tokenizer.convert_tokens_to_ids(final_tokens)

        except Exception as e:
            logger.debug(f"Frag parsing failed for SMILES: {smiles}. Error: {e}")
            input_ids = [self.bos_id, self.bom_id, self.unk_id, self.eom_id]
            orig_to_new_map = [2]

        # Padding 与截断逻辑
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len - 1] + [self.eom_id]
            orig_to_new_map = [idx for idx in orig_to_new_map if idx < self.max_len - 1]
        elif padding and len(input_ids) < self.max_len:
            input_ids += [self.pad_id] * (self.max_len - len(input_ids))

        result = torch.tensor(input_ids, dtype=torch.long) if return_tensors == 'pt' else input_ids

        if return_mapping:
            return result, orig_to_new_map
        return result

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)