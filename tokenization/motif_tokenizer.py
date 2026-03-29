import os
import sys
import torch
import logging
import re
from typing import List, Union
from transformers import T5Tokenizer, AddedToken

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from model.CAMT5.representation import linearize, Frag
except ImportError:
    try:
        from most_t5.model.CAMT5.representation import linearize, Frag
    except ImportError:
        linearize = None

logger = logging.getLogger(__name__)


class MotifTokenizer:
    """
    基团分词器 (Motif Modality) - 锚点解耦与拓扑保留版
    """

    def __init__(self,
                 vocab_file: str = "asset/mol_vocabs/vocab_phase2_25k.txt",
                 base_tokenizer=None,
                 model_name: str = "google/t5-v1_1-base",
                 max_len: int = 768):

        if base_tokenizer is not None:
            self.tokenizer = base_tokenizer
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=False)
        self.max_len = max_len
        self.vocab = self.load_vocab(vocab_file)

        # ====================================================================
        # 🚀 核心一：批量注册拓扑锚点 (Topology Anchors)
        # 假设一分子最多有 200 个连接点，我们一次性将 <1*> 到 <200*> 设为神圣不可侵犯的 Token
        # ====================================================================
        anchor_tokens = [
            AddedToken(f"<{i}*>", lstrip=False, rstrip=False, normalized=False, special=True)
            for i in range(1, 201)
        ]

        special_control_tokens = [
            AddedToken("<bom>", lstrip=False, rstrip=False, normalized=False, special=True),
            AddedToken("<eom>", lstrip=False, rstrip=False, normalized=False, special=True),
            AddedToken("<s>", lstrip=False, rstrip=False, normalized=False, special=True)
        ]

        # 将边界符和所有的锚点一起注册
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_control_tokens + anchor_tokens})

        # 加入您的 25k 纯净结构词表 (如 [C()=O])
        self.tokenizer.add_tokens(self.vocab, special_tokens=False)
        self.token2id = self.tokenizer.get_vocab()

        self.bom_id = self.token2id.get("<bom>")
        self.eom_id = self.token2id.get("<eom>")
        self.bos_id = self.token2id.get("<s>")
        self.pad_id = self.tokenizer.pad_token_id
        self.unk_id = self.tokenizer.unk_token_id

    def load_vocab(self, filepath: str) -> List[str]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocab file not found: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def encode(self, smiles: str, return_tensors: str = 'pt', padding: bool = False, return_mapping: bool = False):
        try:
            # ====================================================================
            # 🚀 核心二：使用 linearize 获取包含锚点的完整 1D 拓扑序列
            # 例如: frag_str = "[C] <1*> [C()=O] <1*>"
            # ====================================================================
            frag_str, _, _ = linearize(smiles)
            raw_tokens = frag_str.split()

            final_tokens = ['<s>', '<bom>']
            orig_to_new_map = []  # 记录 RDKit 的原子映射应该指向哪个绝对位置

            for tok in raw_tokens:
                # 判断当前 token 是锚点还是 Motif
                if re.match(r'^<\d+\*>$', tok):
                    # 这是一个锚点 (如 <1*>)
                    if tok in self.token2id:
                        final_tokens.append(tok)
                    else:
                        final_tokens.append('<unk>')
                else:
                    # 这是一个 Motif (如 [C()=O])
                    # (由于您的 linearize 已经输出了带 [] 和 () 的格式，直接查表即可)
                    if tok in self.token2id:
                        final_tokens.append(tok)
                    else:
                        final_tokens.append('<unk>')

                    # 🚀 核心三：仅当遇到 Motif 时，才记录它的绝对索引位置！
                    # 这保证了 atom_to_motif_map 永远指向真实的化学实体，而不是锚点！
                    orig_to_new_map.append(len(final_tokens) - 1)

            final_tokens.append('<eom>')
            input_ids = self.tokenizer.convert_tokens_to_ids(final_tokens)

        except Exception as e:
            logger.debug(f"Frag parsing failed for SMILES: {smiles}. Error: {e}")
            input_ids = [self.bos_id, self.bom_id, self.unk_id, self.eom_id]
            orig_to_new_map = [2]

        # Padding 与截断逻辑 (精确保护 eom_id)
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