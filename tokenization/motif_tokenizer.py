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
    基团分词器 (Motif Modality) - 终极解耦版
    - 拓扑与语义解耦：动态剥离内嵌锚点，将其作为独立的 Token 放入序列。
    - 完美防御：采用 Direct ID Mapping 彻底绕过 T5 的空格切碎 Bug。
    - 严密 OOV 拦截：强制截断非 20k 词表内的长尾 Motif 为 <unk>。
    """

    def __init__(self,
                 vocab_file: str = "asset/mol_vocabs/vocab_20k.txt",
                 model_name: str = "google/t5-v1_1-base",
                 max_len: int = 256):

        if Frag is None:
            raise ImportError("❌ 无法导入 model.CAMT5.representation.Frag")

        self.max_len = max_len
        self.frag_processor = Frag()

        logger.info(f"Initializing MotifTokenizer (Base: {model_name})")

        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        except Exception:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # 1. 加载定制的 20k Motif 纯净词表
        if not os.path.exists(vocab_file):
            logger.warning(f"⚠️ Vocab file not found: {vocab_file}. Will use empty motif vocab.")
            self.motif_vocab_set = set()
        else:
            with open(vocab_file, "r") as f:
                self.motif_vocab_set = set([line.strip() for line in f if line.strip()])
            logger.info(f"✅ Loaded {len(self.motif_vocab_set)} pure motifs from {vocab_file}")

        # 2. 将 20k Motif 注册为常规 added_tokens
        new_tokens = list(self.motif_vocab_set)
        self.tokenizer.add_tokens(new_tokens)

        # 3. 补全特殊控制符与【拓扑锚点】
        extra_ids = [f"<extra_id_{i}>" for i in range(100)]
        anchors = [f"<{i}*>" for i in range(100)]

        special_tokens_list = ['<bom>', '<eom>', '[.]'] + extra_ids + anchors
        special_tokens = {'additional_special_tokens': special_tokens_list}
        self.tokenizer.add_special_tokens(special_tokens)

        self.pad_id = self.tokenizer.pad_token_id
        self.bom_id = self.tokenizer.convert_tokens_to_ids('<bom>')
        self.eom_id = self.tokenizer.convert_tokens_to_ids('<eom>')
        self.unk_id = self.tokenizer.unk_token_id

    def encode(self, smiles: str, return_tensors: str = "pt", padding: bool = False) -> torch.Tensor:
        """
        SMILES -> 动态解耦 -> Direct Token IDs
        """
        try:
            result = self.frag_processor.encode(smiles)
            motif_str = result[0] if isinstance(result, tuple) else result
            motifs = motif_str.split()

            final_tokens = ['<bom>']

            for m in motifs:
                # 遇到多组分分隔符直接放行
                if m == "[.]":
                    final_tokens.append(m)
                    continue

                # 🚀 1. 动态剥离：提取所有的内嵌锚点
                anchors = re.findall(r'<\d+\*>', m)

                # 🚀 2. 提取纯净骨架
                if m.startswith("[") and m.endswith("]"):
                    inner = m[1:-1]
                else:
                    inner = m
                pure_inner = re.sub(r'<\d+\*>', '', inner)
                pure_motif = f"[{pure_inner}]"

                # 🚀 3. 拓扑与语义解耦入列 (先放锚点，再放骨架)
                final_tokens.extend(anchors)

                if pure_motif in self.motif_vocab_set:
                    final_tokens.append(pure_motif)
                else:
                    final_tokens.append(self.tokenizer.unk_token)

            final_tokens.append('<eom>')

            # 降维打击：直接转化为 ID，绝不给 T5 切碎的机会！
            input_ids = self.tokenizer.convert_tokens_to_ids(final_tokens)

        except Exception as e:
            logger.debug(f"Frag parsing failed for SMILES: {smiles}. Error: {e}")
            input_ids = [self.bom_id, self.unk_id, self.eom_id]

        # Padding 与截断逻辑
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len - 1] + [self.eom_id]
        elif padding and len(input_ids) < self.max_len:
            input_ids += [self.pad_id] * (self.max_len - len(input_ids))

        if return_tensors == "pt":
            return torch.tensor([input_ids])
        return input_ids

    def decode(self, token_ids: Union[torch.Tensor, List[int]], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        clean_ids = [tid for tid in token_ids if tid not in [self.pad_id, self.bom_id, self.eom_id]]
        return self.tokenizer.decode(clean_ids, skip_special_tokens=skip_special_tokens)