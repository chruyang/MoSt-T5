import os
import sys
import torch
import logging
import re  # 新增正则库用于精确提取基团
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
    基团分词器 (Motif Modality)
    - 已集成 OOV 拦截器：确保 1D 序列与 3D atom_mapping 长度绝对 1:1 对齐
    """

    def __init__(self,
                 vocab_file: str = "asset/mol_vocabs/frag_merged.txt",
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

        # 1. 加载定制的 Motif 词表
        if not os.path.exists(vocab_file):
            logger.warning(f"⚠️ Vocab file not found: {vocab_file}. Will use empty motif vocab.")
            self.motif_vocab_set = set()
        else:
            with open(vocab_file, "r") as f:
                # 假设文件中的每行格式为 "[c1ccccc1]"
                self.motif_vocab_set = set([line.strip() for line in f])
            logger.info(f"Loaded {len(self.motif_vocab_set)} motifs from {vocab_file}")

        # 2. 将加载的 Motif 注册到 T5 的词表中
        new_tokens = list(self.motif_vocab_set)
        self.tokenizer.add_tokens(new_tokens)

        # 3. 补全特殊控制符与掩码哨兵
        extra_ids = [f"<extra_id_{i}>" for i in range(100)]
        special_tokens_list = ['<bom>', '<eom>', '[.]'] + extra_ids

        special_tokens = {'additional_special_tokens': special_tokens_list}
        self.tokenizer.add_special_tokens(special_tokens)

        self.pad_id = self.tokenizer.pad_token_id
        self.bom_id = self.tokenizer.convert_tokens_to_ids('<bom>')
        self.eom_id = self.tokenizer.convert_tokens_to_ids('<eom>')
        self.unk_id = self.tokenizer.unk_token_id

    def encode(self, smiles: str, return_tensors: str = "pt", padding: bool = False) -> torch.Tensor:
        """
        SMILES -> Frag String -> Token IDs
        """
        try:
            result = self.frag_processor.encode(smiles)
            motif_str = result[0] if isinstance(result, tuple) else result

            motifs = motif_str.split()  # 自动按空格完美切分
            safe_motifs = []

            for m in motifs:
                # 判断：是否在已知词表中，或者是特殊的多组分连接符 [.]
                if m in self.motif_vocab_set or m == "[.]":
                    safe_motifs.append(m)
                else:
                    safe_motifs.append(self.tokenizer.unk_token)

            final_motif_str = " ".join(safe_motifs)

        except Exception as e:
            logger.debug(f"Frag parsing failed for SMILES: {smiles}. Error: {e}")
            final_motif_str = self.tokenizer.unk_token

            # 首尾拼接哨兵
        full_str = f"<bom> {final_motif_str} <eom>"

        encoding = self.tokenizer(
            full_str,
            max_length=self.max_len,
            padding="max_length" if padding else False,
            truncation=True,
            return_tensors=return_tensors,
            add_special_tokens=False
        )

        if return_tensors == "pt":
            return encoding.input_ids.squeeze(0)
        return encoding.input_ids

    def decode(self, token_ids: Union[torch.Tensor, List[int]], skip_special_tokens: bool = True) -> str:
        """
        Token IDs -> Motif String
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # 过滤掉填充符和首尾哨兵，防止干扰字符串解析
        clean_ids = [tid for tid in token_ids if tid not in [self.pad_id, self.bom_id, self.eom_id]]
        decoded_str = self.tokenizer.decode(clean_ids, skip_special_tokens=skip_special_tokens)

        return decoded_str