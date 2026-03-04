import os
import sys
import torch
import logging
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
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")

        # 路径解析与词表加载
        env_vocab = os.getenv("FRAG_VOCAB_PATH")
        candidates = [vocab_file, os.path.join(project_root, vocab_file), env_vocab]
        valid_vocab_path = None
        for path in candidates:
            if path and os.path.exists(path):
                valid_vocab_path = path
                break

        if not valid_vocab_path:
            raise FileNotFoundError(f"❌ Motif Vocab file not found. Checked: {candidates}")

        with open(valid_vocab_path, 'r', encoding='utf-8') as f:
            new_tokens = [line.strip() for line in f if line.strip()]

        self.tokenizer.add_tokens(new_tokens)

        # 🚀 核心升级二：彻底补全特殊控制符，包括 [.] 与 100 个 extra_id 掩码哨兵
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
        except Exception:
            motif_str = "[C]"

        full_str = f"<bom> {motif_str} <eom>"

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

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)