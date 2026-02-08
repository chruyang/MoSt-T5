import logging
import torch
from typing import List, Union, Dict
from transformers import T5Tokenizer, AutoTokenizer

logger = logging.getLogger(__name__)


class TextTokenizer:
    """
    文本分词器 (Text Modality)
    - 基于 google/t5-v1_1-base
    - model_max_length 设为 1e9 以避免 Tokenizer 阶段的硬截断 (遵循 3D-MolT5 设计)
    """

    def __init__(self, model_name: str = "google/t5-v1_1-base", max_len: int = int(1e9)):
        self.max_len = max_len
        self.model_name = model_name

        logger.info(f"Loading TextTokenizer: {model_name} (max_len={max_len})")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                model_max_length=max_len,
                use_fast=True,
                legacy=False
            )
        except Exception as e:
            logger.warning(f"Fast tokenizer failed, falling back to slow version: {e}")
            self.tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                model_max_length=max_len,
                legacy=False
            )

        # 仅添加必要的 3D 边界符
        special_tokens_dict = {'additional_special_tokens': ['<bom>', '<eom>']}
        self.tokenizer.add_special_tokens(
            special_tokens_dict,
            replace_additional_special_tokens=False
        )

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def __call__(self, text: Union[str, List[str]], padding: bool = True, truncation: bool = True) -> Dict[
        str, torch.Tensor]:
        return self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length" if padding else False,
            truncation=truncation,
            return_tensors="pt"
        )

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self):
        return len(self.tokenizer)