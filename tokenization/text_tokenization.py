import logging
import torch
from typing import List, Union, Dict
from transformers import T5Tokenizer

logger = logging.getLogger(__name__)


class TextTokenizer:
    """
    文本分词器 (Text Modality)
    - 严格基于 3D-MolT5 使用的 google/t5-v1_1-base
    - 包含特殊标记 <bom>, <eom>
    """

    def __init__(self, model_name: str = "google/t5-v1_1-base", max_len: int = 512):
        self.max_len = max_len
        self.model_name = model_name

        logger.info(f"Loading TextTokenizer: {model_name}")
        try:
            # legacy=False 确保行为确定性
            self.tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                model_max_length=max_len,
                legacy=False
            )
        except Exception as e:
            logger.warning(f"Download failed, trying t5-base fallback: {e}")
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=max_len)

        # 添加 3D 分子相关特殊 Token (与 3d_tokenize.py 保持一致)
        # 注意：MotifTokenizer 也会添加这些，但在这里添加是为了文本解码时的完整性
        special_tokens_dict = {'additional_special_tokens': ['<bom>', '<eom>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)

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