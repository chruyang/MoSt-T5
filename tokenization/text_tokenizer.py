from transformers import AutoTokenizer, AddedToken
import logging

logger = logging.getLogger(__name__)

class TextTokenizer:
    """
    文本分词器 (Text Modality) - 模态免疫重构版
    负责处理所有自然语言指令和多模态前缀标签。
    """
    def __init__(self, model_name: str = "google/t5-v1_1-base", max_len: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.max_len = max_len

        # ====================================================================
        # 🚀 极致对齐与免疫：使用 AddedToken 注册所有的多模态特殊符号！
        # 确保诸如 "[MMM]:" 绝对不会被意外切分成 "[", "MMM", "]:"
        # ====================================================================
        special_tokens = [
            AddedToken("<bom>", lstrip=False, rstrip=False, normalized=False, special=True),
            AddedToken("<eom>", lstrip=False, rstrip=False, normalized=False, special=True),
            AddedToken("[MMM]:", lstrip=False, rstrip=False, normalized=False, special=True),
            AddedToken("[Caption]:", lstrip=False, rstrip=False, normalized=False, special=True),
            AddedToken("[Text2Mol]:", lstrip=False, rstrip=False, normalized=False, special=True),
            AddedToken("[Denoise]:", lstrip=False, rstrip=False, normalized=False, special=True)
        ]

        digits = [f"{i}" for i in range(10)]
        self.tokenizer.add_tokens(digits, special_tokens=True)

        # 将这些符号加入分词器，防止它们被切碎
        num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        if num_added > 0:
            logger.info(f"✅ Successfully added {num_added} special multimodal tokens to TextTokenizer.")

    # 🌟 完美代理：使用 **kwargs 接收并透传所有 HuggingFace 原生参数
    def __call__(self, text, **kwargs):
        # 如果调用方没有传 max_length，我们可以给个默认兜底
        if 'max_length' not in kwargs and 'truncation' in kwargs and kwargs['truncation']:
            kwargs['max_length'] = self.max_len
        elif 'max_length' not in kwargs:
            kwargs['max_length'] = self.max_len
            if 'truncation' not in kwargs:
                kwargs['truncation'] = True

        return self.tokenizer(text, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)