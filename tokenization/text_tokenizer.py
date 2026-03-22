from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class TextTokenizer:
    def __init__(self, model_name: str = "google/t5-v1_1-base", max_len: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

        # 🚀 改善二（见下文）：在这里一次性注册所有的多模态特殊符号！
        special_tokens = [
            "<bom>", "<eom>",  # 分子边界符
            "[MMM]:",  # 多模态掩码任务前缀
            "[Caption]:",  # 看图说话任务前缀
            "[Text2Mol]:",  # 文本生成分子前缀
            "[Denoise]:"  # 纯文本降噪前缀
        ]

        # 将这些符号加入分词器，防止它们被切碎
        num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        if num_added > 0:
            logger.info(f"Successfully added {num_added} special multimodal tokens to TextTokenizer.")

    # 🌟 完美代理：使用 **kwargs 接收并透传所有 HuggingFace 原生参数
    def __call__(self, text, **kwargs):
        # 如果调用方没有传 max_length，我们可以给个默认兜底
        if 'max_length' not in kwargs and 'truncation' in kwargs and kwargs['truncation']:
            kwargs['max_length'] = self.max_len

        return self.tokenizer(text, **kwargs)

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)