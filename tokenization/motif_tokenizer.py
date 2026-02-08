import os
import sys
import torch
import logging
from typing import List, Union
from transformers import T5Tokenizer

# ... (路径注入部分保持不变) ...
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

    [修改注] 增加了 padding 参数，支持动态 Padding 策略。
    """

    def __init__(self,
                 vocab_file: str = "asset/mol_vocabs/frag_merged.txt",
                 model_name: str = "google/t5-v1_1-base",
                 max_len: int = 256):

        if Frag is None:
            raise ImportError("❌ 无法导入 model.CAMT5.representation.Frag")

        self.max_len = max_len
        self.frag_processor = Frag()

        # ... (初始化逻辑保持不变) ...
        logger.info(f"Initializing MotifTokenizer (Base: {model_name})")

        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        except Exception:
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")

        # 路径解析与词表加载 (保持不变)
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

        special_tokens = {'additional_special_tokens': ['<bom>', '<eom>']}
        self.tokenizer.add_special_tokens(special_tokens)

        self.pad_id = self.tokenizer.pad_token_id
        self.bom_id = self.tokenizer.convert_tokens_to_ids('<bom>')
        self.eom_id = self.tokenizer.convert_tokens_to_ids('<eom>')
        self.unk_id = self.tokenizer.unk_token_id

    def encode(self, smiles: str, return_tensors: str = "pt", padding: bool = False) -> torch.Tensor:
        """
        SMILES -> Frag String -> Token IDs

        Args:
            padding (bool): 是否填充到 max_len。
                            默认为 False (推荐)，以便在 Collator 中进行动态 Padding。
        """
        # 1. 切分
        try:
            result = self.frag_processor.encode(smiles)
            motif_str = result[0] if isinstance(result, tuple) else result
        except Exception:
            motif_str = "[C]"

            # 2. 构造字符串
        full_str = f"<bom> {motif_str} <eom>"

        # 3. 编码
        # [关键修改] 根据 padding 参数决定行为
        encoding = self.tokenizer(
            full_str,
            max_length=self.max_len,
            padding="max_length" if padding else False,  # 修正这里！
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