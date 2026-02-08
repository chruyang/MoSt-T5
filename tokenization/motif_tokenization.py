import os
import sys
import torch
import logging
from typing import List, Union
from transformers import T5Tokenizer

# 路径注入 (保持不变以支持 CAMT5 导入)
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
        # 仅在运行时报错，允许静态分析通过
        Frag = None

logger = logging.getLogger(__name__)


class MotifTokenizer:
    """
    基团分词器 (Motif Modality)

    基于 T5 Tokenizer 扩展，将化学片段 (Frags) 添加到词表中。
    保证 Motif ID 与 Text ID 在同一 Embedding 空间且不冲突。

    Args:
        vocab_file (str): 词表文件路径。支持绝对路径或相对于项目根目录的路径。
        model_name (str): 基座模型名称 (Default: "google/t5-v1_1-base")
        max_len (int): 最大序列长度 (Default: 256, 通常 Motif 序列比文本短)
    """

    def __init__(self,
                 vocab_file: str = "asset/mol_vocabs/frag_merged.txt",
                 model_name: str = "google/t5-v1_1-base",
                 max_len: int = 256):

        if Frag is None:
            raise ImportError("❌ 无法导入 model.CAMT5.representation.Frag，请检查环境。")

        self.max_len = max_len
        self.frag_processor = Frag()

        logger.info(f"Initializing MotifTokenizer (Base: {model_name})")

        # 1. 加载基座 Tokenizer
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        except Exception:
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")

        # 2. 智能解析词表路径
        # 优先级: 传入参数 -> 环境变量 -> 默认相对路径
        env_vocab = os.getenv("FRAG_VOCAB_PATH")
        candidates = [
            vocab_file,
            os.path.join(project_root, vocab_file),
            env_vocab
        ]

        valid_vocab_path = None
        for path in candidates:
            if path and os.path.exists(path):
                valid_vocab_path = path
                break

        if not valid_vocab_path:
            raise FileNotFoundError(f"❌ Motif Vocab file not found. Checked: {candidates}")

        # 3. 加载并追加化学词表
        with open(valid_vocab_path, 'r', encoding='utf-8') as f:
            new_tokens = [line.strip() for line in f if line.strip()]

        num_added = self.tokenizer.add_tokens(new_tokens)
        logger.info(f"✅ Added {num_added} chemical tokens from {os.path.basename(valid_vocab_path)}")

        # 4. 添加特殊 Token
        special_tokens = {'additional_special_tokens': ['<bom>', '<eom>']}
        self.tokenizer.add_special_tokens(special_tokens)

        # 缓存 ID
        self.pad_id = self.tokenizer.pad_token_id
        self.bom_id = self.tokenizer.convert_tokens_to_ids('<bom>')
        self.eom_id = self.tokenizer.convert_tokens_to_ids('<eom>')
        self.unk_id = self.tokenizer.unk_token_id

    def encode(self, smiles: str, return_tensors: str = "pt") -> torch.Tensor:
        """
        SMILES -> Frag String -> Token IDs
        """
        # 1. 切分 (Frag)
        try:
            result = self.frag_processor.encode(smiles)
            motif_str = result[0] if isinstance(result, tuple) else result
        except Exception:
            motif_str = "[C]"  # Fallback

        # 2. 构造输入字符串
        full_str = f"<bom> {motif_str} <eom>"

        # 3. 编码
        encoding = self.tokenizer(
            full_str,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
            add_special_tokens=False  # 手动添加了 bom/eom
        )

        if return_tensors == "pt":
            return encoding.input_ids.squeeze(0)
        return encoding.input_ids

    @property
    def vocab_size(self) -> int:
        """返回扩充后的词表大小 (用于 resize model embeddings)"""
        return len(self.tokenizer)


# ================= 单元测试 =================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🧪 Testing MotifTokenizer...")

    # 注意：运行测试需要确保 vocab 文件存在
    # 这里使用 try-except 包裹以便在无文件环境下也能提示
    try:
        tokenizer = MotifTokenizer(max_len=32)
        smiles = "CC(=O)O"
        ids = tokenizer.encode(smiles)

        print(f"✅ Input SMILES: {smiles}")
        print(f"✅ Encoded IDs Shape: {ids.shape}")
        print(f"✅ First 3 IDs: {ids[:3].tolist()}")

        # 验证特殊 Token
        decoded = tokenizer.tokenizer.decode(ids, skip_special_tokens=False)
        print(f"✅ Decoded (Raw): {decoded}")
        assert "<bom>" in decoded

        print("🎉 MotifTokenizer Tests Passed!")
    except FileNotFoundError as e:
        print(f"⚠️ Test Skipped: {e}")
    except Exception as e:
        print(f"❌ Test Failed: {e}")