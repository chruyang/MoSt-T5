from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    模型架构参数
    """
    model_name_or_path: str = field(
        default="google/t5-v1_1-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    vocab_path: str = field(
        default="asset/mol_vocabs/frag_merged.txt",
        metadata={"help": "Path to the motif vocabulary file"}
    )
    # E3FP 参数
    e3fp_num_levels: int = field(
        default=4,
        metadata={"help": "Number of E3FP levels (e.g. 4 means Level 0,1,2,3)"}
    )
    e3fp_vocab_size: int = field(
        default=4096,
        metadata={"help": "Number of E3FP bits (excluding padding)"}
    )
    # 架构参数
    fusion_type: str = field(
        default="residual",
        metadata={"help": "Fusion type: 'residual' or 'gate'"}
    )
    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "Dropout rate for the model"}
    )

@dataclass
class DataArguments:
    """
    数据路径与处理参数
    """
    train_file: str = field(
        default=None, metadata={"help": "Path to training LMDB file"}
    )
    validation_file: str = field(
        default=None, metadata={"help": "Path to validation LMDB file"}
    )
    test_file: str = field(
        default=None, metadata={"help": "Path to test LMDB file"}
    )
    max_len: int = field(
        default=512, metadata={"help": "Max sequence length for text/motif"}
    )
    task_type: str = field(
        default="mol2text",
        metadata={"help": "Task type: 'mol2text' (Captioning) or 'text2mol' (Generation)"}
    )
    num_workers: int = field(
        default=4, metadata={"help": "Number of workers for dataloader"}
    )