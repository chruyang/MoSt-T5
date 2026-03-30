import os
import sys
import torch
import logging
import re
from typing import List, Union, Dict, Tuple
from transformers import T5Tokenizer
from rdkit import Chem

# 确保路径兼容性
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 安全导入 Frag 处理器
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
    基团分词器 (Motif Modality) - 工业级重构版
    - 索引固化：采用“特殊词先行”策略，防止微调扩容导致锚点 ID 漂移。
    - 预留扩展：内置 100 个 [RESERVED_i] 占位符，支持未来任务无缝接入。
    - 接口对齐：完美适配 dataset.py 的 return_mapping 请求。
    """

    def __init__(self,
                 vocab_file: str = "asset/mol_vocabs/vocab_20k.txt",
                 base_tokenizer=None,
                 model_name: str = "google/t5-v1_1-base",
                 max_len: int = 256):

        if Frag is None:
            raise ImportError("❌ 无法导入 model.CAMT5.representation.Frag，请检查路径。")

        self.max_len = max_len
        self.frag_processor = Frag()
        logger.info(f"Initializing MotifTokenizer (Base: {model_name})")

        # 🚀 优先使用共享的 TextTokenizer 底层实例以确保跨模态对齐
        if base_tokenizer is not None:
            self.tokenizer = base_tokenizer
        else:
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
            except Exception:
                self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # =========================================================================
        # 🚀 步骤 1：固化特殊词索引 (优先级最高)
        # 顺序：基础任务符 -> 预留位 -> 哨兵位 -> 拓扑锚点
        # =========================================================================
        # 1.1 基础任务标签
        self.task_tokens = ['<bom>', '<eom>', '[.]']

        # 1.2 预留缓冲区：为后续任务额外留出 100 个位置，防止未来扩容冲突
        self.reserved_tokens = [f"[RESERVED_{i}]" for i in range(100)]

        # 1.3 哨兵位与 100 个拓扑锚点
        self.extra_ids = [f"<extra_id_{i}>" for i in range(100)]
        self.anchors = [f"<{i}*>" for i in range(100)]

        # 合并所有特殊符号并一次性注入
        special_tokens_list = self.task_tokens + self.reserved_tokens + self.extra_ids + self.anchors
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_list})

        # =========================================================================
        # 🚀 步骤 2：加载化学基团词表 (排在特殊词之后)
        # =========================================================================
        self.motif_vocab_set = set()
        if not os.path.exists(vocab_file):
            logger.warning(f"⚠️ Vocab file not found: {vocab_file}.")
        else:
            with open(vocab_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # 假设格式为 "token\tcount" 或 "token"
                        tok = line.split('\t')[0].strip()
                        self.motif_vocab_set.add(tok)
            logger.info(f"Loaded {len(self.motif_vocab_set)} pure motifs from {vocab_file}")

        # 使用 add_tokens 批量添加基团
        self.tokenizer.add_tokens(list(self.motif_vocab_set))

        # 3. 记录关键 ID
        self.pad_id = self.tokenizer.pad_token_id
        self.bom_id = self.tokenizer.convert_tokens_to_ids('<bom>')
        self.eom_id = self.tokenizer.convert_tokens_to_ids('<eom>')
        self.unk_id = self.tokenizer.unk_token_id

    def encode(self, smiles: str, return_tensors: str = "pt", padding: bool = False, return_mapping: bool = False):
        """兼容接口，直接调用 __call__"""
        return self.__call__(smiles, return_tensors, padding, return_mapping)

    def __call__(self, smiles: str, return_tensors: str = "pt", padding: bool = False, return_mapping: bool = False):
        """
        核心分词逻辑：将 SMILES 转换为包含锚点和基团的 ID 序列
        """
        try:
            # 获取 Frag 1D 序列
            result = self.frag_processor.encode(smiles)
            motif_str = result[0] if isinstance(result, tuple) else result
            motifs = motif_str.split()

            final_tokens = ['<bom>']
            orig_to_new_map = []

            for m in motifs:
                if m == "[.]":
                    final_tokens.append(m)
                    orig_to_new_map.append(len(final_tokens) - 1)
                    continue

                # 提取锚点（如 <0*>）和纯基团内容（如 [C]）
                anchors = re.findall(r'<\d+\*>', m)
                pure_inner = re.sub(r'<\d+\*>', '', m[1:-1] if m.startswith("[") else m)
                pure_motif = f"[{pure_inner}]"

                # 先放锚点，再放实体
                final_tokens.extend(anchors)
                if pure_motif in self.motif_vocab_set:
                    final_tokens.append(pure_motif)
                else:
                    final_tokens.append(self.tokenizer.unk_token)

                # 🚀 记录化学实体在 final_tokens 中的索引，用于 3D 特征映射
                orig_to_new_map.append(len(final_tokens) - 1)

            final_tokens.append('<eom>')
            input_ids = self.tokenizer.convert_tokens_to_ids(final_tokens)

        except Exception as e:
            logger.debug(f"Frag parsing failed for {smiles}. Error: {e}")
            input_ids = [self.bom_id, self.unk_id, self.eom_id]
            orig_to_new_map = [1]

        # 截断处理
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len - 1] + [self.eom_id]
            orig_to_new_map = [idx for idx in orig_to_new_map if idx < self.max_len - 1]

        # 补齐处理
        if padding and len(input_ids) < self.max_len:
            input_ids += [self.pad_id] * (self.max_len - len(input_ids))

        # 返回格式转换
        if return_tensors == "pt":
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
            if return_mapping:
                return input_ids_tensor, orig_to_new_map
            return input_ids_tensor

        if return_mapping:
            return input_ids, orig_to_new_map
        return input_ids

    def decode(self, token_ids: Union[torch.Tensor, List[int]], skip_special_tokens: bool = True) -> str:
        """
        解码逻辑：Token IDs -> 格式化基团序列 -> 拓扑还原 SMILES
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # 移除 Padding 和边界符
        clean_ids = [tid for tid in token_ids if tid not in [self.pad_id, self.bom_id, self.eom_id]]
        tokens = [self.tokenizer.convert_ids_to_tokens(tid).replace(' ', '') for tid in clean_ids]

        fragments = []
        current_anchors = []

        for tok in tokens:
            if not tok or tok in ['<bom>', '<eom>', '<pad>', '<unk>']:
                continue

            if tok in self.anchors:
                current_anchors.append(tok)
            elif tok == '[.]':
                fragments.append(tok)
                current_anchors = []
            else:
                # 🚀 将积累的锚点回填到基团的连接位点 () 中
                if tok.startswith('[') and tok.endswith(']'):
                    inner = tok[1:-1]
                    num_slots = inner.count("()")

                    # 处理前置锚点
                    if len(current_anchors) > num_slots:
                        prefix = current_anchors.pop(0)
                        inner = prefix + inner

                    # 填充占位符
                    for _ in range(num_slots):
                        if current_anchors:
                            a = current_anchors.pop(0)
                            inner = inner.replace("()", f"({a})", 1)

                    # 处理剩余后置锚点
                    suffix = "".join(current_anchors)
                    fragments.append(f"[{inner}{suffix}]")
                else:
                    fragments.append("".join(current_anchors) + tok)
                current_anchors = []

        # 拼装为 Frag 格式字符串并调用 RDKit 还原
        assembled_string = " ".join(fragments)
        try:
            raw_smiles = self.frag_processor.decode(assembled_string)
            mol = Chem.MolFromSmiles(raw_smiles)
            if mol:
                Chem.RemoveStereochemistry(mol)
                return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return raw_smiles
        except Exception:
            return assembled_string