import os
import sys
import logging
import torch
import numpy as np
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem

# ================= 路径注入 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, "3d_tokenization")
if lib_path not in sys.path:
    sys.path.append(lib_path)

try:
    from e3fp.pipeline import fprints_from_mol_verbose
    from e3fp.fingerprint.fprinter import signed_to_unsigned_int
except ImportError:
    fprints_from_mol_verbose = None

logger = logging.getLogger(__name__)


# ================= 辅助函数 =================
def identifier_to_bit(identifier: int, n_bits: int):
    return signed_to_unsigned_int(identifier) % n_bits


def all_shell_identifier_to_fp(fingerprinter, mol, level, n_bits):
    num_atom = mol.GetNumAtoms()
    fprints_all_atom = -1 * np.ones((num_atom, level + 1), dtype=np.int32)
    if len(fingerprinter.level_shells.keys()) == 0: return fprints_all_atom

    for i, shell in enumerate(fingerprinter.all_shells):
        if shell.center_atom < num_atom:
            lvl = i // num_atom
            if lvl <= level:
                fprints_all_atom[shell.center_atom, lvl] = identifier_to_bit(shell.identifier, n_bits)
    return fprints_all_atom


def check_identifier_in_fprints_list(fprints_list, fingerprinter, fprint_params):
    target_indices = fprints_list[0].indices
    fingerprinter_modify_dict = {}
    for i in range(len(fingerprinter.all_shells)):
        shell_i = fingerprinter.all_shells[i]
        id_i_bit = identifier_to_bit(shell_i.identifier, fprint_params['bits'])
        if id_i_bit not in target_indices:
            for shell_j in fingerprinter.all_shells:
                id_j_bit = identifier_to_bit(shell_j.identifier, fprint_params['bits'])
                if (
                        shell_i.substruct == shell_j.substruct and shell_i.identifier != shell_j.identifier and id_j_bit in target_indices):
                    fingerprinter_modify_dict[i] = shell_j.identifier
                    break
    for k, v in fingerprinter_modify_dict.items():
        fingerprinter.all_shells[k].identifier = v


# ================= 主类 =================

class E3FPTokenizer:
    """
    3D E3FP 特征提取器 (严格对齐 3D-MolT5 配置)

    功能：
    - 将分子转换为 3D E3FP 指纹 Tensor
    - 支持自动 3D 构象生成与优化
    - 严格复刻 3D-MolT5 的参数配置与 Padding 策略

    使用建议：
    - 单个 SMILES：使用 `from_smiles(smi)` (推荐，自动处理 3D)
    - 批量 SMILES：使用 `from_smiles_batch(smi_list)`
    - 已有 RDKit 对象：使用 `encode(mol)` (需确保 mol 已包含 3D 构象)

    Args:
        fp_bits (int): 指纹比特数 (Default: 4096)
        fp_level (int): 半径层级 (Default: 3)
        max_atoms (int): 最大原子数截断 (Default: 256)
        padding_idx (int): 填充值 (Default: -1, 对应模型 Embedding 处理逻辑)
    """

    def __init__(self, fp_bits: int = 4096, fp_level: int = 3, max_atoms: int = 256, padding_idx: int = -1):
        if fprints_from_mol_verbose is None:
            logger.error("❌ e3fp library not found. Please ensure '3d_tokenization' folder is present.")

        self.max_atoms = max_atoms
        self.fp_level = fp_level
        self.padding_idx = padding_idx

        # 3D-MolT5 核心参数 (不可修改)
        self.fprint_params = {
            'bits': fp_bits,
            'rdkit_invariants': True,
            'level': fp_level,
            'all_iters': True,
            'exclude_floating': False,
            'stereo': True
        }
        logger.info(f"Initialized E3FPTokenizer (Bits={fp_bits}, Level={fp_level}, Pad={padding_idx})")

    def _get_empty_tensor(self):
        """返回标准填充的空 Tensor"""
        return torch.full((self.max_atoms, self.fp_level + 1), self.padding_idx, dtype=torch.long)

    def from_smiles(self, smiles: str, random_seed: int = 42) -> torch.Tensor:
        """
        从单个 SMILES 字符串直接生成 3D E3FP Token。
        流程: SMILES -> Mol -> AddHs -> Embed3D -> Optimize -> Tokenize
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return self._get_empty_tensor()

            # E3FP 需要显式氢原子
            mol = Chem.AddHs(mol)

            # 生成 3D 构象
            res = AllChem.EmbedMolecule(mol, randomSeed=random_seed)
            if res == -1:
                # 尝试更宽松的参数
                res = AllChem.EmbedMolecule(mol, randomSeed=random_seed, useRandomCoords=True)
                if res == -1:
                    logger.debug(f"Failed to embed 3D coords for: {smiles}")
                    return self._get_empty_tensor()

            # MMFF 力场优化 (提升 3D 结构质量)
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass

            return self.encode(mol)

        except Exception as e:
            logger.warning(f"SMILES processing error: {e}")
            return self._get_empty_tensor()

    def from_smiles_batch(self, smiles_list: List[str], random_seed: int = 42) -> torch.Tensor:
        """
        批量处理 SMILES 列表。
        Returns:
            Tensor shape (batch_size, max_atoms, level+1)
        """
        tensor_list = [self.from_smiles(smi, random_seed) for smi in smiles_list]
        return torch.stack(tensor_list)

    def encode(self, mol: Chem.Mol, padding: bool = True) -> torch.Tensor:
        """
        从 RDKit Mol 生成 Tensor。如果缺少 3D 构象，尝试自动补全。
        """
        if mol is None:
            return self._get_empty_tensor()

        try:
            # 检查并自动补全 3D 构象
            if mol.GetNumConformers() == 0:
                try:
                    mol_h = Chem.AddHs(mol)
                    res = AllChem.EmbedMolecule(mol_h, randomSeed=42)
                    if res == 0:
                        mol = mol_h
                    else:
                        return self._get_empty_tensor()
                except Exception:
                    return self._get_empty_tensor()

            if fprints_from_mol_verbose is None:
                return self._get_empty_tensor()

            # 生成指纹
            fprints_list, fingerprinter = fprints_from_mol_verbose(mol, fprint_params=self.fprint_params)
            check_identifier_in_fprints_list(fprints_list, fingerprinter, self.fprint_params)

            fprints_np = all_shell_identifier_to_fp(fingerprinter, mol, self.fp_level, self.fprint_params['bits'])
            feats = torch.tensor(fprints_np, dtype=torch.long)

            # Pad / Truncate atoms
            num_atoms = feats.shape[0]
            if num_atoms > self.max_atoms:
                feats = feats[:self.max_atoms, :]
            elif padding and num_atoms < self.max_atoms:
                pad_len = self.max_atoms - num_atoms
                # 使用 padding_idx (-1) 填充
                pad_tensor = torch.full((pad_len, self.fp_level + 1), self.padding_idx, dtype=torch.long)
                feats = torch.cat([feats, pad_tensor], dim=0)

            return feats

        except Exception as e:
            logger.warning(f"E3FP encoding error: {e}")
            return self._get_empty_tensor()

    def get_embedding_dim(self):
        """返回 Embedding 层需要的词表大小 (bits + 1)"""
        return self.fprint_params['bits'] + 1


if __name__ == "__main__":
    # 简单的自我测试
    logging.basicConfig(level=logging.INFO)
    try:
        tokenizer = E3FPTokenizer()

        print("🧪 Test 1: Single SMILES (Auto 3D)")
        smiles = "C1=CC=CC=C1"  # Benzene
        feats = tokenizer.from_smiles(smiles)
        print(f"✅ Shape: {feats.shape}")

        print("\n🧪 Test 2: Batch SMILES")
        batch = ["C", "CC", "CCC"]
        batch_feats = tokenizer.from_smiles_batch(batch)
        print(f"✅ Batch Shape: {batch_feats.shape}")

    except Exception as e:
        print(f"❌ Test Failed: {e}")