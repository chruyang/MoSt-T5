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

logger = logging.getLogger(__name__)

# ================= 安全导入 =================
try:
    from e3fp.pipeline import fprints_from_mol_verbose
    from e3fp.fingerprint.fprinter import signed_to_unsigned_int
    E3FP_AVAILABLE = True
    logger.info("✅ E3FP library loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ E3FP library import failed: {e}")
    fprints_from_mol_verbose = None
    signed_to_unsigned_int = None
    E3FP_AVAILABLE = False


# ================= 辅助函数 =================
def identifier_to_bit(identifier: int, n_bits: int):
    if signed_to_unsigned_int is None:
        raise RuntimeError("E3FP library not available")
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
    def __init__(self, fp_bits: int = 4096, fp_level: int = 3, max_atoms: int = 256, padding_idx: int = -1):
        if not E3FP_AVAILABLE:
            logger.error("❌ e3fp library not found.")

        self.max_atoms = max_atoms
        self.fp_level = fp_level
        self.padding_idx = padding_idx

        self.fprint_params = {
            'bits': fp_bits,
            'rdkit_invariants': True,
            'level': fp_level,
            'all_iters': True,
            'exclude_floating': False,
            'stereo': True
        }

    def _get_empty_tensor(self):
        return torch.full((self.max_atoms, self.fp_level + 1), self.padding_idx, dtype=torch.long)

    def from_smiles(self, smiles: str, random_seed: int = 42) -> torch.Tensor:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._get_empty_tensor()

            mol = Chem.AddHs(mol)

            # 🚀 核心熔断机制 1：限制 EmbedMolecule 最多只尝试 50 次，否则直接放弃
            res = AllChem.EmbedMolecule(mol, randomSeed=random_seed, maxAttempts=50)
            if res == -1:
                res = AllChem.EmbedMolecule(mol, randomSeed=random_seed, useRandomCoords=True, maxAttempts=50)
                if res == -1:
                    return self._get_empty_tensor()

            try:
                # 🚀 核心熔断机制 2：限制力场优化最多迭代 100 步
                AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
            except:
                pass

            return self.encode(mol)

        except Exception as e:
            return self._get_empty_tensor()

    def from_smiles_batch(self, smiles_list: List[str], random_seed: int = 42) -> torch.Tensor:
        tensor_list = [self.from_smiles(smi, random_seed) for smi in smiles_list]
        return torch.stack(tensor_list)

    def encode(self, mol: Chem.Mol, padding: bool = True) -> torch.Tensor:
        if mol is None:
            return self._get_empty_tensor()

        try:
            if not E3FP_AVAILABLE:
                return self._get_empty_tensor()

            if mol.GetNumConformers() == 0:
                try:
                    mol_h = Chem.AddHs(mol)
                    # 🚀 同步添加熔断机制
                    res = AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=50)
                    if res == 0:
                        mol = mol_h
                    else:
                        return self._get_empty_tensor()
                except Exception as e:
                    return self._get_empty_tensor()

            if not mol.HasProp('_Name') or mol.GetProp('_Name') == "":
                mol.SetProp('_Name', 'dummy_molecule')

            fprints_list, fingerprinter = fprints_from_mol_verbose(mol, fprint_params=self.fprint_params)
            check_identifier_in_fprints_list(fprints_list, fingerprinter, self.fprint_params)

            fprints_np = all_shell_identifier_to_fp(fingerprinter, mol, self.fp_level, self.fprint_params['bits'])
            feats = torch.tensor(fprints_np, dtype=torch.long)

            num_atoms = feats.shape[0]
            if num_atoms > self.max_atoms:
                feats = feats[:self.max_atoms, :]
            elif padding and num_atoms < self.max_atoms:
                pad_len = self.max_atoms - num_atoms
                pad_tensor = torch.full((pad_len, self.fp_level + 1), self.padding_idx, dtype=torch.long)
                feats = torch.cat([feats, pad_tensor], dim=0)

            return feats

        except Exception as e:
            return self._get_empty_tensor()

    def get_embedding_dim(self):
        return self.fprint_params['bits'] + 1