import logging
import re
from itertools import product
from typing import Dict, List, NewType, Protocol, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops

from model.CAMT5.config import RepresentationType

logger = logging.getLogger(__name__)
RDLogger.DisableLog('rdApp.*')

SMILES = NewType("SMILES", str)
DUMMY_SMILES = "C"


class Representation(Protocol):
    def encode(self, mol: SMILES, verbose=False) -> str: ...

    def decode(self, text_mol: str, verbose=False) -> SMILES: ...

    def get_size(self, text_mol: str) -> int: ...

    def get_atom_weighted_score(self, text_mol: str, atom_scores: Dict[str, float]) -> float: ...


class Smiles(Representation):
    def encode(self, mol: SMILES, verbose=False) -> str:
        try:
            return Chem.MolToSmiles(Chem.MolFromSmiles(mol), kekuleSmiles=True)
        except Exception as e:
            # 拒绝静默吞没，强制输出错误原因
            logger.warning(f"[Smiles Encode Error] Failed for {mol}: {e}")
            return DUMMY_SMILES

    def decode(self, text_mol: str, verbose=False) -> SMILES:
        try:
            return Chem.MolToSmiles(Chem.MolFromSmiles(text_mol), kekuleSmiles=True)
        except Exception as e:
            logger.warning(f"[Smiles Decode Error] Failed for {text_mol}: {e}")
            return DUMMY_SMILES

    def get_size(self, text_mol: str) -> int:
        mol = Chem.MolFromSmiles(self.decode(text_mol))
        return mol.GetNumAtoms() if mol else 0

    def get_atom_weighted_score(self, text_mol: str, atom_scores: Dict[str, float]) -> float:
        mol = Chem.MolFromSmiles(self.decode(text_mol))
        if not mol: return 0
        return sum([atom_scores.get(atom.GetSymbol(), 0) for atom in mol.GetAtoms()])


class Frag(Representation):
    """
    专为 MoSt-T5 优化的片段处理器 (终极纯净版)
    - 采用 In-place 聚类，誓死捍卫 1D-3D 原子索引的绝对对齐。
    - 采用 虚拟锚点注入 (<X*>)，完美保留跨模态生成能力。
    - 剔除 Branch/Ring 伪需求，完美对齐您的原生词表。
    """

    def encode(self, mol: SMILES, verbose=False) -> str:
        try:
            frag_str, _, _ = linearize(mol)
            return frag_str
        except Exception as e:
            # 核心链路发生错误，必须抛出 Error 级别日志
            logger.error(f"[Frag Encode Error] Critical failure on {mol}: {e}")
            raise e

    def decode(self, text_mol: str, verbose=False) -> SMILES:
        try:
            # 多组分解码：按 [.] 拆分后独立重建，再用 . 缝合
            return ".".join([decode_linear(s) for s in text_mol.split("[.]")])
        except Exception as e:
            logger.warning(f"[Frag Decode Error] Failed to rebuild from '{text_mol}': {e}")
            return DUMMY_SMILES

    def get_size(self, text_mol: str) -> int:
        mol = Chem.MolFromSmiles(self.decode(text_mol))
        return mol.GetNumAtoms() if mol else 0

    def get_atom_weighted_score(self, text_mol: str, atom_scores: Dict[str, float]) -> float:
        mol = Chem.MolFromSmiles(self.decode(text_mol))
        if not mol: return 0
        return sum([atom_scores.get(atom.GetSymbol(), 0) for atom in mol.GetAtoms()])


def get_representation(representation_type) -> Representation:
    if representation_type == RepresentationType.SELFIES.value:
        raise NotImplementedError("SELFIES has been completely removed to ensure pure 2D topology focus.")
    return {RepresentationType.SMILES.value: Smiles, RepresentationType.FRAG.value: Frag}[representation_type]()


def get_importance(tokens: List[str], representation: Representation) -> List[int]:
    return [representation.get_size(t) for t in tokens]


# =====================================================================
# 🚀 核心编码引擎：图节点聚类 + 虚拟锚点注入
# =====================================================================
def linearize(smile: str) -> Tuple[str, List[List[int]], List[Tuple[int, int]]]:
    # 1. 多组分处理 (盐类) 绝对对齐闭环
    splits = smile.split(".")
    if len(splits) > 1:
        linear_smiles_list, atom_mapping, bonds_mapping = [], [], []
        offset = 0
        for idx, s in enumerate(splits):
            frag_str, mapping, bonds = linearize(s)
            if frag_str: linear_smiles_list.append(frag_str)
            atom_mapping.extend([[i + offset for i in m] for m in mapping])
            bonds_mapping.extend([(b[0] + offset, b[1] + offset) for b in bonds])

            mol = Chem.MolFromSmiles(s)
            if mol is not None: offset += mol.GetNumAtoms()
            if idx < len(splits) - 1:
                linear_smiles_list.append("[.]")
                atom_mapping.append([])
        return " ".join(linear_smiles_list), atom_mapping, bonds_mapping

    # 2. In-place 物理级去立体化
    mol = Chem.MolFromSmiles(smile)
    if mol is None: raise ValueError(f"Invalid SMILES: {smile}")
    Chem.RemoveStereochemistry(mol)
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception as e:
        logger.debug(f"[Kekulize Warning] Fallback to normal graph for {smile}: {e}")
        pass

    m_kekule = mol

    # 3. 提取拓扑图 (环与多重键合并，绝不物理切断分子)
    rings = [list(ring) for ring in Chem.GetSymmSSSR(m_kekule)]
    non_single_bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in m_kekule.GetBonds() if
                        b.GetBondType() != Chem.rdchem.BondType.SINGLE]

    motifs = rings + non_single_bonds
    merged_motifs = []
    while motifs:
        current_motif = set(motifs.pop(0))
        merged = True
        while merged:
            merged = False
            for i, other_motif in enumerate(motifs):
                if current_motif.intersection(set(other_motif)):
                    current_motif.update(motifs.pop(i))
                    merged = True
                    break
        merged_motifs.append(list(current_motif))

    # 孤立原子防坍缩兜底
    atoms_in_motifs = {atom for motif in merged_motifs for atom in motif}
    for atom in m_kekule.GetAtoms():
        if atom.GetIdx() not in atoms_in_motifs:
            merged_motifs.append([atom.GetIdx()])

    atom_to_motif = {atom: i for i, motif in enumerate(merged_motifs) for atom in motif}

    # 4. 提取跨界边，并分配全局锚点 ID
    cross_bonds = []
    edges = set()
    for bond in m_kekule.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        mu, mv = atom_to_motif[u], atom_to_motif[v]
        if mu != mv:
            edges.add((min(mu, mv), max(mu, mv)))
            cross_bonds.append({
                'motif_u': mu, 'motif_v': mv,
                'atom_u': u, 'atom_v': v,
                'bond_type': bond.GetBondType(),
                'anchor_id': len(cross_bonds)
            })

    edges = list(edges)

    # 5. DFS 纯净序列化与 虚拟锚点注入 (彻底剔除 Branch 和 Ring 标签)
    frag_list, atom_mapping = [], []
    visited = set()

    adjacency_list = {i: [] for i in range(len(merged_motifs))}
    for edge in edges:
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])

    start_node = max(adjacency_list, key=lambda k: len(adjacency_list[k]), default=0)

    def dfs(node):
        visited.add(node)
        motif_atoms = merged_motifs[node]

        # 构造带有锚点的子图副本，绝不污染原有的 3D 映射
        em = Chem.EditableMol(Chem.Mol())
        parent_to_sub = {}
        for a_idx in motif_atoms:
            atom = m_kekule.GetAtomWithIdx(a_idx)
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetIsotope(atom.GetIsotope())
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
            parent_to_sub[a_idx] = em.AddAtom(new_atom)

        for a1, a2 in product(motif_atoms, motif_atoms):
            if a1 < a2:
                bond = m_kekule.GetBondBetweenAtoms(a1, a2)
                if bond: em.AddBond(parent_to_sub[a1], parent_to_sub[a2], bond.GetBondType())

        # 给跨界原子挂载虚拟 Dummy 原子
        for cb in cross_bonds:
            if node == cb['motif_u'] or node == cb['motif_v']:
                local_atom = cb['atom_u'] if node == cb['motif_u'] else cb['atom_v']
                dummy = Chem.Atom(0)
                dummy.SetIsotope(10000 + cb['anchor_id'])  # RDKit 特殊同位素标记
                dummy_idx = em.AddAtom(dummy)
                em.AddBond(parent_to_sub[local_atom], dummy_idx, cb['bond_type'])

        submol = em.GetMol()
        Chem.SanitizeMol(submol)
        frag_smiles = Chem.MolToSmiles(submol, kekuleSmiles=True)

        # 完美转换回 CAMT5 原生的锚点语法
        frag_smiles = re.sub(r'\[(1\d{4})\*\]', lambda m: f"<{int(m.group(1)) - 10000}*>", frag_smiles)

        frag_list.append(frag_smiles)
        atom_mapping.append(motif_atoms)

        # 继续 DFS 遍历
        neighbors = [n for n in adjacency_list[node] if n not in visited]
        for neighbor in neighbors:
            dfs(neighbor)

    if merged_motifs:
        dfs(start_node)
        for i in range(len(merged_motifs)):
            if i not in visited: dfs(i)

    # 6. 用空格安全拼接 Token，绝对纯净的结构表示
    frag_string = " ".join([f"[{frag}]" for frag in frag_list])
    return frag_string, atom_mapping, []


# =====================================================================
# 🚀 极简降维解码引擎：O(N) 全局锚点重组 (无视排列顺序)
# =====================================================================
def decode_linear(linear_smiles: str) -> SMILES:
    if not linear_smiles.strip() or linear_smiles == DUMMY_SMILES:
        return ""

    raw_tokens = linear_smiles.split()
    frags = []
    for token in raw_tokens:
        if token.startswith("[") and token.endswith("]"):
            inner = token[1:-1]
            if inner == ".": continue
            frags.append(inner)
        else:
            frags.append(token)

    m = Chem.RWMol()
    anchor_registry = {}

    for frag_idx, frag_smiles in enumerate(frags):
        rdkit_smiles = re.sub(r'<(\d+)\*>', lambda m: f"[{10000 + int(m.group(1))}*]", frag_smiles)
        rdkit_smiles = re.sub(r'[@/\\\\]', '', rdkit_smiles)  # 强力剥离文本生成的立体残渣

        mol_frag = Chem.MolFromSmiles(rdkit_smiles)
        if not mol_frag:
            logger.debug(f"[Decode Skip] Submol invalid: {rdkit_smiles}")
            continue

        sub_to_global = {}
        for atom in mol_frag.GetAtoms():
            if atom.GetAtomicNum() == 0 and atom.GetIsotope() >= 10000:
                anchor_id = atom.GetIsotope() - 10000
                neighbors = atom.GetNeighbors()
                if not neighbors: continue
                neighbor = neighbors[0]
                bond = mol_frag.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())

                if anchor_id not in anchor_registry: anchor_registry[anchor_id] = []
                anchor_registry[anchor_id].append({
                    'frag_idx': frag_idx,
                    'sub_neighbor_idx': neighbor.GetIdx(),
                    'bond_type': bond.GetBondType()
                })
            else:
                new_atom = Chem.Atom(atom.GetAtomicNum())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                new_atom.SetIsotope(atom.GetIsotope())
                new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
                global_idx = m.AddAtom(new_atom)
                sub_to_global[atom.GetIdx()] = global_idx

        for bond in mol_frag.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if u in sub_to_global and v in sub_to_global:
                try:
                    m.AddBond(sub_to_global[u], sub_to_global[v], bond.GetBondType())
                except Exception as e:
                    logger.debug(f"[Decode Bond Sub] Failed inner bond: {e}")

        for anchor_id, entries in anchor_registry.items():
            for entry in entries:
                if entry['frag_idx'] == frag_idx:
                    entry['global_idx'] = sub_to_global[entry['sub_neighbor_idx']]

    # 依据全局锚点一键缝合
    for anchor_id, entries in anchor_registry.items():
        if len(entries) == 2:
            u, v = entries[0]['global_idx'], entries[1]['global_idx']
            bond_type = entries[0]['bond_type'] if entries[0]['bond_type'] == entries[1][
                'bond_type'] else Chem.rdchem.BondType.SINGLE
            try:
                m.AddBond(u, v, bond_type)
            except Exception as e:
                logger.debug(f"[Decode Bond Cross] Failed across fragments: {e}")

    try:
        Chem.SanitizeMol(m)
    except Exception as e:
        logger.debug(f"[Sanitize Fallback] Strict sanitize failed, applying bypass: {e}")
        for a in m.GetAtoms():
            m.UpdatePropertyCache(strict=False)
            a.SetNumExplicitHs(a.GetNumImplicitHs())
            a.SetNoImplicit(True)
        try:
            Chem.SanitizeMol(m)
        except Exception as fallback_e:
            logger.debug(f"[Sanitize Failed] Bypass also failed: {fallback_e}")

    return Chem.MolToSmiles(m, kekuleSmiles=True)