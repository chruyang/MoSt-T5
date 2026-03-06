import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem
from tqdm import tqdm

# 导入 baseline 逻辑
from moleculenet.loader import MoleculeDataset, _load_bbbp_dataset, _load_bace_dataset
from moleculenet.splitters import scaffold_split


def generate_atom_to_motif_map_online(smiles, motif_ids):
    # 动态导入以防路径循环
    from model.CAMT5.representation import linearize

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []
    try:
        Chem.Kekulize(mol)
        smi_kekule = Chem.MolToSmiles(mol, kekuleSmiles=True)
    except:
        smi_kekule = smiles

    full_mapping = []
    atom_offset = 0
    sub_mols = smi_kekule.split(".")
    for sub_smi in sub_mols:
        try:
            _, _, mapping = linearize(sub_smi)
            for frag_indices in mapping:
                full_mapping.append([idx + atom_offset for idx in frag_indices])
            m_tmp = Chem.MolFromSmiles(sub_smi)
            if m_tmp: atom_offset += m_tmp.GetNumAtoms()
        except:
            pass

    num_atoms = mol.GetNumAtoms()
    atom_to_motif_list = [-1] * num_atoms
    for motif_idx, atom_indices in enumerate(full_mapping):
        token_idx = motif_idx + 1
        if token_idx >= len(motif_ids): break
        for a_idx in atom_indices:
            if a_idx < num_atoms: atom_to_motif_list[a_idx] = token_idx
    return atom_to_motif_list


class MoleculeNetDataset(Dataset):
    # 🚀 新增一个 cache_name 参数
    def __init__(self, smiles_list, labels, motif_tokenizer, e3fp_tokenizer, cache_name=None):
        self.motif_tokenizer = motif_tokenizer
        self.e3fp_tokenizer = e3fp_tokenizer
        self.valid_data = []

        # 🚀 1. 尝试直接加载缓存
        if cache_name is not None:
            cache_path = f"dataset/{cache_name}_cache.pt"
            if os.path.exists(cache_path):
                print(f"📦 秒速加载预计算缓存: {cache_path}")
                self.valid_data = torch.load(cache_path)
                return

        # 如果没有缓存，则老老实实计算
        for smi, lbl in tqdm(zip(smiles_list, labels), total=len(smiles_list), desc=f"预处理 {cache_name} 3D 构象"):
            mol = Chem.MolFromSmiles(smi)
            if not mol: continue
            try:
                Chem.Kekulize(mol)
                smi_3d = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)
            except:
                smi_3d = smi

            smi_2d = smi_3d.replace('@', '')

            e3fp_tensor = self.e3fp_tokenizer.from_smiles(smi_3d)
            if torch.all(e3fp_tensor == self.e3fp_tokenizer.padding_idx):
                continue

            motif_ids = self.motif_tokenizer.tokenizer.encode(smi_2d, add_special_tokens=True)
            atom_map_list = generate_atom_to_motif_map_online(smi_2d, motif_ids)

            max_ats = self.e3fp_tokenizer.max_atoms
            atom_map_list = (atom_map_list[:max_ats] + [-1] * max_ats)[:max_ats]
            atom_mask = (e3fp_tensor[:, 0] != self.e3fp_tokenizer.padding_idx).long().tolist()

            self.valid_data.append({
                "m_ids": motif_ids,
                "e_ids": e3fp_tensor,
                "a_map": atom_map_list,
                "a_mask": atom_mask,
                "label": lbl
            })

        # 🚀 2. 计算完毕后，保存到硬盘
        if cache_name is not None:
            os.makedirs("dataset", exist_ok=True)
            torch.save(self.valid_data, cache_path)
            print(f"💾 已将预处理数据保存至: {cache_path}")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        return {
            "motif_input_ids": torch.tensor(item["m_ids"], dtype=torch.long),
            "e3fp_input_ids": item["e_ids"].clone().detach(),
            "atom_to_motif_map": torch.tensor(item["a_map"], dtype=torch.long),
            "atom_attention_mask": torch.tensor(item["a_mask"], dtype=torch.long),
            "label": item["label"]
        }


class PropertyCollator:
    def __call__(self, batch):
        # 🛡️ 对应 __getitem__ 返回的键名
        motif_ids = [item['motif_input_ids'] for item in batch]
        e3fp_ids = [item['e3fp_input_ids'] for item in batch]
        atom_maps = [item['atom_to_motif_map'] for item in batch]
        atom_masks = [item['atom_attention_mask'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        # 进行 Padding
        batch_motif = pad_sequence(motif_ids, batch_first=True, padding_value=0)
        batch_mask = (batch_motif != 0).long()
        batch_e3fp = pad_sequence(e3fp_ids, batch_first=True, padding_value=-1)
        batch_map = pad_sequence(atom_maps, batch_first=True, padding_value=-1)
        batch_atom_mask = pad_sequence(atom_masks, batch_first=True, padding_value=0)

        # 🚀 返回键名必须对应 MoStT5ForSequenceClassification.forward 的参数名
        return {
            "input_ids": batch_motif,
            "attention_mask": batch_mask,
            "e3fp_ids": batch_e3fp,
            "atom_to_motif_map": batch_map,
            "atom_attention_mask": batch_atom_mask,
            "labels": labels
        }


def get_official_scaffold_split(dataset_name="bbbp", raw_csv_path="dataset/bbbp/raw/BBBP.csv"):
    if dataset_name == "bbbp":
        smiles_list, _, labels = _load_bbbp_dataset(raw_csv_path)
    elif dataset_name == "bace":
        smiles_list, _, _, labels = _load_bace_dataset(raw_csv_path)
    else:
        raise NotImplementedError

    valid_idx = [i for i, s in enumerate(smiles_list) if s is not None]
    smiles_list = [smiles_list[i] for i in valid_idx]
    labels = labels[valid_idx]

    class DummyDataset:
        def __init__(self, l): self.labels = l

        def __len__(self): return len(self.labels)

        def __getitem__(self, i): return self.labels[i]

    _, _, _, (tr_s, va_s, te_s) = scaffold_split(
        DummyDataset(labels), smiles_list, task_idx=None, null_value=0,
        frac_train=0.8, frac_valid=0.1, frac_test=0.1, return_smiles=True
    )

    smi_to_lbl = {s: l for s, l in zip(smiles_list, labels)}

    def wash(l):
        v = int(l.item() if hasattr(l, 'item') else l)
        return 0 if v == -1 else v

    return (tr_s, [wash(smi_to_lbl[s]) for s in tr_s],
            va_s, [wash(smi_to_lbl[s]) for s in va_s],
            te_s, [wash(smi_to_lbl[s]) for s in te_s])