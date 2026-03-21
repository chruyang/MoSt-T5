import os
import sys
import lmdb
import pickle
import json
import csv
import re
import multiprocessing
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from multiprocessing import cpu_count

RDLogger.DisableLog('rdApp.*')

# ================= 动态包路径挂载 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from model.CAMT5.representation import linearize
except ImportError:
    raise ImportError("请确保项目根目录下存在 model/CAMT5/representation.py")

# ================= 绝对路径与配置 =================
BASE_DIR = os.path.expanduser("~/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem")

SPLITS = {
    "Pretrain": {
        "json": os.path.join(BASE_DIR, "pretrain", "2d_computed_properties.json"),
        "lmdb": os.path.join(BASE_DIR, "pretrain", "3d-pubchem.lmdb")
    },
    "Train": {
        "json": os.path.join(BASE_DIR, "train", "2d_computed_properties.json"),
        "lmdb": os.path.join(BASE_DIR, "train", "3d-pubchem.lmdb")
    },
    "Valid": {
        "json": os.path.join(BASE_DIR, "valid", "2d_computed_properties.json"),
        "lmdb": os.path.join(BASE_DIR, "valid", "3d-pubchem.lmdb")
    },
    "Test": {
        "json": os.path.join(BASE_DIR, "test", "2d_computed_properties.json"),
        "lmdb": os.path.join(BASE_DIR, "test", "3d-pubchem.lmdb")
    }
}

CSV_PATH = os.path.join(PROJECT_ROOT, "asset/base_motif_frequencies.csv")
VOCAB_TIERS = [10000, 20000, 30000, 40000, 50000, 62196]
NUM_WORKERS = max(1, cpu_count() - 2)

# ================= 核心分析逻辑 =================

def load_base_vocab_tiers(csv_path, tiers):
    vocab_list = []
    if not os.path.exists(csv_path):
        print(f"❌ 找不到词表文件: {csv_path}")
        sys.exit(1)
        
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            vocab_list.append(row[0])

    tier_sets = {}
    for size in tiers:
        actual_size = min(size, len(vocab_list))
        tier_sets[actual_size] = set(vocab_list[:actual_size])
    return tier_sets

def process_smiles_to_pure_motifs(raw_smiles):
    if not raw_smiles: return []
    try:
        frag_str, _, _ = linearize(raw_smiles)
        raw_motifs = frag_str.split()
        pure_motifs = []
        for motif in raw_motifs:
            if motif.startswith("[") and motif.endswith("]"):
                base_motif = motif[1:-1]
            else:
                base_motif = motif
            base_motif = re.sub(r'<\d+\*>', '', base_motif)
            if base_motif:
                pure_motifs.append(base_motif)
        return pure_motifs
    except Exception:
        return []

def extract_smiles_from_pubchem_split(json_path, lmdb_path):
    safe_keys = set()
    if not os.path.exists(json_path) or not os.path.exists(lmdb_path):
        return []
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                mol_id = item.get('input')
                if mol_id is not None:
                    safe_keys.add(str(mol_id))
                    
    all_smiles = []
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        for key in safe_keys:
            val_bytes = txn.get(key.encode('utf-8'))
            if val_bytes:
                record = pickle.loads(val_bytes)
                smi = record.get('smiles')
                if smi:
                    all_smiles.append(smi)
    env.close()
    return all_smiles

def main():
    print("🚀 开始进行【PubChemQC Base + PubChem Pretrain 混合】词表极限消融实验...")
    
    # 1. 加载所有基础候选词表 (PubChemQC 提取)
    base_vocab_tiers = load_base_vocab_tiers(CSV_PATH, VOCAB_TIERS)
    
    # 2. 提取 PubChem Pretrain 集的专属化学骨架库
    print("\n" + "="*60)
    print("🧬 正在从 PubChem Pretrain (30万分子) 中提取全景化学骨架...")
    pretrain_smiles = extract_smiles_from_pubchem_split(SPLITS["Pretrain"]["json"], SPLITS["Pretrain"]["lmdb"])
    pretrain_vocab = set()
    
    if pretrain_smiles:
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            iterator = pool.imap_unordered(process_smiles_to_pure_motifs, pretrain_smiles, chunksize=500)
            for motifs in tqdm(iterator, total=len(pretrain_smiles), desc="切分 Pretrain"):
                if motifs: 
                    pretrain_vocab.update(motifs)
    print(f"🎯 成功提取！PubChem Pretrain 共包含 Motif 种类数: {len(pretrain_vocab):,}")

    # 3. 评估 Train / Valid / Test 泛化能力
    for split_name in ["Train", "Valid", "Test"]:
        print("\n" + "="*60)
        print(f"📥 正在读取 {split_name} 数据集...")
        smiles_list = extract_smiles_from_pubchem_split(SPLITS[split_name]["json"], SPLITS[split_name]["lmdb"])
        if not smiles_list: continue
            
        all_molecules_motifs = []
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            iterator = pool.imap_unordered(process_smiles_to_pure_motifs, smiles_list, chunksize=500)
            for motifs in tqdm(iterator, total=len(smiles_list), desc=f"切分 {split_name}"):
                if motifs: 
                    all_molecules_motifs.append(motifs)

        total_molecules = len(all_molecules_motifs)
        total_tokens = sum(len(m) for m in all_molecules_motifs)
        
        print(f"\n📊 【{split_name}】混合词表测试报告 (共 {total_tokens:,} 个 Tokens):")
        print(f"{'Base Size':<11} | {'Combined Size':<15} | {'Token OOV Rate':<18} | {'Molecule 完美复原率':<18}")
        print("-" * 75)
        
        for size in VOCAB_TIERS:
            base_set = base_vocab_tiers[size]
            # 🔥 核心：将 PubChemQC 基础词表与 PubChem Pretrain 词表融合
            combined_vocab_set = base_set.union(pretrain_vocab)
            
            oov_tokens = 0
            perfect_molecules = 0
            
            for mol_motifs in all_molecules_motifs:
                mol_oov_count = sum(1 for motif in mol_motifs if motif not in combined_vocab_set)
                oov_tokens += mol_oov_count
                if mol_oov_count == 0:
                    perfect_molecules += 1
            
            token_oov_rate = (oov_tokens / total_tokens) * 100
            mol_perfect_rate = (perfect_molecules / total_molecules) * 100
            
            print(f"Top {size:<7} | {len(combined_vocab_set):<15,} | {token_oov_rate:>6.2f}% (<unk>)     | {mol_perfect_rate:>6.2f}% (完整)")

if __name__ == "__main__":
    main()