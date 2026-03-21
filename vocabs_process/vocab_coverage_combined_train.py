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
BASE_DIR = os.path.expanduser("~/autodl-tmp/3D-MoIT/3d-mol-dataset")
PUBCHEMQC_LMDB = f"{BASE_DIR}/pubchemqc/pubchemqc_database.lmdb"

SPLITS = {
    "Train": f"{BASE_DIR}/pubchemqc/train/3d_computed_properties_unit.json",
    "Valid": f"{BASE_DIR}/pubchemqc/valid/3d_computed_properties_unit.json",
    "Test": f"{BASE_DIR}/pubchemqc/test/3d_computed_properties_unit.json"
}

CSV_PATH = os.path.join(PROJECT_ROOT, "asset/base_motif_frequencies.csv")
VOCAB_TIERS = [10000, 20000, 30000, 40000, 50000, 62196]
NUM_WORKERS = max(1, cpu_count() - 2)

# ================= 核心分析逻辑 =================

def load_base_vocab_tiers(csv_path, tiers):
    vocab_list = []
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

def extract_smiles_from_split(json_path):
    safe_keys = set()
    if not os.path.exists(json_path):
        print(f"⚠️ 找不到文件: {json_path}")
        return []
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                mol_id = item.get('input')
                if mol_id is not None:
                    safe_keys.add(str(mol_id))
                    
    all_smiles = []
    env = lmdb.open(PUBCHEMQC_LMDB, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        for key in safe_keys:
            val_bytes = txn.get(key.encode('utf-8'))
            if val_bytes:
                record = pickle.loads(val_bytes)
                smi = record.get('smi')
                if smi:
                    all_smiles.append(smi)
    env.close()
    return all_smiles

def main():
    print("🚀 开始进行【PubChemQC Pretrain + Train 混合】词表覆盖率消融实验...")
    
    # 1. 加载所有基础候选词表 (Pretrain 生成)
    base_vocab_tiers = load_base_vocab_tiers(CSV_PATH, VOCAB_TIERS)
    
    # 2. 提前提取和切分所有的分子
    split_motifs_dict = {}
    for split_name, json_path in SPLITS.items():
        print(f"\n📥 正在读取 {split_name} 数据集...")
        smiles_list = extract_smiles_from_split(json_path)
        if not smiles_list: continue
            
        print(f"🔥 正在切分 {split_name} 集 ({len(smiles_list):,} 个分子)...")
        motifs_list = []
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            iterator = pool.imap_unordered(process_smiles_to_pure_motifs, smiles_list, chunksize=500)
            for motifs in tqdm(iterator, total=len(smiles_list)):
                if motifs: motifs_list.append(motifs)
        split_motifs_dict[split_name] = motifs_list

    # 3. 提取 PubChemQC Train 集的专属词汇库
    print("\n" + "="*50)
    print("🧬 正在从 PubChemQC Train 集中提取独有化学骨架...")
    train_vocab = set()
    if "Train" in split_motifs_dict:
        for mol_motifs in split_motifs_dict["Train"]:
            train_vocab.update(mol_motifs)
    print(f"🎯 Train 集共包含 Motif 种类数: {len(train_vocab):,}")

    # 4. 构建混合词表并评估 Valid / Test
    print("="*50)
    for split_name in ["Valid", "Test"]:
        if split_name not in split_motifs_dict: continue
        
        all_molecules_motifs = split_motifs_dict[split_name]
        total_molecules = len(all_molecules_motifs)
        total_tokens = sum(len(m) for m in all_molecules_motifs)
        
        print(f"\n📊 【{split_name}】测试报告 (共 {total_tokens:,} 个 Tokens):")
        print(f"{'Base Size':<11} | {'Combined Size':<15} | {'Token OOV Rate':<18} | {'Molecule 完美复原率':<18}")
        print("-" * 75)
        
        for size in VOCAB_TIERS:
            base_set = base_vocab_tiers[size]
            # 🔥 核心：将 Pretrain 基础词表与 Train 词表融合
            combined_vocab_set = base_set.union(train_vocab)
            
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