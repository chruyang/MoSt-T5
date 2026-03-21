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
        print(f"⚠️ 找不到对应的 JSON 或 LMDB 文件:\n  - {json_path}\n  - {lmdb_path}")
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
                # 🔥 核心修复点：将 'smi' 改为 'smiles'
                smi = record.get('smiles')
                if smi:
                    all_smiles.append(smi)
    env.close()
    return all_smiles

def main():
    print("🚀 开始执行【PubChemQC 词表 -> PubChem 全集】的跨域 Zero-shot 覆盖率测试...")
    
    base_vocab_tiers = load_base_vocab_tiers(CSV_PATH, VOCAB_TIERS)
    
    for split_name, paths in SPLITS.items():
        print("\n" + "="*60)
        print(f"🔬 正在处理数据集: 【PubChem {split_name}】")
        print("="*60)
        
        json_path = paths["json"]
        lmdb_path = paths["lmdb"]
        
        print(f"📥 正在从 LMDB 数据库中读取 SMILES...")
        smiles_list = extract_smiles_from_pubchem_split(json_path, lmdb_path)
        
        if not smiles_list:
            print("⚠️ 警告：未能提取到任何 SMILES，可能数据为空或路径错误。")
            continue
            
        print(f"🔥 成功提取 {len(smiles_list):,} 个有效分子。正在进行多进程切词...")
        all_molecules_motifs = []
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            iterator = pool.imap_unordered(process_smiles_to_pure_motifs, smiles_list, chunksize=500)
            for motifs in tqdm(iterator, total=len(smiles_list)):
                if motifs: 
                    all_molecules_motifs.append(motifs)

        total_molecules = len(all_molecules_motifs)
        total_tokens = sum(len(m) for m in all_molecules_motifs)
        
        print(f"\n📈 {split_name} 集泛化覆盖率报告 (共 {total_tokens:,} 个 Tokens):")
        print(f"{'Vocab Size':<12} | {'Token OOV Rate':<18} | {'Molecule 完美复原率':<18}")
        print("-" * 55)
        
        for size in VOCAB_TIERS:
            base_set = base_vocab_tiers[size]
            oov_tokens = 0
            perfect_molecules = 0
            
            for mol_motifs in all_molecules_motifs:
                mol_oov_count = sum(1 for motif in mol_motifs if motif not in base_set)
                oov_tokens += mol_oov_count
                if mol_oov_count == 0:
                    perfect_molecules += 1
            
            token_oov_rate = (oov_tokens / total_tokens) * 100
            mol_perfect_rate = (perfect_molecules / total_molecules) * 100
            
            print(f"Top {size:<8} | {token_oov_rate:>6.2f}% (<unk>)     | {mol_perfect_rate:>6.2f}% (完整)")

if __name__ == "__main__":
    main()