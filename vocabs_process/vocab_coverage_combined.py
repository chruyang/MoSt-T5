import os
import sys
import csv
import re
import multiprocessing
import pandas as pd
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
BASE_DIR = os.path.expanduser("~/autodl-tmp/e3fp-mol-instructions-qm9/data")

SPLITS = {
    "Train": os.path.join(BASE_DIR, "train-00000-of-00001-89b835c5dcb34f25.parquet"),
    # 注意：如果 Valid 和 Test 实际文件名不同，请在此处修正
    "Valid": os.path.join(BASE_DIR, "validation-00000-of-00001-0d70ffd341948dee.parquet"),
    "Test":  os.path.join(BASE_DIR, "test-00000-of-00001-0d70ffd341948dee.parquet")
}

CSV_PATH = os.path.join(PROJECT_ROOT, "asset/base_motif_frequencies.csv")
VOCAB_TIERS = [10000, 20000, 30000, 40000, 50000, 62196]
NUM_WORKERS = max(1, cpu_count() - 2)

# ================= 核心分析逻辑 =================

def load_base_vocab_tiers(csv_path, tiers):
    """读取预训练生成的 CSV 并生成不同大小的词表 Set"""
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

def extract_smiles_from_parquet(parquet_path):
    if not os.path.exists(parquet_path):
        return []
    df = pd.read_parquet(parquet_path, columns=['smiles'])
    return df['smiles'].dropna().tolist()

def main():
    print("🚀 开始进行【PubChemQC + 下游 qm9 Train 增强】的混合词表覆盖率测试...")
    
    # 1. 加载所有基础候选词表
    base_vocab_tiers = load_base_vocab_tiers(CSV_PATH, VOCAB_TIERS)
    
    # 2. 提前切分所有数据集 (避免重复计算 Train)
    split_motifs_dict = {}
    for split_name, parquet_path in SPLITS.items():
        smiles_list = extract_smiles_from_parquet(parquet_path)
        if not smiles_list:
            print(f"⚠️ 找不到或无法读取 {split_name} 的数据: {parquet_path}")
            continue
            
        print(f"\n🔥 正在切分 {split_name} 集 ({len(smiles_list):,} 个分子)...")
        motifs_list = []
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            iterator = pool.imap_unordered(process_smiles_to_pure_motifs, smiles_list, chunksize=500)
            for motifs in tqdm(iterator, total=len(smiles_list)):
                if motifs: motifs_list.append(motifs)
        split_motifs_dict[split_name] = motifs_list

    # 3. 提取下游 Train 集的专属词汇库
    print("\n" + "="*50)
    print("🧬 正在从下游 Train 集中提取专属词汇...")
    downstream_train_vocab = set()
    if "Train" in split_motifs_dict:
        for mol_motifs in split_motifs_dict["Train"]:
            downstream_train_vocab.update(mol_motifs)
    print(f"🎯 成功提取到下游 Train 专属 Motif 种类总数: {len(downstream_train_vocab):,}")

    # 4. 构建混合词表并开始评估
    print("="*50)
    for split_name, all_molecules_motifs in split_motifs_dict.items():
        total_molecules = len(all_molecules_motifs)
        total_tokens = sum(len(m) for m in all_molecules_motifs)
        
        print(f"\n📊 【{split_name}】测试报告 (共 {total_tokens:,} 个 Tokens):")
        print(f"{'Base Vocab Size':<15} | {'Combined Vocab Size':<20} | {'Token OOV Rate':<18} | {'Molecule 完美复原率':<18}")
        print("-" * 80)
        
        for size in VOCAB_TIERS:
            base_set = base_vocab_tiers[size]
            # 🔥 核心：将基础词表与下游 Train 词表无缝融合 (Set Union)
            combined_vocab_set = base_set.union(downstream_train_vocab)
            
            oov_tokens = 0
            perfect_molecules = 0
            
            for mol_motifs in all_molecules_motifs:
                mol_oov_count = sum(1 for motif in mol_motifs if motif not in combined_vocab_set)
                oov_tokens += mol_oov_count
                if mol_oov_count == 0:
                    perfect_molecules += 1
            
            token_oov_rate = (oov_tokens / total_tokens) * 100
            mol_perfect_rate = (perfect_molecules / total_molecules) * 100
            
            print(f"Top {size:<11} | {len(combined_vocab_set):<19,} | {token_oov_rate:>6.2f}% (<unk>)     | {mol_perfect_rate:>6.2f}% (完整)")

if __name__ == "__main__":
    main()