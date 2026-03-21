import os
import sys
import csv
import re
import multiprocessing
import pandas as pd
import selfies as sf
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
BASE_DIR = os.path.expanduser("~/autodl-tmp/e3fp-mol-instructions-forward-reaction-prediction/data")

SPLITS = {
    "Train": os.path.join(BASE_DIR, "train-00000-of-00001.parquet"),
    "Valid": os.path.join(BASE_DIR, "validation-00000-of-00001.parquet"),
    "Test":  os.path.join(BASE_DIR, "test-00000-of-00001.parquet")
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

def decode_selfies(selfies_str):
    if not selfies_str or not isinstance(selfies_str, str): return ""
    try:
        return sf.decoder(selfies_str)
    except Exception:
        return ""

def process_and_convert_parquet(parquet_path):
    if not os.path.exists(parquet_path):
        print(f"⚠️ 找不到文件: {parquet_path}")
        return []

    df = pd.read_parquet(parquet_path)
    new_path = parquet_path.replace(".parquet", "_smiles.parquet")
    
    # 复用上一轮生成的 _smiles 缓存，或者重新生成
    if 'input_smiles' not in df.columns or 'output_smiles' not in df.columns:
        print(f"🔄 正在将 SELFIES 极速解码为 SMILES: {os.path.basename(parquet_path)}...")
        tqdm.pandas(desc="解码反应物")
        df['input_smiles'] = df['input'].progress_apply(decode_selfies)
        tqdm.pandas(desc="解码产物")
        df['output_smiles'] = df['output'].progress_apply(decode_selfies)
        df.to_parquet(new_path)

    combined_smiles = df['input_smiles'] + "." + df['output_smiles']
    return combined_smiles.dropna().tolist()

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

def main():
    print("🚀 开始进行【PubChemQC + Mol-Inst-Forward Train 混合】反应词表覆盖率消融实验...")
    
    # 1. 加载所有基础候选词表 (Pretrain 生成)
    base_vocab_tiers = load_base_vocab_tiers(CSV_PATH, VOCAB_TIERS)
    
    # 2. 提前提取和切分所有的分子
    split_motifs_dict = {}
    for split_name, parquet_path in SPLITS.items():
        print(f"\n📥 正在读取 {split_name} 数据集...")
        smiles_list = process_and_convert_parquet(parquet_path)
        if not smiles_list: continue
            
        print(f"🔥 正在切分 {split_name} 反应集 ({len(smiles_list):,} 条反应)...")
        motifs_list = []
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            iterator = pool.imap_unordered(process_smiles_to_pure_motifs, smiles_list, chunksize=500)
            for motifs in tqdm(iterator, total=len(smiles_list)):
                if motifs: motifs_list.append(motifs)
        split_motifs_dict[split_name] = motifs_list

    # 3. 提取下游 Train 集的专属词汇库
    print("\n" + "="*50)
    print("🧬 正在从 Mol-Inst-Forward Train 集中提取独有反应骨架...")
    train_vocab = set()
    if "Train" in split_motifs_dict:
        for rxn_motifs in split_motifs_dict["Train"]:
            train_vocab.update(rxn_motifs)
    print(f"🎯 Train 集共包含 Motif 种类数: {len(train_vocab):,}")

    # 4. 构建混合词表并评估 Valid / Test
    print("="*50)
    for split_name in ["Valid", "Test"]:
        if split_name not in split_motifs_dict: continue
        
        all_reactions_motifs = split_motifs_dict[split_name]
        total_reactions = len(all_reactions_motifs)
        total_tokens = sum(len(m) for m in all_reactions_motifs)
        
        print(f"\n📊 【{split_name}】测试报告 (共 {total_tokens:,} 个 Tokens):")
        print(f"{'Base Size':<11} | {'Combined Size':<15} | {'Token OOV Rate':<18} | {'Reaction 完美复原率':<18}")
        print("-" * 75)
        
        for size in VOCAB_TIERS:
            base_set = base_vocab_tiers[size]
            # 🔥 核心：将 Pretrain 基础词表与 Train 词表融合
            combined_vocab_set = base_set.union(train_vocab)
            
            oov_tokens = 0
            perfect_reactions = 0
            
            for rxn_motifs in all_reactions_motifs:
                mol_oov_count = sum(1 for motif in rxn_motifs if motif not in combined_vocab_set)
                oov_tokens += mol_oov_count
                if mol_oov_count == 0:
                    perfect_reactions += 1
            
            token_oov_rate = (oov_tokens / total_tokens) * 100
            rxn_perfect_rate = (perfect_reactions / total_reactions) * 100
            
            print(f"Top {size:<7} | {len(combined_vocab_set):<15,} | {token_oov_rate:>6.2f}% (<unk>)     | {rxn_perfect_rate:>6.2f}% (完整)")

if __name__ == "__main__":
    main()