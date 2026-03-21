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
# ⚠️ 请根据您实际存放 e3fp-chebi-molgen 的路径修改 BASE_DIR
BASE_DIR = os.path.expanduser("~/autodl-tmp/e3fp-mol-instructions-qm9/data")

# 自动匹配您的 Parquet 文件
SPLITS = {
    "Train": os.path.join(BASE_DIR, "train-00000-of-00001-89b835c5dcb34f25.parquet"),
    "Valid": os.path.join(BASE_DIR, "validation-00000-of-00001-0d70ffd341948dee.parquet"),
    "Test":  os.path.join(BASE_DIR, "test-00000-of-00001-0d70ffd341948dee.parquet")
}

# 我们依然使用 PubChemQC 提取出来的那个原始词表来做 Zero-shot 测试
CSV_PATH = os.path.join(PROJECT_ROOT, "asset/base_motif_frequencies.csv")

# 候选词表大小阶梯
VOCAB_TIERS = [10000, 20000, 30000, 40000, 50000, 62196]

NUM_WORKERS = max(1, cpu_count() - 2)

# ================= 核心分析逻辑 =================

def load_vocab_tiers(csv_path, tiers):
    """读取预训练生成的 CSV 并生成不同大小的词表 Set"""
    print(f"📥 正在加载基准词表统计 (源自 PubChemQC): {csv_path}")
    vocab_list = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            vocab_list.append(row[0])

    tier_sets = {}
    for size in tiers:
        actual_size = min(size, len(vocab_list))
        tier_sets[actual_size] = set(vocab_list[:actual_size])
    return tier_sets

def process_smiles_to_pure_motifs(raw_smiles):
    """子进程任务：将 SMILES 转换为纯净的 Base Motifs 列表"""
    if not raw_smiles:
        return []
    try:
        frag_str, _, _ = linearize(raw_smiles)
        raw_motifs = frag_str.split()
        
        pure_motifs = []
        for motif in raw_motifs:
            if motif.startswith("[") and motif.endswith("]"):
                base_motif = motif[1:-1]
            else:
                base_motif = motif
            
            # 剥离锚点
            base_motif = re.sub(r'<\d+\*>', '', base_motif)
            if base_motif:
                pure_motifs.append(base_motif)
                
        return pure_motifs
    except Exception:
        return []

def extract_smiles_from_parquet(parquet_path):
    """直接从 Parquet 文件中极速提取 smiles 列"""
    if not os.path.exists(parquet_path):
        print(f"⚠️ 找不到文件: {parquet_path}")
        return []
    
    # 相比 LMDB，用 Pandas 读 Parquet 简直是降维打击，一行代码搞定！
    df = pd.read_parquet(parquet_path, columns=['smiles'])
    # 去除空值并转换为列表
    return df['smiles'].dropna().tolist()

def main():
    print("🚀 开始进行 QM9 下游数据集的 Zero-shot 词表覆盖率测试...")
    if not os.path.exists(CSV_PATH):
        print(f"❌ 找不到词表文件 {CSV_PATH}！")
        return

    # 1. 加载所有候选词表
    vocab_tiers = load_vocab_tiers(CSV_PATH, VOCAB_TIERS)
    
    # 2. 遍历每一个下游分割集
    for split_name, parquet_path in SPLITS.items():
        print("\n" + "="*50)
        print(f"🔬 正在分析下游数据集: 【{split_name}】")
        print("="*50)
        
        smiles_list = extract_smiles_from_parquet(parquet_path)
        print(f"🧪 从 Parquet 极速提取到 {len(smiles_list):,} 个有效分子。")
        
        if not smiles_list:
            continue
            
        all_molecules_motifs = []
        print(f"🔥 启动多进程切词 (请稍候)...")
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            iterator = pool.imap_unordered(process_smiles_to_pure_motifs, smiles_list, chunksize=500)
            for motifs in tqdm(iterator, total=len(smiles_list), desc=f"{split_name} 切词进度"):
                if motifs: 
                    all_molecules_motifs.append(motifs)

        total_molecules = len(all_molecules_motifs)
        total_tokens = sum(len(m) for m in all_molecules_motifs)
        print(f"📊 {split_name} 集总 Tokens 数量: {total_tokens:,}")

        # 3. 对不同大小的词表进行覆盖率压力测试
        print("\n📈 下游泛化测试报告 (Zero-shot Coverage):")
        print(f"{'Vocab Size':<12} | {'Token OOV Rate':<18} | {'Molecule 完美复原率':<18}")
        print("-" * 55)
        
        for size, vocab_set in sorted(vocab_tiers.items()):
            oov_tokens = 0
            perfect_molecules = 0
            
            for mol_motifs in all_molecules_motifs:
                mol_oov_count = sum(1 for motif in mol_motifs if motif not in vocab_set)
                oov_tokens += mol_oov_count
                
                if mol_oov_count == 0:
                    perfect_molecules += 1
            
            token_oov_rate = (oov_tokens / total_tokens) * 100
            mol_perfect_rate = (perfect_molecules / total_molecules) * 100
            
            print(f"Top {size:<8} | {token_oov_rate:>6.2f}% (<unk>)     | {mol_perfect_rate:>6.2f}% (完整)")

if __name__ == "__main__":
    main()