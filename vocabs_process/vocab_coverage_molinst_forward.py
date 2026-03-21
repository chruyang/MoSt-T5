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

# ================= 核心分析与转换逻辑 =================

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
    """将单个 SELFIES 字符串安全解码为 SMILES"""
    if not selfies_str or not isinstance(selfies_str, str):
        return ""
    try:
        return sf.decoder(selfies_str)
    except Exception:
        return ""

def process_and_convert_parquet(parquet_path):
    """读取 Parquet，转换 SELFIES，保存新文件，并返回用于评估的 SMILES"""
    if not os.path.exists(parquet_path):
        print(f"⚠️ 找不到文件: {parquet_path}")
        return []

    df = pd.read_parquet(parquet_path)
    new_path = parquet_path.replace(".parquet", "_smiles.parquet")
    
    # 检查是否已经转换过，避免重复工作
    if 'input_smiles' not in df.columns or 'output_smiles' not in df.columns:
        print(f"🔄 正在将 SELFIES 极速解码为 SMILES: {os.path.basename(parquet_path)}...")
        # 解码 input (反应物)
        tqdm.pandas(desc="解码反应物 (input)")
        df['input_smiles'] = df['input'].progress_apply(decode_selfies)
        
        # 解码 output (产物)
        tqdm.pandas(desc="解码产物 (output)")
        df['output_smiles'] = df['output'].progress_apply(decode_selfies)
        
        # 存盘！后续微调时直接读取这个带有 _smiles 的新 parquet 文件
        df.to_parquet(new_path)
        print(f"💾 转换完成！全新数据集已保存至: {new_path}")
    else:
        print(f"✅ 检测到已有的 SMILES 列，直接复用。")

    # 对于覆盖率测试，模型必须认识反应物和产物。
    # 我们用 '.' 把两边连起来，当作一个超级分子复合体进行测试
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
    print("🚀 开始执行 Mol-Inst 正向反应预测数据集升维 (SELFIES -> SMILES) 及覆盖率测试...")
    
    # 1. 加载所有基础候选词表 (Pretrain 生成)
    base_vocab_tiers = load_base_vocab_tiers(CSV_PATH, VOCAB_TIERS)
    
    # 2. 遍历每一个下游分割集
    for split_name, parquet_path in SPLITS.items():
        print("\n" + "="*60)
        print(f"🔬 正在处理下游任务: 【Forward Reaction {split_name}】")
        print("="*60)
        
        # 转换并提取 SMILES
        smiles_list = process_and_convert_parquet(parquet_path)
        if not smiles_list: continue
            
        print(f"🔥 正在对翻译后的 SMILES 进行拓扑切分 ({len(smiles_list):,} 条化学反应)...")
        all_molecules_motifs = []
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            iterator = pool.imap_unordered(process_smiles_to_pure_motifs, smiles_list, chunksize=500)
            for motifs in tqdm(iterator, total=len(smiles_list)):
                if motifs: 
                    all_molecules_motifs.append(motifs)

        total_reactions = len(all_molecules_motifs)
        total_tokens = sum(len(m) for m in all_molecules_motifs)
        
        print(f"\n📈 泛化覆盖率报告 (共 {total_tokens:,} 个 Tokens):")
        print(f"{'Vocab Size':<12} | {'Token OOV Rate':<18} | {'Reaction 完美复原率':<18}")
        print("-" * 55)
        
        for size in VOCAB_TIERS:
            base_set = base_vocab_tiers[size]
            oov_tokens = 0
            perfect_reactions = 0
            
            for rxn_motifs in all_molecules_motifs:
                mol_oov_count = sum(1 for motif in rxn_motifs if motif not in base_set)
                oov_tokens += mol_oov_count
                if mol_oov_count == 0:
                    perfect_reactions += 1
            
            token_oov_rate = (oov_tokens / total_tokens) * 100
            rxn_perfect_rate = (perfect_reactions / total_reactions) * 100
            
            print(f"Top {size:<8} | {token_oov_rate:>6.2f}% (<unk>)     | {rxn_perfect_rate:>6.2f}% (完整)")

if __name__ == "__main__":
    main()