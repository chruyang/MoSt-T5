import os
import sys
import lmdb
import pickle
import json
import csv
import re
import multiprocessing
from collections import Counter
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
PUBCHEM_PRETRAIN_JSON = os.path.expanduser("~/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/2d_computed_properties.json")
PUBCHEM_PRETRAIN_LMDB = os.path.expanduser("~/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/3d-pubchem.lmdb")

CSV_PATH = os.path.join(PROJECT_ROOT, "asset/base_motif_frequencies.csv")
BASE_VOCAB_SIZE = 30000
NUM_WORKERS = max(1, cpu_count() - 2)

# ================= 核心分析逻辑 =================

def load_base_vocab(csv_path, size):
    """加载 Phase 1 的核心 30k 词表"""
    vocab_set = set()
    if not os.path.exists(csv_path):
        print(f"❌ 找不到基础词表文件: {csv_path}")
        sys.exit(1)
        
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) 
        count = 0
        for row in reader:
            if count >= size: break
            vocab_set.add(row[0])
            count += 1
    return vocab_set

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

def extract_smiles_from_pubchem_pretrain():
    safe_keys = set()
    if not os.path.exists(PUBCHEM_PRETRAIN_JSON) or not os.path.exists(PUBCHEM_PRETRAIN_LMDB):
        return []
        
    with open(PUBCHEM_PRETRAIN_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                mol_id = item.get('input')
                if mol_id is not None:
                    safe_keys.add(str(mol_id))
                    
    all_smiles = []
    env = lmdb.open(PUBCHEM_PRETRAIN_LMDB, subdir=False, readonly=True, lock=False)
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
    print(f"🚀 开始执行【Phase 2 词表扩充收益分析】 (基于 Base Top {BASE_VOCAB_SIZE})...")
    
    # 1. 加载 30k 基础词表
    base_vocab = load_base_vocab(CSV_PATH, BASE_VOCAB_SIZE)
    print(f"✅ 成功加载 Phase 1 基础词表: {len(base_vocab):,} 个 Motif")
    
    # 2. 读取并切分 PubChem Pretrain
    print("\n📥 正在从 PubChem Pretrain 读取 SMILES...")
    pretrain_smiles = extract_smiles_from_pubchem_pretrain()
    
    if not pretrain_smiles:
        print("⚠️ 未能提取到 SMILES 数据。")
        return

    # 3. 统计新词频次
    print(f"🔥 正在多进程切分 {len(pretrain_smiles):,} 个分子，并筛查出专属新词...")
    novel_motifs_counter = Counter()
    total_tokens_in_dataset = 0
    total_novel_tokens = 0
    
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        iterator = pool.imap_unordered(process_smiles_to_pure_motifs, pretrain_smiles, chunksize=500)
        for motifs in tqdm(iterator, total=len(pretrain_smiles)):
            if motifs:
                total_tokens_in_dataset += len(motifs)
                for m in motifs:
                    # 只有不在 30k 基础词表中的，才算作“新词”
                    if m not in base_vocab:
                        novel_motifs_counter[m] += 1
                        total_novel_tokens += 1

    unique_novel_motifs = len(novel_motifs_counter)
    print("\n" + "="*60)
    print(f"🎯 分析完成！")
    print(f"   - 数据集总 Token 数量: {total_tokens_in_dataset:,}")
    print(f"   - 属于新词 (不在 30k 中) 的 Token 总数: {total_novel_tokens:,}")
    print(f"   - 新词种类总数 (独特的 OOV 骨架): {unique_novel_motifs:,}")
    
    # 4. 阶梯收益报告
    print("\n📈 【Phase 2 扩充收益阶梯报告】")
    print("如果您在 30k 基础上继续扩充 N 个新词，您能挽回多少 OOV Token？")
    print(f"{'新增扩充量 (N)':<15} | {'新词总大小 (30k+N)':<18} | {'解决的新词 OOV 比例':<20} | {'残余的长尾种类数'}")
    print("-" * 80)
    
    # 按频次从高到低排序
    sorted_novel = novel_motifs_counter.most_common()
    
    # 我们测试增加 1000, 2000, 3000, 5000, 10000, 20000 个新词的效果
    checkpoints = [500, 1000, 2000, 3000, 5000, 10000, 20000, unique_novel_motifs]
    
    cumulative_tokens_saved = 0
    current_idx = 0
    
    for n in checkpoints:
        if n > unique_novel_motifs: n = unique_novel_motifs
        
        # 累加从 current_idx 到 n 的新词频次
        while current_idx < n:
            cumulative_tokens_saved += sorted_novel[current_idx][1]
            current_idx += 1
            
        saved_ratio = (cumulative_tokens_saved / total_novel_tokens) * 100 if total_novel_tokens > 0 else 0
        remaining_types = unique_novel_motifs - n
        
        print(f"+ {n:<13,} | {BASE_VOCAB_SIZE + n:<18,} | {saved_ratio:>6.2f}% ({cumulative_tokens_saved:,} 词) | {remaining_types:,} 种骨架")
        if n == unique_novel_motifs: break

if __name__ == "__main__":
    main()