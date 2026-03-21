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

# 要探查的分割集
SPLITS = {
    # 考虑到 Pretrain 跑一遍要 8 分钟，如果您想提速，可以注释掉 pretrain，只看下游的 generalization
    "Pretrain": f"{BASE_DIR}/pubchemqc/pretrain/3d_computed_properties_unit.json",
    "Train": f"{BASE_DIR}/pubchemqc/train/3d_computed_properties_unit.json",
    "Valid": f"{BASE_DIR}/pubchemqc/valid/3d_computed_properties_unit.json",
    "Test": f"{BASE_DIR}/pubchemqc/test/3d_computed_properties_unit.json"
}

CSV_PATH = os.path.join(PROJECT_ROOT, "asset/base_motif_frequencies.csv")

# 设定的候选词表大小阶梯
VOCAB_TIERS = [10000, 20000, 30000, 40000, 50000, 62196]

NUM_WORKERS = max(1, cpu_count() - 2)

# ================= 核心分析逻辑 =================

def load_vocab_tiers(csv_path, tiers):
    """读取 CSV 并生成不同大小的词表 Set"""
    print(f"📥 正在加载基准词表统计: {csv_path}")
    vocab_list = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            vocab_list.append(row[0])  # 只需要 Base_Motif 字符串

    tier_sets = {}
    for size in tiers:
        # 取 Top N 个作为词表，并转为 set 以实现 O(1) 极速查询
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

def extract_smiles_from_split(json_path):
    """从单个 JSON 分割集里提取对应的全量 SMILES"""
    safe_keys = set()
    if not os.path.exists(json_path):
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
    print("🚀 开始进行多尺度词表覆盖率消融实验...")
    if not os.path.exists(CSV_PATH):
        print(f"❌ 找不到词表文件 {CSV_PATH}，请确认上一步已成功执行！")
        return

    # 1. 加载所有候选词表
    vocab_tiers = load_vocab_tiers(CSV_PATH, VOCAB_TIERS)
    
    # 2. 遍历每一个分割集
    for split_name, json_path in SPLITS.items():
        print("\n" + "="*50)
        print(f"🔬 正在分析数据集: 【{split_name}】")
        print("="*50)
        
        smiles_list = extract_smiles_from_split(json_path)
        print(f"🧪 提取到 {len(smiles_list):,} 个有效分子。")
        
        if not smiles_list:
            continue
            
        # 多进程切词
        all_molecules_motifs = []
        print(f"🔥 启动多进程切词 (请稍候)...")
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            iterator = pool.imap_unordered(process_smiles_to_pure_motifs, smiles_list, chunksize=500)
            for motifs in tqdm(iterator, total=len(smiles_list), desc=f"{split_name} 切词进度"):
                if motifs: # 排除极少数解析失败的空列表
                    all_molecules_motifs.append(motifs)

        total_molecules = len(all_molecules_motifs)
        total_tokens = sum(len(m) for m in all_molecules_motifs)
        print(f"📊 {split_name} 集总 Tokens 数量: {total_tokens:,}")

        # 3. 对不同大小的词表进行覆盖率压力测试
        print("\n📈 覆盖率测试报告 (Coverage Analysis):")
        print(f"{'Vocab Size':<12} | {'Token OOV Rate':<18} | {'Molecule 完美复原率':<18}")
        print("-" * 55)
        
        for size, vocab_set in sorted(vocab_tiers.items()):
            oov_tokens = 0
            perfect_molecules = 0
            
            for mol_motifs in all_molecules_motifs:
                mol_oov_count = sum(1 for motif in mol_motifs if motif not in vocab_set)
                oov_tokens += mol_oov_count
                
                # 如果这个分子没有任何 OOV (未登录词)，说明它能仅靠 2D 词表完美复原
                if mol_oov_count == 0:
                    perfect_molecules += 1
            
            # 计算指标
            token_oov_rate = (oov_tokens / total_tokens) * 100
            mol_perfect_rate = (perfect_molecules / total_molecules) * 100
            
            print(f"Top {size:<8} | {token_oov_rate:>6.2f}% (<unk>)     | {mol_perfect_rate:>6.2f}% (完整)")

if __name__ == "__main__":
    main()