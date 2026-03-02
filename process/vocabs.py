import os
import sys
import lmdb
import pickle
import csv
import multiprocessing
from collections import Counter
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from multiprocessing import Pool, cpu_count

RDLogger.DisableLog('rdApp.*')
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.getcwd())
try:
    from model.CAMT5.representation import linearize
except ImportError:
    raise ImportError("请确保 model/CAMT5/representation.py 存在")

# ================= 配置 =================
INPUT_DB = "../dataset/3d-pubchem-all.lmdb"
OUTPUT_MY_VOCAB = "../asset/my_dataset_vocab.txt"
OUTPUT_CSV = "../asset/motif_frequencies.csv"  # 📊 频率统计 CSV 输出路径

NUM_WORKERS = max(1, cpu_count() - 2)

# 🚀 核心逻辑：基于纯词频的智能截断
MIN_FREQ = 3  # 坚决抛弃频次 <3 的长尾巨型骨架，交由 <UNK> 处理
MAX_VOCAB_SIZE = 100000  # 放宽硬边界，完全由 MIN_FREQ 决定淘汰
# =======================================

def process_batch(data_batch):
    local_frags = Counter()
    for value_bytes in data_batch:
        try:
            record = pickle.loads(value_bytes)
            raw_smiles = record.get('smiles')
            if not raw_smiles: continue

            mol = Chem.MolFromSmiles(raw_smiles)
            if mol is None: continue

            for sub_smi in raw_smiles.split("."):
                res = linearize(sub_smi)
                local_frags.update(res[1])
        except Exception:
            pass
    return local_frags

def main():
    if not os.path.exists(INPUT_DB): return
    print(f"🚀 开始生成专属宏观词表 (纯高频 Motif + 词频>={MIN_FREQ} 截断)...")
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    all_values = []
    env = lmdb.open(INPUT_DB, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for _, value in tqdm(cursor.iternext(keys=True, values=True), desc="读取 LMDB"):
            all_values.append(value)
    env.close()

    chunk_size = 100
    chunks = [all_values[i:i + chunk_size] for i in range(0, len(all_values), chunk_size)]
    global_frag_counter = Counter()

    print(f"🔥 启动 {NUM_WORKERS} 个进程进行提取与统计...")
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        iterator = pool.imap_unordered(process_batch, chunks)
        for frags_counter in tqdm(iterator, total=len(chunks), desc="处理进度"):
            global_frag_counter.update(frags_counter)

    # ================= 核心分析与截断逻辑 =================
    total_vocab_size = len(global_frag_counter)
    print(f"\n📊 截断前完整词表大小 (全量基团种类): {total_vocab_size}")

    # 1. 初始化有效词表：只放入 [.] (因为它在 CAMT5 中既是化学分隔符，也是词)
    valid_vocab_set = {"[.]"}

    # 2. 严格按照频率进行淘汰，只保留宏观高频 Motif
    filtered_frags = {k: v for k, v in global_frag_counter.items() if v >= MIN_FREQ}
    top_k_frags = Counter(filtered_frags).most_common(MAX_VOCAB_SIZE - 1)

    for frag, count in top_k_frags:
        valid_vocab_set.add(frag)

    valid_vocab = list(valid_vocab_set)

    # 3. 按字母顺序保存
    os.makedirs(os.path.dirname(OUTPUT_MY_VOCAB), exist_ok=True)
    with open(OUTPUT_MY_VOCAB, "w", encoding='utf-8') as f:
        for frag in sorted(valid_vocab):
            f.write(frag + "\n")


    print(f"💾 最终用于训练的截断宏观词表已保存至: {OUTPUT_MY_VOCAB}")
    print(f"🔥 词表实际尺寸已由 {total_vocab_size} 精简至 ---> 【 {len(valid_vocab)} 】")

if __name__ == "__main__":
    main()