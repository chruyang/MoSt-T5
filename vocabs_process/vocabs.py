import os
import sys
import lmdb
import pickle
import json
import multiprocessing
from collections import Counter
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from multiprocessing import cpu_count

RDLogger.DisableLog('rdApp.*')
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.getcwd())

try:
    from model.CAMT5.representation import linearize
except ImportError:
    raise ImportError("请确保 model/CAMT5/representation.py 存在")

# ================= 配置区 =================
# 数据集基础路径 (请确保路径正确)
BASE_DIR = "3d-mol-dataset"

# 1. 核心 LMDB 数据库
PUBCHEMQC_LMDB = f"~/autodl-tmp/MoSt-T5/{BASE_DIR}/pubchemqc/pubchemqc_database.lmdb"

# 2. 绝对安全的白名单索引文件 (只用 pretrain 和 train)
SAFE_JSONS = [
    f"{BASE_DIR}/pubchemqc/pretrain/3d_computed_properties_unit.json",
    f"{BASE_DIR}/pubchemqc/train/3d_computed_properties_unit.json"
]

OUTPUT_MY_VOCAB = "asset/mol_vocabs/my_dataset_vocab.txt"

NUM_WORKERS = max(1, cpu_count() - 2)

# 🚀 核心逻辑：基于纯词频的智能截断
MIN_FREQ = 3  # 坚决抛弃频次 < 3 的长尾巨型骨架，交由 <UNK> 和 3D E3FP 兜底
MAX_VOCAB_SIZE = 100000  # 放宽硬边界，完全由 MIN_FREQ 决定淘汰
# =======================================

def process_smiles(raw_smiles):
    """
    子进程任务：接收 SMILES -> 返回提取的 Motif 计数器
    """
    local_frags = Counter()
    if not raw_smiles:
        return local_frags
    try:
        # 🚀 升级点 1：新版 linearize 内部已经处理了 '.' 多组分，无需外部再 split
        frag_str, _, _ = linearize(raw_smiles)
        
        # 🚀 升级点 2：新版返回的 frag_str 是空格分隔的，直接 split() 即可完美切词！
        motifs = frag_str.split()
        
        # 过滤掉可能的控制符，只保留形如 [...] 的实体基团
        valid_motifs = [m for m in motifs if m.startswith("[") and m.endswith("]")]
        local_frags.update(valid_motifs)
        
    except Exception:
        pass # 极少数 RDKit 无法解析的奇葩分子直接跳过
        
    return local_frags

def main():
    if not os.path.exists(PUBCHEMQC_LMDB):
        print(f"❌ 找不到 LMDB: {PUBCHEMQC_LMDB}")
        return
        
    print(f"🚀 开始生成专属宏观词表 (来源: PubChemQC Pretrain/Train | 词频>={MIN_FREQ})...")

    # ================= 阶段 1：构建安全白名单 =================
    safe_keys = set()
    for json_path in SAFE_JSONS:
        if os.path.exists(json_path):
            print(f"📥 正在读取白名单索引: {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
                safe_keys.update(data.keys())
        else:
            print(f"⚠️ 警告: 找不到 JSON 文件 {json_path}")

    if not safe_keys:
        print("❌ 安全白名单为空，请检查 JSON 路径！")
        return
        
    print(f"🛡️ 共锁定 {len(safe_keys)} 个合法的安全分子 (绝不触碰 Test/Valid)！")

    # ================= 阶段 2：轻量化读取 SMILES =================
    # 🚀 升级点 3：不存 bytes，只存 SMILES 字符串，彻底解决 OOM 内存爆炸危机
    all_smiles = []
    env = lmdb.open(PUBCHEMQC_LMDB, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        for key in tqdm(safe_keys, desc="从 LMDB 提取安全 SMILES"):
            val_bytes = txn.get(key.encode('utf-8'))
            if val_bytes:
                record = pickle.loads(val_bytes)
                # 优先取 kekule 格式，退而求其次取普通 smiles
                smi = record.get('smiles_kekule') or record.get('smiles')
                if smi:
                    all_smiles.append(smi)
    env.close()

    # ================= 阶段 3：多进程高速提取 =================
    global_frag_counter = Counter()
    print(f"🔥 启动 {NUM_WORKERS} 个进程进行 Motif 提取与统计...")
    
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        # 直接使用 imap_unordered 处理轻量级的 SMILES 列表，速度起飞
        iterator = pool.imap_unordered(process_smiles, all_smiles, chunksize=100)
        for frags_counter in tqdm(iterator, total=len(all_smiles), desc="处理进度"):
            global_frag_counter.update(frags_counter)

    # ================= 阶段 4：核心分析与截断逻辑 =================
    total_vocab_size = len(global_frag_counter)
    print(f"\n📊 截断前完整词表大小 (全量基团种类): {total_vocab_size}")

    # 1. 初始化有效词表：只放入 [.] (多组分分隔符)
    valid_vocab_set = {"[.]"}

    # 2. 严格按照频率进行淘汰，只保留高频 Motif
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