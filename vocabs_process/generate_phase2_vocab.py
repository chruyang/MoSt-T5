import os
import sys
import lmdb
import pickle
import json
import re
import multiprocessing
from collections import Counter
from tqdm import tqdm
from rdkit import RDLogger

# 关闭 RDKit 烦人的底层警告
RDLogger.DisableLog('rdApp.*')

# ================= 动态包路径挂载 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from model.CAMT5.representation import linearize
except ImportError:
    raise ImportError("❌ 请确保项目根目录下存在 model/CAMT5/representation.py")

# ================= 绝对路径与配置 =================
# 您的 Phase 1 基础词表 (绝对不能打乱顺序！)
BASE_VOCAB_PATH = os.path.join(PROJECT_ROOT, "asset/mol_vocabs/vocab_20k.txt")
# 即将生成的 Phase 2 词表
NEW_VOCAB_OUTPUT = os.path.join(PROJECT_ROOT, "asset/mol_vocabs/vocab_phase2_25k.txt")

# 严格指向 Phase 2 (PubChem) 的图文对预训练数据
PUBCHEM_PRETRAIN_JSON = os.path.expanduser("~/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/2d_computed_properties.json")
PUBCHEM_PRETRAIN_LMDB = os.path.expanduser("~/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/3d-pubchem.lmdb")

# 目标设定：总词表 25,000
TARGET_VOCAB_SIZE = 25000
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# ================= 核心工作函数 =================

def process_smiles_to_pure_motifs(raw_smiles):
    """独立的 Worker 函数：将 SMILES 切分为纯净的 Motif"""
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
    """提取 Phase 2 所有的合法 SMILES"""
    safe_keys = set()
    if not os.path.exists(PUBCHEM_PRETRAIN_JSON) or not os.path.exists(PUBCHEM_PRETRAIN_LMDB):
        print(f"❌ 找不到预训练数据，请检查路径: \nJSON: {PUBCHEM_PRETRAIN_JSON}\nLMDB: {PUBCHEM_PRETRAIN_LMDB}")
        return []
        
    with open(PUBCHEM_PRETRAIN_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                mol_id = item.get('input') # 兼容您前面发现的 instruction 格式
                if not mol_id: 
                    mol_id = item.get('cid')
                if mol_id is not None:
                    safe_keys.add(str(mol_id))
                    
    all_smiles = []
    env = lmdb.open(PUBCHEM_PRETRAIN_LMDB, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        for key in safe_keys:
            val_bytes = txn.get(key.encode('utf-8'))
            if val_bytes:
                try:
                    record = pickle.loads(val_bytes)
                    smi = record.get('smiles')
                    if smi:
                        all_smiles.append(smi)
                except:
                    pass
    env.close()
    return all_smiles

# ================= 主流程 =================

def main():
    print(f"🚀 开始执行【Phase 2 专属词表生成】 (目标总大小: {TARGET_VOCAB_SIZE})...")
    
    # 1. 严格加载并保护 20k 基础词表
    if not os.path.exists(BASE_VOCAB_PATH):
        print(f"❌ 找不到基础词表文件: {BASE_VOCAB_PATH}")
        sys.exit(1)
        
    base_vocab_list = []
    with open(BASE_VOCAB_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip()
            if w:
                base_vocab_list.append(w)
                
    base_vocab_set = set(base_vocab_list)
    current_size = len(base_vocab_list)
    print(f"✅ 成功加载 Phase 1 基础词表: {current_size:,} 个 Token (顺序已锁定)")
    
    needed_new_tokens = TARGET_VOCAB_SIZE - current_size
    if needed_new_tokens <= 0:
        print(f"⚠️ 基础词表大小 ({current_size}) 已经 >= 目标大小 ({TARGET_VOCAB_SIZE})，无需扩充！")
        return

    print(f"🎯 本次任务需要从 PubChem 中挖掘最核心的 {needed_new_tokens:,} 个全新化学骨架。")
    
    # 2. 读取并切分 PubChem Pretrain
    print("\n📥 正在提取 Phase 2 预训练分子的 SMILES...")
    pretrain_smiles = extract_smiles_from_pubchem_pretrain()
    if not pretrain_smiles: return
    print(f"   -> 成功提取 {len(pretrain_smiles):,} 个有效分子。")

    # 3. 统计新词频次
    print(f"\n🔥 启动多进程切词与新词统频 (Workers: {NUM_WORKERS})...")
    novel_motifs_counter = Counter()
    
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        iterator = pool.imap_unordered(process_smiles_to_pure_motifs, pretrain_smiles, chunksize=500)
        for motifs in tqdm(iterator, total=len(pretrain_smiles)):
            if motifs:
                for m in motifs:
                    # 🛡️ 核心过滤：只要它在 Base 20k 里，我们就不管它
                    if m not in base_vocab_set:
                        novel_motifs_counter[m] += 1

    # 4. 截取最高频的新词
    sorted_new_motifs = novel_motifs_counter.most_common(needed_new_tokens)
    
    if len(sorted_new_motifs) < needed_new_tokens:
        print(f"⚠️ 警告: 数据集中总共只有 {len(sorted_new_motifs)} 种新词，无法满足扩充 {needed_new_tokens} 个的需求。将全部加入！")
        actual_new_tokens = [m for m, freq in sorted_new_motifs]
    else:
        actual_new_tokens = [m for m, freq in sorted_new_motifs]
        cutoff_freq = sorted_new_motifs[-1][1]
        print(f"📊 截断线报告: 扩充的第 {needed_new_tokens} 个词 (垫底词) 在数据集中出现了 {cutoff_freq} 次。")

    # 5. 物理拼接并保存 (先放旧词，再追加新词)
    final_vocab = base_vocab_list + actual_new_tokens
    
    print("\n💾 正在保存终极 Phase 2 词表...")
    with open(NEW_VOCAB_OUTPUT, 'w', encoding='utf-8') as f:
        for w in final_vocab:
            f.write(w + "\n")
            
    print("="*60)
    print(f"🎉 恭喜！词表扩充大功告成！")
    print(f"   ✅ 原词表大小: {current_size:,}")
    print(f"   ✅ 新增核心词: {len(actual_new_tokens):,}")
    print(f"   ✅ 最终总大小: {len(final_vocab):,}")
    print(f"   📂 文件路径:   {NEW_VOCAB_OUTPUT}")
    print("="*60)
    print("👉 下一步提醒: \n在接下来跑 Step 2 和 train2.py 时，请务必将 MotifTokenizer 的 vocab_file 指向这个新文件！")

if __name__ == "__main__":
    main()