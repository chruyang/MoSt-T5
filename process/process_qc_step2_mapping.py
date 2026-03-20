import os
import sys
import lmdb
import pickle
import multiprocessing
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# 动态挂载
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)

from model.CAMT5.representation import linearize

# ================= ⚙️ PubChemQC 专属路径配置 =================
BASE_DIR = os.path.expanduser("~/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchemqc")
INPUT_DB = os.path.join(BASE_DIR, "pubchemqc_e3fp.lmdb")   # 上一步的输出
OUTPUT_DB = os.path.join(BASE_DIR, "pubchemqc_final.lmdb") # 最终用于预训练的库

NUM_WORKERS = max(1, cpu_count() - 2)

def worker_mapping(item):
    key, value_bytes = item
    try:
        record = pickle.loads(value_bytes)
        
        # 优先获取稳定的 Kekule SMILES
        smiles = record.get('smiles_kekule') or record.get('smiles')
        if not smiles: return None

        # 🚀 极致优雅：一次调用，获取拓扑序列与底层原子映射地图
        frag_str, atom_mapping, _ = linearize(smiles)

        record['motif_seq'] = f"<bom> {frag_str} <eom>"
        record['atom_mapping'] = atom_mapping

        return key, pickle.dumps(record)
    except Exception:
        # 跳过极个别无法成环/解析报错的奇葩量子分子
        return None

def main():
    if not os.path.exists(INPUT_DB): 
        print("❌ 找不到 E3FP 数据库，请先运行 step1")
        return
        
    print(f"🚀 开始生成 PubChemQC 终极映射数据库...")
    env_in = lmdb.open(INPUT_DB, subdir=False, readonly=True, lock=False)
    env_out = lmdb.open(OUTPUT_DB, map_size=int(1e12), subdir=False, readonly=False, map_async=True)

    with env_in.begin() as txn:
        total = txn.stat()['entries']
        keys_values = [(k, v) for k, v in txn.cursor().iternext(keys=True, values=True)]

    txn_out = env_out.begin(write=True)
    success = 0
    
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(worker_mapping, keys_values, chunksize=100), total=total, desc="注入 Mapping"):
            if result:
                txn_out.put(result[0], result[1])
                success += 1
                if success % 5000 == 0:
                    txn_out.commit()
                    txn_out = env_out.begin(write=True)
        txn_out.commit()

    env_in.close()
    env_out.close()
    print(f"🎉 全部打包完成！最终可用于 Phase 1 预训练的高质量数据: {success} 条")

if __name__ == "__main__":
    main()