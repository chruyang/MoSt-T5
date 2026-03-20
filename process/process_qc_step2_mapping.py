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

# ================= ⚙️ PubChemQC 专属配置 =================
BASE_DIR = os.path.expanduser("~/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchemqc")
INPUT_DB = os.path.join(BASE_DIR, "pubchemqc_e3fp.lmdb")
OUTPUT_DB = os.path.join(BASE_DIR, "pubchemqc_final.lmdb")

NUM_WORKERS = max(1, cpu_count() - 2)
CHUNK_SIZE = 1000  # 1D 线性化速度比 3D 快，Chunk 设大一点
COMMIT_BATCH = 50000  # 每 5 万条写入一次硬盘


def worker_mapping(item):
    key, value_bytes = item
    try:
        record = pickle.loads(value_bytes)

        smiles = record.get('smiles_kekule') or record.get('smiles')
        if not smiles: return None

        # 🚀 极致优雅：一次调用，获取拓扑序列与底层原子映射地图
        # 注意：这里严格提取 res[1] 作为 atom_mapping (修复了之前的 Bug)
        frag_str, atom_mapping, _ = linearize(smiles)

        record['motif_seq'] = f"<bom> {frag_str} <eom>"
        record['atom_mapping'] = atom_mapping

        return key, pickle.dumps(record)
    except Exception:
        return None


def main():
    if not os.path.exists(INPUT_DB):
        print(f"❌ 找不到输入 LMDB: {INPUT_DB}，请先运行 step1")
        return

    print(f"🚀 启动 PubChemQC 1D 拓扑映射生成 (Worker 数: {NUM_WORKERS})...")

    env_in = lmdb.open(INPUT_DB, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    env_out = lmdb.open(OUTPUT_DB, map_size=int(1e12), subdir=False, readonly=False,
                        writemap=True, map_async=True, meminit=False)

    # ================= 🛡️ O(1) 断点续传逻辑 =================
    existing_keys = set()
    with env_out.begin() as txn:
        if txn.stat()['entries'] > 0:
            print("🔍 正在扫描已处理的数据，建立断点续传索引...")
            for k in txn.cursor().iternext(keys=True, values=False):
                existing_keys.add(k)
            print(f"✅ 成功加载 {len(existing_keys):,} 条历史记录，将自动跳过。")

    # ================= 📦 任务生成器 =================
    def task_generator():
        with env_in.begin() as txn:
            for k, v in txn.cursor().iternext(keys=True, values=True):
                if k not in existing_keys:
                    yield (k, v)

    total_in = env_in.stat()['entries']
    total_est = total_in - len(existing_keys)

    if total_est <= 0:
        print("🎉 所有数据已处理完毕！无需运行。")
        return

    txn_out = env_out.begin(write=True)
    success = 0

    try:
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            iterator = pool.imap_unordered(worker_mapping, task_generator(), chunksize=CHUNK_SIZE)

            for result in tqdm(iterator, total=total_est, desc="🧩 生成映射"):
                if result:
                    txn_out.put(result[0], result[1])
                    success += 1

                    if success % COMMIT_BATCH == 0:
                        txn_out.commit()
                        txn_out = env_out.begin(write=True)

            txn_out.commit()
    except Exception as e:
        print(f"❌ 发生异常，回滚最后未提交的事务: {e}")
        txn_out.abort()

    env_in.close()
    env_out.close()
    print(f"🎉 映射生成任务结束！本次成功注入: {success:,} 条")


if __name__ == "__main__":
    main()