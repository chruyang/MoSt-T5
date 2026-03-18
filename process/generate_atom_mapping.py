import lmdb
import pickle
import sys
import os
import multiprocessing
from tqdm import tqdm
from rdkit import Chem
from multiprocessing import Pool, cpu_count

# 路径配置和模块导入
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

os.environ['PYTHONPATH'] = parent_path + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, parent_path)

model_path = os.path.join(parent_path, 'model')
camt5_path = os.path.join(parent_path, 'model', 'CAMT5')

for path in [parent_path, model_path, camt5_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from model.CAMT5.representation import Frag, linearize
    print("✅ 成功导入 model.CAMT5.representation 模块")
except ImportError as e:
    print(f"⚠️ 主要导入失败: {e}")
    try:
        from representation import Frag, linearize
        print("✅ 通过备用方式成功导入 representation 模块")
    except ImportError:
        raise ImportError("无法导入 representation 模块，请检查路径和文件")

# ================= 配置 =================
DB_PATH = "../dataset/3d-pubchem-final.lmdb"
NUM_WORKERS = max(1, cpu_count() - 4)
# =======================================

def worker(item):
    key, value_bytes = item
    try:
        record = pickle.loads(value_bytes)

        # 使用 smiles_kekule 作为基准，因为它与 Motif 严格对应
        smiles_kekule = record.get('smiles_kekule')
        if not smiles_kekule:
            return None

        # =====================================================================
        # 🚀 极其优雅的调用：将一切重担交给新版的 linearize
        # 1. 它内置了盐类 (A.B) 的分割、偏移和空列表 [] 补齐。
        # 2. 它内置了孤立原子的兜底，不会让任何特征坍缩到 idx=0。
        # 3. 返回值顺序：(字符串, 原子映射, 边映射)
        # =====================================================================
        frag_str, atom_mapping, bonds_mapping = linearize(smiles_kekule)

        # 直接将完美对齐的 atom_mapping 存入记录
        record['atom_mapping'] = atom_mapping

        return key, pickle.dumps(record)

    except Exception as e:
        # 记录因遇到极端奇葩分子而解析失败的样本，在训练中可以直接跳过
        return None

def main():
    print(f"🚀 开始计算 Atom-to-Motif Mapping 并更新数据库...")
    print(f"📂 目标: {DB_PATH}")

    # 1. 读取所有数据
    env = lmdb.open(DB_PATH, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        total = txn.stat()['entries']
        cursor = txn.cursor()
        print("📥 正在读取 Keys...")
        all_data = []
        for k, v in tqdm(cursor.iternext(keys=True, values=True), total=total):
            all_data.append((k, v))
    env.close()

    # 2. 并行计算
    print(f"⚙️ 启动 {NUM_WORKERS} 个进程进行计算...")
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = pool.imap_unordered(worker, all_data, chunksize=20)

        # 3. 写入 (原地更新)
        env_write = lmdb.open(DB_PATH, subdir=False, map_size=int(1e12), readonly=False, lock=False)
        with env_write.begin(write=True) as txn_write:
            count = 0
            for res in tqdm(results, total=total, desc="写入 Mapping"):
                if res is None: continue

                key, new_value = res
                txn_write.put(key, new_value)
                count += 1

                if count % 5000 == 0:
                    txn_write.commit()
                    txn_write = env_write.begin(write=True)

            txn_write.commit()

    env_write.close()
    print(f"✅ 完成！成功为 {count} 条数据注入了 'atom_mapping' 字段。")

    # 打印一条样例数据以验证
    verify_env = lmdb.open(DB_PATH, subdir=False, readonly=True)
    with verify_env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()
        k, v = cursor.item()
        rec = pickle.loads(v)
        print("\n🔍 样例数据检查:")
        print(f"SMILES: {rec['smiles_kekule']}")
        print(f"Motif Seq: {rec.get('motif_seq', '未找到')[:50]}...")
        print(f"Atom Mapping (前3个Motif): {rec.get('atom_mapping', [])[:3]}")
    verify_env.close()

if __name__ == "__main__":
    main()