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

# 方法1: 设置环境变量
os.environ['PYTHONPATH'] = parent_path + os.pathsep + os.environ.get('PYTHONPATH', '')

# 方法2: 直接插入到sys.path
sys.path.insert(0, parent_path)

# 方法3: 添加特定的子目录路径（提高导入成功率）
model_path = os.path.join(parent_path, 'model')
camt5_path = os.path.join(parent_path, 'model', 'CAMT5')

for path in [parent_path, model_path, camt5_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# 尝试导入模块
try:
    from model.CAMT5.representation import Frag, linearize
    print("✅ 成功导入 model.CAMT5.representation 模块")
except ImportError as e:
    print(f"⚠️ 主要导入失败: {e}")
    # 尝试备用导入方式
    try:
        from representation import Frag, linearize
        print("✅ 通过备用方式成功导入 representation 模块")
    except ImportError:
        # 最后的尝试：检查文件是否存在
        rep_file = os.path.join(camt5_path, 'representation.py')
        if os.path.exists(rep_file):
            print(f"🔍 文件存在但导入失败: {rep_file}")
            print("💡 可能的原因：")
            print("   1. representation.py 中存在语法错误")
            print("   2. 缺少必要的依赖包")
            print("   3. Python路径配置问题")
        else:
            print(f"❌ 文件不存在: {rep_file}")
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

        # 处理多组分 (用 . 分割)
        full_mapping = []
        atom_offset = 0  # 用于处理多组分时的索引偏移

        # 注意：linearize 是针对单分子设计的
        # 如果 smiles_kekule 包含 "." (如盐类)，我们需要分别处理并累加索引偏移
        sub_mols = smiles_kekule.split(".")

        for sub_smi in sub_mols:
            # 调用修改后的 linearize，接收 3 个返回值
            # frag_str, vocab_update, mapping
            _, _, mapping = linearize(sub_smi)

            # mapping 是一个 list of lists，例如 [[0,1], [2,3,4]]
            # 我们需要加上偏移量，因为 RDKit 解析 sub_smi 时索引是从 0 开始的
            # 但在整个分子的视角下，第二个组分的原子索引要接着第一个组分算

            shifted_mapping = []
            for frag_indices in mapping:
                shifted_indices = [idx + atom_offset for idx in frag_indices]
                shifted_mapping.append(shifted_indices)

            full_mapping.extend(shifted_mapping)

            # 更新偏移量：加上当前组分的原子数
            mol_tmp = Chem.MolFromSmiles(sub_smi)
            if mol_tmp:
                atom_offset += mol_tmp.GetNumAtoms()

        # 将计算出的 mapping 存入记录
        # 格式: List[List[int]], 对应 motif_seq 中的 <bom>...<eom> 之间的 motif
        # 注意: 我们的 motif_seq 有 <bom> 和 <eom>，以及中间可能的连接符 [.]
        # 这里的 mapping 只对应实际的化学片段。
        # 您在 Dataset __getitem__ 里使用时，要注意索引对齐。
        record['atom_mapping'] = full_mapping

        return key, pickle.dumps(record)

    except Exception as e:
        # print(f"Error: {e}")
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
            # 这里的 total 也是进度条参考
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
        print(f"Motif Seq: {rec['motif_seq'][:50]}...")
        print(f"Atom Mapping (前3个Motif): {rec.get('atom_mapping', [])[:3]}")
    verify_env.close()


if __name__ == "__main__":
    main()