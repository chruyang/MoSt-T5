import os
import sys
import lmdb
import pickle
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Geometry import Point3D
from multiprocessing import Pool, cpu_count
import logging

# ================= 🚀 动态挂载 E3FP 源码路径 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
e3fp_lib_path = os.path.join(project_root, "tokenization", "3d_tokenization")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if e3fp_lib_path not in sys.path:
    sys.path.insert(0, e3fp_lib_path)

logging.getLogger('e3fp').setLevel(logging.ERROR)

try:
    from e3fp.pipeline import fprints_from_mol_verbose
    from e3fp.fingerprint.fprinter import signed_to_unsigned_int

    print("✅ E3FP 源码模块动态挂载成功！")
except ImportError as e:
    raise ImportError(f"❌ 错误: 无法导入 e3fp，详细报错: {e}")

# ================= ⚙️ PubChemQC 专属配置 =================
BASE_DIR = os.path.expanduser("~/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchemqc")
INPUT_DB_PATH = os.path.join(BASE_DIR, "pubchemqc_database.lmdb")
OUTPUT_DB_PATH = os.path.join(BASE_DIR, "pubchemqc_e3fp.lmdb")

# 榨干 3090 工作站的 CPU 多核性能
NUM_WORKERS = max(1, cpu_count() - 2)
CHUNK_SIZE = 500  # 提升 IPC 效率
COMMIT_BATCH = 20000  # 减少 I/O 阻塞频率

FP_BITS = 4096
FP_LEVEL = 3


def identifier_to_bit(identifier: int, bits=4096):
    return signed_to_unsigned_int(identifier) % bits


def worker_process_molecule(data_tuple):
    key, value_bytes = data_tuple
    try:
        record = pickle.loads(value_bytes)

        smiles = record.get('smi')
        coordinates_list = record.get('coordinates_list')

        if not smiles or not coordinates_list:
            return None

        # 提取量子计算的三维坐标 (保留纯重原子)
        coords_array = np.array(coordinates_list[0], dtype=np.float64)
        coords_array = coords_array - coords_array.mean(axis=0)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        mol.SetProp("_Name", str(smiles))

        num_atoms = mol.GetNumAtoms()
        if num_atoms != len(coords_array):
            return None

        conf = Chem.Conformer(num_atoms)
        for i in range(num_atoms):
            conf.SetAtomPosition(i, Point3D(float(coords_array[i][0]), float(coords_array[i][1]),
                                            float(coords_array[i][2])))
        mol.AddConformer(conf)

        fprint_params = {'bits': FP_BITS, 'rdkit_invariants': True, 'level': FP_LEVEL, 'all_iters': True,
                         'exclude_floating': False}
        _, fingerprinter = fprints_from_mol_verbose(mol, fprint_params=fprint_params)

        # feature_matrix = -1 * np.ones((num_atoms, FP_LEVEL + 1), dtype=np.int32)
        # for shell in fingerprinter.all_shells:
        #     c_idx, r_idx = int(shell.center_atom), int(shell.radius)
        #     if c_idx < num_atoms and r_idx <= FP_LEVEL:
        #         feature_matrix[c_idx, r_idx] = identifier_to_bit(shell.identifier, bits=FP_BITS)
        feature_matrix = -1 * np.ones((num_atoms, FP_LEVEL + 1), dtype=np.int32)

        if len(fingerprinter.level_shells.keys()) > 0:
            fp_num_atom = len(fingerprinter.all_shells) // len(fingerprinter.level_shells.keys())

            for i, shell in enumerate(fingerprinter.all_shells):
                c_idx = int(shell.center_atom)
                if c_idx < num_atoms:
                    lvl = i // fp_num_atom
                    if lvl <= FP_LEVEL:
                        feature_matrix[c_idx, lvl] = identifier_to_bit(shell.identifier, bits=FP_BITS)
        record['smiles'] = smiles
        record['e3fp'] = feature_matrix

        return (key, pickle.dumps(record))
    except Exception:
        return None


def main():
    print(f"🚀 启动 PubChemQC 3D E3FP 高性能提取 (Worker 数: {NUM_WORKERS})...")

    if not os.path.exists(INPUT_DB_PATH):
        print(f"❌ 找不到输入 LMDB 文件: {INPUT_DB_PATH}")
        return

    env_in = lmdb.open(INPUT_DB_PATH, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    # 开启极限 I/O 模式：writemap, map_async
    env_out = lmdb.open(OUTPUT_DB_PATH, map_size=int(1e12), subdir=False, readonly=False,
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
    success_count = 0

    try:
        with Pool(processes=NUM_WORKERS) as pool:
            iterator = pool.imap_unordered(worker_process_molecule, task_generator(), chunksize=CHUNK_SIZE)

            for result in tqdm(iterator, total=total_est, desc="🚀 计算 E3FP"):
                if result:
                    txn_out.put(result[0], result[1])
                    success_count += 1

                    if success_count % COMMIT_BATCH == 0:
                        txn_out.commit()
                        txn_out = env_out.begin(write=True)

            txn_out.commit()
    except Exception as e:
        print(f"❌ 发生异常，回滚最后未提交的事务: {e}")
        txn_out.abort()

    env_in.close();
    env_out.close()
    print(f"🎉 E3FP 提取任务结束！本次成功提取并写入: {success_count:,} 条")


if __name__ == "__main__":
    main()