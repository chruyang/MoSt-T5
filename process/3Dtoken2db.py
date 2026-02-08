# @Time    : 2026/2/05 参考3DMOLT5的3d_tokenize.py
import os
import lmdb
import pickle
import numpy as np
import time
from tqdm import tqdm
from rdkit import Chem
from rdkit.Geometry import Point3D
from multiprocessing import Pool, cpu_count

# 尝试导入 e3fp
try:
    from e3fp.pipeline import fprints_from_mol_verbose
    from e3fp.fingerprint.fprinter import signed_to_unsigned_int
except ImportError:
    raise ImportError("错误: 无法导入 e3fp，请确保已安装或源码在当前路径下。")

# ================= ⚙️ 配置区域 =================
INPUT_DB_PATH = "../data/3d-pubchem-all.lmdb"
if not os.path.exists(INPUT_DB_PATH):
    INPUT_DB_PATH = "3d-pubchem-all.lmdb"

OUTPUT_DB_PATH = "3d-pubchem-all-e3fp.lmdb"

# 并行进程数 (默认使用 CPU 核心数 - 2，防止卡死系统)
NUM_WORKERS = max(1, cpu_count() - 2)
# 每次提交给进程的任务块大小
CHUNK_SIZE = 100

FP_BITS = 4096
FP_LEVEL = 3


# ===============================================

def identifier_to_bit(identifier: int, bits=4096):
    return signed_to_unsigned_int(identifier) % bits


def worker_process_molecule(data_tuple):
    """
    工作进程函数：接收原始数据，返回处理后的数据
    输入: (key, value_bytes)
    输出: (key, result_bytes) 或 None
    """
    key, value_bytes = data_tuple

    try:
        # 1. 反序列化 (在子进程中进行，减少主进程负担)
        record = pickle.loads(value_bytes)

        smiles = record.get('smiles')
        coordinates_list = record.get('coordinates')
        original_atoms_list = record.get('atoms')

        # --- E3FP 计算逻辑 (复用之前的稳健逻辑) ---
        if not smiles or not coordinates_list:
            return None

        # 坐标预处理
        coords_array = np.array(coordinates_list[0], dtype=np.float64)
        coords_array = coords_array - coords_array.mean(axis=0)  # 中心化

        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        mol = Chem.AddHs(mol)
        mol.SetProp("_Name", str(smiles))  # 修复 _Name

        # 校验
        num_atoms = mol.GetNumAtoms()
        if num_atoms != len(coords_array):
            return None

        if original_atoms_list:
            rdkit_atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            if len(rdkit_atoms) == len(original_atoms_list):
                if tuple(rdkit_atoms) != tuple(original_atoms_list):
                    return None

        # 注入坐标
        conf = Chem.Conformer(num_atoms)
        for i in range(num_atoms):
            conf.SetAtomPosition(i, Point3D(float(coords_array[i][0]),
                                            float(coords_array[i][1]),
                                            float(coords_array[i][2])))
        mol.AddConformer(conf)

        # 计算特征
        fprint_params = {
            'bits': FP_BITS, 'rdkit_invariants': True, 'level': FP_LEVEL,
            'all_iters': True, 'exclude_floating': False
        }

        # 禁止 e3fp 在子进程疯狂打印 log
        import logging
        logging.getLogger('e3fp').setLevel(logging.ERROR)

        _, fingerprinter = fprints_from_mol_verbose(mol, fprint_params=fprint_params)

        feature_matrix = -1 * np.ones((num_atoms, FP_LEVEL + 1), dtype=np.int32)

        for shell in fingerprinter.all_shells:
            c_idx = int(shell.center_atom)
            r_idx = int(shell.radius)
            if c_idx < num_atoms and r_idx <= FP_LEVEL:
                val = identifier_to_bit(shell.identifier, bits=FP_BITS)
                feature_matrix[c_idx, r_idx] = val

        # --- 更新记录 ---
        record['e3fp'] = feature_matrix

        # 序列化结果返回
        return (key, pickle.dumps(record))

    except Exception as e:
        # 子进程报错不打印，以免刷屏，返回 None 即可
        return None


def main():
    print(f"🚀 启动并行处理 | 进程数: {NUM_WORKERS}")
    print(f"📂 输入: {INPUT_DB_PATH}")
    print(f"💾 输出: {OUTPUT_DB_PATH}")

    if not os.path.exists(INPUT_DB_PATH):
        print("❌ 输入文件不存在")
        return

    # 打开环境
    env_in = lmdb.open(INPUT_DB_PATH, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    # map_size 设大点防止溢出
    env_out = lmdb.open(OUTPUT_DB_PATH, map_size=int(1e12), subdir=False, readonly=False, meminit=False, map_async=True)

    # 1. 扫描已完成的数据 (断点续传核心)
    existing_keys = set()
    with env_out.begin() as txn:
        entries = txn.stat()['entries']
        if entries > 0:
            print(f"🔄 检测到库中已有 {entries} 条数据，正在建立索引以跳过...")
            with txn.cursor() as curs:
                for k in curs.iternext(keys=True, values=False):
                    existing_keys.add(k)
            print(f"   ✅ 成功索引已处理 Key: {len(existing_keys)} 条 (将自动跳过)")

    # 2. 准备任务生成器
    def task_generator():
        with env_in.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor.iternext(keys=True, values=True):
                if key in existing_keys:
                    continue  # 跳过已存在的
                yield (key, value)

    # 计算剩余任务量
    total_in = env_in.stat()['entries']
    total_est = total_in - len(existing_keys)

    if total_est == 0:
        print("🎉 所有数据已处理完毕！无需运行。")
        env_in.close()
        env_out.close()
        return

    # 3. 启动进程池
    success_count = 0
    write_batch_size = 2000

    # --- 修复点：不再使用 with env_out.begin() ---
    # 手动开启第一个事务
    txn_out = env_out.begin(write=True)

    try:
        with Pool(processes=NUM_WORKERS) as pool:
            # 使用 imap_unordered 并行计算
            iterator = pool.imap_unordered(worker_process_molecule, task_generator(), chunksize=CHUNK_SIZE)

            pbar = tqdm(iterator, total=total_est, unit="mol", mininterval=1.0)

            for result in pbar:
                if result is None:
                    continue

                key, pickled_data = result
                txn_out.put(key, pickled_data)
                success_count += 1

                # 定期提交事务
                if success_count % write_batch_size == 0:
                    txn_out.commit()
                    # 立即开启下一轮事务
                    txn_out = env_out.begin(write=True)
                    pbar.set_description(f"New Writes: {success_count}")

            # 循环结束，提交最后剩余的数据
            txn_out.commit()
            print(f"✅ 最后批次提交完成。")

    except Exception as e:
        print(f"❌ 发生异常，尝试回滚最后未提交的事务: {e}")
        txn_out.abort()
    finally:
        env_in.close()
        env_out.close()

    print(f"\n🎉 全部完成! 本次补充处理: {success_count} 条")


if __name__ == "__main__":
    main()