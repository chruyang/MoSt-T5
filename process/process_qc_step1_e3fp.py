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
except ImportError as e:
    raise ImportError(f"❌ 错误: 无法导入 e3fp，详细报错: {e}")

# ================= ⚙️ PubChemQC 专属路径配置 =================
BASE_DIR = os.path.expanduser("~/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchemqc")
INPUT_DB_PATH = os.path.join(BASE_DIR, "pubchemqc_database.lmdb")
OUTPUT_DB_PATH = os.path.join(BASE_DIR, "pubchemqc_e3fp.lmdb")

NUM_WORKERS = max(1, cpu_count() - 2)
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

        # 提取量子计算的三维坐标
        coords_array = np.array(coordinates_list[0], dtype=np.float64)
        coords_array = coords_array - coords_array.mean(axis=0)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None

        # 🚀 核心修复：绝对禁止调用 Chem.AddHs(mol)！
        # 保持重原子骨架，完美匹配 PubChemQC 提供的坐标长度！
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

        feature_matrix = -1 * np.ones((num_atoms, FP_LEVEL + 1), dtype=np.int32)
        for shell in fingerprinter.all_shells:
            c_idx, r_idx = int(shell.center_atom), int(shell.radius)
            if c_idx < num_atoms and r_idx <= FP_LEVEL:
                feature_matrix[c_idx, r_idx] = identifier_to_bit(shell.identifier, bits=FP_BITS)

        # 写入标准化字段
        record['smiles'] = smiles
        record['e3fp'] = feature_matrix

        return (key, pickle.dumps(record))
    except Exception:
        return None


def main():
    print(f"🚀 开始提取 PubChemQC 3D E3FP 特征 (纯重原子 QM 坐标模式)...")

    if not os.path.exists(INPUT_DB_PATH):
        print(f"❌ 找不到输入 LMDB 文件: {INPUT_DB_PATH}")
        return

    env_in = lmdb.open(INPUT_DB_PATH, subdir=False, readonly=True, lock=False)
    env_out = lmdb.open(OUTPUT_DB_PATH, map_size=int(1e12), subdir=False, readonly=False, map_async=True)

    with env_in.begin() as txn:
        total_in = txn.stat()['entries']
        keys_values = [(k, v) for k, v in txn.cursor().iternext(keys=True, values=True)]

    txn_out = env_out.begin(write=True)
    success_count = 0
    with Pool(processes=NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(worker_process_molecule, keys_values, chunksize=100), total=total_in):
            if result:
                txn_out.put(result[0], result[1])
                success_count += 1
                if success_count % 5000 == 0:
                    txn_out.commit()
                    txn_out = env_out.begin(write=True)
        txn_out.commit()
    env_in.close();
    env_out.close()
    print(f"🎉 E3FP 提取完成！有效数据: {success_count}")


if __name__ == "__main__":
    main()