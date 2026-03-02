import lmdb
import pickle
import sys
import os
import multiprocessing
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
    raise ImportError("❌ 错误: 请确保 model/CAMT5/representation.py 存在")

# ================= ⚙️ 配置区域 =================
INPUT_DB = "../dataset/3d-pubchem-all-e3fp.lmdb"
OUTPUT_DB = "../dataset/3d-pubchem-final.lmdb"
NUM_WORKERS = max(1, cpu_count() - 2)


# ===================================================

def worker_process(item):
    key, value_bytes = item
    try:
        record = pickle.loads(value_bytes)
        raw_smiles = record.get('smiles')
        if not raw_smiles: return None

        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None: return None

        linear_smiles = ""
        atom_mapping = []
        atom_offset = 0

        for sub_smi in raw_smiles.split("."):
            res = linearize(sub_smi)
            frag_str = res[0]

            # 🚀 保持宏观图结构：直接将完整的切块拼接，绝不拆解
            linear_smiles += frag_str + "[.]"

            # 提取 2D 文本 Token 到 3D 原子索引的绝对映射
            if len(res) == 3:
                # 加上全局偏移量，保证多片段分子(A.B)的原子序号不冲突
                shifted_mapping = [[idx + atom_offset for idx in motif] for motif in res[2]]
                atom_mapping.extend(shifted_mapping)
            else:
                # 兜底：处理没有被切断的极简分子，整体作为一个 Token
                m_temp = Chem.MolFromSmiles(sub_smi)
                if m_temp:
                    shifted_mapping = [idx + atom_offset for idx in range(m_temp.GetNumAtoms())]
                    atom_mapping.append(shifted_mapping)

            m_sub = Chem.MolFromSmiles(sub_smi)
            if m_sub:
                atom_offset += m_sub.GetNumAtoms()

        if linear_smiles.endswith("[.]"):
            linear_smiles = linear_smiles[:-3]

        # 完美打包数据
        record['motif_seq'] = f"<bom>{linear_smiles}<eom>"
        record['raw_smiles'] = raw_smiles
        record['atom_to_motif_map'] = atom_mapping

        return key, pickle.dumps(record)

    except Exception:
        return None


def main():
    if not os.path.exists(INPUT_DB): return
    print(f"🚀 开始生成最终数据集: {OUTPUT_DB}")
    env_in = lmdb.open(INPUT_DB, subdir=False, readonly=True, lock=False)
    env_out = lmdb.open(OUTPUT_DB, map_size=int(1e12), subdir=False, readonly=False, meminit=False, map_async=True)

    def data_generator():
        with env_in.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor.iternext(keys=True, values=True):
                yield (k, v)

    total_entries = env_in.stat()['entries']

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        txn_out = env_out.begin(write=True)
        try:
            iterator = pool.imap_unordered(worker_process, data_generator(), chunksize=100)
            success, filtered = 0, 0
            pbar = tqdm(iterator, total=total_entries, unit="mol", desc="处理进度")

            for result in pbar:
                if result is None:
                    filtered += 1
                    pbar.set_description(f"✅ {success} | 🗑️ {filtered}")
                    continue

                key, data = result
                txn_out.put(key, data)
                success += 1

                if success % 5000 == 0:
                    txn_out.commit()
                    txn_out = env_out.begin(write=True)

            txn_out.commit()
        except Exception as e:
            txn_out.abort()
            raise e

    env_in.close()
    env_out.close()
    print(f"🎉 全部打包完成！有效数据: {success}, 丢弃: {filtered}")


if __name__ == "__main__":
    main()