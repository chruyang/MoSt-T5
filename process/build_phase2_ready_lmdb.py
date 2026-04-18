import os
import lmdb
import pickle
import sys
import multiprocessing
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger

# 关闭 RDKit 烦人的底层警告
RDLogger.DisableLog('rdApp.*')

# 挂载路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tokenization.e3fp_tokenizer import E3FPTokenizer
from model.CAMT5.representation import linearize


# 移除 motif_tokenizer 参数，彻底解耦
def get_atom_mapping(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return [], smiles
    try:
        Chem.Kekulize(mol)
        smi_kekule = Chem.MolToSmiles(mol, kekuleSmiles=True)
    except:
        smi_kekule = smiles

    atom_mapping = []
    atom_offset = 0
    sub_mols = smi_kekule.split(".")
    for sub_smi in sub_mols:
        try:
            _, _, mapping = linearize(sub_smi)
            for frag_indices in mapping:
                atom_mapping.append([idx + atom_offset for idx in frag_indices])
            m_tmp = Chem.MolFromSmiles(sub_smi)
            if m_tmp: atom_offset += m_tmp.GetNumAtoms()
        except:
            pass

    return atom_mapping, smi_kekule


# ========================================================
# 🚀 多进程 Worker 函数：每个 CPU 核心独立计算化学力场
# ========================================================
# 因为多进程无法传递复杂的 Tokenizer 实例，我们在进程内部懒加载
global_e3fp_tokenizer = None


def init_worker():
    global global_e3fp_tokenizer
    # 在每个子进程中初始化 E3FP 计算引擎
    global_e3fp_tokenizer = E3FPTokenizer(fp_level=3, fp_bits=4096)


def process_single_molecule(item):
    key, value = item
    try:
        data = pickle.loads(value)
        smiles = data.get('smiles', '')
        if not smiles:
            return None

        # 1. 生成 Mapping 和基准 SMILES
        mapping, smi_kekule = get_atom_mapping(smiles)
        data['atom_mapping'] = mapping
        data['smiles_kekule'] = smi_kekule

        # 2. 算 3D 特征 (极其耗时，由多进程分担)
        e3fp_tensor = global_e3fp_tokenizer.from_smiles(smi_kekule)

        # 3D 构象生成失败防范
        if e3fp_tensor.shape[0] == 0:
            return None

        data['e3fp'] = e3fp_tensor.numpy()

        return pickle.dumps(data)
    except Exception as e:
        return None


# ========================================================
# 主控管线
# ========================================================
def build_lmdb():
    input_lmdb_path = "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_pubchem.lmdb"
    output_lmdb_path = "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_pubchem_final.lmdb"

    # 获取可用 CPU 核心数，留 2 个核心给系统调度
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    commit_batch = 10000  # 每处理完 1 万条，写一次硬盘防 OOM

    print(f"📂 打开输入数据库: {input_lmdb_path}")
    env_in = lmdb.open(input_lmdb_path, readonly=True, lock=False)

    with env_in.begin() as txn:
        total_entries = txn.stat()['entries']

    print(f"📝 创建输出数据库: {output_lmdb_path} (启用 {num_workers} 个 CPU 核心加速)")
    env_out = lmdb.open(output_lmdb_path, map_size=int(1e12))

    # 任务生成器
    def task_generator():
        with env_in.begin() as txn_in:
            for key, value in txn_in.cursor().iternext():
                yield (key, value)

    txn_out = env_out.begin(write=True)
    valid_count = 0

    # 🚀 启动多进程引擎
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker) as pool:
        iterator = pool.imap_unordered(process_single_molecule, task_generator(), chunksize=500)

        for result_bytes in tqdm(iterator, total=total_entries, desc="🛠️ 并行计算 3D & Mapping"):
            if result_bytes is not None:
                # 写入连续的 ID
                txn_out.put(str(valid_count).encode('utf-8'), result_bytes)
                valid_count += 1

                # 🛡️ 内存救星：分块落盘
                if valid_count % commit_batch == 0:
                    txn_out.commit()
                    txn_out = env_out.begin(write=True)

    # 提交最后一点尾巴数据
    txn_out.put(b'__len__', str(valid_count).encode('utf-8'))
    txn_out.commit()

    env_in.close()
    env_out.close()
    print(f"✅ 处理完成！利用多核加速，成功写入 {valid_count} 条有效数据。")


if __name__ == "__main__":
    build_lmdb()