import os
import lmdb
import pickle
import sys
import traceback
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

from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
from model.CAMT5.representation import linearize


def get_atom_mapping(smiles, motif_tokenizer):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return [], smiles  # 🚀 修复1：失败时也把字符串退回去
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

    # 🚀 修复2：强制返回生成 Mapping 的基准字符串！
    return atom_mapping, smi_kekule


def build_lmdb():
    input_lmdb_path = "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_pubchem.lmdb"
    output_lmdb_path = "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_pubchem_final.lmdb"
    vocab_path = "asset/mol_vocabs/vocab_phase2_25k.txt"

    print("🚀 初始化 Tokenizers...")
    motif_tokenizer = MotifTokenizer(vocab_file=vocab_path)
    e3fp_tokenizer = E3FPTokenizer(fp_level=3, fp_bits=4096)

    print(f"📂 打开输入数据库: {input_lmdb_path}")
    env_in = lmdb.open(input_lmdb_path, readonly=True, lock=False)

    # 获取总条目数
    with env_in.begin() as txn:
        total_entries = txn.stat()['entries']

    print(f"📝 创建输出数据库: {output_lmdb_path}")
    env_out = lmdb.open(output_lmdb_path, map_size=int(1e12))

    valid_count = 0
    error_count = 0

    with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
        cursor = txn_in.cursor()

        for key, value in tqdm(cursor.iternext(), total=total_entries, desc="🛠️ 计算 Mapping & E3FP"):
            data = pickle.loads(value)
            smiles = data.get('smiles', '')

            if smiles:
                try:
                    # 🚀 修复3：接收 Mapping 的同时，接收基准字符串
                    mapping, smi_kekule = get_atom_mapping(smiles, motif_tokenizer)
                    data['atom_mapping'] = mapping

                    # 🚀 修复4：把基准字符串硬编码进数据库，供 dataset.py 切词使用！
                    data['smiles_kekule'] = smi_kekule

                    # 🚀 修复5：强制 E3FP 使用这个重排后的基准字符串生成 3D 坐标！
                    e3fp_tensor = e3fp_tokenizer.from_smiles(smi_kekule)

                    data['e3fp'] = e3fp_tensor.numpy()

                    # 写入新库
                    txn_out.put(str(valid_count).encode('utf-8'), pickle.dumps(data))
                    valid_count += 1
                except Exception as e:
                    error_count += 1
                    if error_count <= 3:
                        print(f"\n⚠️ [Debug] 样本处理报错 (SMILES: {smiles[:30]}...): {e}")

        txn_out.put(b'__len__', str(valid_count).encode('utf-8'))

    env_in.close()
    env_out.close()
    print(f"✅ 处理完成！成功写入 {valid_count} 条有效数据，失败 {error_count} 条。")


if __name__ == "__main__":
    build_lmdb()