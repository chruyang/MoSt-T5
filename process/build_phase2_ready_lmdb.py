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
    if not mol: return []
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
    return atom_mapping

def main():
    input_lmdb = "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/3d-pubchem.lmdb"
    output_lmdb = "/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_pubchem_ready.lmdb"
    
    # 🚨 词表与特征器初始化
    vocab_path = os.path.join(PROJECT_ROOT, "asset/mol_vocabs/vocab_phase2_25k.txt")
    
    print("🚀 加载 Motif 与 E3FP Tokenizers...")
    motif_tokenizer = MotifTokenizer(vocab_file=vocab_path, model_name="google/t5-v1_1-base")
    e3fp_tokenizer = E3FPTokenizer() # 初始化 E3FP 计算器

    env_in = lmdb.open(input_lmdb, subdir=False, readonly=True, lock=False)
    env_out = lmdb.open(output_lmdb, subdir=False, map_size=50*1024*1024*1024)
    
    valid_count = 0
    error_count = 0
    
    with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
        total_entries = txn_in.stat()['entries']
        cursor = txn_in.cursor()
        
        for key, value in tqdm(cursor, total=total_entries, desc="🛠️ 计算 Mapping & E3FP"):
            data = pickle.loads(value)
            smiles = data.get('smiles', '')
            
            if smiles:
                try:
                    # 1. 替代 process_qc_step2_mapping.py 的功能
                    mapping = get_atom_mapping(smiles, motif_tokenizer)
                    data['atom_mapping'] = mapping
                    
                    # 2. 替代 process_qc_step1_e3fp.py 的功能 (预计算 3D 矩阵)
                    # 🚀 核心修复：强制使用 from_smiles，避免不存在的方法报错
                    e3fp_tensor = e3fp_tokenizer.from_smiles(smiles)
                        
                    data['e3fp'] = e3fp_tensor.numpy() # 转回 numpy 存储以节省 LMDB 空间

                    # 3. 写入新库 (原封不动保留了 description 文本)
                    txn_out.put(str(valid_count).encode('utf-8'), pickle.dumps(data))
                    valid_count += 1
                except Exception as e:
                    error_count += 1
                    # 打印前 3 个错误以防再次出现不可见的致命错误
                    if error_count <= 3:
                        print(f"\n⚠️ [Debug] 样本处理报错 (SMILES: {smiles[:30]}...): {e}")
                        traceback.print_exc()
                
        txn_out.put(b'__len__', str(valid_count).encode('utf-8'))
        
    env_in.close()
    env_out.close()
    
    print(f"\n✅ Phase 2 满血版 LMDB 构建完成！")
    print(f"   -> 成功处理并入库: {valid_count} 条多模态数据。")
    print(f"   -> 异常跳过: {error_count} 条。")
    print(f"💾 保存在: {output_lmdb}")

if __name__ == "__main__":
    main()