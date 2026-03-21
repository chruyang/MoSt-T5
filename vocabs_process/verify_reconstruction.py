import os
import sys
import lmdb
import pickle
import random
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger

# 关闭 RDKit 烦人的底层警告
RDLogger.DisableLog('rdApp.*')

# ================= 动态包路径挂载 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from model.CAMT5.representation import linearize, decode_linear
except ImportError:
    raise ImportError("请确保项目根目录下存在 model/CAMT5/representation.py")

# ================= 绝对路径与配置 =================
BASE_DIR = os.path.expanduser("~/autodl-tmp/3D-MoIT/3d-mol-dataset")
PUBCHEMQC_LMDB = f"{BASE_DIR}/pubchemqc/pubchemqc_database.lmdb"
NUM_SAMPLES = 10000  # 随机抽取 1万个分子进行严酷测试

def verify_molecule(raw_smiles):
    """
    核心验证逻辑：测试分子能否经历“粉碎 -> 重组”的完美闭环
    """
    # 1. 原始分子标准化 (因为 linearize 内部会去除立体化学)
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None: 
        return True, "Invalid original SMILES" # 忽略原本就坏掉的数据
    
    Chem.RemoveStereochemistry(mol)
    standard_smi = Chem.MolToSmiles(mol) # 使用 RDKit 默认的权威 Canonical 格式

    # 2. 编码：粉碎成带有虚拟锚点的 1D 序列
    try:
        frag_str, _, _ = linearize(standard_smi)
    except Exception as e:
        return False, f"Linearize Error: {e}"

    # 3. 解码：根据序列中的 <0*>, <1*> 等锚点重新拼装
    try:
        recon_smi = decode_linear(frag_str)
        recon_mol = Chem.MolFromSmiles(recon_smi)
        if recon_mol is None:
            return False, "Decode produced invalid SMILES"
            
        Chem.RemoveStereochemistry(recon_mol)
        final_recon_smi = Chem.MolToSmiles(recon_mol)
    except Exception as e:
        return False, f"Decode Error: {e}"

    # 4. 终极审判：重组后的图拓扑，必须与原始图拓扑 100% 一致！
    if standard_smi == final_recon_smi:
        return True, ""
    else:
        return False, f"Mismatch!\nOrig: {standard_smi}\nRecon: {final_recon_smi}\nFragStr: {frag_str}"

def main():
    print(f"🚀 开始进行 MoSt-T5 架构的分子复原闭环验证 (Sanity Check)...")
    if not os.path.exists(PUBCHEMQC_LMDB):
        print(f"❌ 找不到 LMDB 数据库！")
        return

    # 从数据库中快速提取所有 SMILES
    all_smiles = []
    env = lmdb.open(PUBCHEMQC_LMDB, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for idx, (key, val_bytes) in enumerate(cursor):
            if idx > 100000: break # 只扫前 10 万条来做随机池，加快速度
            record = pickle.loads(val_bytes)
            smi = record.get('smi')
            if smi: all_smiles.append(smi)
    env.close()

    # 随机采样
    samples = random.sample(all_smiles, min(NUM_SAMPLES, len(all_smiles)))
    print(f"🧪 成功抽取 {len(samples):,} 个分子作为验证集。")

    success_count = 0
    fail_count = 0
    failed_examples = []

    # 单线程遍历，方便捕获错误 (一万条通常不到 1 分钟就能跑完)
    for smi in tqdm(samples, desc="验证中"):
        is_success, error_msg = verify_molecule(smi)
        if is_success:
            success_count += 1
        else:
            fail_count += 1
            failed_examples.append((smi, error_msg))

    # ================= 输出报告 =================
    print("\n" + "="*40)
    print(f"📊 闭环验证报告")
    print("="*40)
    print(f"✅ 测试总数: {len(samples):,}")
    print(f"🟢 完美复原: {success_count:,}")
    print(f"🔴 复原失败: {fail_count:,}")
    
    success_rate = (success_count / len(samples)) * 100
    print(f"🏆 完美重组率 (Exact Match): {success_rate:.2f}%")

    if fail_count > 0:
        print("\n⚠️ 失败样例追溯 (前 3 个):")
        for orig, err in failed_examples[:3]:
            print(f"   [原分子] {orig}")
            print(f"   [错误详情] {err}\n")

if __name__ == "__main__":
    main()