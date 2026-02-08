import os
import sys
import lmdb
import pickle
import multiprocessing
from collections import Counter
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from multiprocessing import Pool, cpu_count
# 屏蔽 RDKit 产生的非致命警告
RDLogger.DisableLog('rdApp.*')

# 确保能导入 model.representation
sys.path.append(os.getcwd())
try:
    from model.representation import Frag, linearize
except ImportError:
    raise ImportError("请确保 model/representation.py 存在于当前目录")

# ================= 配置 =================
INPUT_DB = "3d-pubchem-all-e3fp.lmdb"  # 输入数据集
OUTPUT_MY_VOCAB = "./my_dataset_vocab.txt"  # 输出词表
NUM_WORKERS = max(1, cpu_count() - 2)  # 进程数


# =======================================

def process_batch(data_batch):
    """
    子进程：接收一批数据，返回 (片段集合, 失败次数, 总处理数)
    """
    local_frags = set()
    local_fail = 0
    local_total = 0

    # 实例化 Frag 用于解码校验 (参考代码逻辑)
    frag_processor = Frag()

    for value_bytes in data_batch:
        try:
            record = pickle.loads(value_bytes)
            smiles = record.get('smiles')
            if not smiles: continue

            local_total += 1

            # 1. 标准化 (参考代码是直接读 csv，这里我们加一步 Kekulize 保证稳健性)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue
            raw_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)  # 保持 Kekule 形式

            # 2. 切分并收集 (核心逻辑)
            linear_smiles = ""
            for sub_smi in raw_smiles.split("."):
                frag_str, frag_dict = linearize(sub_smi)
                local_frags.update(frag_dict)  # 更新词表
                linear_smiles += frag_str + "[.]"

            # 去除末尾连接符
            linear_smiles = linear_smiles[:-3]

            # 3. 【补全】一致性校验 (Sanity Check)
            # 这是您提到的参考代码中包含的逻辑：
            # 尝试将切分后的序列还原，看是否和原分子一致
            try:
                result_smiles = frag_processor.decode(linear_smiles)

                # 计算 InChI 进行对比
                mol_orig = Chem.MolFromSmiles(raw_smiles)
                mol_recon = Chem.MolFromSmiles(result_smiles)

                if mol_orig and mol_recon:
                    if Chem.MolToInchi(mol_orig) != Chem.MolToInchi(mol_recon):
                        local_fail += 1
                else:
                    local_fail += 1
            except:
                local_fail += 1

        except Exception:
            pass

    return local_frags, local_fail, local_total


def main():
    if not os.path.exists(INPUT_DB):
        print(f"❌ 错误: 找不到文件 {INPUT_DB}")
        return

    print(f"🚀 开始从数据集生成词表 (含 InChI 校验): {INPUT_DB}")

    # 1. 读取所有数据到内存
    all_values = []
    env = lmdb.open(INPUT_DB, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for _, value in tqdm(cursor.iternext(keys=True, values=True), desc="读取 LMDB"):
            all_values.append(value)
    env.close()

    # 2. 多进程处理
    print(f"🔥 启动 {NUM_WORKERS} 个进程进行提取与校验...")
    chunk_size = len(all_values) // NUM_WORKERS + 1
    chunks = [all_values[i:i + chunk_size] for i in range(0, len(all_values), chunk_size)]

    global_frag_set = set(["[.]"])
    total_fail = 0
    total_processed = 0

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_batch, chunks)

    print("∑ 正在合并结果...")
    for frags, fail_count, count in results:
        global_frag_set.update(frags)
        total_fail += fail_count
        total_processed += count

    print(f"📊 统计报告:")
    print(f"   - 总处理分子: {total_processed}")
    print(f"   - 校验失败数: {total_fail}")
    print(f"   - 失败率: {total_fail / total_processed * 100:.2f}%")
    print(f"   - 唯一片段数: {len(global_frag_set)}")

    # 3. 保存词表
    os.makedirs(os.path.dirname(OUTPUT_MY_VOCAB), exist_ok=True)
    with open(OUTPUT_MY_VOCAB, "w") as f:
        for frag in sorted(list(global_frag_set)):  # 排序一下更好看
            f.write(frag + "\n")

    print(f"💾 专属词表已保存至: {OUTPUT_MY_VOCAB}")


if __name__ == "__main__":
    main()