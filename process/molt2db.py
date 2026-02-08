import lmdb
import pickle
import sys
import os
import multiprocessing
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from multiprocessing import Pool, cpu_count

# 屏蔽 RDKit 产生的非致命警告，保持输出清爽
RDLogger.DisableLog('rdApp.*')

# 确保能导入 model.representation
sys.path.append(os.getcwd())
try:
    from model.representation import Frag, linearize
except ImportError:
    raise ImportError("❌ 错误: 请确保 model/representation.py 存在于当前目录")

# ================= ⚙️ 最终配置区域 =================
INPUT_DB = "3d-pubchem-all-e3fp.lmdb"  # 源数据 (含3D特征)
OUTPUT_DB = "3d-pubchem-final.lmdb"  # 目标数据 (训练用)
# 请确保这里指向您刚刚合并好的词表文件
VOCAB_FILE = "asset/mol_vocabs/frag_pubchem_merged.txt"
NUM_WORKERS = max(1, cpu_count() - 2)  # CPU 核心数
# ===================================================

# 全局变量，用于子进程共享词表
vocab_set = None


def init_worker(vocab_list):
    """子进程初始化：加载词表到内存"""
    global vocab_set
    vocab_set = set(vocab_list)


def worker_process(item):
    """
    处理单条数据：切分 -> 词表检查 -> 结构校验 -> 封装
    """
    key, value_bytes = item
    try:
        record = pickle.loads(value_bytes)
        raw_smiles = record.get('smiles')
        if not raw_smiles: return None

        # 1. 标准化 (Kekulize)
        # 这一步保证了输入给 CAMT5 算法的是标准形式
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None: return None
        smiles_kekule = Chem.MolToSmiles(mol, kekuleSmiles=True)

        # 2. 切分 (Linearize)
        linear_smiles = ""
        all_frags = []

        # 处理可能存在的混合物/盐
        for sub_smi in smiles_kekule.split("."):
            frag_str, frag_dict = linearize(sub_smi)
            linear_smiles += frag_str + "[.]"
            all_frags.extend(frag_dict)

        if linear_smiles.endswith("[.]"):
            linear_smiles = linear_smiles[:-3]

        # 3. 词表兼容性检查 (Safety Check)
        # 理论上现在应该 100% 通过，除非 linearize 产生了非常奇怪的边缘情况
        for frag in all_frags:
            check_frag = frag
            if not check_frag.startswith("["):
                check_frag = f"[{frag}]"

            if check_frag not in vocab_set:
                # 如果这一步还在报错，说明合并词表可能有遗漏，或者格式不对
                return None

        # 4. 结构一致性校验 (InChI Check)
        # 这是为了剔除那 0.37% 无法还原的坏数据，保证训练质量
        frag_processor = Frag()
        result_smiles = frag_processor.decode(linear_smiles)

        # 生成 InChI 比较
        mol_orig = Chem.MolFromSmiles(smiles_kekule)
        mol_recon = Chem.MolFromSmiles(result_smiles)

        if mol_orig is None or mol_recon is None:
            return None

        if Chem.MolToInchi(mol_orig) != Chem.MolToInchi(mol_recon):
            return None

        # 5. 成功：封装数据
        # 添加 CAMT5 必需的起始/结束标记
        final_motif_seq = f"<bom>{linear_smiles}<eom>"

        record['motif_seq'] = final_motif_seq
        record['smiles_kekule'] = smiles_kekule  # 保存一份标准化的 SMILES 备用

        return key, pickle.dumps(record)

    except Exception:
        return None


def main():
    if not os.path.exists(INPUT_DB):
        print(f"❌ 输入数据库不存在: {INPUT_DB}")
        return
    if not os.path.exists(VOCAB_FILE):
        print(f"❌ 词表文件不存在: {VOCAB_FILE}")
        return

    # 1. 加载词表
    print(f"📖 正在加载合并词表: {VOCAB_FILE} ...")
    with open(VOCAB_FILE, 'r') as f:
        vocab_list = [line.strip() for line in f if line.strip()]

    # 确保连接符存在
    if "[.]" not in vocab_list:
        vocab_list.append("[.]")

    print(f"✅ 词表加载完成，共 {len(vocab_list)} 个 Token")

    # 2. 准备处理
    print(f"🚀 开始生成最终数据集: {OUTPUT_DB}")
    env_in = lmdb.open(INPUT_DB, subdir=False, readonly=True, lock=False)
    # map_size 设为 1TB，防止溢出
    env_out = lmdb.open(OUTPUT_DB, map_size=int(1e12), subdir=False, readonly=False, meminit=False, map_async=True)

    def data_generator():
        with env_in.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor.iternext(keys=True, values=True):
                yield (k, v)

    total_entries = env_in.stat()['entries']

    # 3. 多进程处理 (修复版：移除 with 上下文管理器)
    with multiprocessing.Pool(NUM_WORKERS, initializer=init_worker, initargs=(vocab_list,)) as pool:

        # [修复点 1] 手动开启第一个事务，不要用 'with'
        txn_out = env_out.begin(write=True)

        try:
            # 使用 imap 配合 tqdm 显示实时进度
            iterator = pool.imap_unordered(worker_process, data_generator(), chunksize=20)

            success = 0
            filtered = 0

            pbar = tqdm(iterator, total=total_entries, unit="mol", desc="处理进度")
            for result in pbar:
                if result is None:
                    filtered += 1
                    pbar.set_description(f"✅ {success} | 🗑️ {filtered}")
                    continue

                key, data = result
                txn_out.put(key, data)
                success += 1

                # 每 5000 条提交一次事务
                if success % 5000 == 0:
                    txn_out.commit()
                    # [修复点 2] 立即开启下一轮事务
                    txn_out = env_out.begin(write=True)

            # [修复点 3] 循环结束，提交最后剩余的数据
            txn_out.commit()
            print("✅ 最后批次提交完成。")

        except Exception as e:
            # 发生意外时回滚
            print(f"❌ 处理中断，正在回滚未提交的事务: {e}")
            txn_out.abort()
            raise e

    env_in.close()
    env_out.close()

    print(f"\n{'=' * 30}")
    print(f"🎉 全部完成！")
    print(f"✅ 最终有效数据: {success}")
    print(f"🗑️ 过滤坏数据: {filtered} (占比 {filtered / total_entries * 100:.2f}%)")
    print(f"💾 数据已保存至: {OUTPUT_DB}")
    print(f"{'=' * 30}")
    print(f"💡 接下来，您可以在 DataLoader 中使用 '{OUTPUT_DB}' 进行训练了！")


if __name__ == "__main__":
    main()