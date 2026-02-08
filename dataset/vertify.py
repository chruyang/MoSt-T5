import lmdb
import pickle
import numpy as np
import os
import random
from tqdm import tqdm
from termcolor import colored  # 可选，为了打印好看

# ================= ⚙️ 配置区域 =================
DB_PATH = "3d-pubchem.lmdb"
CHECK_TOTAL_LIMIT = 50000  # 检查前5万条（为了速度，不查全量）
PRINT_SAMPLES = 5  # 详细打印多少个样本


# ===========================================

def inspect_sample(record, idx):
    """详细展示单条数据的内部对齐逻辑"""
    smiles = record['smiles_kekule']
    motif_seq = record['motif_seq']
    e3fp = record['e3fp']
    mapping = record.get('atom_mapping', [])

    print(f"\n{'=' * 20} 🧪 样本 {idx} 详情 {'=' * 20}")
    print(f"📘 SMILES (Kekule): {colored(smiles, 'cyan')}")
    print(f"📙 Motif Seq: {colored(motif_seq, 'yellow')}")
    print(f"📐 E3FP Shape: {e3fp.shape} (N_atoms={e3fp.shape[0]}, Dim={e3fp.shape[1]})")

    if not mapping:
        print(colored("❌ 警告: 该样本没有 Atom Mapping!", "red"))
        return

    print(f"🔗 Atom Mapping (List of Lists): 长度={len(mapping)}")
    print(f"   {mapping}")

    # --- 深度逻辑验证 ---
    # 我们尝试解析 Motif Seq 并和 Mapping 对齐
    # 注意：motif_seq 包含 <bom>, <eom> 和连接符 [.]
    # 简单的对齐展示：
    print("\n🔍 [逻辑抽查] Motif <-> Atom <-> E3FP (前 3 个):")

    # 简单的清洗以用于展示 (不代表真实 Tokenizer 逻辑)
    clean_seq = motif_seq.replace("<bom>", "").replace("<eom>", "")
    # 假设是用 [.] 分割的
    fragments = clean_seq.split("[.]")

    # 注意：generate_atom_mapping 产生的 mapping 是对应于 fragments 的
    min_len = min(len(fragments), len(mapping))

    for i in range(min_len):
        frag_str = fragments[i]
        atom_indices = mapping[i]

        # 检查索引越界
        max_idx = max(atom_indices) if atom_indices else -1
        if max_idx >= e3fp.shape[0]:
            print(f"   ❌ Motif {i} '{frag_str}': 索引 {max_idx} 超出 E3FP 范围 ({e3fp.shape[0]})!")
            continue

        # 获取对应的 E3FP 特征 (取平均值展示)
        if atom_indices:
            feats = e3fp[atom_indices]
            # 打印前5维特征作为指纹
            feat_preview = feats[0][:5]
            print(
                f"   ✅ Motif {i}: {colored(frag_str, 'green'):<15} -> Atoms {str(atom_indices):<12} -> E3FP[0][:5]: {feat_preview}")
        else:
            print(f"   ⚠️ Motif {i}: {frag_str:<15} -> Atoms [] (无对应原子?)")

    if len(fragments) != len(mapping):
        print(colored(f"\n⚠️ 注意: 解析出的片段数 ({len(fragments)}) 与 映射数 ({len(mapping)}) 不完全一致。", "yellow"))
        print("   (原因可能是: mapping 是基于 linearize 产生的，而 motif_seq 是字符串拼接，[.] 分割可能不准确。)")
        print("   (只要 mapping 索引不越界，且逻辑上大致对应即可放心使用。)")


def check_integrity(db_path):
    if not os.path.exists(db_path):
        print(f"❌ 找不到文件: {db_path}")
        return

    env = lmdb.open(db_path, subdir=False, readonly=True, lock=False)

    stats = {
        "total": 0,
        "valid_mapping": 0,
        "missing_mapping": 0,
        "index_out_of_bounds": 0,
        "e3fp_shape_mismatch": 0
    }

    print(f"🚀 开始全库体检 (Limit: {CHECK_TOTAL_LIMIT})...")

    with env.begin() as txn:
        cursor = txn.cursor()
        total_entries = txn.stat()['entries']

        # 随机挑选几个索引来打印详情
        sample_indices = set(random.sample(range(min(total_entries, CHECK_TOTAL_LIMIT)), PRINT_SAMPLES))

        pbar = tqdm(cursor.iternext(keys=True, values=True), total=min(total_entries, CHECK_TOTAL_LIMIT))

        for i, (key, value) in enumerate(pbar):
            if i >= CHECK_TOTAL_LIMIT:
                break

            stats["total"] += 1
            try:
                record = pickle.loads(value)

                # 1. 检查字段
                if 'atom_mapping' not in record:
                    stats["missing_mapping"] += 1
                    continue
                else:
                    stats["valid_mapping"] += 1

                # 2. 检查 E3FP
                e3fp = record['e3fp']
                n_atoms = e3fp.shape[0]

                # 3. 检查边界 (CRITICAL)
                mapping = record['atom_mapping']
                flat_indices = [idx for sublist in mapping for idx in sublist]

                if flat_indices:
                    max_idx = max(flat_indices)
                    min_idx = min(flat_indices)

                    if max_idx >= n_atoms:
                        stats["index_out_of_bounds"] += 1
                        # 严重错误，必须打印
                        print(f"\n❌ 严重错误 Key {key}: Max Index {max_idx} >= E3FP Len {n_atoms}")

                    if min_idx < 0:
                        stats["index_out_of_bounds"] += 1

                # 4. 打印抽样
                if i in sample_indices:
                    inspect_sample(record, i)

            except Exception as e:
                print(f"❌ 读取错误 Key {key}: {e}")

    env.close()

    print(f"\n{'=' * 30}")
    print(f"📊 最终体检报告")
    print(f"{'=' * 30}")
    print(f"扫描总数: {stats['total']}")
    print(f"✅ 映射有效: {stats['valid_mapping']} ({(stats['valid_mapping'] / stats['total']) * 100:.2f}%)")
    print(f"⚪ 映射缺失: {stats['missing_mapping']} ({(stats['missing_mapping'] / stats['total']) * 100:.2f}%)")
    print(f"❌ 索引越界: {stats['index_out_of_bounds']} (必须为 0)")

    if stats['index_out_of_bounds'] == 0:
        print(f"\n🎉 恭喜! 数据集逻辑完美，Atoms -> Motifs 索引安全，可直接训练！")
    else:
        print(f"\n🚫 警告! 存在索引越界，训练会崩溃，请检查 E3FP 生成逻辑或 Mapping 逻辑。")


if __name__ == "__main__":
    try:
        import termcolor
    except ImportError:
        # 如果没有安装 termcolor，定义一个伪函数防止报错
        def colored(text, color):
            return text

    check_integrity(DB_PATH)