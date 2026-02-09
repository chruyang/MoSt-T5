import lmdb
import pickle
import random
import os
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def split_lmdb_3way(source_path, train_path, valid_path, test_path,
                    ratios=(0.8, 0.1, 0.1), seed=42):
    """
    将单个 LMDB 数据集分割为训练集、验证集和测试集。

    Args:
        source_path: 源 LMDB 路径
        train_path: 输出训练集路径
        valid_path: 输出验证集路径
        test_path:  输出测试集路径
        ratios: (train_ratio, valid_ratio, test_ratio) tuple, sum should be 1.0 or close
        seed: 随机种子
    """

    if not os.path.exists(source_path):
        logger.error(f"❌ Source file not found: {source_path}")
        return

    # 归一化比例（防止用户输入 90, 5, 5 这种）
    total_ratio = sum(ratios)
    ratios = [r / total_ratio for r in ratios]

    logger.info(f"🚀 Starting 3-way split: {source_path}")
    logger.info(f"   Ratios (Train/Valid/Test): {ratios[0]:.2f} / {ratios[1]:.2f} / {ratios[2]:.2f}")

    # 1. 读取所有 Keys
    env_src = lmdb.open(source_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
    with env_src.begin() as txn:
        try:
            length = int(txn.get(b'__len__'))
        except:
            length = txn.stat()['entries']

        # 生成索引列表
        all_indices = list(range(length))

    env_src.close()

    # 2. 随机打乱
    random.seed(seed)
    random.shuffle(all_indices)

    # 3. 计算分割点
    n_total = len(all_indices)
    n_train = int(n_total * ratios[0])
    n_valid = int(n_total * ratios[1])
    # 剩下的都给 test，确保总数对齐

    train_indices = all_indices[:n_train]
    valid_indices = all_indices[n_train: n_train + n_valid]
    test_indices = all_indices[n_train + n_valid:]

    logger.info(f"📊 Total: {n_total}")
    logger.info(f"   Train: {len(train_indices)}")
    logger.info(f"   Valid: {len(valid_indices)}")
    logger.info(f"   Test : {len(test_indices)}")

    # 4. 定义写入函数 (复用之前的逻辑)
    def write_dataset(indices, output_path, desc_key_priority=['enriched_description', 'description']):
        if os.path.exists(output_path):
            os.remove(output_path)

        # 修改为：
        import psutil

        def calculate_optimal_map_size(num_entries, avg_entry_size=2048):
            """根据数据量动态计算合适的map_size"""
            # 估算总大小 = 条目数 × 平均大小 × 安全系数
            estimated_size = num_entries * avg_entry_size * 3  # 3倍安全系数
            # 获取可用磁盘空间
            disk_usage = psutil.disk_usage(os.path.dirname(os.path.abspath(output_path)))
            available_space = disk_usage.free

            # 取估算值和可用空间的较小值，但至少保证100MB
            optimal_size = min(estimated_size, available_space // 2)  # 使用一半可用空间
            optimal_size = max(optimal_size, 100 * 1024 * 1024)  # 至少100MB

            return optimal_size

        # 在write_dataset函数中使用
        map_size = calculate_optimal_map_size(len(indices))
        env_out = lmdb.open(output_path, map_size=map_size, subdir=False)
        env_src = lmdb.open(source_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)

        with env_src.begin() as txn_src, env_out.begin(write=True) as txn_out:
            for new_idx, original_idx in enumerate(tqdm(indices, desc=f"Writing {os.path.basename(output_path)}")):
                # 读取
                data_bytes = txn_src.get(str(original_idx).encode())
                if not data_bytes: continue
                entry = pickle.loads(data_bytes)

                # --- 数据清洗: Description 选择 ---
                # 您的数据有 description 和 enriched_description
                # 策略: 优先用 enriched，没有则用 description，统一存为 'text'
                text_content = ""
                for k in desc_key_priority:
                    if k in entry and entry[k] and isinstance(entry[k], str) and len(entry[k].strip()) > 0:
                        text_content = entry[k]
                        break
                entry['text'] = text_content

                # --- 写入 ---
                # Key 重置为连续整数 0, 1, 2...
                txn_out.put(str(new_idx).encode(), pickle.dumps(entry))

                if new_idx % 1000 == 0: pass

            # 写入长度元数据
            txn_out.put(b'__len__', str(len(indices)).encode())

        env_src.close()
        env_out.close()
        logger.info(f"✅ Saved to {output_path}")

    # 5. 执行写入
    write_dataset(train_indices, train_path)
    write_dataset(valid_indices, valid_path)
    write_dataset(test_indices, test_path)


if __name__ == "__main__":
    # 配置路径
    # 请根据您 view_lmdb.py 的路径调整
    SOURCE_LMDB = "../dataset/3d-pubchem.lmdb"

    TRAIN_LMDB = "../dataset/3d-pubchem-train.lmdb"
    VALID_LMDB = "../dataset/3d-pubchem-valid.lmdb"
    TEST_LMDB = "../dataset/3d-pubchem-test.lmdb"

    # 分割比例: 80% 训练, 10% 验证, 10% 测试
    split_lmdb_3way(SOURCE_LMDB, TRAIN_LMDB, VALID_LMDB, TEST_LMDB, ratios=(0.80, 0.1, 0.1))