import lmdb
import pickle
import random
import os
import logging
import psutil
from tqdm import tqdm

# 配置日志
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def split_lmdb_3way(source_path, train_path, valid_path, test_path,
                    ratios=(0.8, 0.1, 0.1), seed=42):
    """
    将单个 LMDB 数据集分割为训练集、验证集和测试集。
    """

    if not os.path.exists(source_path):
        logger.error(f"❌ Source file not found: {source_path}")
        return

    # 归一化比例
    total_ratio = sum(ratios)
    ratios = [r / total_ratio for r in ratios]

    logger.info(f"🚀 Starting 3-way split: {source_path}")
    logger.info(f"   Ratios (Train/Valid/Test): {ratios[0]:.2f} / {ratios[1]:.2f} / {ratios[2]:.2f}")

    # ================================================================
    # 💡 核心修复 1: 不要用 range，直接用 cursor 获取所有真实的 Key
    # ================================================================
    logger.info("Reading all valid keys from source LMDB...")
    env_src = lmdb.open(source_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)

    all_keys = []
    with env_src.begin() as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            if key != b'__len__':  # 排除长度元数据
                all_keys.append(key)
    env_src.close()

    n_total = len(all_keys)
    logger.info(f"📊 Found {n_total} real data entries in source.")

    # 2. 随机打乱真实的 Keys
    random.seed(seed)
    random.shuffle(all_keys)

    # 3. 计算分割点
    n_train = int(n_total * ratios[0])
    n_valid = int(n_total * ratios[1])

    train_keys = all_keys[:n_train]
    valid_keys = all_keys[n_train: n_train + n_valid]
    test_keys = all_keys[n_train + n_valid:]

    logger.info(f"   Train: {len(train_keys)}")
    logger.info(f"   Valid: {len(valid_keys)}")
    logger.info(f"   Test : {len(test_keys)}")

    # 4. 定义写入函数
    def write_dataset(keys_to_write, output_path, desc_key_priority=['enriched_description', 'description']):
        logger.info(f"\n📝 Processing {os.path.basename(output_path)} ({len(keys_to_write)} entries)")
        
        if os.path.exists(output_path):
            logger.info(f"   Removing existing file: {output_path}")
            os.remove(output_path)

        def calculate_optimal_map_size(num_entries, avg_entry_size=4096):
            # 增加平均条目大小估计（从2048到4096）
            estimated_size = num_entries * avg_entry_size * 5  # 增加安全系数
            disk_usage = psutil.disk_usage(os.path.dirname(os.path.abspath(output_path)))
            available_space = disk_usage.free
            # 使用更保守的空间分配策略
            optimal_size = min(estimated_size, available_space // 3)  # 改为使用1/3可用空间
            optimal_size = max(optimal_size, 500 * 1024 * 1024)  # 提高最小值到500MB
            logger.info(f"   Estimated map_size: {optimal_size / (1024*1024):.1f} MB for {num_entries} entries")
            return optimal_size

        map_size = calculate_optimal_map_size(len(keys_to_write))
        logger.info(f"   Opening LMDB with map_size: {map_size / (1024*1024):.1f} MB")
        
        env_src = lmdb.open(source_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
        
        # 分两步进行：先写入数据，再写入长度信息
        # 第一步：写入所有数据
        env_out = lmdb.open(output_path, map_size=map_size, subdir=False)
        valid_count = 0
        
        with env_src.begin() as txn_src:
            pbar = tqdm(keys_to_write, desc=f"Writing {os.path.basename(output_path)}")
            for original_key in pbar:
                try:
                    data_bytes = txn_src.get(original_key)
                    if not data_bytes: 
                        continue

                    entry = pickle.loads(data_bytes)

                    # --- 数据清洗 ---
                    text_content = ""
                    for k in desc_key_priority:
                        if k in entry and entry[k] and isinstance(entry[k], str) and len(entry[k].strip()) > 0:
                            text_content = entry[k]
                            break
                    entry['text'] = text_content

                    # --- 写入新库 ---
                    # 直接在循环中写入，不使用事务批处理以避免复杂性
                    with env_out.begin(write=True) as txn_out:
                        txn_out.put(str(valid_count).encode(), pickle.dumps(entry))
                    valid_count += 1
                    pbar.set_postfix({'written': valid_count})
                        
                except Exception as e:
                    logger.warning(f"   Warning: Failed to process key {original_key}: {e}")
                    continue
        
        env_src.close()
        env_out.close()
        
        # 第二步：写入长度信息
        env_out = lmdb.open(output_path, map_size=map_size, subdir=False)
        with env_out.begin(write=True) as txn_final:
            txn_final.put(b'__len__', str(valid_count).encode())
        env_out.close()
        
        logger.info(f"✅ Saved {valid_count} entries to {output_path}")

    # 5. 执行写入
    write_dataset(train_keys, train_path)
    write_dataset(valid_keys, valid_path)
    write_dataset(test_keys, test_path)


if __name__ == "__main__":
    SOURCE_LMDB = "../dataset/3d-pubchem-final.lmdb"

    TRAIN_LMDB = "../dataset/3d-pubchem-train.lmdb"
    VALID_LMDB = "../dataset/3d-pubchem-valid.lmdb"
    TEST_LMDB = "../dataset/3d-pubchem-test.lmdb"

    split_lmdb_3way(SOURCE_LMDB, TRAIN_LMDB, VALID_LMDB, TEST_LMDB, ratios=(0.80, 0.1, 0.1))