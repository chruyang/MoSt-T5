import os
import lmdb
from tqdm import tqdm

# ================= 配置 =================
# 指向您刚刚用 molt2db.py 生成的最终大数据库
INPUT_DB = "../dataset/3d-pubchem-final.lmdb"
# 新的测试小数据库路径
OUTPUT_DB = "../dataset/3d-pubchem-mini.lmdb"
NUM_SAMPLES = 100  # 提取 100 条数据用于过拟合测试


# =======================================

def create_mini_lmdb():
    if not os.path.exists(INPUT_DB):
        print(f"❌ 找不到输入文件: {INPUT_DB}")
        return

    print(f"🚀 开始从 {INPUT_DB} 提取前 {NUM_SAMPLES} 条数据...")

    # 打开源数据库 (只读)
    env_in = lmdb.open(INPUT_DB, subdir=False, readonly=True, lock=False)
    # 创建新数据库 (设置一个小一点的 map_size，100MB 绝对够了)
    env_out = lmdb.open(OUTPUT_DB, subdir=False, map_size=104857600, readonly=False, meminit=False, map_async=True)

    count = 0
    with env_in.begin() as txn_in:
        cursor = txn_in.cursor()
        with env_out.begin(write=True) as txn_out:
            for key, value in tqdm(cursor.iternext(keys=True, values=True), total=NUM_SAMPLES, desc="复制数据"):
                txn_out.put(key, value)
                count += 1
                if count >= NUM_SAMPLES:
                    break

    env_in.close()
    env_out.close()
    print(f"🎉 成功创建 Mini LMDB！文件保存在: {OUTPUT_DB}，共包含 {count} 条数据。")


if __name__ == "__main__":
    create_mini_lmdb()