import polaris as po
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

print("正在获取数据集元信息...")
dataset = po.load_dataset("asap-discovery/antiviral-admet-2025-unblinded")

TARGET_ENDPOINT = "HLM"

# ==========================================
# 增加：防断线、自动重试机制
# ==========================================
def fetch_row_with_retry(row_idx, max_retries=5):
    for attempt in range(max_retries):
        try:
            return {
                'smiles': dataset.get_data(row=row_idx, col='CXSMILES'),
                TARGET_ENDPOINT: dataset.get_data(row=row_idx, col=TARGET_ENDPOINT),
                'Set': dataset.get_data(row=row_idx, col='Set')
            }
        except Exception as e:
            if attempt < max_retries - 1:
                # 等待 1~3 秒后重试
                time.sleep(1.5)
            else:
                print(f"行 {row_idx} 在 {max_retries} 次重试后彻底失败: {e}")
                return None

# ==========================================
# 降低并发数到安全范围，避免被 Polaris 封禁 IP
# ==========================================
safe_workers = 16

print(f"开启 {safe_workers} 线程并发，并启动【自动断线重试】保护...")

data_list = []
with ThreadPoolExecutor(max_workers=safe_workers) as executor:
    # 映射任务并展示进度条
    for result in tqdm(executor.map(fetch_row_with_retry, dataset.rows), total=len(dataset.rows), desc="极速下载中"):
        if result is not None:
            data_list.append(result)

df = pd.DataFrame(data_list)

# 确保所有数据都下载下来了
if len(df) < len(dataset.rows) * 0.9:
    print(f"警告：由于网络极差，只下载到了 {len(df)} 条数据，建议使用本地电脑下载！")

# 严格遵循官方切分
df_train_official = df[df['Set'] == 'train'].copy()
df_test_official = df[df['Set'] == 'test'].copy()

# 处理稀疏数据
df_train_official = df_train_official[['smiles', TARGET_ENDPOINT]].dropna(subset=[TARGET_ENDPOINT])
train_df, valid_df = train_test_split(df_train_official, test_size=0.1, random_state=42)

test_df = df_test_official[['smiles', TARGET_ENDPOINT]].copy()
test_df[TARGET_ENDPOINT] = test_df[TARGET_ENDPOINT].fillna(-999.0)

# 保存文件
output_dir = f"/root/autodl-tmp/molformer/data/admet_{TARGET_ENDPOINT.lower()}"
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(os.path.join(output_dir, f"admet_{TARGET_ENDPOINT.lower()}_train.csv"), index=False)
valid_df.to_csv(os.path.join(output_dir, f"admet_{TARGET_ENDPOINT.lower()}_valid.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, f"admet_{TARGET_ENDPOINT.lower()}_test.csv"), index=False)

print(f"\n=== {TARGET_ENDPOINT} 数据集构建完成 ===")
print(f"训练集: {len(train_df)} | 验证集: {len(valid_df)} | 官方测试集: {len(test_df)}")
print(f"数据已保存在: {output_dir}")