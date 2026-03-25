#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMDB 数据集 ID 连续性验证工具
功能：扫描 LMDB 数据库，分析 ID 分布情况，判断是否连续
"""

import lmdb
import pickle
import argparse
from typing import List, Tuple
from tqdm import tqdm


def analyze_lmdb_keys(lmdb_path: str, show_samples: int = 20):
    """
    分析 LMDB 数据库的 Key 分布
    
    Args:
        lmdb_path (str): LMDB 数据库路径
        show_samples (int): 显示的样本数量
    """
    print("="*70)
    print("🔬 LMDB 数据集 ID 连续性分析工具")
    print("="*70)
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    all_keys = []
    numeric_keys = []
    
    print("\n📋 正在扫描所有 Key...")
    with env.begin() as txn:
        cursor = txn.cursor()
        total_entries = txn.stat()['entries']
        
        for k in tqdm(cursor.iternext(keys=True, values=False), 
                     total=total_entries, desc="扫描中"):
            all_keys.append(k)
            
            # 尝试解析为整数
            try:
                key_str = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                numeric_keys.append(int(key_str))
            except:
                pass
    
    env.close()
    
    # ================= 分析报告 =================
    print("\n" + "="*70)
    print("📊 分析报告")
    print("="*70)
    
    print(f"\n【基本信息】")
    print(f"  • 总记录数：{len(all_keys):,}")
    print(f"  • 可解析为数字的 Key 数量：{len(numeric_keys):,}")
    
    if len(numeric_keys) > 0:
        min_id = min(numeric_keys)
        max_id = max(numeric_keys)
        id_range = max_id - min_id + 1
        density = len(numeric_keys) / id_range * 100
        
        print(f"\n【ID 范围统计】")
        print(f"  • 最小 ID: {min_id:,}")
        print(f"  • 最大 ID: {max_id:,}")
        print(f"  • ID 跨度：{id_range:,}")
        print(f"  • 密度：{density:.2f}% ({len(numeric_keys):,} / {id_range:,})")
        
        # 判断是否连续
        expected_keys = set(range(min_id, max_id + 1))
        actual_keys = set(numeric_keys)
        missing_keys = sorted(expected_keys - actual_keys)
        
        if len(missing_keys) == 0:
            print(f"\n✅ 完美：ID 完全连续！")
        else:
            print(f"\n❌ 警告：发现 {len(missing_keys):,} 个缺失的 ID")
            if len(missing_keys) <= 50:
                print(f"   缺失的 ID: {missing_keys}")
            else:
                print(f"   前 50 个缺失的 ID: {missing_keys[:50]}")
        
        # 显示样本
        print(f"\n【Key 样本展示 (前 {show_samples} 个)】")
        for i, k in enumerate(all_keys[:show_samples]):
            key_str = k.decode('utf-8') if isinstance(k, bytes) else str(k)
            print(f"   {i+1}. {key_str}")
        
        # 密度评估
        print(f"\n【数据完整性评估】")
        if density >= 99.9:
            print(f"  ✅ 优秀：密度 > 99.9%，几乎所有 ID 都连续")
        elif density >= 90:
            print(f"  ⚠️  良好：密度 {density:.1f}%，存在少量缺失")
        elif density >= 50:
            print(f"  ❌ 警告：密度 {density:.1f}%，大量 ID 缺失！")
            print(f"     当前代码只能利用约 {density:.1f}% 的数据")
        else:
            print(f"  🚨 严重：密度 < 50%，超过一半的数据无法访问！")
            print(f"         当前代码只能利用约 {density:.1f}% 的数据")
        
        # 训练影响评估
        print(f"\n【对第一阶段训练的影响】")
        print(f"  • DataLoader 会尝试访问 idx: 0 ~ {len(all_keys)-1}")
        print(f"  • 实际存在的 ID 范围：{min_id} ~ {max_id}")
        
        if min_id > len(all_keys):
            print(f"  🚨 严重不匹配：最小 ID ({min_id}) > 记录数 ({len(all_keys)})")
            print(f"     预测数据利用率：< 1% (几乎全部落空)")
        elif density < 50:
            overlap = len(set(range(len(all_keys))) & actual_keys)
            utilization = overlap / len(all_keys) * 100
            print(f"  🚨 低效：预计数据利用率仅约 {utilization:.1f}%")
        else:
            print(f"  ✅ 影响较小：大部分 ID 在合理范围内")
    
    else:
        print(f"\n⚠️  警告：没有发现数字类型的 Key")
        print(f"   样本：{all_keys[:10]}")
    
    print("\n" + "="*70)
    print("💡 建议:")
    if len(numeric_keys) > 0 and (density < 99 or min_id > 100):
        print("  1. 修改 dataset.py 使用游标遍历而非索引访问")
        print("  2. 参考 process_qc_step1/2.py 的实现模式")
        print("  3. 预加载所有 available_keys 到内存中")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LMDB ID 连续性分析工具')
    parser.add_argument('lmdb_path', type=str, help='LMDB 数据库路径')
    parser.add_argument('-s', '--samples', type=int, default=20, 
                       help='显示的 Key 样本数量 (默认：20)')
    
    args = parser.parse_args()
    
    analyze_lmdb_keys(args.lmdb_path, args.samples)
