#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMDB 数据集重复检测工具
功能：检测 LMDB 数据库中是否存在重复的记录
"""

import lmdb
import pickle
import argparse
from typing import Dict, Set, Tuple, List
from tqdm import tqdm
from collections import defaultdict


class LMDPDuplicateChecker:
    """LMDB 重复检测器"""
    
    def __init__(self, db_path: str):
        """
        初始化检测器
        
        Args:
            db_path (str): LMDB 数据库路径
        """
        self.db_path = db_path
        self.env = None
    
    def connect(self):
        """连接到 LMDB 数据库"""
        try:
            self.env = lmdb.open(
                self.db_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
            print(f"✅ 成功连接到数据库：{self.db_path}")
            return True
        except Exception as e:
            print(f"❌ 连接数据库失败：{e}")
            return False
    
    def check_duplicates_by_smiles(self, show_samples: int = 10) -> Tuple[int, int, Dict]:
        """
        通过 SMILES 检测重复的分子
        
        Args:
            show_samples (int): 显示前 N 个重复样本
            
        Returns:
            Tuple[int, int, Dict]: (总记录数，重复分子数，重复详情字典)
        """
        if not self.env:
            return 0, 0, {}
        
        smiles_to_ids: Dict[str, List[int]] = defaultdict(list)
        total_count = 0
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            
            for key, value in tqdm(cursor.iternext(), desc="检测 SMILES 重复", unit="条"):
                try:
                    # 解析 ID
                    current_id = int(key.decode('utf-8'))
                    
                    # 解析记录
                    record = pickle.loads(value)
                    smiles = record.get('smiles', '') or record.get('smiles_kekule', '')
                    
                    if smiles:
                        smiles_to_ids[smiles].append(current_id)
                        total_count += 1
                        
                except Exception as e:
                    print(f"⚠️ 解析记录时出错 (key: {key}): {e}")
        
        # 找出重复的 SMILES
        duplicates = {smiles: ids for smiles, ids in smiles_to_ids.items() if len(ids) > 1}
        duplicate_count = len(duplicates)
        
        # 显示部分重复样本
        if duplicates and show_samples > 0:
            print(f"\n🔍 重复 SMILES 样本（前 {show_samples} 个）:")
            print("=" * 80)
            
            for i, (smiles, ids) in enumerate(list(duplicates.items())[:show_samples]):
                print(f"\n【重复样本 {i+1}】")
                print(f"  SMILES: {smiles[:100]}{'...' if len(smiles) > 100 else ''}")
                print(f"  重复 ID: {ids}")
                print(f"  重复次数：{len(ids)}")
        
        return total_count, duplicate_count, duplicates
    
    def check_duplicates_by_content(self, show_samples: int = 10) -> Tuple[int, int]:
        """
        通过完整记录内容检测重复（更严格）
        
        Args:
            show_samples (int): 显示前 N 个重复样本
            
        Returns:
            Tuple[int, int]: (总记录数，重复记录数)
        """
        if not self.env:
            return 0, 0
        
        content_to_ids: Dict[str, List[int]] = defaultdict(list)
        total_count = 0
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            
            for key, value in tqdm(cursor.iternext(), desc="检测内容重复", unit="条"):
                try:
                    # 解析 ID
                    current_id = int(key.decode('utf-8'))
                    
                    # 解析记录并生成内容的哈希
                    record = pickle.loads(value)
                    # 使用 pickle 序列化后转字符串作为唯一标识
                    content_hash = str(hash(pickle.dumps(record, protocol=2)))
                    
                    content_to_ids[content_hash].append(current_id)
                    total_count += 1
                    
                except Exception as e:
                    print(f"⚠️ 解析记录时出错 (key: {key}): {e}")
        
        # 找出重复的内容
        duplicates = {content: ids for content, ids in content_to_ids.items() if len(ids) > 1}
        duplicate_records = sum(len(ids) - 1 for ids in duplicates.values())  # 计算多余的记录数
        
        # 显示部分重复样本
        if duplicates and show_samples > 0:
            print(f"\n🔍 重复内容样本（前 {show_samples} 个）:")
            print("=" * 80)
            
            for i, (content, ids) in enumerate(list(duplicates.items())[:show_samples]):
                print(f"\n【重复内容 {i+1}】")
                print(f"  重复 ID: {ids}")
                print(f"  重复次数：{len(ids)}")
        
        return total_count, duplicate_records
    
    def check_duplicate_ids(self, show_samples: int = 10) -> Tuple[int, List[int]]:
        """
        检测是否有重复的 ID
        
        Args:
            show_samples (int): 显示前 N 个重复 ID
            
        Returns:
            Tuple[int, List[int]]: (重复 ID 数量，重复 ID 列表)
        """
        if not self.env:
            return 0, []
        
        seen_ids: Set[int] = set()
        duplicate_ids: List[int] = []
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            
            for key, value in tqdm(cursor.iternext(), desc="检测 ID 重复", unit="条"):
                try:
                    current_id = int(key.decode('utf-8'))
                    
                    if current_id in seen_ids:
                        duplicate_ids.append(current_id)
                    else:
                        seen_ids.add(current_id)
                        
                except Exception as e:
                    print(f"⚠️ 解析记录时出错 (key: {key}): {e}")
        
        # 显示部分重复 ID
        if duplicate_ids and show_samples > 0:
            print(f"\n🔍 重复 ID 列表（前 {show_samples} 个）:")
            print("=" * 80)
            print(f"  {duplicate_ids[:show_samples]}")
        
        return len(duplicate_ids), duplicate_ids
    
    def close(self):
        """关闭数据库连接"""
        if self.env:
            self.env.close()
            print("\n✅ 数据库连接已关闭")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LMDB 数据集重复检测工具')
    parser.add_argument('db_path', nargs='?', default='3d-pubchem-train.lmdb',
                       help='LMDB 数据库路径 (默认：3d-pubchem-train.lmdb)')
    parser.add_argument('--mode', type=str, choices=['smiles', 'content', 'id'], 
                       default='smiles',
                       help='检测模式：smiles(基于 SMILES 检测), content(基于内容检测), id(基于 ID 检测)')
    parser.add_argument('-s', '--show_samples', type=int, default=10,
                       help='显示的重复样本数量 (默认：10)')
    parser.add_argument('--all', action='store_true',
                       help='执行所有检测模式')
    
    args = parser.parse_args()
    
    # 创建检测器实例
    checker = LMDPDuplicateChecker(args.db_path)
    
    # 连接数据库
    if not checker.connect():
        return
    
    try:
        if args.all:
            # 执行所有检测
            print("\n" + "="*80)
            print("🔍 模式 1: 基于 SMILES 检测重复")
            print("="*80)
            total1, dup1, _ = checker.check_duplicates_by_smiles(args.show_samples)
            
            print("\n" + "="*80)
            print("🔍 模式 2: 基于内容检测重复")
            print("="*80)
            total2, dup2 = checker.check_duplicates_by_content(args.show_samples)
            
            print("\n" + "="*80)
            print("🔍 模式 3: 基于 ID 检测重复")
            print("="*80)
            total3, dup_ids = checker.check_duplicate_ids(args.show_samples)
            
            # 汇总报告
            print("\n" + "="*80)
            print("📊 重复检测总结报告")
            print("="*80)
            print(f"总记录数：{total1:,}")
            print(f"重复 SMILES 分子数：{dup1:,} ({dup1/total1*100:.2f}%)" if total1 > 0 else "N/A")
            print(f"重复记录数（内容相同）: {dup2:,} ({dup2/total2*100:.2f}%)" if total2 > 0 else "N/A")
            print(f"重复 ID 数：{total3:,}")
            
            if dup1 == 0 and dup2 == 0 and total3 == 0:
                print("\n✅ 数据集无重复记录！")
            else:
                print(f"\n⚠️  数据集存在重复记录，建议清理！")
                
        elif args.mode == 'smiles':
            total, duplicates, _ = checker.check_duplicates_by_smiles(args.show_samples)
            print(f"\n📊 检测结果:")
            print(f"  总记录数：{total:,}")
            print(f"  重复 SMILES 分子数：{duplicates:,} ({duplicates/total*100:.2f}%)" if total > 0 else "N/A")
            
            if duplicates == 0:
                print(f"\n✅ 数据集无重复 SMILES！")
            else:
                print(f"\n⚠️  数据集存在 {duplicates} 个重复的 SMILES！")
                
        elif args.mode == 'content':
            total, duplicates = checker.check_duplicates_by_content(args.show_samples)
            print(f"\n📊 检测结果:")
            print(f"  总记录数：{total:,}")
            print(f"  重复记录数：{duplicates:,} ({duplicates/total*100:.2f}%)" if total > 0 else "N/A")
            
            if duplicates == 0:
                print(f"\n✅ 数据集无重复记录！")
            else:
                print(f"\n⚠️  数据集存在 {duplicates} 条重复记录！")
                
        elif args.mode == 'id':
            count, dup_ids = checker.check_duplicate_ids(args.show_samples)
            print(f"\n📊 检测结果:")
            print(f"  重复 ID 数量：{count:,}")
            
            if count == 0:
                print(f"\n✅ 数据集无重复 ID！")
            else:
                print(f"\n⚠️  数据集存在 {count} 个重复 ID！")
    
    finally:
        checker.close()


if __name__ == "__main__":
    main()
