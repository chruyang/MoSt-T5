#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMDB数据库查看工具
功能: 查看和分析LMDB格式的分子数据集
"""

import lmdb
import pickle
import argparse
import json
from typing import Dict, Any, List
import numpy as np

class LMDBViewer:
    """LMDB数据库查看器类"""
    
    def __init__(self, db_path: str):
        """
        初始化LMDB查看器
        
        Args:
            db_path (str): LMDB数据库路径
        """
        self.db_path = db_path
        self.env = None
    
    def connect(self):
        """连接到LMDB数据库"""
        try:
            self.env = lmdb.open(
                self.db_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
            print(f"✅ 成功连接到数据库: {self.db_path}")
            return True
        except Exception as e:
            print(f"❌ 连接数据库失败: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        获取数据库基本信息
        
        Returns:
            Dict[str, Any]: 数据库统计信息
        """
        if not self.env:
            return {}
        
        with self.env.begin() as txn:
            stat = txn.stat()
            return {
                'entries': stat['entries'],
                'depth': stat['depth'],
                'branch_pages': stat['branch_pages'],
                'leaf_pages': stat['leaf_pages'],
                'overflow_pages': stat['overflow_pages'],
                'file_size_mb': stat['leaf_pages'] * 4096 / (1024 * 1024)  # 估算文件大小
            }
    
    def sample_records(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """
        随机采样记录
        
        Args:
            num_samples (int): 采样数量
            
        Returns:
            List[Dict[str, Any]]: 采样的记录列表
        """
        samples = []
        if not self.env:
            return samples
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            count = 0
            for key, value in cursor.iternext(keys=True, values=True):
                if count >= num_samples:
                    break
                
                try:
                    record = pickle.loads(value)
                    samples.append({
                        'key': key.decode('utf-8') if isinstance(key, bytes) else key,
                        'record': record
                    })
                    count += 1
                except Exception as e:
                    print(f"⚠️ 解析记录时出错 (key: {key}): {e}")
        
        return samples
    
    def analyze_record_structure(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析单条记录的结构
        
        Args:
            record (Dict[str, Any]): 记录数据
            
        Returns:
            Dict[str, Any]: 结构分析结果
        """
        analysis = {}
        
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                analysis[key] = {
                    'type': 'numpy_array',
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'sample_values': value.flatten()[:5].tolist() if value.size > 0 else []
                }
            elif isinstance(value, list):
                analysis[key] = {
                    'type': 'list',
                    'length': len(value),
                    'first_element_type': type(value[0]).__name__ if value else 'empty'
                }
            elif isinstance(value, dict):
                analysis[key] = {
                    'type': 'dict',
                    'keys': list(value.keys())
                }
            else:
                analysis[key] = {
                    'type': type(value).__name__,
                    'value': str(value)[:100] + '...' if len(str(value)) > 100 else str(value)
                }
        
        return analysis
    
    def close(self):
        """关闭数据库连接"""
        if self.env:
            self.env.close()
            print("✅ 数据库连接已关闭")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LMDB数据库查看工具')
    parser.add_argument('db_path', nargs='?', default='3d-pubchem.lmdb',
                       help='LMDB数据库路径 (默认: 3d-pubchem.lmdb)')
    parser.add_argument('-n', '--num_samples', type=int, default=5, 
                       help='采样记录数量 (默认: 5)')
    parser.add_argument('--info_only', action='store_true',
                       help='只显示数据库基本信息')
    
    args = parser.parse_args()
    
    # 创建查看器实例
    viewer = LMDBViewer(args.db_path)
    
    # 连接数据库
    if not viewer.connect():
        return
    
    try:
        # 显示数据库基本信息
        print("\n" + "="*50)
        print("📊 数据库基本信息")
        print("="*50)
        info = viewer.get_database_info()
        for key, value in info.items():
            print(f"{key}: {value}")
        
        if args.info_only:
            return
        
        # 采样并分析记录
        print("\n" + "="*50)
        print(f"📝 采样记录分析 (采样数量: {args.num_samples})")
        print("="*50)
        
        samples = viewer.sample_records(args.num_samples)
        
        for i, sample in enumerate(samples, 1):
            print(f"\n--- 记录 {i} (Key: {sample['key']}) ---")
            
            # 分析记录结构
            structure = viewer.analyze_record_structure(sample['record'])
            
            # 打印结构信息
            for field, details in structure.items():
                print(f"  {field}:")
                for detail_key, detail_value in details.items():
                    print(f"    {detail_key}: {detail_value}")
    
    finally:
        viewer.close()

if __name__ == "__main__":
    main()