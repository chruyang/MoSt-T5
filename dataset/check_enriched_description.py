#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 LMDB 数据库中 enriched_description 字段的覆盖率
功能：统计有多少数据包含 enriched_description 字段
"""

import lmdb
import pickle
import argparse
from typing import Dict, Any, Tuple
from tqdm import tqdm


class EnrichedDescriptionChecker:
    """enriched_description 字段检查器"""
    
    def __init__(self, db_path: str):
        """
        初始化检查器
        
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
    
    def check_field_coverage(self) -> Tuple[Dict[str, int], int]:
        """
        检查 enriched_description 字段的覆盖情况
        
        Returns:
            Tuple[Dict[str, int], int]: (字段统计字典，总记录数)
        """
        if not self.env:
            return {}, 0
        
        stats = {
            'has_enriched_description': 0,
            'has_description': 0,
            'has_text': 0,
            'no_text_field': 0,
            'empty_enriched_description': 0,
            'empty_description': 0,
            'empty_text': 0,
            'meaningful_enriched': 0,  # 有实际内容的 enriched
            'meaningful_desc': 0,      # 有实际内容的 description
        }
        
        total_count = 0
        
        with self.env.begin() as txn:
            # 获取总记录数
            try:
                length_bytes = txn.get(b'__len__')
                if length_bytes:
                    total_expected = int(length_bytes)
                    print(f"📊 数据库声明的总记录数：{total_expected}")
            except Exception:
                pass
            
            cursor = txn.cursor()
            
            for key, value in tqdm(cursor.iternext(), desc="检查记录", unit="条"):
                try:
                    record = pickle.loads(value)
                    total_count += 1
                    
                    # 检查 enriched_description
                    enriched_desc = record.get('enriched_description', '')
                    if enriched_desc is not None and enriched_desc.strip():
                        stats['has_enriched_description'] += 1
                        # 检查是否有实际内容（排除 N/A, null, None 等占位符）
                        clean_enriched = enriched_desc.strip().lower()
                        if clean_enriched not in ['', 'n/a', 'null', 'none', 'na', 'not available']:
                            stats['meaningful_enriched'] += 1
                    else:
                        stats['empty_enriched_description'] += 1
                    
                    # 检查 description
                    desc = record.get('description', '')
                    if desc is not None and desc.strip():
                        stats['has_description'] += 1
                        # 检查是否有实际内容
                        clean_desc = desc.strip().lower()
                        if clean_desc not in ['', 'n/a', 'null', 'none', 'na', 'not available']:
                            stats['meaningful_desc'] += 1
                    else:
                        stats['empty_description'] += 1
                    
                    # 检查 text
                    text = record.get('text', '')
                    if text is not None and text.strip():
                        stats['has_text'] += 1
                    else:
                        stats['empty_text'] += 1
                    
                    # 检查是否没有任何文本字段
                    if not (enriched_desc or desc or text):
                        stats['no_text_field'] += 1
                        
                except Exception as e:
                    print(f"⚠️ 解析记录时出错 (key: {key}): {e}")
        
        return stats, total_count
    
    def print_statistics(self, stats: Dict[str, int], total_count: int):
        """打印统计结果"""
        print("\n" + "="*70)
        print("📊 enriched_description 字段覆盖率统计")
        print("="*70)
        print(f"总记录数：{total_count:,}")
        print()
        
        # 计算百分比
        def calc_percent(count):
            return (count / total_count * 100) if total_count > 0 else 0
        
        print("【各字段非空统计】")
        print(f"  enriched_description 非空：{stats['has_enriched_description']:>12,} "
              f"({calc_percent(stats['has_enriched_description']):>6.2f}%)")
        print(f"    - 有实际内容：          {stats['meaningful_enriched']:>12,} "
              f"({calc_percent(stats['meaningful_enriched']):>6.2f}%)")
        print(f"  description 非空：        {stats['has_description']:>12,} "
              f"({calc_percent(stats['has_description']):>6.2f}%)")
        print(f"    - 有实际内容：          {stats['meaningful_desc']:>12,} "
              f"({calc_percent(stats['meaningful_desc']):>6.2f}%)")
        print(f"  text 非空：              {stats['has_text']:>12,} "
              f"({calc_percent(stats['has_text']):>6.2f}%)")
        print()
        
        print("【各字段为空统计】")
        print(f"  enriched_description 为空：{stats['empty_enriched_description']:>12,} "
              f"({calc_percent(stats['empty_enriched_description']):>6.2f}%)")
        print(f"  description 为空：        {stats['empty_description']:>12,} "
              f"({calc_percent(stats['empty_description']):>6.2f}%)")
        print(f"  text 为空：              {stats['empty_text']:>12,} "
              f"({calc_percent(stats['empty_text']):>6.2f}%)")
        print()
        
        print("【完全缺失文本字段】")
        print(f"  三个字段都为空：        {stats['no_text_field']:>12,} "
              f"({calc_percent(stats['no_text_field']):>6.2f}%)")
        print()
        
        # 回退策略分析
        print("【回退策略分析】")
        has_any_text = total_count - stats['no_text_field']
        only_enriched = stats['has_enriched_description']
        need_description_fallback = stats['empty_enriched_description'] - stats['empty_description']
        need_text_fallback = stats['empty_enriched_description'] + stats['empty_description'] - stats['empty_text']
        
        print(f"  仅需 enriched_description:  {only_enriched:>12,} "
              f"({calc_percent(only_enriched):>6.2f}%)")
        print(f"  需回退到 description:      {need_description_fallback:>12,} "
              f"({calc_percent(need_description_fallback):>6.2f}%)")
        print(f"  需回退到 text:             {need_text_fallback:>12,} "
              f"({calc_percent(need_text_fallback):>6.2f}%)")
        print(f"  有任何文本字段的数据：     {has_any_text:>12,} "
              f"({calc_percent(has_any_text):>6.2f}%)")
        print("="*70)
        
        # 结论
        print("\n【结论】")
        coverage = calc_percent(stats['has_enriched_description'])
        if coverage == 100:
            print("  ✅ 所有数据都有 enriched_description 字段")
        elif coverage >= 95:
            print(f"  ⚠️  绝大多数数据 ({coverage:.2f}%) 有 enriched_description 字段")
        elif coverage >= 80:
            print(f"  ⚠️  大部分数据 ({coverage:.2f}%) 有 enriched_description 字段，需要回退策略")
        else:
            print(f"  ❌ enriched_description 字段覆盖率较低 ({coverage:.2f}%)，严重依赖回退策略")
        print()
    
    def sample_missing_records(self, num_samples: int = 5):
        """采样显示缺少 enriched_description 的记录"""
        print(f"\n📝 采样显示前 {num_samples} 条缺少 enriched_description 的记录:")
        print("-" * 70)
        
        count = 0
        displayed = 0
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            
            for key, value in cursor.iternext():
                if displayed >= num_samples:
                    break
                
                try:
                    record = pickle.loads(value)
                    enriched_desc = record.get('enriched_description', '')
                    
                    if not enriched_desc:
                        count += 1
                        print(f"\n记录 {count} (Key: {key.decode('utf-8') if isinstance(key, bytes) else key})")
                        
                        # 显示其他字段
                        desc = record.get('description', '')
                        text = record.get('text', '')
                        smiles = record.get('smiles', '')
                        
                        print(f"  enriched_description: '{enriched_desc}'")
                        print(f"  description: '{desc[:100]}...' if len(str(desc)) > 100 else '{desc}'")
                        print(f"  text: '{text[:100]}...' if len(str(text)) > 100 else '{text}'")
                        print(f"  smiles: {smiles[:50]}..." if len(smiles) > 50 else f"  smiles: {smiles}")
                        
                        displayed += 1
                        
                except Exception as e:
                    print(f"⚠️ 解析记录时出错：{e}")
        
        if displayed == 0:
            print("  ✅ 所有记录都有 enriched_description 字段！")
    
    def sample_enriched_descriptions(self, num_samples: int = 5):
        """采样显示包含 enriched_description 的记录"""
        print(f"\n📝 采样显示前 {num_samples} 条包含 enriched_description 的记录:")
        print("=" * 70)
        
        count = 0
        displayed = 0
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            
            for key, value in cursor.iternext():
                if displayed >= num_samples:
                    break
                
                try:
                    record = pickle.loads(value)
                    enriched_desc = record.get('enriched_description', '')
                    
                    if enriched_desc and enriched_desc.strip():
                        count += 1
                        print(f"\n【样本 {count}】(Key: {key.decode('utf-8') if isinstance(key, bytes) else key})")
                        print(f"  enriched_description:\n  {enriched_desc[:300]}{'...' if len(enriched_desc) > 300 else ''}")
                        
                        # 显示长度和字数统计
                        word_count = len(enriched_desc.split())
                        char_count = len(enriched_desc)
                        print(f"  📊 长度：{char_count} 字符，{word_count} 单词")
                        
                        # 可选显示其他信息
                        smiles = record.get('smiles', '')
                        if smiles:
                            print(f"  SMILES: {smiles[:80]}{'...' if len(smiles) > 80 else ''}")
                        
                        displayed += 1
                        
                except Exception as e:
                    print(f"⚠️ 解析记录时出错：{e}")
    
    def close(self):
        """关闭数据库连接"""
        if self.env:
            self.env.close()
            print("\n✅ 数据库连接已关闭")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='检查 enriched_description 字段覆盖率')
    parser.add_argument('db_path', nargs='?', default='3d-pubchem-pretrain2.lmdb',
                       help='LMDB 数据库路径 (默认：3d-pubchem.lmdb)')
    parser.add_argument('-n', '--num_samples', type=int, default=5,
                       help='采样显示缺少 enriched_description 的记录数量 (默认：5)')
    parser.add_argument('-s', '--show_samples', type=int, default=0,
                       help='采样显示包含 enriched_description 的记录数量 (默认：0，不显示)')
    parser.add_argument('--info_only', action='store_true',
                       help='只显示统计信息，不显示采样记录')
    
    args = parser.parse_args()
    
    # 创建检查器实例
    checker = EnrichedDescriptionChecker(args.db_path)
    
    # 连接数据库
    if not checker.connect():
        return
    
    try:
        # 执行检查
        stats, total_count = checker.check_field_coverage()
        
        # 打印统计结果
        checker.print_statistics(stats, total_count)
        
        # 显示采样记录
        if not args.info_only:
            # 显示包含 enriched_description 的样本
            if args.show_samples > 0:
                checker.sample_enriched_descriptions(args.show_samples)
            
            # 显示缺少 enriched_description 的样本
            checker.sample_missing_records(args.num_samples)
    
    finally:
        checker.close()


if __name__ == "__main__":
    main()
