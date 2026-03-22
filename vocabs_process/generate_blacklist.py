import json
import os

def extract_ids_from_json(json_path, split_name):
    print(f"📥 正在解析 {split_name} 集合: {json_path}")
    if not os.path.exists(json_path):
        print(f"   -> ❌ 文件不存在！")
        return set()
        
    ids = set()
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        ids.update(data.keys())
        print(f"   -> 🟢 检测到 Dict 格式，提取了 {len(ids)} 个 ID。")
        
    elif isinstance(data, list):
        if len(data) == 0:
            print("   -> ⚠️ 列表为空！")
            return ids

        sample = data[0]
        if isinstance(sample, dict):
            # 🚀 核心修复：加入 'input' 字段来兼容 Instruction 微调格式
            possible_keys = ['input', 'cid', 'CID', 'index', 'Index', 'id', 'ID', 'pubchem_id']
            id_key = None
            for k in possible_keys:
                if k in sample:
                    id_key = k
                    break
                    
            if id_key == 'input':
                # 指令微调数据：同一个分子有多种属性任务，需要 set 自动去重
                for item in data:
                    val = item.get('input', '')
                    if str(val).strip():
                        ids.add(str(val).strip())
                print(f"   -> 🟢 检测到 Instruction 格式，使用 'input' 成功提取并去重了 {len(ids)} 个 ID。")
            elif id_key:
                for item in data:
                    ids.add(str(item[id_key]))
                print(f"   -> 🟢 检测到 List[Dict] 格式，识别到主键 '{id_key}'，提取了 {len(ids)} 个 ID。")
            else:
                print(f"   -> ❌ 无法识别字典中的主键！当前样本的可用键名: {list(sample.keys())}")
                
        elif isinstance(sample, (int, str)):
            ids.update([str(x) for x in data])
            print(f"   -> 🟢 检测到纯 List 格式，提取了 {len(ids)} 个 ID。")

    return ids

def generate_strict_splits():
    # 严格对齐您的 PubChemQC 目录
    base_dir = os.path.expanduser("~/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchemqc")
    print("="*60)
    print(f"🚀 启动严格数据隔离流水线: {base_dir}")
    print("="*60)

    json_name = "3d_computed_properties_unit.json"

    # 1. 提取 Pretrain (Phase 1/2 唯一合法的护城河)
    pretrain_path = os.path.join(base_dir, "pretrain", json_name)
    pretrain_ids = extract_ids_from_json(pretrain_path, "Pretrain")

    # 2. 提取 Downstream (Train, Valid, Test)
    downstream_ids = set()
    for split in ["train", "valid", "test"]:
        path = os.path.join(base_dir, split, json_name)
        split_ids = extract_ids_from_json(path, split.capitalize())
        downstream_ids.update(split_ids)

    # 3. 交叉校验，确保零重合 (Zero Leakage)
    overlap = pretrain_ids.intersection(downstream_ids)
    if overlap:
        print(f"\n⚠️ 警告: 发现 {len(overlap)} 个分子同时出现在 Pretrain 和下游任务中！")
        # 强制剔除，守护预训练底线
        pretrain_ids = pretrain_ids - downstream_ids
        print(f"   -> 已启动免疫机制：将这 {len(overlap)} 个分子从预训练白名单中强制抹除。")
    else:
        print("\n✨ 完美！Pretrain 与下游任务之间零重合，数据绝对隔离。")

    # 4. 保存最终的白名单与黑名单
    whitelist_path = os.path.join(base_dir, "pretrain_whitelist.json")
    blacklist_path = os.path.join(base_dir, "downstream_blacklist.json")

    with open(whitelist_path, 'w', encoding='utf-8') as f:
        json.dump(list(pretrain_ids), f)
    with open(blacklist_path, 'w', encoding='utf-8') as f:
        json.dump(list(downstream_ids), f)

    print("="*60)
    print(f"🎯 最终成果:")
    print(f"   ✅ Pretrain 白名单保存: {len(pretrain_ids):,} 个分子 (预期 ~3,119,717)")
    print(f"   🛑 Downstream 黑名单保存: {len(downstream_ids):,} 个分子 (预期 ~779,930)")
    print("="*60)

if __name__ == "__main__":
    generate_strict_splits()