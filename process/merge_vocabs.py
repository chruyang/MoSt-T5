import os

# ================= 配置 =================
# CAMT5 原始词表路径 (请根据实际情况修改)
ORIGINAL_VOCAB = "asset/mol_vocabs/frag_pubchem_v3.txt"
# 上一步生成的新词表
MY_VOCAB = "asset/mol_vocabs/my_dataset_vocab.txt"
# 最终合并后的输出路径
MERGED_OUTPUT = "asset/mol_vocabs/frag_pubchem_merged.txt"


# =======================================

def load_vocab(path):
    if not os.path.exists(path):
        print(f"⚠️ 警告: 找不到文件 {path}")
        return set()
    with open(path, 'r') as f:
        # 读取每一行，去除空白符
        return set(line.strip() for line in f if line.strip())


def main():
    print("🔄 开始合并词表...")

    # 1. 加载
    vocab_orig = load_vocab(ORIGINAL_VOCAB)
    vocab_new = load_vocab(MY_VOCAB)

    print(f"   - 原词表大小: {len(vocab_orig)}")
    print(f"   - 新词表大小: {len(vocab_new)}")

    # 2. 合并 (Set Union 自动去重)
    vocab_merged = vocab_orig.union(vocab_new)

    # 确保基础 Token 存在
    if "[.]" not in vocab_merged:
        vocab_merged.add("[.]")

    print(f"   - 合并后大小: {len(vocab_merged)}")
    print(f"   - 新增片段数: {len(vocab_merged) - len(vocab_orig)}")

    # 3. 保存
    # 注意：通常需要按一定顺序保存，或者不需要。
    # CAMT5 的 tokenizer 可能会添加特殊 token (<pad>, <s> 等)，
    # 这里我们只负责保存化学片段部分。
    with open(MERGED_OUTPUT, 'w') as f:
        # 如果您希望特殊 token 写在文件里，可以在这里添加，
        # 但根据 generate_pubchem_dict.py 的逻辑，它只保存了片段。
        # 建议保持纯净，由 Tokenizer 类处理特殊 token。
        for frag in sorted(list(vocab_merged)):
            f.write(frag + "\n")

    print(f"✅ 合并完成！最终词表已保存至: {MERGED_OUTPUT}")
    print(f"👉 下一步：请在预处理脚本 (preprocess_motifs_rigorous.py) 中将 VOCAB_FILE 设置为 '{MERGED_OUTPUT}'")


if __name__ == "__main__":
    main()