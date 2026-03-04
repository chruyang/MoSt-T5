import os
import json
import math
import pickle
import lmdb
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenization.text_tokenizer import TextTokenizer

import nltk

try:
    # 尝试加载，如果失败则下载
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    # 强制下载最新命名的资源
    nltk.download('averaged_perceptron_tagger_eng')


def build_text_importance_dict(lmdb_path, model_name_or_path, output_path):
    tokenizer = TextTokenizer(model_name=model_name_or_path, max_len=512)
    is_subdir = os.path.isdir(lmdb_path)  # 判断是否是目录
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=is_subdir  # <--- 自动适配目录模式或文件模式
    )
    texts = []
    with env.begin() as txn:
        length = int(txn.get(b'__len__'))
        for idx in tqdm(range(length), desc="Reading LMDB"):
            data = txn.get(str(idx).encode())
            if data:
                entry = pickle.loads(data)
                text = entry.get('enriched_description', '') or entry.get('description', '') or entry.get('text', '')
                if text: texts.append(text)

    vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b[A-Za-z]+\b')
    vectorizer.fit(texts)
    idf_dict = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    max_idf = max(idf_dict.values()) if idf_dict else 1.0

    print("3. 结合词性 (POS) 映射到 T5 词表...")
    vocab = tokenizer.tokenizer.get_vocab()
    weight_dict = {}

    # 预处理：提取所有需要标注的候选词
    token_list = []
    token_ids = []
    for token_str, token_id in vocab.items():
        weight_dict[token_id] = 0.0  # 默认权重
        clean_word = token_str.replace(' ', '').strip().lower()
        if clean_word.isalpha() and token_str not in tokenizer.tokenizer.all_special_tokens:
            token_list.append(clean_word)
            token_ids.append(token_id)

    # 🚀 批量标注词性，效率远高于循环单次调用
    print(f"   开始对 {len(token_list)} 个候选词进行词性标注...")
    tagged_tokens = nltk.pos_tag(token_list)

    # 映射回权重字典
    for i, (word, pos_tag) in enumerate(tagged_tokens):
        if pos_tag.startswith('NN') or pos_tag.startswith('JJ'):
            current_id = token_ids[i]
            # 获取 IDF 权重
            weight_dict[current_id] = idf_dict.get(word, max_idf * 0.1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(weight_dict, f)
    print(f"✅ Text weights saved to: {output_path}")


if __name__ == "__main__":
    build_text_importance_dict("../dataset/official_pretrain_lmdb", "google/t5-v1_1-base",
                               "../asset/text_idf_weights.json")