from tokenization.motif_tokenizer import MotifTokenizer

def main():
    tokenizer = MotifTokenizer()
    
    # 测试阿司匹林
    smiles = "c1ccccc1C(=O)O"
    print(f"\n🧪 测试 SMILES: {smiles}")
    
    # 调用 encode
    token_ids = tokenizer.encode(smiles, return_tensors="list")
    print(f"🔢 转化后的 Token IDs: {token_ids}")
    
    # 逆向还原看结果
    tokens = tokenizer.tokenizer.convert_ids_to_tokens(token_ids)
    print(f"🧩 逆向还原的 Tokens: {tokens}")
    
    # 顺便测一下带电荷分子
    smiles2 = "CC[O-]"
    print(f"\n🧪 测试带电荷分子: {smiles2}")
    tokens2 = tokenizer.tokenizer.convert_ids_to_tokens(tokenizer.encode(smiles2, return_tensors="list"))
    print(f"🧩 逆向还原的 Tokens: {tokens2}")

if __name__ == "__main__":
    main()