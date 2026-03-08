import torch
import numpy as np
import os
import pickle
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments  # 修复了生成任务的导入
from peft import get_peft_model, LoraConfig, TaskType
import nltk

# 导入您的原生模型和分词器
from model.modeling import MoStT5ForConditionalGeneration
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# 🚀 动态导入您本地生成 mapping 的关键函数
from finetune_dataset import generate_atom_to_motif_map_online

# === 🚀 核心魔法补丁：绕过 Hugging Face 的 Generate 严格审查 ===
from transformers.generation.utils import GenerationMixin

original_validate = GenerationMixin._validate_model_kwargs

def _patched_validate_model_kwargs(self, model_kwargs):
    # 在验证时，暂时隐藏我们自定义的 3D 特征参数，防止被拦截
    custom_keys = ['e3fp_ids', 'atom_to_motif_map', 'atom_attention_mask']
    filtered_kwargs = {k: v for k, v in model_kwargs.items() if k not in custom_keys}
    original_validate(self, filtered_kwargs)

# 替换底层的验证函数
GenerationMixin._validate_model_kwargs = _patched_validate_model_kwargs
# ==========================================================
# ==========================================
# 1. HuggingFace 数据集加载与特征缓存
# ==========================================
class HFChEBI20Dataset(Dataset):
    def __init__(self, hf_repo_id, split_name, motif_tokenizer, e3fp_tokenizer, cache_name):
        self.motif_tokenizer = motif_tokenizer
        self.e3fp_tokenizer = e3fp_tokenizer
        self.data = []
        cache_path = f"dataset/chebi20_hf_cache_{cache_name}.pt"

        if os.path.exists(cache_path):
            print(f"📦 Loading cached data from {cache_path}...")
            with open(cache_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            print(f"📥 Downloading HF dataset '{hf_repo_id}' [{split_name}]...")
            raw_dataset = load_dataset(hf_repo_id, split=split_name)

            print(f"⚙️ Processing raw data and building 3D spatial cache...")
            for i, item in enumerate(raw_dataset):
                smiles = item['smiles']
                description = item['input']

                if not smiles or not description:
                    continue

                try:
                    # 1. 先提取完整的 2D Motif 和 Map (保证映射函数绝对不报错)
                    motif_ids = self.motif_tokenizer.tokenizer.encode(smiles, add_special_tokens=True)
                    atom_map_list = generate_atom_to_motif_map_online(smiles, motif_ids)

                    # 2. 🚀 安全截断逻辑 (必须在生成 Map 之后执行)
                    max_allowable_len = 400
                    if len(motif_ids) > max_allowable_len:
                        motif_ids = motif_ids[:max_allowable_len - 1] + [self.motif_tokenizer.tokenizer.eos_token_id]

                    # 3. 🛡️ 3D 映射保护：将超出截断范围的 3D 原子映射强制设为无效 (-1)
                    atom_map_list = [-1 if idx >= max_allowable_len else idx for idx in atom_map_list]

                    # 4. 提取 3D E3FP (彻底修复 Callable 问题，万无一失版)
                    if hasattr(self.e3fp_tokenizer, 'encode'):
                        e3fp_out = self.e3fp_tokenizer.encode(smiles)
                    else:
                        e3fp_out = self.e3fp_tokenizer(smiles)

                    # 统一转为 Tensor 并处理维度
                    if isinstance(e3fp_out, dict):
                        e3fp_tensor = torch.tensor(e3fp_out['input_ids'], dtype=torch.long)
                    else:
                        e3fp_tensor = torch.tensor(e3fp_out, dtype=torch.long)

                    if e3fp_tensor.dim() > 2:
                        e3fp_tensor = e3fp_tensor.squeeze(0)

                    # 5. 对齐 Map 长度到 max_atoms
                    max_ats = self.e3fp_tokenizer.max_atoms
                    atom_map_list = (atom_map_list[:max_ats] + [-1] * max_ats)[:max_ats]

                    # 6. 生成 Mask
                    atom_mask = (e3fp_tensor[:, 0] != self.e3fp_tokenizer.padding_idx).long().tolist()

                    self.data.append({
                        'smiles': smiles,
                        'motif_input_ids': torch.tensor(motif_ids, dtype=torch.long),
                        'e3fp_input_ids': e3fp_tensor.clone().detach(),
                        'atom_to_motif_map': torch.tensor(atom_map_list, dtype=torch.long),
                        'atom_attention_mask': torch.tensor(atom_mask, dtype=torch.long),
                        'target_text': description
                    })
                except Exception as e:
                    # 🚨 撕开面具：如果是代码逻辑错误，必须立刻崩溃并打印出来！
                    print(f"\n🚨 [致命错误] Failed on SMILES: {smiles}")
                    print(f"🚨 [错误详情]: {e}")
                    raise e

                if (i + 1) % 2000 == 0:
                    print(f"Processed {i + 1}/{len(raw_dataset)} molecules...")

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"✅ Cache saved to {cache_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ==========================================
# 2. 100% 对齐 CAMT5 源码的 Collator
# ==========================================
class CAMT5StyleCollator:
    def __init__(self, tokenizer, task_definition, e3fp_padding_idx=-1):
        self.tokenizer = tokenizer.tokenizer
        self.e3fp_pad = e3fp_padding_idx

        # 定义前缀和后缀 (严格遵守 CAMT5 的 NI 模板)
        prefix_str = f"Definition: {task_definition}\n\nNow complete the following example -\nInput: "
        self.prefix_ids = self.tokenizer.encode(prefix_str, add_special_tokens=False)

        suffix_str = ".\nOutput: "
        self.suffix_ids = self.tokenizer.encode(suffix_str, add_special_tokens=False)

    def __call__(self, batch):
        input_ids_list, e3fp_ids_list, atom_map_list, atom_mask_list, labels_list = [], [], [], [], []

        for item in batch:
            motif_ids = item['motif_input_ids'].tolist()
            e3fp = item['e3fp_input_ids']
            atom_map = item['atom_to_motif_map'].clone()
            atom_mask = item['atom_attention_mask'].clone()

            # 核心逻辑：切除 Motif 自带的 </s>，确保 Suffix 能被正确读取
            if motif_ids[-1] == self.tokenizer.eos_token_id:
                motif_ids = motif_ids[:-1]
                # 您的 e3fp 维度是 [max_atoms, feature_dim]，不需要根据 motif 长度切片
                # 但需要注意，我们修改的是 1D 序列，所以 3D 的不需要跟 1D 同步切片

            prefix_len = len(self.prefix_ids)
            suffix_len = len(self.suffix_ids) + 1  # 加上最后的 EOS token

            # 1. 组装 1D 序列: [Prefix] + [Motif] + [Suffix] + [EOS]
            full_input_ids = self.prefix_ids + motif_ids + self.suffix_ids + [self.tokenizer.eos_token_id]
            input_ids_list.append(torch.tensor(full_input_ids, dtype=torch.long))

            # 2. 组装 3D 特征 (头尾填 Padding)
            prefix_e3fp = torch.full((prefix_len, e3fp.shape[1]), self.e3fp_pad, dtype=torch.long)
            suffix_e3fp = torch.full((suffix_len, e3fp.shape[1]), self.e3fp_pad, dtype=torch.long)
            full_e3fp = torch.cat([prefix_e3fp, e3fp, suffix_e3fp], dim=0)
            e3fp_ids_list.append(full_e3fp)

            # 3. 映射桥梁平移 🚀
            valid_mask = atom_map != -1
            atom_map[valid_mask] += prefix_len  # 仅受 prefix 长度影响，因为后缀在最后

            prefix_map = torch.full((prefix_len,), -1, dtype=torch.long)
            suffix_map = torch.full((suffix_len,), -1, dtype=torch.long)
            full_map = torch.cat([prefix_map, atom_map, suffix_map], dim=0)
            atom_map_list.append(full_map)

            # 4. 3D Attention Mask
            prefix_atom_mask = torch.zeros(prefix_len, dtype=torch.long)
            suffix_atom_mask = torch.zeros(suffix_len, dtype=torch.long)
            full_atom_mask = torch.cat([prefix_atom_mask, atom_mask, suffix_atom_mask], dim=0)
            atom_mask_list.append(full_atom_mask)

            # 5. 目标文本 (Labels)
            target_text = item['target_text']
            label_ids = self.tokenizer.encode(target_text, add_special_tokens=True, max_length=384, truncation=True)
            labels_list.append(torch.tensor(label_ids, dtype=torch.long))

        batch_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
        batch_attention_mask = (batch_input_ids != 0).long()
        batch_e3fp = pad_sequence(e3fp_ids_list, batch_first=True, padding_value=self.e3fp_pad)
        batch_map = pad_sequence(atom_map_list, batch_first=True, padding_value=-1)
        batch_atom_mask = pad_sequence(atom_mask_list, batch_first=True, padding_value=0)
        batch_labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "e3fp_ids": batch_e3fp,
            "atom_to_motif_map": batch_map,
            "atom_attention_mask": batch_atom_mask,
            "labels": batch_labels
        }


# ==========================================
# 3. 评估指标 (BLEU-4)
# ==========================================
def compute_generation_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        ref = [label.strip().split()]
        hyp = pred.strip().split()
        try:
            score = nltk.translate.bleu_score.sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25))
            bleu_scores.append(score)
        except:
            bleu_scores.append(0.0)

    return {"bleu": np.mean(bleu_scores) * 100}


def main():
    pretrained_path = "./most_t5_phase2_alignment_v2"
    print("Loading Pretrained Generative Model with LoRA...")

    base_model = MoStT5ForConditionalGeneration.from_pretrained(pretrained_path)

    # 🚀 生成长文本，放开关键模块以提高模型容量
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "k", "v", "o"]
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    motif_tokenizer = MotifTokenizer(vocab_file="asset/mol_vocabs/my_dataset_vocab.txt",
                                     model_name="google/t5-v1_1-base")
    e3fp_tokenizer = E3FPTokenizer(fp_level=4, fp_bits=4096)

    print("Loading ChEBI-20 from HuggingFace...")
    hf_repo = "QizhiPei/e3fp-chebi-molgen"

    # 强制重新生成一次缓存，以应用最新的字段映射
    train_dataset = HFChEBI20Dataset(hf_repo, "train", motif_tokenizer, e3fp_tokenizer, "train_v3")
    eval_dataset = HFChEBI20Dataset(hf_repo, "validation", motif_tokenizer, e3fp_tokenizer, "eval_v3")
    test_dataset = HFChEBI20Dataset(hf_repo, "test", motif_tokenizer, e3fp_tokenizer, "test_v3")

    # 定义我们的 Mol2Text 任务指令
    task_def = "Generate a comprehensive text description for the given molecule based on its structure and physical properties"
    collator = CAMT5StyleCollator(tokenizer=motif_tokenizer, task_definition=task_def)

    training_args = Seq2SeqTrainingArguments(  # 🚀 使用了修复后的 Seq2SeqTrainingArguments
        output_dir="./finetune_mol2text",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,  # 推理时稍微加大 BatchSize 提速
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        predict_with_generate=True,  # 🚀 这里就不会再报错了！
        generation_max_length=256,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        save_safetensors=False,
        save_total_limit=1
    )

    trainer = Seq2SeqTrainer(  # 🚀 使用了修复后的 Seq2SeqTrainer
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=lambda eval_pred: compute_generation_metrics(eval_pred, motif_tokenizer.tokenizer)
    )

    print("Starting Mol2Text Generative Fine-tuning...")
    trainer.train()

    print("Evaluating on Test Set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"🎉 Final Test BLEU Score: {test_results['eval_bleu']:.4f}")


if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    main()