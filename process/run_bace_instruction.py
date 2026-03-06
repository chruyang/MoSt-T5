import torch
import numpy as np
from transformers import Trainer, TrainingArguments
from sklearn.metrics import roc_auc_score
import scipy.special
from peft import get_peft_model, LoraConfig, TaskType

# 导入原生生成式模型和分词器
from model.modeling import MoStT5ForConditionalGeneration
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
from finetune_dataset import MoleculeNetDataset, get_official_scaffold_split
from torch.nn.utils.rnn import pad_sequence


# ==========================================
# 1. 核心创新：跨模态指令拼接 Collator
# ==========================================
class InstructionCollator:
    def __init__(self, tokenizer, e3fp_padding_idx=-1):
        self.tokenizer = tokenizer.tokenizer  # 提取底层的 HuggingFace tokenizer
        self.e3fp_pad = e3fp_padding_idx
        # 设定自然语言指令
        self.prompt = "Does this molecule inhibit BACE-1? "
        self.prompt_ids = self.tokenizer.encode(self.prompt, add_special_tokens=False)

    def __call__(self, batch):
        input_ids_list, e3fp_ids_list, atom_map_list, atom_mask_list, labels_list = [], [], [], [], []

        for item in batch:
            text_len = len(self.prompt_ids)
            motif_ids = item['motif_input_ids'].tolist()

            # 1. 序列物理拼接: [Text] + [Motif]
            full_input_ids = self.prompt_ids + motif_ids
            input_ids_list.append(torch.tensor(full_input_ids, dtype=torch.long))

            # 2. 3D特征填充: 文本部分没有3D特征，填充 -1
            e3fp = item['e3fp_input_ids']
            dummy_e3fp = torch.full((text_len, e3fp.shape[1]), self.e3fp_pad, dtype=torch.long)
            full_e3fp = torch.cat([dummy_e3fp, e3fp], dim=0)
            e3fp_ids_list.append(full_e3fp)

            # 3. 映射桥梁平移 (核心机制！激活 Phase 2 的图文对齐)
            atom_map = item['atom_to_motif_map'].clone()
            valid_mask = atom_map != -1
            atom_map[valid_mask] += text_len  # 将分子的映射索引向右推移 text_len 个单位
            dummy_map = torch.full((text_len,), -1, dtype=torch.long)
            full_map = torch.cat([dummy_map, atom_map], dim=0)
            atom_map_list.append(full_map)

            # 4. 3D Attention Mask
            atom_mask = item['atom_attention_mask']
            dummy_atom_mask = torch.zeros(text_len, dtype=torch.long)
            full_atom_mask = torch.cat([dummy_atom_mask, atom_mask], dim=0)
            atom_mask_list.append(full_atom_mask)

            # 5. 生成式标签: 把 1/0 转为 "Yes"/"No"
            label_str = "Yes" if item['label'] == 1 else "No"
            label_ids = self.tokenizer.encode(label_str, add_special_tokens=True)  # 自带 </s>
            labels_list.append(torch.tensor(label_ids, dtype=torch.long))

        # Pad Sequences
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
# 2. 评估逻辑：提取 Yes/No 的 Logits 计算 AUC
# ==========================================
def preprocess_logits_for_metrics(logits, labels):
    # 提取 Decoder 生成第一个 Token (Yes 或 No) 时的全词表 Logits
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits[:, 0, :]


def compute_metrics(eval_pred, tokenizer):
    logits, labels = eval_pred

    # 拿到 Yes 和 No 在词表中的具体 Token ID
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    # 切片提取概率
    yes_logits = logits[:, yes_id]
    no_logits = logits[:, no_id]

    # 对 Yes 和 No 的 Logits 进行 Softmax 归一化
    probs = scipy.special.softmax(np.stack([no_logits, yes_logits], axis=-1), axis=-1)[:, 1]

    # 还原真实标签：如果是 Yes 的 ID，记为 1，否则为 0
    true_labels = (labels[:, 0] == yes_id).astype(int)

    auc = roc_auc_score(true_labels, probs)
    return {"roc_auc": auc}


def main():
    pretrained_path = "./most_t5_phase2_alignment_v2/checkpoint-127500"
    print("Loading Pretrained Generative Model with LoRA...")

    # 🚀 直接加载完全体生成模型
    base_model = MoStT5ForConditionalGeneration.from_pretrained(pretrained_path)

    # 🚀 引入 LoRA 冻结主体参数，只微调 Attention 的 q, v 层
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"]
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()  # 打印可训练参数量，通常在 1% 左右

    motif_tokenizer = MotifTokenizer(vocab_file="asset/mol_vocabs/my_dataset_vocab.txt",
                                     model_name="google/t5-v1_1-base")
    e3fp_tokenizer = E3FPTokenizer(fp_level=4, fp_bits=4096)

    print("Loading Cached BACE Dataset...")
    csv_path = "dataset/bace/raw/bace.csv"
    tr_smiles, tr_y, va_smiles, va_y, te_smiles, te_y = get_official_scaffold_split("bace", raw_csv_path=csv_path)

    # 🌟 直接秒读之前的 Cache
    train_dataset = MoleculeNetDataset(tr_smiles, tr_y, motif_tokenizer, e3fp_tokenizer, cache_name="bace_train")
    eval_dataset = MoleculeNetDataset(va_smiles, va_y, motif_tokenizer, e3fp_tokenizer, cache_name="bace_eval")
    test_dataset = MoleculeNetDataset(te_smiles, te_y, motif_tokenizer, e3fp_tokenizer, cache_name="bace_test")

    collator = InstructionCollator(tokenizer=motif_tokenizer)

    training_args = TrainingArguments(
        output_dir="./finetune_bace_instruction",
        num_train_epochs=30,
        per_device_train_batch_size=8,  # Seq2Seq 显存占用较大，适当减小 Batch
        per_device_eval_batch_size=16,
        learning_rate=3e-4,  # 🚀 LoRA 通常需要比全参数微调更大的学习率
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        save_safetensors=False,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, motif_tokenizer.tokenizer)
    )

    print("Starting Generative Instruction Fine-tuning...")
    trainer.train()

    print("Evaluating on Test Set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"🎉 Final Generative Test ROC-AUC on BACE: {test_results['eval_roc_auc']:.4f}")


if __name__ == "__main__":
    main()