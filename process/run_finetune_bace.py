import torch
import numpy as np
from transformers import Trainer, TrainingArguments
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

from model.modeling import MoStT5ForConditionalGeneration
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
from finetune_cls_model import MoStT5ForSequenceClassification
from finetune_dataset import MoleculeNetDataset, PropertyCollator, get_official_scaffold_split


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probs = softmax(predictions, axis=1)[:, 1]
    auc = roc_auc_score(labels, probs)
    return {"roc_auc": auc}


def main():
    # 1. 加载相同的预训练权重
    pretrained_path = "./checkpoint"

    print("Loading Pretrained Model...")
    base_model = MoStT5ForConditionalGeneration.from_pretrained(pretrained_path)
    model = MoStT5ForSequenceClassification(pretrained_model=base_model, num_labels=2)

    motif_tokenizer = MotifTokenizer(vocab_file="asset/mol_vocabs/my_dataset_vocab.txt",
                                     model_name="google/t5-v1_1-base")
    e3fp_tokenizer = E3FPTokenizer(fp_level=4, fp_bits=4096)

    # 2. 准备 BACE 数据
    print("Loading and Splitting BACE Dataset (Scaffold Split)...")
    # 🚀 替换为 bace，并指定 csv 路径（如果您的路径不同，请在这里修改）
    csv_path = "dataset/bace/raw/bace.csv"
    tr_smiles, tr_y, va_smiles, va_y, te_smiles, te_y = get_official_scaffold_split("bace", raw_csv_path=csv_path)

    # 🚀 更改 cache_name，建立 BACE 的独立缓存
    train_dataset = MoleculeNetDataset(tr_smiles, tr_y, motif_tokenizer, e3fp_tokenizer, cache_name="bace_train")
    eval_dataset = MoleculeNetDataset(va_smiles, va_y, motif_tokenizer, e3fp_tokenizer, cache_name="bace_eval")
    test_dataset = MoleculeNetDataset(te_smiles, te_y, motif_tokenizer, e3fp_tokenizer, cache_name="bace_test")

    collator = PropertyCollator()

    # 3. 设定微调参数
    training_args = TrainingArguments(
        output_dir="./finetune_bace_results",  # 🚀 更改输出文件夹
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=1e-5,  # BACE 数据量也小，保持低学习率
        warmup_ratio=0.1,
        weight_decay=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        save_safetensors=False,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    # 4. 开始微调
    print("Starting Fine-tuning...")
    trainer.train()

    # 5. 在 Test Set 上给出最终分数
    print("Evaluating on Test Set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"🎉 Final Test ROC-AUC on BACE: {test_results['eval_roc_auc']:.4f}")


if __name__ == "__main__":
    main()