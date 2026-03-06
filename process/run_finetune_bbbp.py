import torch
import numpy as np
from transformers import Trainer, TrainingArguments
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

# 导入您的自定义模块
from model.modeling import MoStT5ForConditionalGeneration
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
from finetune_cls_model import MoStT5ForSequenceClassification

# 导入处理好缓存逻辑的数据集组件
from finetune_dataset import MoleculeNetDataset, PropertyCollator, get_official_scaffold_split


def compute_metrics(eval_pred):
    """计算学术标准的 ROC-AUC"""
    predictions, labels = eval_pred
    probs = softmax(predictions, axis=1)[:, 1]  # 获取正类概率
    auc = roc_auc_score(labels, probs)
    return {"roc_auc": auc}


def main():
    # 1. 加载预训练权重 (⚠️ 请确保这里填的是您目前 loss 最低的那个 checkpoint 的路径)
    pretrained_path = "./checkpoint1"

    print("Loading Pretrained Model...")
    base_model = MoStT5ForConditionalGeneration.from_pretrained(pretrained_path)
    model = MoStT5ForSequenceClassification(pretrained_model=base_model, num_labels=2)

    # 💡 [可选策略]: 冻结底层 Encoder 仅微调分类头 (如果您依然发现严重过拟合，可以取消下面两行的注释)
    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    motif_tokenizer = MotifTokenizer(vocab_file="asset/mol_vocabs/my_dataset_vocab.txt",
                                     model_name="google/t5-v1_1-base")
    e3fp_tokenizer = E3FPTokenizer(fp_level=4, fp_bits=4096)

    # 2. 准备 Scaffold 数据
    print("Loading and Splitting BBBP Dataset (Scaffold Split)...")
    tr_smiles, tr_y, va_smiles, va_y, te_smiles, te_y = get_official_scaffold_split("bbbp")

    # 🚀 引入 cache_name 启用一键缓存机制，第一次会计算 20 分钟并保存，之后 1 秒加载！
    train_dataset = MoleculeNetDataset(tr_smiles, tr_y, motif_tokenizer, e3fp_tokenizer, cache_name="bbbp_train")
    eval_dataset = MoleculeNetDataset(va_smiles, va_y, motif_tokenizer, e3fp_tokenizer, cache_name="bbbp_eval")
    test_dataset = MoleculeNetDataset(te_smiles, te_y, motif_tokenizer, e3fp_tokenizer, cache_name="bbbp_test")

    collator = PropertyCollator()

    # 3. 设定微调参数 (🚀 已开启防爆盘、去报错，并应用抗过拟合的最优策略)
    training_args = TrainingArguments(
        output_dir="./finetune_bbbp_results",
        num_train_epochs=20,  # 💡 降低 Epoch 数，20 轮已足够
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=1e-5,  # 💡 学习率降低，减缓死记硬背
        warmup_ratio=0.1,
        weight_decay=0.05,  # 💡 提升正则化，增加泛化能力
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,  # 🛡️ 严禁 Trainer 删列 (解决 KeyError)
        save_safetensors=False,  # 🛡️ 回退 .bin 保存 (解决权重共享报错)
        save_total_limit=2  # 🛡️ 限制保存数量 (防止硬盘爆满)
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
    print(f"🎉 Final Test ROC-AUC: {test_results['eval_roc_auc']:.4f}")


if __name__ == "__main__":
    main()