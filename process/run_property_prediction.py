import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import pandas as pd
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import roc_auc_score, mean_squared_error

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

# 🚀 导入您现有的生成模型和分词器
from model.modeling import MoStT5ForConditionalGeneration
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from finetune_dataset import generate_atom_to_motif_map_online


# ==========================================
# 0. 🚀 动态构建的属性预测器 (无需修改 modeling.py)
# ==========================================
class MoStT5PropertyPredictor(nn.Module):
    def __init__(self, lora_model, num_labels):
        super().__init__()
        # 只提取底座模型的 Encoder 部分
        self.encoder = lora_model.get_encoder()
        self.config = lora_model.config
        self.num_labels = num_labels

        # 获取 Encoder 的隐藏层维度 (T5-base 通常是 768)
        hidden_size = self.config.d_model

        # 构建一个全新的分类/回归头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, e3fp_ids, atom_to_motif_map, atom_attention_mask, labels=None,
                **kwargs):
        # 1. 抽取 3D-2D 融合特征
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            e3fp_ids=e3fp_ids,
            atom_to_motif_map=atom_to_motif_map,
            atom_attention_mask=atom_attention_mask,
            return_dict=True
        )
        hidden_states = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        # 2. 平均池化 (Mean Pooling)，获取整个分子的全局表征
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # 3. 通过预测头输出数值
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # 回归任务
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # 分类任务
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ==========================================
# 1. 极其稳健的 MoleculeNet 官方数据 + Scaffold Split
# ==========================================
class MoleculeNetDataset(Dataset):
    def __init__(self, dataset_name, split_name, motif_tokenizer, e3fp_tokenizer, task_type="classification"):
        self.motif_tokenizer = motif_tokenizer
        self.e3fp_tokenizer = e3fp_tokenizer
        self.task_type = task_type
        self.data = []

        cache_path = f"dataset/{dataset_name}_cache_{split_name}.pt"

        if os.path.exists(cache_path):
            print(f"📦 Loading cached data from {cache_path}...")
            with open(cache_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            print(f"📥 Downloading raw official CSV for '{dataset_name}'...")
            if dataset_name == 'clintox':
                url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
                df = pd.read_csv(url)
                df = df.dropna(subset=['smiles', 'FDA_APPROVED'])
                smiles_list = df['smiles'].tolist()
                labels_list = df['FDA_APPROVED'].tolist()
            elif dataset_name == 'esol':
                url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
                df = pd.read_csv(url)
                df = df.dropna(subset=['smiles', 'measured log solubility in mols per litre'])
                smiles_list = df['smiles'].tolist()
                labels_list = df['measured log solubility in mols per litre'].tolist()
            elif dataset_name == 'bbbp':  # 🚀 新增 BBBP 解析逻辑
                url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
                df = pd.read_csv(url)
                df = df.dropna(subset=['smiles', 'p_np'])
                smiles_list = df['smiles'].tolist()
                labels_list = df['p_np'].tolist()
            elif dataset_name == 'bace':  # 🚀 新增 BACE 解析逻辑
                url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
                df = pd.read_csv(url)
                df = df.dropna(subset=['mol', 'Class'])
                smiles_list = df['mol'].tolist()  # BACE 的 SMILES 列叫 'mol'
                labels_list = df['Class'].tolist()
            else:
                raise ValueError(f"Dataset {dataset_name} direct link not configured.")

            # 🧪 执行严格的 Scaffold Split (分子骨架划分)
            print("🧪 Performing rigorous Scaffold Split (MoleculeNet Standard)...")
            scaffolds = defaultdict(list)
            for i, smi in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                    scaffolds[scaffold].append(i)
                else:
                    scaffolds[''].append(i)

            # 按骨架包含的分子数量从大到小排序 (引入 key 保证确定性)
            scaffold_sets = [scaffolds[s] for s in
                             sorted(scaffolds.keys(), key=lambda k: (len(scaffolds[k]), k), reverse=True)]

            train_idx, val_idx, test_idx = [], [], []
            train_cutoff = int(0.8 * len(smiles_list))
            val_cutoff = int(0.9 * len(smiles_list))

            # 将同一骨架的分子打包放入对应的集合
            for scaffold_set in scaffold_sets:
                if len(train_idx) + len(scaffold_set) <= train_cutoff:
                    train_idx.extend(scaffold_set)
                elif len(train_idx) + len(val_idx) + len(scaffold_set) <= val_cutoff:
                    val_idx.extend(scaffold_set)
                else:
                    test_idx.extend(scaffold_set)

            if split_name == "train":
                target_smi = [smiles_list[i] for i in train_idx]
                target_labels = [labels_list[i] for i in train_idx]
            elif split_name in ["val", "validation"]:
                target_smi = [smiles_list[i] for i in val_idx]
                target_labels = [labels_list[i] for i in val_idx]
            elif split_name == "test":
                target_smi = [smiles_list[i] for i in test_idx]
                target_labels = [labels_list[i] for i in test_idx]

            print(f"⚙️ Processing {len(target_smi)} molecules for [{split_name}] split...")
            for smiles, label in zip(target_smi, target_labels):
                try:
                    motif_ids = self.motif_tokenizer.tokenizer.encode(smiles, add_special_tokens=True)
                    if len(motif_ids) > 400: continue

                    atom_map_list = generate_atom_to_motif_map_online(smiles, motif_ids)

                    if hasattr(self.e3fp_tokenizer, 'encode'):
                        e3fp_out = self.e3fp_tokenizer.encode(smiles)
                    else:
                        e3fp_out = self.e3fp_tokenizer(smiles)

                    # 消除 UserWarning
                    e3fp_tensor = torch.tensor(e3fp_out['input_ids'] if isinstance(e3fp_out, dict) else e3fp_out,
                                               dtype=torch.long).clone().detach()
                    if e3fp_tensor.dim() > 2: e3fp_tensor = e3fp_tensor.squeeze(0)

                    max_ats = self.e3fp_tokenizer.max_atoms
                    atom_map_list = (atom_map_list[:max_ats] + [-1] * max_ats)[:max_ats]
                    atom_mask = (e3fp_tensor[:, 0] != self.e3fp_tokenizer.padding_idx).long().tolist()

                    self.data.append({
                        'smiles': smiles,
                        'motif_input_ids': torch.tensor(motif_ids, dtype=torch.long),
                        'e3fp_input_ids': e3fp_tensor,
                        'atom_to_motif_map': torch.tensor(atom_map_list, dtype=torch.long),
                        'atom_attention_mask': torch.tensor(atom_mask, dtype=torch.long),
                        'label': float(label) if task_type == "regression" else int(label)
                    })
                except Exception as e:
                    continue

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"✅ Cache saved to {cache_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ==========================================
# 2. 纯 Encoder 的 Collator
# ==========================================
class PropertyCollator:
    def __init__(self, padding_value=0, e3fp_pad=-1):
        self.pad = padding_value
        self.e3fp_pad = e3fp_pad

    def __call__(self, batch):
        motif_ids = [item['motif_input_ids'] for item in batch]
        e3fp_ids = [item['e3fp_input_ids'] for item in batch]
        atom_maps = [item['atom_to_motif_map'] for item in batch]
        atom_masks = [item['atom_attention_mask'] for item in batch]

        if isinstance(batch[0]['label'], float):
            labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)
        else:
            labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        batch_motif = pad_sequence(motif_ids, batch_first=True, padding_value=self.pad)
        batch_mask = (batch_motif != self.pad).long()
        batch_e3fp = pad_sequence(e3fp_ids, batch_first=True, padding_value=self.e3fp_pad)
        batch_map = pad_sequence(atom_maps, batch_first=True, padding_value=-1)
        batch_atom_mask = pad_sequence(atom_masks, batch_first=True, padding_value=0)

        return {
            "input_ids": batch_motif,
            "attention_mask": batch_mask,
            "e3fp_ids": batch_e3fp,
            "atom_to_motif_map": batch_map,
            "atom_attention_mask": batch_atom_mask,
            "labels": labels
        }


# ==========================================
# 3. 评估指标 (ROC-AUC 或 RMSE)
# ==========================================
def compute_metrics(eval_pred, task_type="classification"):
    logits, labels = eval_pred
    if task_type == "classification":
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.5
        return {"roc_auc": auc}
    else:
        rmse = np.sqrt(mean_squared_error(labels, logits))
        return {"rmse": rmse}


def main():
    # 🎯 切换任务：'clintox' (分类) 或 'esol' (回归)
    DATASET_NAME = 'bbbp'
    TASK_TYPE = 'classification' if DATASET_NAME in ['clintox', 'bbbp', 'bace'] else 'regression'
    NUM_LABELS = 2 if TASK_TYPE == "classification" else 1

    pretrained_path = "./checkpoint-127500"
    print(f"Loading Pure Encoder for {DATASET_NAME} {TASK_TYPE}...")

    # 加载生成模型底座
    base_model = MoStT5ForConditionalGeneration.from_pretrained(pretrained_path)

    peft_config = LoraConfig(
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "k", "v", "o"]
    )
    lora_model = get_peft_model(base_model, peft_config)

    # 挂载属性预测头
    model = MoStT5PropertyPredictor(lora_model, num_labels=NUM_LABELS)
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters for Property Prediction: {trainable_params:,}")

    motif_tokenizer = MotifTokenizer(vocab_file="asset/mol_vocabs/my_dataset_vocab.txt",
                                     model_name="google/t5-v1_1-base")
    e3fp_tokenizer = E3FPTokenizer(fp_level=4, fp_bits=4096)

    train_dataset = MoleculeNetDataset(DATASET_NAME, "train", motif_tokenizer, e3fp_tokenizer, TASK_TYPE)
    val_dataset = MoleculeNetDataset(DATASET_NAME, "val", motif_tokenizer, e3fp_tokenizer, TASK_TYPE)
    test_dataset = MoleculeNetDataset(DATASET_NAME, "test", motif_tokenizer, e3fp_tokenizer, TASK_TYPE)
    collator = PropertyCollator()

    training_args = TrainingArguments(
        output_dir=f"./finetune_{DATASET_NAME}",
        num_train_epochs=50,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=3e-4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc" if TASK_TYPE == "classification" else "rmse",
        greater_is_better=(TASK_TYPE == "classification"),
        bf16=True,
        report_to="none",
        save_total_limit=1,
        remove_unused_columns=False,  # 🚀 关键修复：防止丢弃 motif_input_ids
        save_safetensors=False  # 🚀 关键修复：允许保存 shared 权重
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, TASK_TYPE)
    )

    print(f"Starting {DATASET_NAME} Fine-tuning...")
    trainer.train()
    test_results = trainer.evaluate(test_dataset)
    print(f"🎉 Final Test Results: {test_results}")


if __name__ == "__main__":
    main()