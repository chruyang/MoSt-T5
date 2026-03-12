import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import pandas as pd
from argparse import ArgumentParser
from transformers import Trainer, TrainingArguments, set_seed, EarlyStoppingCallback
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import get_peft_model, LoraConfig
from sklearn.metrics import roc_auc_score, mean_squared_error
from rdkit import Chem

from model.modeling import MoStT5ForConditionalGeneration
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# ==========================================
# 🌍 1. MoleculeNet 全局任务注册表 (精简与更新版)
# ==========================================
MOLECULE_NET_CONFIG = {
    # --- 您专属的数据集 ---
    "estrogen": {"task_type": "classification", "smiles_col": "smiles", "target_cols": ["alpha", "beta"]},
    "metstab": {"task_type": "classification", "smiles_col": "smiles", "target_cols": ["high", "low"]},
    
    # --- 分类任务 (Classification) ---
    "bbbp": {"task_type": "classification", "smiles_col": "smiles", "target_cols": ["p_np"]},
    "bace": {"task_type": "classification", "smiles_col": "smiles", "target_cols": ["Class"]},
    "clintox": {"task_type": "classification", "smiles_col": "smiles", "target_cols": ["FDA_APPROVED", "CT_TOX"]}, 
    "sider": {"task_type": "classification", "smiles_col": "smiles", "target_cols": ['Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders', 'Investigations', 'Musculoskeletal and connective tissue disorders', 'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders', 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)', 'General disorders and administration site conditions', 'Endocrine disorders', 'Surgical and medical procedures', 'Vascular disorders', 'Blood and lymphatic system disorders', 'Skin and subcutaneous tissue disorders', 'Congenital, familial and genetic disorders', 'Infections and infestations', 'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders', 'Renal and urinary disorders', 'Pregnancy, puerperium and perinatal conditions', 'Ear and labyrinth disorders', 'Cardiac disorders', 'Nervous system disorders', 'Injury, poisoning and procedural complications']},
    
    # --- 毒性预测双雄 ---
    "tox21": {"task_type": "classification", "smiles_col": "smiles", "target_cols": ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']},
    "toxcast": {"task_type": "classification", "smiles_col": "smiles", "target_cols": "AUTO"}, # 🌟 核心：使用 AUTO 自动解析 617 个任务

    # --- 其他备用任务 ---
    "hiv": {"task_type": "classification", "smiles_col": "smiles", "target_cols": ["HIV_active"]},
    "muv": {"task_type": "classification", "smiles_col": "smiles", "target_cols": ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689', 'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810', 'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']},
    "esol": {"task_type": "regression", "smiles_col": "smiles", "target_cols": ["measured log solubility in mols per litre"]},
    "freesolv": {"task_type": "regression", "smiles_col": "smiles", "target_cols": ["expt"]},
    "lipophilicity": {"task_type": "regression", "smiles_col": "smiles", "target_cols": ["exp"]},
}

# ==========================================
# 🚀 2. 原子级到 Motif 级映射
# ==========================================
def generate_atom_to_motif_map_online(smiles, motif_ids):
    from model.CAMT5.representation import linearize

    mol = Chem.MolFromSmiles(smiles)
    if not mol: return []
    try:
        Chem.Kekulize(mol)
        smi_kekule = Chem.MolToSmiles(mol, kekuleSmiles=True)
    except:
        smi_kekule = smiles

    full_mapping = []
    atom_offset = 0
    sub_mols = smi_kekule.split(".")
    for sub_smi in sub_mols:
        try:
            _, _, mapping = linearize(sub_smi)
            for frag_indices in mapping:
                full_mapping.append([idx + atom_offset for idx in frag_indices])
            m_tmp = Chem.MolFromSmiles(sub_smi)
            if m_tmp: atom_offset += m_tmp.GetNumAtoms()
        except: pass

    num_atoms = mol.GetNumAtoms()
    atom_to_motif_list = [0] * num_atoms  
    
    for motif_idx, atom_indices in enumerate(full_mapping):
        token_idx = motif_idx + 1  
        if token_idx >= len(motif_ids): break
        for a_idx in atom_indices:
            if a_idx < num_atoms: atom_to_motif_list[a_idx] = token_idx
    return atom_to_motif_list

# ==========================================
# 🚀 3. 多任务属性预测器
# ==========================================
class MoStT5PropertyPredictor(nn.Module):
    def __init__(self, encoder, config, num_tasks, task_type):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.num_tasks = num_tasks
        self.task_type = task_type
        
        hidden_size = self.config.d_model
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_tasks)
        )
        
    def forward(self, input_ids, attention_mask, e3fp_ids, atom_to_motif_map, atom_attention_mask, labels=None, **kwargs):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            e3fp_ids=e3fp_ids,
            atom_to_motif_map=atom_to_motif_map,
            atom_attention_mask=atom_attention_mask,
            return_dict=True
        )
        hidden_states = encoder_outputs.last_hidden_state 
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            valid_mask = (labels != -100.0).float()
            safe_labels = labels.clone()
            safe_labels[labels == -100.0] = 0.0
            
            if self.task_type == "classification":
                loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            else:
                loss_fct = nn.MSELoss(reduction='none')
                
            loss_matrix = loss_fct(logits, safe_labels)
            loss = (loss_matrix * valid_mask).sum() / torch.clamp(valid_mask.sum(), min=1e-9)
                
        return SequenceClassifierOutput(loss=loss, logits=logits)

SPLIT_TO_ID = {'train': 0, 'val': 1, 'test': 2, 'validation': 1}

# ==========================================
# 🚀 4. Dataset (含全局特征缓存与 AUTO 自愈机制)
# ==========================================
class MoleculeNetDataset(Dataset):
    def __init__(self, data_dir, dataset_name, split_name, motif_tokenizer, e3fp_tokenizer, split_type="scaffold", seed=42):
        self.data = []
        
        config = MOLECULE_NET_CONFIG[dataset_name]
        smiles_col = config["smiles_col"]
        target_cols = config["target_cols"]
        self.task_type = config["task_type"]
        
        csv_path = f"{data_dir}/{dataset_name}/{dataset_name}.csv"
        global_cache_path = f"{data_dir}/{dataset_name}_global_feature_cache.pt"

        # 🌟 核心：防御型 AUTO 解析机制
        if target_cols == "AUTO":
            temp_df = pd.read_csv(csv_path, nrows=0, sep=None, engine='python')
            target_cols = [col.strip() for col in temp_df.columns if col.strip() != smiles_col]
            config["target_cols"] = target_cols 
            if split_name == "train":
                print(f"🔍 [{dataset_name}] AUTO: Automatically detected {len(target_cols)} task columns.")

        if os.path.exists(global_cache_path):
            if split_name == "train": 
                print(f"📦 Loading GLOBAL feature cache from {global_cache_path}...")
            with open(global_cache_path, 'rb') as f:
                global_features = pickle.load(f)
        else:
            print(f"⏳ Global cache not found. Building for the ENTIRE [{dataset_name}] dataset (This only runs ONCE)...")
            df_full = pd.read_csv(csv_path, sep=None, engine='python')
            df_full.columns = [col.strip() for col in df_full.columns]
            df_full = df_full.dropna(subset=[smiles_col])
            
            global_features = {}
            unique_smiles = df_full[smiles_col].unique()
            print(f"⚙️ Extracting 3D/2D features for {len(unique_smiles)} unique molecules...")
            
            for smiles in unique_smiles:
                try:
                    motif_ids = motif_tokenizer.tokenizer.encode(smiles, add_special_tokens=True)
                    if len(motif_ids) > 400: continue
                    
                    atom_map_list = generate_atom_to_motif_map_online(smiles, motif_ids)
                    
                    if hasattr(e3fp_tokenizer, 'from_smiles'): e3fp_out = e3fp_tokenizer.from_smiles(smiles)
                    elif hasattr(e3fp_tokenizer, 'encode'): e3fp_out = e3fp_tokenizer.encode(smiles)
                    else: e3fp_out = e3fp_tokenizer(smiles)
                        
                    e3fp_tensor = torch.tensor(e3fp_out['input_ids'] if isinstance(e3fp_out, dict) else e3fp_out, dtype=torch.long).clone().detach()
                    if e3fp_tensor.dim() > 2: e3fp_tensor = e3fp_tensor.squeeze(0)

                    num_e3fp_atoms = e3fp_tensor.shape[0]
                    atom_map_list = atom_map_list[:num_e3fp_atoms]
                    if len(atom_map_list) < num_e3fp_atoms:
                        atom_map_list.extend([0] * (num_e3fp_atoms - len(atom_map_list)))
                    atom_mask = (e3fp_tensor[:, 0] != e3fp_tokenizer.padding_idx).long().tolist()

                    global_features[smiles] = {
                        'motif_input_ids': torch.tensor(motif_ids, dtype=torch.long),
                        'e3fp_input_ids': e3fp_tensor,
                        'atom_to_motif_map': torch.tensor(atom_map_list, dtype=torch.long),
                        'atom_attention_mask': torch.tensor(atom_mask, dtype=torch.long)
                    }
                except Exception:
                    continue
                    
            os.makedirs(os.path.dirname(global_cache_path), exist_ok=True)
            with open(global_cache_path, 'wb') as f:
                pickle.dump(global_features, f)
            print(f"✅ Global cache built successfully with {len(global_features)} valid molecules!")

        if split_type == "scaffold":
            split_path = f"{data_dir}/{dataset_name}/splits/scaffold-0.npy" if os.path.exists(f"{data_dir}/{dataset_name}/splits/scaffold-0.npy") else f"{data_dir}/{dataset_name}/splits/scaffold.npy"
        else:
            split_path = f"{data_dir}/{dataset_name}/splits/random-0.npy" if os.path.exists(f"{data_dir}/{dataset_name}/splits/random-0.npy") else f"{data_dir}/{dataset_name}/splits/random_seed{seed}.npy"

        if not os.path.exists(split_path):
            raise FileNotFoundError(f"❌ Split index missing: {split_path}")

        df = pd.read_csv(csv_path, sep=None, engine='python')
        df.columns = [col.strip() for col in df.columns]
        
        split_key = "val" if split_name == "validation" else split_name
        use_idxs = np.load(split_path, allow_pickle=True)[SPLIT_TO_ID[split_key]]
        df_split = df.iloc[use_idxs].dropna(subset=[smiles_col])
        
        target_smiles = df_split[smiles_col].tolist()
        target_labels = np.nan_to_num(df_split[target_cols].values, nan=-100.0)

        for i, smiles in enumerate(target_smiles):
            if smiles in global_features:
                feat_dict = global_features[smiles].copy() 
                feat_dict['smiles'] = smiles
                feat_dict['label'] = torch.tensor(target_labels[i], dtype=torch.float)
                self.data.append(feat_dict)
                
        if split_name == "train":
            print(f"✅ Successfully loaded {len(self.data)} molecules for [{split_name}] from global cache using {split_type} split.")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

class PropertyCollator:
    def __init__(self, padding_value=0, e3fp_pad=-1):
        self.pad = padding_value
        self.e3fp_pad = e3fp_pad

    def __call__(self, batch):
        motif_ids = [item['motif_input_ids'] for item in batch]
        e3fp_ids = [item['e3fp_input_ids'] for item in batch]
        atom_maps = [item['atom_to_motif_map'] for item in batch]
        atom_masks = [item['atom_attention_mask'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])

        batch_motif = pad_sequence(motif_ids, batch_first=True, padding_value=self.pad)
        batch_mask = (batch_motif != self.pad).long()
        batch_e3fp = pad_sequence(e3fp_ids, batch_first=True, padding_value=self.e3fp_pad)
        batch_map = pad_sequence(atom_maps, batch_first=True, padding_value=0)
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
# 🚀 5. 对齐 Molformer 的宏平均评估
# ==========================================
def compute_metrics(eval_pred, task_type="classification"):
    logits, labels = eval_pred
    valid_mask = (labels != -100.0)
    
    if task_type == "classification":
        probs = 1.0 / (1.0 + np.exp(-logits))
        aucs = []
        
        for i in range(labels.shape[1]):
            task_labels = labels[:, i]
            task_probs = probs[:, i]
            task_mask = valid_mask[:, i]
            
            valid_task_labels = task_labels[task_mask]
            valid_task_probs = task_probs[task_mask]
            
            if len(np.unique(valid_task_labels)) > 1:
                auc = roc_auc_score(valid_task_labels, valid_task_probs)
                aucs.append(auc)
                
        return {"roc_auc": np.mean(aucs) if aucs else 0.5}
        
    else:
        rmses = []
        for i in range(labels.shape[1]):
            task_labels = labels[:, i]
            task_logits = logits[:, i]
            task_mask = valid_mask[:, i]
            
            valid_task_labels = task_labels[task_mask]
            valid_task_logits = task_logits[task_mask]
            
            if len(valid_task_labels) > 0:
                rmses.append(np.sqrt(mean_squared_error(valid_task_labels, valid_task_logits)))
                
        return {"rmse": np.mean(rmses) if rmses else 0.0}

# ==========================================
# 🚀 6. 主函数流程
# ==========================================
def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split_type", type=str, default="scaffold", choices=["scaffold", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    set_seed(args.seed)
    
    # 🌟 核心：在模型初始化前拦截并解析 AUTO
    dataset_cfg = MOLECULE_NET_CONFIG[args.dataset_name]
    if dataset_cfg["target_cols"] == "AUTO":
        csv_path = f"{args.data_dir}/{args.dataset_name}/{args.dataset_name}.csv"
        temp_df = pd.read_csv(csv_path, nrows=0, sep=None, engine='python')
        dataset_cfg["target_cols"] = [col.strip() for col in temp_df.columns if col.strip() != dataset_cfg["smiles_col"]]
        
    TASK_TYPE = dataset_cfg["task_type"]
    NUM_TASKS = len(dataset_cfg["target_cols"])

    print(f"\n🚀 启动任务: {args.dataset_name.upper()} | 类型: {TASK_TYPE} | 标签数量: {NUM_TASKS}")
    print(f"Loading base model from {args.model_path}...")
    
    from transformers import AutoConfig
    from model.configuration import MoStT5Config
    AutoConfig.register("most-t5", MoStT5Config)
    
    base_model = MoStT5ForConditionalGeneration.from_pretrained(args.model_path)

    peft_config = LoraConfig(
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "k", "v", "o"] 
    )
    encoder_with_lora = get_peft_model(base_model.encoder, peft_config)
    
    model = MoStT5PropertyPredictor(encoder=encoder_with_lora, config=base_model.config, num_tasks=NUM_TASKS, task_type=TASK_TYPE)
    
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Trainable params (Encoder LoRA + Head): {trainable_params:,}")

    motif_tokenizer = MotifTokenizer(vocab_file="asset/mol_vocabs/my_dataset_vocab.txt", model_name="google/t5-v1_1-base")
    e3fp_tokenizer = E3FPTokenizer(fp_level=4, fp_bits=4096)

    train_dataset = MoleculeNetDataset(args.data_dir, args.dataset_name, "train", motif_tokenizer, e3fp_tokenizer, split_type=args.split_type, seed=args.seed)
    val_dataset = MoleculeNetDataset(args.data_dir, args.dataset_name, "val", motif_tokenizer, e3fp_tokenizer, split_type=args.split_type, seed=args.seed)
    test_dataset = MoleculeNetDataset(args.data_dir, args.dataset_name, "test", motif_tokenizer, e3fp_tokenizer, split_type=args.split_type, seed=args.seed)
    
    collator = PropertyCollator()

    training_args = TrainingArguments(
        output_dir=f"./checkpoints/{args.dataset_name}_seed{args.seed}",
        num_train_epochs=args.epochs,  
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc" if TASK_TYPE == "classification" else "rmse",
        greater_is_better=(TASK_TYPE == "classification"),
        save_total_limit=1,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        save_safetensors=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, TASK_TYPE),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=30)]
    )

    print(f"🔥 Starting Fine-tuning for {args.dataset_name.upper()}...")
    trainer.train()

    print(f"🧪 Evaluating on Blind Test Set...")
    test_results = trainer.predict(test_dataset)
    
    print("\n" + "=" * 50)
    print(f"✅ [FINAL RESULT] Dataset: {args.dataset_name.upper()} | Seed: {args.seed}")
    if TASK_TYPE == "classification": print(f"✅ Test Macro-ROC-AUC: {test_results.metrics['test_roc_auc']:.4f}")
    else: print(f"✅ Test Macro-RMSE: {test_results.metrics['test_rmse']:.4f}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()