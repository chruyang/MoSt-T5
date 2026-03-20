import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import pandas as pd
import logging
import sys
import glob
from argparse import ArgumentParser
from transformers import Trainer, TrainingArguments, set_seed, EarlyStoppingCallback, TrainerCallback
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import get_peft_model, LoraConfig
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rdkit import Chem
import transformers

from model.modeling import MoStT5ForConditionalGeneration
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

ADMET_CONFIG = {
    "task_type": "regression",
    "smiles_col": "CXSMILES",
    "target_cols": ["MLM", "HLM", "KSOL", "LogD", "MDR1-MDCKII"],
    "log_transform_cols": ["MLM", "HLM", "KSOL", "MDR1-MDCKII"],
    "split_col": "Set"
}


class LoggingCallback(TrainerCallback):
    def __init__(self, logger, target_name):
        self.logger = logger
        self.target_name = target_name

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            epoch = state.epoch
            rmse = metrics.get('eval_rmse', 0.0)
            mae = metrics.get('eval_mae', 0.0)
            loss = metrics.get('eval_loss', 0.0)
            self.logger.info(
                f"[{self.target_name}] Epoch: {epoch:.0f} | Dev Loss: {loss:.4f} | Dev RMSE: {rmse:.4f} | Dev MAE: {mae:.4f}")


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
        except:
            pass

    num_atoms = mol.GetNumAtoms()
    atom_to_motif_list = [0] * num_atoms

    for motif_idx, atom_indices in enumerate(full_mapping):
        token_idx = motif_idx + 1
        if token_idx >= len(motif_ids): break
        for a_idx in atom_indices:
            if a_idx < num_atoms: atom_to_motif_list[a_idx] = token_idx
    return atom_to_motif_list


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
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_tasks)
        )

    def forward(self, input_ids, attention_mask, e3fp_ids, atom_to_motif_map, atom_attention_mask, labels=None,
                **kwargs):
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, e3fp_ids=e3fp_ids,
            atom_to_motif_map=atom_to_motif_map, atom_attention_mask=atom_attention_mask, return_dict=True
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

            loss_fct = nn.MSELoss(reduction='none')
            loss_matrix = loss_fct(logits, safe_labels)
            loss = (loss_matrix * valid_mask).sum() / torch.clamp(valid_mask.sum(), min=1e-9)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class ADMETDataset(Dataset):
    def __init__(self, data_dir, dataset_name, split_name, motif_tokenizer, e3fp_tokenizer, target_cols,
                 log_transform_cols, seed=42, logger=None):
        self.data = []
        config = ADMET_CONFIG
        smiles_col = config["smiles_col"]
        split_col = config["split_col"]

        csv_path = f"{data_dir}/{dataset_name}/{dataset_name}.csv"
        global_cache_path = f"{data_dir}/{dataset_name}_global_feature_cache.pt"

        if os.path.exists(global_cache_path):
            with open(global_cache_path, 'rb') as f:
                global_features = pickle.load(f)
        else:
            df_full = pd.read_csv(csv_path, sep=None, engine='python')
            df_full.columns = [col.strip() for col in df_full.columns]
            df_full = df_full.dropna(subset=[smiles_col])

            global_features = {}
            unique_smiles = df_full[smiles_col].unique()
            for smiles in unique_smiles:
                try:
                    motif_ids = motif_tokenizer.tokenizer.encode(smiles, add_special_tokens=True)
                    if len(motif_ids) > 400: continue
                    atom_map_list = generate_atom_to_motif_map_online(smiles, motif_ids)

                    if hasattr(e3fp_tokenizer, 'from_smiles'):
                        e3fp_out = e3fp_tokenizer.from_smiles(smiles)
                    elif hasattr(e3fp_tokenizer, 'encode'):
                        e3fp_out = e3fp_tokenizer.encode(smiles)
                    else:
                        e3fp_out = e3fp_tokenizer(smiles)

                    e3fp_tensor = torch.tensor(e3fp_out['input_ids'] if isinstance(e3fp_out, dict) else e3fp_out,
                                               dtype=torch.long).clone().detach()
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

        df = pd.read_csv(csv_path, sep=None, engine='python')
        df.columns = [col.strip() for col in df.columns]

        if split_name == "train":
            df_split = df[df[split_col].str.lower() == 'train'].copy()
        elif split_name in ["val", "validation"]:
            df_split = df[df[split_col].str.lower().isin(['val', 'validation'])].copy()
            if len(df_split) == 0:
                df_split = df[df[split_col].str.lower() == 'test'].copy()
        elif split_name == "test":
            df_split = df[df[split_col].str.lower() == 'test'].copy()

        df_split = df_split.dropna(subset=[smiles_col])
        target_smiles = df_split[smiles_col].tolist()
        raw_labels = df_split[target_cols].astype(float).values

        for i, col in enumerate(target_cols):
            if col in log_transform_cols:
                valid_mask = ~np.isnan(raw_labels[:, i])
                clipped_vals = np.clip(raw_labels[valid_mask, i], a_min=0, a_max=None)
                raw_labels[valid_mask, i] = np.log10(clipped_vals + 1)

        target_labels = np.nan_to_num(raw_labels, nan=-100.0)

        for i, smiles in enumerate(target_smiles):
            if smiles in global_features:
                feat_dict = global_features[smiles].copy()
                feat_dict['smiles'] = smiles
                feat_dict['label'] = torch.tensor(target_labels[i], dtype=torch.float)
                self.data.append(feat_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PropertyCollator:
    def __init__(self, padding_value=0, e3fp_pad=-1):
        self.pad = padding_value
        self.e3fp_pad = e3fp_pad

    def __call__(self, batch):
        motif_ids = pad_sequence([item['motif_input_ids'] for item in batch], batch_first=True, padding_value=self.pad)
        batch_mask = (motif_ids != self.pad).long()
        e3fp_ids = pad_sequence([item['e3fp_input_ids'] for item in batch], batch_first=True,
                                padding_value=self.e3fp_pad)
        atom_maps = pad_sequence([item['atom_to_motif_map'] for item in batch], batch_first=True, padding_value=0)
        atom_masks = pad_sequence([item['atom_attention_mask'] for item in batch], batch_first=True, padding_value=0)
        labels = torch.stack([item['label'] for item in batch])
        return {"input_ids": motif_ids, "attention_mask": batch_mask, "e3fp_ids": e3fp_ids,
                "atom_to_motif_map": atom_maps, "atom_attention_mask": atom_masks, "labels": labels}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    valid_mask = (labels != -100.0)
    rmses, maes = [], []
    for i in range(labels.shape[1]):
        task_labels = labels[:, i][valid_mask[:, i]]
        task_logits = logits[:, i][valid_mask[:, i]]
        if len(task_labels) > 0:
            rmses.append(np.sqrt(mean_squared_error(task_labels, task_logits)))
            maes.append(mean_absolute_error(task_labels, task_logits))
    return {"rmse": np.mean(rmses) if rmses else 0.0, "mae": np.mean(maes) if maes else 0.0}


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--dataset_name", type=str, default="antiviral_admet_2025")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    os.makedirs("./ablation_logs", exist_ok=True)
    log_file = f"./ablation_logs/admet_{args.seed}.log"

    logger = logging.getLogger("ADMET_Logger")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    transformers.logging.set_verbosity_error()

    set_seed(args.seed)
    config = ADMET_CONFIG
    all_targets = config["target_cols"]

    motif_tokenizer = MotifTokenizer(vocab_file="asset/mol_vocabs/my_dataset_vocab.txt",
                                     model_name="google/t5-v1_1-base")
    e3fp_tokenizer = E3FPTokenizer(fp_level=4, fp_bits=4096)
    collator = PropertyCollator()

    y_pred_my_model = {}

    csv_path = f"{args.data_dir}/{args.dataset_name}/{args.dataset_name}.csv"
    df = pd.read_csv(csv_path, sep=None, engine='python')
    split_col = config["split_col"]
    df_test = df[df[split_col].str.lower() == 'test'].copy()
    y_true_dict = {}

    from transformers import AutoConfig
    from model.configuration import MoStT5Config
    AutoConfig.register("most-t5", MoStT5Config)

    for target in all_targets:
        current_target_cols = [target]
        current_log_cols = [target] if target in config["log_transform_cols"] else []
        y_true_dict[target] = df_test[target].astype(float).values

        train_dataset = ADMETDataset(args.data_dir, args.dataset_name, "train", motif_tokenizer, e3fp_tokenizer,
                                     current_target_cols, current_log_cols, args.seed, logger)
        val_dataset = ADMETDataset(args.data_dir, args.dataset_name, "val", motif_tokenizer, e3fp_tokenizer,
                                   current_target_cols, current_log_cols, args.seed, logger)
        test_dataset = ADMETDataset(args.data_dir, args.dataset_name, "test", motif_tokenizer, e3fp_tokenizer,
                                    current_target_cols, current_log_cols, args.seed, logger)

        base_model = MoStT5ForConditionalGeneration.from_pretrained(args.model_path)

        peft_config = LoraConfig(
            inference_mode=False,
            r=64,
            lora_alpha=128,
            lora_dropout=0.1,
            target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
        )
        encoder_with_lora = get_peft_model(base_model.encoder, peft_config)

        model = MoStT5PropertyPredictor(encoder=encoder_with_lora, config=base_model.config, num_tasks=1,
                                        task_type="regression")
        for param in model.classifier.parameters(): param.requires_grad = True

        training_args = TrainingArguments(
            output_dir=f"./checkpoints/{args.dataset_name}_{target}_seed{args.seed}",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            learning_rate=args.lr,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="rmse",
            greater_is_better=False,
            save_total_limit=1,
            bf16=True,
            report_to="none",
            disable_tqdm=True,
            remove_unused_columns=False,
            save_safetensors=False
        )

        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
            data_collator=collator, compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=50),
                LoggingCallback(logger, target)
            ]
        )

        ckpt_dir = f"./checkpoints/{args.dataset_name}_{target}_seed{args.seed}"
        ckpt_paths = glob.glob(f"{ckpt_dir}/checkpoint-*")

        if len(ckpt_paths) > 0:
            best_ckpt = sorted(ckpt_paths, key=lambda x: int(x.split('-')[-1]))[-1]
            weight_path = os.path.join(best_ckpt, "pytorch_model.bin")
            model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        else:
            trainer.train()

        test_results = trainer.predict(test_dataset)
        pred_col = test_results.predictions[:, 0]

        if target in current_log_cols:
            pred_col = np.power(10, pred_col) - 1.0
            pred_col = np.clip(pred_col, a_min=0.0, a_max=None)

        y_pred_my_model[target] = pred_col

        del trainer, model, base_model, encoder_with_lora
        torch.cuda.empty_cache()

    export_data = {
        "y_true": y_true_dict,
        "all_y_pred": {"MoSt-T5 (Ours)": y_pred_my_model}
    }

    dump_path = f"./eval_data_seed{args.seed}.pkl"
    with open(dump_path, "wb") as f:
        pickle.dump(export_data, f)


if __name__ == "__main__":
    main()