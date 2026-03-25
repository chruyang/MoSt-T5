import torch
import json
import lmdb
import pickle
import logging
import numpy as np
import os
import re
from typing import List, Dict, Any
from argparse import ArgumentParser
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    EarlyStoppingCallback
)
from sklearn.metrics import mean_absolute_error

from model.modeling import MoStT5ForConditionalGeneration
from tokenization.text_tokenizer import TextTokenizer
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================
# 1. 终极 Dataset (绝对 -1 填充版)
# ==========================================
class GenerativeHybridDataset(Dataset):
    def __init__(self, json_path: str, lmdb_path: str,
                 text_tokenizer: TextTokenizer,
                 motif_tokenizer: MotifTokenizer,
                 e3fp_tokenizer: E3FPTokenizer,
                 max_source_length: int = 768,
                 max_target_length: int = 64,
                 is_eval: bool = False):

        self.json_path, self.lmdb_path = json_path, lmdb_path
        self.text_tokenizer, self.motif_tokenizer, self.e3fp_tokenizer = text_tokenizer, motif_tokenizer, e3fp_tokenizer
        self.e3fp_width = self.e3fp_tokenizer.fp_level + 1
        self.max_source_length, self.max_target_length = max_source_length, max_target_length
        self.is_eval = is_eval

        logger.info(f"📂 加载指令数据: {os.path.basename(json_path)}")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.qa_data = json.load(f)
        self.length = len(self.qa_data)

        logger.info(f"📂 挂载特征库: {os.path.basename(lmdb_path)}")
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False,
                             subdir=os.path.isdir(lmdb_path))

        self.cid_to_key = {}
        with self.env.begin() as txn:
            for key, value in txn.cursor():
                if key == b'__len__': continue
                try:
                    raw_cid = str(pickle.loads(value).get('cid', ''))
                    match = re.search(r'\d+', raw_cid)
                    if match: self.cid_to_key[match.group()] = key
                except:
                    continue

    def __len__(self):
        return self.length

    def _get_val(self, item: dict, possible_keys: list) -> str:
        for k in possible_keys:
            if k in item: return str(item[k])
            for d_k in item.keys():
                if d_k.lower().strip() == k.lower(): return str(item[d_k])
        return ""

    def handle_dimension_mismatch(self, e3fp_ids: torch.Tensor) -> torch.Tensor:
        current_width = e3fp_ids.shape[1]
        if current_width == self.e3fp_width: return e3fp_ids
        if current_width < self.e3fp_width:
            # 🚨🚨🚨 核心修正 1：维度不足时，绝对使用 -1 填充，杜绝随机特征注入！
            pad_tensor = torch.full((e3fp_ids.shape[0], self.e3fp_width - current_width), -1, dtype=torch.long)
            return torch.cat([e3fp_ids, pad_tensor], dim=1)
        return e3fp_ids[:, :self.e3fp_width]

    def __getitem__(self, idx):
        for attempt in range(10):
            try:
                item = self.qa_data[idx]

                prompt = self._get_val(item, ["Instruction", "instruction", "prompt"])
                target_text = self._get_val(item, ["Target", "target", "output"])
                raw_cid_str = self._get_val(item, ["Input", "input", "cid", "id"])

                match = re.search(r'\d+', raw_cid_str)
                if not match: raise ValueError(f"无效 CID: {raw_cid_str}")
                clean_cid = match.group()

                lmdb_key = self.cid_to_key.get(clean_cid)
                if not lmdb_key: raise ValueError(f"CID {clean_cid} 不在特征库中")

                with self.env.begin() as txn:
                    db_entry = pickle.loads(txn.get(lmdb_key))

                smiles = db_entry.get('smiles', db_entry.get('smi', ''))

                text_ids = \
                self.text_tokenizer.tokenizer(prompt, truncation=True, max_length=self.max_source_length // 2,
                                              return_tensors="pt")['input_ids'].squeeze(0)
                len_t = len(text_ids)

                motif_result = self.motif_tokenizer.encode(smiles, return_tensors='pt', padding=False,
                                                           return_mapping=True)
                motif_ids, motif_mapping = motif_result if isinstance(motif_result, tuple) else (motif_result, [])
                if motif_ids.dim() > 1: motif_ids = motif_ids.squeeze(0)
                len_m = len(motif_ids)

                if len_t + len_m > self.max_source_length:
                    motif_ids = motif_ids[:self.max_source_length - len_t]
                    len_m = len(motif_ids)

                input_ids = torch.cat([text_ids, motif_ids])
                target_ids = \
                self.text_tokenizer.tokenizer(target_text, truncation=True, max_length=self.max_target_length,
                                              return_tensors="pt")['input_ids'].squeeze(0)

                e3fp_numpy = db_entry.get('e3fp')
                e3fp_ids = torch.tensor(e3fp_numpy, dtype=torch.long)
                e3fp_ids = self.handle_dimension_mismatch(e3fp_ids)

                num_atoms = e3fp_ids.shape[0]
                atom_to_motif_map = torch.full((num_atoms,), -1, dtype=torch.long)
                atom_mapping = db_entry.get('atom_mapping', [])

                seq_len = len(input_ids)
                for motif_idx, atom_indices in enumerate(atom_mapping):
                    if motif_idx >= len(motif_mapping): break
                    real_token_idx = motif_mapping[motif_idx]
                    if real_token_idx < len_m:
                        target_pos = real_token_idx + len_t
                        if target_pos < seq_len:
                            if isinstance(atom_indices, int): atom_indices = [atom_indices]
                            for atom_idx in atom_indices:
                                if atom_idx < num_atoms:
                                    atom_to_motif_map[atom_idx] = target_pos

                # 🚨🚨🚨 核心修正 2：文本假原子绝对使用 -1 填充！
                dummy_e3fp = torch.full((len_t, self.e3fp_width), -1, dtype=torch.long)
                final_e3fp = torch.cat([dummy_e3fp, e3fp_ids])

                dummy_map = torch.full((len_t,), -1, dtype=torch.long)
                final_map = torch.cat([dummy_map, atom_to_motif_map])

                return {
                    "input_ids": input_ids, "labels": target_ids,
                    "e3fp_ids": final_e3fp, "atom_to_motif_map": final_map
                }
            except Exception as e:
                if self.is_eval: return None
                idx = (idx + 1) % self.length
        return None


# ==========================================
# 2. Generative Collator (极简原生版)
# ==========================================
class GenerativeCollator:
    def __init__(self, text_pad_id, e3fp_pad_id=-1):
        self.text_pad_id = text_pad_id
        self.e3fp_pad_id = e3fp_pad_id

    def __call__(self, batch):
        valid_batch = [f for f in batch if f is not None]
        if len(valid_batch) == 0:
            return {
                "input_ids": torch.tensor([[self.text_pad_id]]),
                "attention_mask": torch.tensor([[0]]),
                "labels": torch.tensor([[-100]]),
                "e3fp_ids": torch.tensor([[[-1] * 4]]),
                "atom_attention_mask": torch.tensor([[0]]),
                "atom_to_motif_map": torch.tensor([[-1]])
            }

        input_ids = pad_sequence([f['input_ids'] for f in valid_batch], batch_first=True,
                                 padding_value=self.text_pad_id)
        labels = pad_sequence([f['labels'] for f in valid_batch], batch_first=True, padding_value=-100)

        # 🚨🚨🚨 核心修正 3：直接透传 -1，绝不在 Collator 里画蛇添足！让 modeling.py 去处理它！
        e3fp_ids = pad_sequence([f['e3fp_ids'] for f in valid_batch], batch_first=True, padding_value=self.e3fp_pad_id)
        atom_to_motif_map = pad_sequence([f['atom_to_motif_map'] for f in valid_batch], batch_first=True,
                                         padding_value=-1)

        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.text_pad_id).long(),
            "labels": labels,
            "e3fp_ids": e3fp_ids,
            "atom_attention_mask": (e3fp_ids[:, :, 0] != self.e3fp_pad_id).long(),
            "atom_to_motif_map": atom_to_motif_map
        }


# ==========================================
# 3. Metric Computation
# ==========================================
def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    y_true, y_pred = [], []
    pattern = r'-?\d+\.?\d*'
    for p_str, l_str in zip(decoded_preds, decoded_labels):
        l_matches = re.findall(pattern, l_str)
        if not l_matches: continue
        y_true.append(float(l_matches[0]))
        p_matches = re.findall(pattern, p_str)
        y_pred.append(float(p_matches[0]) if p_matches else 0.0)

    return {"eval_mae": mean_absolute_error(y_true, y_pred)} if y_true else {"eval_mae": 999.0}


# ==========================================
# 4. 主函数
# ==========================================
def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    set_seed(42)

    text_tokenizer = TextTokenizer("google/t5-v1_1-base", max_len=768)
    motif_tokenizer = MotifTokenizer("asset/mol_vocabs/vocab_phase2_25k.txt", "google/t5-v1_1-base", max_len=768)
    e3fp_tokenizer = E3FPTokenizer(fp_level=3, fp_bits=4096)

    model = MoStT5ForConditionalGeneration.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(motif_tokenizer.tokenizer))

    train_dataset = GenerativeHybridDataset(os.path.join(args.data_dir, "train/3d_computed_properties_unit.json"),
                                            os.path.join(args.data_dir, "pubchemqc_final.lmdb"), text_tokenizer,
                                            motif_tokenizer, e3fp_tokenizer, max_source_length=768)
    eval_dataset = GenerativeHybridDataset(os.path.join(args.data_dir, "valid/3d_computed_properties_unit.json"),
                                           os.path.join(args.data_dir, "pubchemqc_final.lmdb"), text_tokenizer,
                                           motif_tokenizer, e3fp_tokenizer, max_source_length=768, is_eval=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints/gen_prop_SingleGPU_Stable",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=32,
        per_device_eval_batch_size=4,
        optim="adafactor",
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5000,
        max_steps=100000,
        eval_strategy="steps", eval_steps=2500,
        save_strategy="steps", save_steps=2500,
        logging_steps=10,
        bf16=False, fp16=False,
        gradient_checkpointing=False,
        predict_with_generate=True,
        generation_max_length=64,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mae",
        greater_is_better=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none"
    )

    collator = GenerativeCollator(text_pad_id=text_tokenizer.tokenizer.pad_token_id, e3fp_pad_id=-1)

    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=lambda e: compute_metrics(e, text_tokenizer.tokenizer),
        callbacks=[EarlyStoppingCallback(5)]
    )

    logger.info("🚀 Starting 100% PURE Aligned Fine-Tuning...")
    trainer.train()


if __name__ == "__main__":
    main()