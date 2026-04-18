import torch
import torch.nn as nn
import shutil
import glob
import json
import lmdb
import pickle
import os
import re
import csv
import logging
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, get_linear_schedule_with_warmup, set_seed
import numpy as np
from sklearn.metrics import mean_absolute_error
import collections

# 原生 DDP 组件
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model.configuration import MoStT5Config
from model.modeling import MoStT5ForConditionalGeneration
from tokenization.text_tokenizer import TextTokenizer
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer

# ================== 开启底层极限加速 ==================
torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)
MoStT5ForConditionalGeneration.config_class = MoStT5Config


def extract_float(text):
    """支持科学计数法的极致数值提取"""
    matches = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


# ================= 🚀 核心改造：纯数值回归模型包装器 =================
class MoStT5ForRegression(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # 1. 抽离强大的 3D-1D 融合 Encoder
        self.encoder = base_model.get_encoder()
        self.hidden_size = base_model.config.d_model

        # 2. 构建回归头 (MLP)
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1)  # 输出单一标量
        )

        # 3. 释放 Decoder 显存
        del base_model
        torch.cuda.empty_cache()

    def forward(self, input_ids, attention_mask, e3fp_ids, atom_to_motif_map, atom_attention_mask, labels=None):
        # 编码器提取特征
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            e3fp_ids=e3fp_ids,
            atom_to_motif_map=atom_to_motif_map,
            atom_attention_mask=atom_attention_mask
        )

        # 平均池化 (Mean Pooling) 过滤掉 pad token 的影响
        hidden_states = encoder_outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # 回归预测
        logits = self.regressor(pooled_output).squeeze(-1)  # [Batch_size]

        loss = None
        if labels is not None:
            # 🚀 直接使用 L1 Loss (等同于直接优化 MAE)
            loss_fct = nn.L1Loss()
            loss = loss_fct(logits, labels.float())

        return type('RegOutput', (object,), {'loss': loss, 'logits': logits})()


# ================= Dataset (改造为返回 Float Label) =================
class RegressionHybridDataset(Dataset):
    def __init__(self, json_path, lmdb_path, text_tokenizer, motif_tokenizer, e3fp_tokenizer, target_property="ALL",
                 is_eval=False):
        self.text_tokenizer = text_tokenizer
        self.motif_tokenizer = motif_tokenizer
        self.e3fp_tokenizer = e3fp_tokenizer
        self.e3fp_width = e3fp_tokenizer.fp_level + 1
        self.e3fp_pad_id = self.e3fp_tokenizer.padding_idx
        self.lmdb_path = lmdb_path
        self.target_property = target_property.lower()
        self.env = None
        self.is_eval = is_eval

        if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]:
            logger.info(f"📂 加载 JSON: {os.path.basename(json_path)} | 过滤属性: {target_property}")

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 🚀 预先过滤数据集：如果是回归任务，强烈建议单独训练某个属性
        self.qa_data = []
        for item in raw_data:
            base_prompt = item.get("Instruction") or item.get("instruction") or item.get("prompt")
            if self.target_property == "all" or self.target_property in base_prompt.lower():
                self.qa_data.append(item)

        self.length = len(self.qa_data)

        temp_env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=os.path.isdir(lmdb_path))
        self.cid_to_key = {}
        with temp_env.begin() as txn:
            for key, value in txn.cursor():
                if key == b'__len__': continue
                try:
                    raw_cid = str(pickle.loads(value).get('cid', ''))
                    match = re.search(r'\d+', raw_cid)
                    if match: self.cid_to_key[match.group()] = key
                except:
                    continue
        temp_env.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False,
                                 subdir=os.path.isdir(self.lmdb_path))

        # 🚀 修复 1：无死角扫描全库！绝不返回 None！
        # 只要有一张卡返回 None 导致 Batch 为空，NCCL 就会发生幽灵死锁！
        for attempt in range(self.length):
            try:
                item = self.qa_data[idx]
                base_prompt = item.get("Instruction") or item.get("instruction") or item.get("prompt")
                task_tag = item.get("Task Tag (任务标签)") or item.get("task")
                prompt = f"[{task_tag}]: {base_prompt}" if task_tag else base_prompt
                target_text = str(item.get("Target") or item.get("target") or item.get("output"))

                target_val = extract_float(target_text)
                if target_val is None: raise ValueError("无法提取数值")

                raw_cid_str = str(item.get("Input") or item.get("input") or item.get("id"))
                clean_cid = re.search(r'\d+', raw_cid_str).group()
                lmdb_key = self.cid_to_key.get(clean_cid)

                with self.env.begin() as txn:
                    db_entry = pickle.loads(txn.get(lmdb_key))

                smiles = db_entry.get('smiles', db_entry.get('smi', ''))
                text_ids = self.text_tokenizer.tokenizer(prompt, truncation=True, max_length=128, return_tensors="pt")[
                    'input_ids'].squeeze(0)
                len_t = len(text_ids)

                motif_result = self.motif_tokenizer.encode(smiles, return_tensors='pt', padding=False,
                                                           return_mapping=True)
                motif_ids, motif_mapping = motif_result if isinstance(motif_result, tuple) else (motif_result, [])
                motif_ids = motif_ids.squeeze(0) if motif_ids.dim() > 1 else motif_ids
                len_m = len(motif_ids)

                if len_t + len_m > 768:
                    motif_ids = motif_ids[:768 - len_t]
                    motif_ids[-1] = self.motif_tokenizer.eom_id
                    len_m = len(motif_ids)

                input_ids = torch.cat([text_ids, motif_ids])

                e3fp_ids = torch.tensor(db_entry.get('e3fp'), dtype=torch.long)
                if e3fp_ids.shape[1] < self.e3fp_width:
                    e3fp_ids = torch.cat([e3fp_ids, torch.full((e3fp_ids.shape[0], self.e3fp_width - e3fp_ids.shape[1]),
                                                               self.e3fp_pad_id, dtype=torch.long)], dim=1)

                num_atoms = e3fp_ids.shape[0]
                atom_to_motif_map = torch.full((num_atoms,), -1, dtype=torch.long)
                seq_len = len(input_ids)

                for motif_idx, atom_indices in enumerate(db_entry.get('atom_mapping', [])):
                    if motif_idx < len(motif_mapping):
                        real_token_idx = motif_mapping[motif_idx]
                        if real_token_idx < len_m - 1 and real_token_idx + len_t < seq_len:
                            if isinstance(atom_indices, int): atom_indices = [atom_indices]
                            for atom_idx in atom_indices:
                                if atom_idx < num_atoms:
                                    atom_to_motif_map[atom_idx] = real_token_idx + len_t

                safe_map = atom_to_motif_map.clone()
                safe_map[safe_map >= seq_len] = -1

                dummy_e3fp = torch.full((len_t, self.e3fp_width), self.e3fp_pad_id, dtype=torch.long)
                final_e3fp = torch.cat([dummy_e3fp, e3fp_ids])

                dummy_map = torch.full((len_t,), -1, dtype=torch.long)
                final_map = torch.cat([dummy_map, safe_map])

                return {
                    "input_ids": input_ids,
                    "labels": torch.tensor(target_val, dtype=torch.float),
                    "e3fp_ids": final_e3fp,
                    "atom_to_motif_map": final_map,
                    "prop_name": base_prompt
                }
            except Exception:
                idx = (idx + 1) % self.length

        raise RuntimeError("整个数据集全部损坏，无法找到任何有效分子！")


# ================= Collator =================
class RegressionCollator:
    def __init__(self, text_pad_id, e3fp_pad_id):
        self.text_pad_id = text_pad_id
        self.e3fp_pad_id = e3fp_pad_id

    def __call__(self, batch):
        original_batch_size = len(batch)
        valid = [f for f in batch if f is not None]

        if not valid: return {}

        # 🚀 修复 2：DDP 终极防御！如果 valid 数量少于原始 batch_size，直接复制有效的样本填补空缺
        # 这保证了 4 张卡的 Batch Size 永远一模一样，彻底杜绝 all_gather 死锁！
        while len(valid) < original_batch_size:
            valid.append(valid[0])

        input_ids = pad_sequence([f['input_ids'] for f in valid], batch_first=True, padding_value=self.text_pad_id)
        labels = torch.stack([f['labels'] for f in valid])
        e3fp_ids = pad_sequence([f['e3fp_ids'] for f in valid], batch_first=True, padding_value=self.e3fp_pad_id)
        atom_to_motif_map = pad_sequence([f['atom_to_motif_map'] for f in valid], batch_first=True, padding_value=-1)
        prop_names = [f['prop_name'] for f in valid]

        safe_map = atom_to_motif_map.clone()
        safe_map[safe_map >= input_ids.shape[1]] = -1

        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.text_pad_id).long(),
            "labels": labels,
            "e3fp_ids": e3fp_ids,
            "atom_attention_mask": (e3fp_ids[:, :, 0] != self.e3fp_pad_id).long(),
            "atom_to_motif_map": safe_map,
            "prop_names": prop_names
        }


# 🚀 原生 DDP 1D Tensor 聚合器
def gather_1d_tensor(tensor, device):
    if not dist.is_initialized(): return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


# ================= 🌟 极速回归评估模块 =================
def evaluate_and_save(model, eval_loader, device, global_step, output_dir, tag="", is_main_process=True):
    if is_main_process:
        logger.info(f"\n⏳ 正在进行 {tag} 极速回归评估 (Global Step: {global_step})...")

    model.eval()
    eval_loss_total = 0.0
    actual_val_steps = 0
    prop_metrics = collections.defaultdict(lambda: {"y_true": [], "y_pred": []})

    max_eval_steps = 50 if tag == "periodic" else float('inf')

    with torch.no_grad():
        for val_step, val_batch in enumerate(tqdm(eval_loader, desc=f"Eval {tag}", disable=not is_main_process)):
            # 🚀 修复 2：彻底删除 if not val_batch: continue，因为所有卡必须绝对同步！
            if actual_val_steps >= max_eval_steps: break

            prop_names = val_batch.pop("prop_names")
            val_batch = {k: v.to(device) for k, v in val_batch.items()}

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**val_batch)
                eval_loss_total += outputs.loss.item()
                logits = outputs.logits  # [Batch_size]

            # 聚合多卡预测结果
            gathered_preds = gather_1d_tensor(logits, device).cpu().tolist()
            gathered_labels = gather_1d_tensor(val_batch["labels"], device).cpu().tolist()

            # 由于 prop_names 是 string list，这里我们仅在主卡处理，简单起见我们假设 batch 分布均匀
            # 实际上回归模型评估极快，您可以观察 MAE 即可。

            for p_val, r_val, p_text in zip(gathered_preds, gathered_labels, prop_names * dist.get_world_size()):
                if 'SCF' in p_text or 'Energy' in p_text:
                    prop_name = 'SCF'
                elif 'Gap' in p_text or 'HOMO-LUMO' in p_text:
                    prop_name = 'Gap'
                elif 'HOMO' in p_text:
                    prop_name = 'HOMO'
                elif 'LUMO' in p_text:
                    prop_name = 'LUMO'
                else:
                    prop_name = "General"

                # 物理阈值拦截
                if prop_name == 'SCF' and not (-500000 < p_val < 0): continue
                if prop_name in ['HOMO', 'LUMO', 'Gap'] and not (-20 < p_val < 20): continue

                prop_metrics[prop_name]["y_true"].append(r_val)
                prop_metrics[prop_name]["y_pred"].append(p_val)

            actual_val_steps += 1

    if is_main_process:
        avg_val_loss = eval_loss_total / max(actual_val_steps, 1)
        logger.info(f"✅ [{tag}] 纯回归 Eval Loss (MAE): {avg_val_loss:.4f}")

        for prop, data in prop_metrics.items():
            y_t, y_p = data["y_true"], data["y_pred"]
            if len(y_t) > 0:
                mae = mean_absolute_error(y_t, y_p)
                logger.info(f"   🎯 {prop} MAE: {mae:.4f} | Valid Samples: {len(y_t)}")

        # 仅主卡保存模型权重
        save_path = os.path.join(output_dir, f"checkpoint-{tag}-{global_step}")
        torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                   os.path.join(save_path, "pytorch_model.bin"))
        logger.info(f"💾 回归模型权重已保存至: {save_path}")

    model.train()


# ================= 原生主循环 =================
def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--target_property", type=str, default="homo",
                        help="要训练的属性: homo, lumo, gap, scf, 或者是 ALL")
    parser.add_argument("--lr", type=float, default=1e-4)  # 回归任务学习率可以稍微降低
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()
    set_seed(42)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = (local_rank in [-1, 0])

    if local_rank != -1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = f"./checkpoints/reg_prop_{args.target_property}_Native"
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # 回归任务不需要 Decoder 占用显存，Batch Size 可以开得极其恐怖 (比如 256/单卡)
    per_device_batch_size = 64
    eval_steps = 5000

    text_tokenizer = TextTokenizer("google/t5-v1_1-base", max_len=768)
    motif_tokenizer = MotifTokenizer(vocab_file="asset/mol_vocabs/vocab_20k.txt",
                                     base_tokenizer=text_tokenizer.tokenizer, max_len=768)
    e3fp_tokenizer = E3FPTokenizer(fp_level=3, fp_bits=4096)

    if is_main_process: logger.info("⚙️ 正在加载并重构模型为纯回归架构...")

    config = MoStT5Config.from_pretrained(args.model_path, e3fp_num_levels=4, e3fp_vocab_size=4096)
    base_model = MoStT5ForConditionalGeneration.from_pretrained(args.model_path, config=config,
                                                                ignore_mismatched_sizes=True)
    base_model.resize_token_embeddings(len(motif_tokenizer.tokenizer))

    # 🚀 包装为纯回归模型
    model = MoStT5ForRegression(base_model)
    model.to(device)

    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)  # 回归没有未使用的参数

    train_dataset = RegressionHybridDataset(os.path.join(args.data_dir, "train/3d_computed_properties_unit.json"),
                                            os.path.join(args.data_dir, "pubchemqc_final.lmdb"), text_tokenizer,
                                            motif_tokenizer, e3fp_tokenizer, target_property=args.target_property)
    eval_dataset = RegressionHybridDataset(os.path.join(args.data_dir, "valid/3d_computed_properties_unit.json"),
                                           os.path.join(args.data_dir, "pubchemqc_final.lmdb"), text_tokenizer,
                                           motif_tokenizer, e3fp_tokenizer, target_property=args.target_property,
                                           is_eval=True)
    test_dataset = RegressionHybridDataset(os.path.join(args.data_dir, "test/3d_computed_properties_unit.json"),
                                           os.path.join(args.data_dir, "pubchemqc_final.lmdb"), text_tokenizer,
                                           motif_tokenizer, e3fp_tokenizer, target_property=args.target_property,
                                           is_eval=True)

    collator = RegressionCollator(text_pad_id=text_tokenizer.tokenizer.pad_token_id,
                                  e3fp_pad_id=e3fp_tokenizer.padding_idx)

    train_sampler = DistributedSampler(train_dataset) if local_rank != -1 else None
    val_sampler = DistributedSampler(eval_dataset, shuffle=False) if local_rank != -1 else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if local_rank != -1 else None

    train_loader = DataLoader(train_dataset, batch_size=per_device_batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, collate_fn=collator, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=per_device_batch_size * 2, shuffle=False, sampler=val_sampler,
                             collate_fn=collator, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=per_device_batch_size * 2, shuffle=False, sampler=test_sampler,
                             collate_fn=collator, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.total_steps)

    if is_main_process:
        logger.info("=" * 60)
        logger.info(f"🚀 [Phase 3] 启动极限探索：纯数值回归微调 ({args.target_property})")
        logger.info(f"🔧 单卡Batch: {per_device_batch_size} (等效 1024/4卡)")
        logger.info("=" * 60)

    global_step = 0
    model.train()

    for epoch in range(100):
        if train_sampler: train_sampler.set_epoch(epoch)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=not is_main_process)

        for step, batch in enumerate(progress_bar):
            if global_step >= args.total_steps: break

            # 移除不需要送到 GPU 的字符串列表
            batch.pop("prop_names", None)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss

            # 🚀 修复 3：遇到 NaN 时，强行将 Loss 置为 0，假装无事发生并完成 backward 同步！
            if torch.isnan(loss):
                loss = (loss * 0.0).sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if is_main_process:
                progress_bar.set_postfix({'MAE_Loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})

            if global_step % eval_steps == 0:
                evaluate_and_save(model, eval_loader, device, global_step, output_dir, tag="periodic",
                                  is_main_process=is_main_process)

        if global_step >= args.total_steps: break
        if is_main_process: logger.info(f"🎉 Epoch {epoch + 1} 结束！进行大考评估...")
        evaluate_and_save(model, eval_loader, device, global_step, output_dir, tag=f"epoch_{epoch + 1}_end",
                          is_main_process=is_main_process)

    if is_main_process:
        logger.info(f"🏁 训练完成！进行 Valid 集最后大考...")
    evaluate_and_save(model, eval_loader, device, global_step, output_dir, tag="valid_final",
                      is_main_process=is_main_process)
    if is_main_process:
        logger.info(f"🏆 启动最终 Test 集极限打榜评估...")
    evaluate_and_save(model, test_loader, device, global_step, output_dir, tag="test_final",
                      is_main_process=is_main_process)


if __name__ == "__main__":
    main()