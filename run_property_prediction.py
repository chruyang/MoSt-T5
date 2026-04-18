import torch
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

# 引入原生 DDP 组件
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from model.configuration import MoStT5Config
from model.modeling import MoStT5ForConditionalGeneration
from tokenization.text_tokenizer import TextTokenizer
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer

torch.backends.cudnn.benchmark = True
logger = logging.getLogger(__name__)
MoStT5ForConditionalGeneration.config_class = MoStT5Config


# ==========================================
# 1. 专为 Mol2Text (属性预测) 打造的 Dataset
# ==========================================
class GenerativeHybridDataset(Dataset):
    def __init__(self, json_path, lmdb_path, text_tokenizer, motif_tokenizer, e3fp_tokenizer, target_property="all",
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
            logger.info(f"📂 加载 JSON: {os.path.basename(json_path)} | 训练属性过滤: {target_property}")

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        self.qa_data = []
        for item in raw_data:
            # 🚀 动态适配 PubChem 的任务字段
            task_name = str(item.get("Task", "Unknown")).lower()
            if self.target_property == "all" or self.target_property in task_name:
                self.qa_data.append(item)

        self.length = len(self.qa_data)

        if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]:
            logger.info(f"✅ 过滤完毕，[{target_property}] 任务共提取出 {self.length} 条纯净数据！")

        temp_env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=os.path.isdir(lmdb_path))
        self.cid_to_key = {}
        with temp_env.begin() as txn:
            for key, value in txn.cursor():
                if key == b'__len__': continue
                try:
                    raw_cid = str(pickle.loads(value).get('cid', ''))
                    numbers = re.findall(r'\d+', raw_cid)
                    if numbers: self.cid_to_key[numbers[-1]] = key
                except:
                    continue
        temp_env.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False,
                                 subdir=os.path.isdir(self.lmdb_path))

        for attempt in range(self.length):
            try:
                item = self.qa_data[idx]
                base_prompt = item.get("Instruction") or item.get("instruction") or item.get("prompt")
                task_tag = item.get("Task") or "Property Prediction"
                prompt = f"[{task_tag}]: {base_prompt}"

                # 🚀 属性预测任务的目标是文本（数值）
                target_text = str(item.get("Output") or item.get("output"))
                real_id = item.get("id")

                numbers = re.findall(r'\d+', str(real_id))
                if not numbers: raise ValueError("无法找到分子的数字ID")
                clean_cid = numbers[-1]

                lmdb_key = self.cid_to_key.get(clean_cid)
                if not lmdb_key: raise ValueError(f"找不到CID {clean_cid} 的LMDB映射")

                with self.env.begin() as txn:
                    db_entry = pickle.loads(txn.get(lmdb_key))

                # 提取去过立体化的 2D 骨架用于生成序列
                smiles = db_entry.get('smiles', db_entry.get('smi', ''))

                # ==========================================
                # Encoder 输入: [Text Prompt] + [Motif Molecule]
                # ==========================================
                text_ids = self.text_tokenizer.tokenizer(
                    prompt, truncation=True, max_length=128, return_tensors="pt"
                )['input_ids'].squeeze(0)

                # 剥离 </s> 以打通跨模态注意力
                if text_ids[-1] == self.text_tokenizer.tokenizer.eos_token_id:
                    text_ids = text_ids[:-1]
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

                input_ids = torch.cat([
                    text_ids,
                    motif_ids,
                    torch.tensor([self.text_tokenizer.tokenizer.eos_token_id], dtype=torch.long)
                ])

                # Decoder 目标: [属性数值文本]
                target_ids = self.text_tokenizer.tokenizer(
                    target_text, truncation=True, max_length=36, return_tensors="pt"
                )['input_ids'].squeeze(0)

                # ==========================================
                # 对齐 3D 坐标
                # ==========================================
                e3fp_ids = torch.tensor(db_entry.get('e3fp'), dtype=torch.long)
                if e3fp_ids.shape[1] < self.e3fp_width:
                    e3fp_ids = torch.cat([e3fp_ids,
                                          torch.full((e3fp_ids.shape[0], self.e3fp_width - e3fp_ids.shape[1]),
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

                # 对齐补丁：前面 Text Token 的部分没有 3D 特征，全部填充 pad
                dummy_e3fp = torch.full((len_t, self.e3fp_width), self.e3fp_pad_id, dtype=torch.long)
                final_e3fp = torch.cat([dummy_e3fp, e3fp_ids])

                dummy_map = torch.full((len_t,), -1, dtype=torch.long)
                final_map = torch.cat([dummy_map, safe_map])

                return {
                    "input_ids": input_ids,
                    "labels": target_ids,
                    "e3fp_ids": final_e3fp,
                    "atom_to_motif_map": final_map,
                    "task_name": task_tag  # 🚀 动态传入 Task Name，供评估使用
                }
            except Exception as e:
                idx = (idx + 1) % self.length

        raise RuntimeError("数据集彻底损坏或过滤后无可用数据！")


class GenerativeCollator:
    def __init__(self, text_pad_id, e3fp_pad_id):
        self.text_pad_id = text_pad_id
        self.e3fp_pad_id = e3fp_pad_id

    def __call__(self, batch):
        original_size = len(batch)
        valid = [f for f in batch if f is not None]
        if not valid: return {}

        while len(valid) < original_size:
            valid.append(valid[0])

        input_ids = pad_sequence([f['input_ids'] for f in valid], batch_first=True, padding_value=self.text_pad_id)
        labels = pad_sequence([f['labels'] for f in valid], batch_first=True, padding_value=-100)
        e3fp_ids = pad_sequence([f['e3fp_ids'] for f in valid], batch_first=True, padding_value=self.e3fp_pad_id)
        atom_to_motif_map = pad_sequence([f['atom_to_motif_map'] for f in valid], batch_first=True, padding_value=-1)

        safe_map = atom_to_motif_map.clone()
        safe_map[safe_map >= input_ids.shape[1]] = -1

        # 🚀 提取出用于指标分类的 Task Name
        task_names = [f['task_name'] for f in valid]

        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.text_pad_id).long(),
            "labels": labels,
            "e3fp_ids": e3fp_ids,
            "atom_attention_mask": (e3fp_ids[:, :, 0] != self.e3fp_pad_id).long(),
            "atom_to_motif_map": safe_map,
            "task_names": task_names
        }


def extract_float(text):
    text = text.replace(" ", "")
    # 🚀 强大的正则：兼容整数、小数、负数、科学计数法
    matches = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', text)
    if matches:
        try:
            # 🚀 对齐 3DMOLT5：提取第一个匹配到的数值（防止后面的单位数字干扰）
            return float(matches[0])
        except ValueError:
            return None
    return None


def pad_and_gather(tensor, pad_value, device):
    if not dist.is_initialized(): return tensor
    local_max = torch.tensor([tensor.shape[1]], dtype=torch.long, device=device)
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
    max_len = global_max.item()

    padded = torch.full((tensor.shape[0], max_len), pad_value, dtype=tensor.dtype, device=device)
    padded[:, :tensor.shape[1]] = tensor

    gathered = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)
    return torch.cat(gathered, dim=0)


# ==========================================
# 🚀 核心评估：动态支持 PubChem 各项属性指标
# ==========================================
def evaluate_and_save(model, eval_loader, text_tokenizer, device, global_step, output_dir, tag="",
                      is_main_process=True):
    if is_main_process:
        logger.info(f"\n⏳ 正在进行 {tag} 评估 (Global Step: {global_step})...")

    model.eval()
    eval_loss_total, actual_val_steps = 0.0, 0
    prop_metrics = collections.defaultdict(lambda: {"y_true": [], "y_pred": [], "total_count": 0})

    current_beams = 5 if tag == "test_final" else 1
    max_eval_steps = float('inf')
    max_gen_length = 36
    example_printed = False
    unwrapped_model = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        for val_step, val_batch in enumerate(tqdm(eval_loader, desc=f"Eval {tag}", disable=not is_main_process)):
            if actual_val_steps >= max_eval_steps: break

            # 🚀 取出非 Tensor 标签
            task_names_list = val_batch.pop("task_names")
            val_batch = {k: v.to(device) for k, v in val_batch.items()}

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                val_outputs = model(**val_batch)
                eval_loss_total += val_outputs.loss.item()

                encoder_outputs = unwrapped_model.get_encoder()(
                    input_ids=val_batch["input_ids"], attention_mask=val_batch["attention_mask"],
                    e3fp_ids=val_batch["e3fp_ids"], atom_to_motif_map=val_batch["atom_to_motif_map"],
                    atom_attention_mask=val_batch["atom_attention_mask"]
                )

                generated_ids = unwrapped_model.generate(
                    attention_mask=val_batch["attention_mask"], encoder_outputs=encoder_outputs,
                    max_length=max_gen_length, num_beams=current_beams, early_stopping=True, do_sample=False
                )

            # Gather 回主卡
            generated_ids = pad_and_gather(generated_ids, text_tokenizer.tokenizer.pad_token_id, device)
            labels_ids = pad_and_gather(val_batch["labels"], -100, device)
            gathered_input_ids = pad_and_gather(val_batch["input_ids"], text_tokenizer.tokenizer.pad_token_id, device)

            # 🚀 跨卡 Gather 字符串列表
            if dist.is_initialized():
                gathered_tasks = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_tasks, task_names_list)
                gathered_tasks = [t for sublist in gathered_tasks for t in sublist]
            else:
                gathered_tasks = task_names_list

            if is_main_process:
                # 🚀 解码均为 Text Tokenizer，因为输出和标签全是数字/自然语言文本
                preds = text_tokenizer.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                labels = torch.where(labels_ids != -100, labels_ids, text_tokenizer.tokenizer.pad_token_id)
                refs = text_tokenizer.tokenizer.batch_decode(labels, skip_special_tokens=True)

                if not example_printed:
                    demo_prompt_str = text_tokenizer.tokenizer.decode(gathered_input_ids[0], skip_special_tokens=False)
                    logger.info("\n" + "✨" * 25)
                    logger.info(f"👀 [实时生成样例观测 - {tag}]")
                    logger.info(f"🔹 【输入 序列】: {demo_prompt_str[:150]}... (截断)")
                    logger.info(f"🔸 【真实 数值】: {refs[0]}")
                    logger.info(f"🤖 【模型 预测】: {preds[0]}")
                    logger.info("✨" * 25 + "\n")
                    example_printed = True

                # 动态分配 MAE 统计
                for p_str, r_str, t_name in zip(preds, refs, gathered_tasks):
                    p_val, r_val = extract_float(p_str), extract_float(r_str)

                    prop_name = t_name if t_name else "Unknown_Property"
                    prop_metrics[prop_name]["total_count"] += 1

                    if r_val is not None and p_val is not None:
                        # 🚀 解除了 QM9 那里 (-20 < p_val < 20) 的硬编码限制！
                        # 现在支持数百(如分子量)甚至更大的合法数值区间
                        if abs(p_val) < 1e6:
                            prop_metrics[prop_name]["y_true"].append(r_val)
                            prop_metrics[prop_name]["y_pred"].append(p_val)

            actual_val_steps += 1

    if is_main_process:
        avg_val_loss = eval_loss_total / max(actual_val_steps, 1)
        logger.info(f"✅ [{tag}] 属性预测 Eval Loss: {avg_val_loss:.4f}")
        eval_result_str = f"Tag: {tag} | Step: {global_step} | Eval Loss: {avg_val_loss:.4f}\n"

        for prop, data in prop_metrics.items():
            y_t, y_p = data["y_true"], data["y_pred"]
            total_samples = data["total_count"]
            valid_samples = len(y_t)
            valid_ratio = (valid_samples / total_samples * 100) if total_samples > 0 else 0.0

            if valid_samples > 0:
                mae = mean_absolute_error(y_t, y_p)
                log_msg = f"   🎯 [{prop}] MAE: {mae:.4f} | Valid Format: {valid_ratio:.1f}% ({valid_samples}/{total_samples})"
            else:
                log_msg = f"   ❌ [{prop}] MAE: N/A | Valid Format: 0.0% (0/{total_samples})"

            logger.info(log_msg)
            eval_result_str += log_msg + "\n"

        with open(os.path.join(output_dir, "eval_results.txt"), "a", encoding="utf-8") as f:
            f.write(eval_result_str + "-" * 40 + "\n")

        # ... (保存模型与 TSV 日志的代码保持不变) ...
        save_path = os.path.join(output_dir, f"checkpoint-{tag}-{global_step}")
        os.makedirs(save_path, exist_ok=True)
        unwrapped_model.save_pretrained(save_path, safe_serialization=False)
        logger.info(f"💾 模型已保存至: {save_path}")

    model.train()


def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--target_property", type=str, default="all")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--total_steps", type=int, default=250000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
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

    output_dir = f"./checkpoints/gen_prop_{args.target_property}_PubChem_BF16"
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                            handlers=[logging.FileHandler(os.path.join(output_dir, "training_property.log"),
                                                          encoding='utf-8'), logging.StreamHandler()])

    per_device_batch_size = 16  # 根据 3090 显存微调
    num_train_epochs = 70
    eval_steps = 10000

    AutoConfig.register("most-t5", MoStT5Config)
    text_tokenizer = TextTokenizer("google/t5-v1_1-base", max_len=768)

    motif_tokenizer = MotifTokenizer(vocab_file="asset/mol_vocabs/vocab_phase2_25k.txt",
                                     base_tokenizer=text_tokenizer.tokenizer, max_len=768)
    e3fp_tokenizer = E3FPTokenizer(fp_level=3, fp_bits=4096)

    if is_main_process: logger.info("⚙️ 正在加载模型 (Mol2Text)...")

    config = MoStT5Config.from_pretrained(args.model_path, e3fp_num_levels=4, e3fp_vocab_size=4096)
    model = MoStT5ForConditionalGeneration.from_pretrained(args.model_path, config=config, ignore_mismatched_sizes=True)
    model.resize_token_embeddings(len(motif_tokenizer.tokenizer))
    model.to(device)

    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 实例化数据集
    train_dataset = GenerativeHybridDataset(os.path.join(args.data_dir, "train/pubchem_properties.json"),
                                            os.path.join(args.data_dir, "pubchem_final.lmdb"), text_tokenizer,
                                            motif_tokenizer, e3fp_tokenizer, target_property=args.target_property)
    eval_dataset = GenerativeHybridDataset(os.path.join(args.data_dir, "valid/pubchem_properties.json"),
                                           os.path.join(args.data_dir, "pubchem_final.lmdb"), text_tokenizer,
                                           motif_tokenizer, e3fp_tokenizer, target_property=args.target_property,
                                           is_eval=True)
    test_dataset = GenerativeHybridDataset(os.path.join(args.data_dir, "test/pubchem_properties.json"),
                                           os.path.join(args.data_dir, "pubchem_final.lmdb"), text_tokenizer,
                                           motif_tokenizer, e3fp_tokenizer, target_property=args.target_property,
                                           is_eval=True)

    collator = GenerativeCollator(text_pad_id=text_tokenizer.tokenizer.pad_token_id,
                                  e3fp_pad_id=e3fp_tokenizer.padding_idx)

    train_sampler = DistributedSampler(train_dataset) if local_rank != -1 else None
    val_sampler = DistributedSampler(eval_dataset, shuffle=False) if local_rank != -1 else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if local_rank != -1 else None

    train_loader = DataLoader(train_dataset, batch_size=per_device_batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, collate_fn=collator, num_workers=4, pin_memory=True,
                              persistent_workers=True)
    eval_loader = DataLoader(eval_dataset, batch_size=per_device_batch_size * 2, shuffle=False, sampler=val_sampler,
                             collate_fn=collator, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=per_device_batch_size * 2, shuffle=False, sampler=test_sampler,
                             collate_fn=collator, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.total_steps)

    if is_main_process:
        logger.info("=" * 60)
        logger.info(f"🚀 启动 PubChem 生成式属性预测微调")
        logger.info("=" * 60)

    global_step = 0
    model.train()

    for epoch in range(num_train_epochs):
        if train_sampler: train_sampler.set_epoch(epoch)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_train_epochs}", disable=not is_main_process,
                            mininterval=2.0)

        for step, batch in enumerate(progress_bar):
            if global_step >= args.total_steps: break

            # 🚀 训练前剥离自定义非 Tensor 参数
            batch.pop("task_names", None)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps

            if torch.isnan(loss) or loss.item() * args.gradient_accumulation_steps > 50.0:
                loss = sum(p.sum() for p in model.parameters() if p.requires_grad) * 0.0

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main_process and global_step % 10 == 0:
                    progress_bar.set_postfix({'loss': f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                                              'lr': f"{scheduler.get_last_lr()[0]:.2e}"})

                if global_step % eval_steps == 0:
                    evaluate_and_save(model, eval_loader, text_tokenizer, device, global_step, output_dir,
                                      tag=f"step_{global_step}", is_main_process=is_main_process)

        if global_step >= args.total_steps: break
        if is_main_process: logger.info(f"🎉 Epoch {epoch + 1} 结束！")

    if is_main_process: logger.info(f"🏁 训练完成！进行 Test 集全量预测...")
    # 🚀 注意：最后评估传递的是 text_tokenizer，负责解出最终自然语言预测值
    evaluate_and_save(model, test_loader, text_tokenizer, device, global_step, output_dir, tag="test_final",
                      is_main_process=is_main_process)


if __name__ == "__main__":
    main()