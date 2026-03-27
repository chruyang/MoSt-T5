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
# 🚀 引入线性调度器代替余弦调度器
from transformers import AutoConfig, get_linear_schedule_with_warmup, set_seed
import numpy as np
from sklearn.metrics import mean_absolute_error
import collections

from model.configuration import MoStT5Config
from model.modeling import MoStT5ForConditionalGeneration
from tokenization.text_tokenizer import TextTokenizer
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer

# ================== 开启底层极限加速 ==================
torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("training_most_t5_sota.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MoStT5ForConditionalGeneration.config_class = MoStT5Config


# ================= Dataset (坚守您完美的 MoSt-T5 专属锚点对齐) =================
class GenerativeHybridDataset(Dataset):
    def __init__(self, json_path, lmdb_path, text_tokenizer, motif_tokenizer, e3fp_tokenizer, is_eval=False):
        self.text_tokenizer = text_tokenizer
        self.motif_tokenizer = motif_tokenizer
        self.e3fp_tokenizer = e3fp_tokenizer
        self.e3fp_width = e3fp_tokenizer.fp_level + 1
        self.lmdb_path = lmdb_path
        self.env = None
        self.is_eval = is_eval

        logger.info(f"📂 加载 JSON: {os.path.basename(json_path)}")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.qa_data = json.load(f)
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

        for attempt in range(50):
            try:
                item = self.qa_data[idx]
                base_prompt = item.get("Instruction") or item.get("instruction") or item.get("prompt")
                task_tag = item.get("Task Tag (任务标签)") or item.get("task")
                prompt = f"[{task_tag}]: {base_prompt}" if task_tag else base_prompt
                target_text = str(item.get("Target") or item.get("target") or item.get("output"))

                raw_cid_str = str(item.get("Input") or item.get("input") or item.get("id"))
                clean_cid = re.search(r'\d+', raw_cid_str).group()
                lmdb_key = self.cid_to_key.get(clean_cid)

                with self.env.begin() as txn:
                    db_entry = pickle.loads(txn.get(lmdb_key))

                smiles = db_entry.get('smiles', db_entry.get('smi', ''))

                text_ids = self.text_tokenizer.tokenizer(
                    prompt, truncation=True, max_length=128, return_tensors="pt"
                )['input_ids'].squeeze(0)
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
                target_ids = \
                self.text_tokenizer.tokenizer(target_text, truncation=True, max_length=36, return_tensors="pt")[
                    'input_ids'].squeeze(0)

                e3fp_ids = torch.tensor(db_entry.get('e3fp'), dtype=torch.long)
                if e3fp_ids.shape[1] < self.e3fp_width:
                    e3fp_ids = torch.cat([e3fp_ids,
                                          torch.full((e3fp_ids.shape[0], self.e3fp_width - e3fp_ids.shape[1]), -1,
                                                     dtype=torch.long)], dim=1)

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

                dummy_e3fp = torch.full((len_t, self.e3fp_width), -1, dtype=torch.long)
                final_e3fp = torch.cat([dummy_e3fp, e3fp_ids])

                dummy_map = torch.full((len_t,), -1, dtype=torch.long)
                final_map = torch.cat([dummy_map, atom_to_motif_map])

                return {
                    "input_ids": input_ids,
                    "labels": target_ids,
                    "e3fp_ids": final_e3fp,
                    "atom_to_motif_map": final_map
                }
            except Exception:
                if self.is_eval: return None
                idx = (idx + 1) % self.length
        return None


# ================= Collator =================
class GenerativeCollator:
    def __init__(self, text_pad_id):
        self.text_pad_id = text_pad_id

    def __call__(self, batch):
        valid = [f for f in batch if f is not None]
        if not valid: return {}

        input_ids = pad_sequence([f['input_ids'] for f in valid], batch_first=True, padding_value=self.text_pad_id)
        labels = pad_sequence([f['labels'] for f in valid], batch_first=True, padding_value=-100)
        e3fp_ids = pad_sequence([f['e3fp_ids'] for f in valid], batch_first=True, padding_value=-1)
        atom_to_motif_map = pad_sequence([f['atom_to_motif_map'] for f in valid], batch_first=True, padding_value=-1)

        safe_map = atom_to_motif_map.clone()
        safe_map[safe_map >= input_ids.shape[1]] = -1

        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.text_pad_id).long(),
            "labels": labels,
            "e3fp_ids": e3fp_ids,
            "atom_attention_mask": (e3fp_ids[:, :, 0] != -1).long(),
            "atom_to_motif_map": safe_map
        }


def extract_float(text):
    # 严格匹配浮点数并取最后一个
    matches = re.findall(r'-?\d+\.\d+', text)
    return float(matches[-1]) if matches else None


# ================= 🌟 学术级评估模块 =================
def evaluate_and_save(model, eval_loader, text_tokenizer, device, global_step, output_dir, tag=""):
    logger.info(f"\n⏳ 正在进行 {tag} 评估 (Global Step: {global_step})...")
    model.eval()
    eval_loss_total = 0.0
    actual_val_steps = 0

    prop_metrics = collections.defaultdict(lambda: {"y_true": [], "y_pred": [], "total_count": 0})

    # 动态 Beam Search：日常评估用 Greedy 加速，Epoch大考用 Beam=5 追求极致性能
    current_beams = 1 if tag == "periodic" else 5
    max_eval_steps = 100 if tag == "periodic" else float('inf')
    example_printed = False
    with torch.no_grad():
        for val_step, val_batch in enumerate(tqdm(eval_loader, desc=f"Eval {tag}")):
            if not val_batch: continue
            if actual_val_steps >= max_eval_steps: break

            val_batch = {k: v.to(device) for k, v in val_batch.items()}

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                val_outputs = model(**val_batch)
                eval_loss_total += val_outputs.loss.item()

                encoder_outputs = model.get_encoder()(
                    input_ids=val_batch["input_ids"],
                    attention_mask=val_batch["attention_mask"],
                    e3fp_ids=val_batch["e3fp_ids"],
                    atom_to_motif_map=val_batch["atom_to_motif_map"],
                    atom_attention_mask=val_batch["atom_attention_mask"]
                )

                generated_ids = model.generate(
                    attention_mask=val_batch["attention_mask"],
                    encoder_outputs=encoder_outputs,
                    max_length=36,
                    num_beams=current_beams,
                    early_stopping=True,
                    do_sample=False
                )

            preds = text_tokenizer.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels = torch.where(val_batch["labels"] != -100, val_batch["labels"],
                                 text_tokenizer.tokenizer.pad_token_id)
            refs = text_tokenizer.tokenizer.batch_decode(labels, skip_special_tokens=True)
            prompts = text_tokenizer.tokenizer.batch_decode(val_batch["input_ids"], skip_special_tokens=True)

            for p_str, r_str, prompt_str in zip(preds, refs, prompts):
                # 🚀 新增：极其直观的 Case Study 打印模块
                if not example_printed:
                    logger.info("\n" + "✨" * 25)
                    logger.info(f"👀 [实时生成样例观测 - {tag}]")
                    logger.info(f"🔹 【输入 Prompt】: {prompt_str}")
                    logger.info(f"🔸 【真实 标签】: {r_str}")
                    logger.info(f"🤖 【模型 预测】: {p_str}")
                    logger.info("✨" * 25 + "\n")

                    example_printed = True
                p_val, r_val = extract_float(p_str), extract_float(r_str)

                if 'SCF' in r_str or 'Energy' in r_str:
                    prop_name = 'SCF'
                elif 'Gap' in r_str or 'HOMO-LUMO' in r_str:
                    prop_name = 'Gap'
                elif 'HOMO' in r_str:
                    prop_name = 'HOMO'
                elif 'LUMO' in r_str:
                    prop_name = 'LUMO'
                else:
                    prop_name = "General"

                prop_metrics[prop_name]["total_count"] += 1

                if r_val is not None and p_val is not None:
                    # 宽松物理截断过滤异常值
                    if prop_name == 'SCF' and not (-100 < p_val < 100): continue
                    if prop_name in ['HOMO', 'LUMO', 'Gap'] and not (-50 < p_val < 50): continue
                    prop_metrics[prop_name]["y_true"].append(r_val)
                    prop_metrics[prop_name]["y_pred"].append(p_val)

            actual_val_steps += 1

    avg_val_loss = eval_loss_total / max(actual_val_steps, 1)

    # 提醒用户忽略 Eval Loss，专注 MAE
    logger.info(
        f"✅ [{tag}] 文本生成 Eval Loss: {avg_val_loss:.4f} (注意: 数值回归任务中该指标不绝对代表精度，请以 MAE 为准)")
    eval_result_str = f"Tag: {tag} | Step: {global_step} | Eval Loss: {avg_val_loss:.4f}\n"

    for prop, data in prop_metrics.items():
        y_t, y_p = data["y_true"], data["y_pred"]
        total_samples = data["total_count"]
        valid_samples = len(y_t)
        valid_ratio = (valid_samples / total_samples * 100) if total_samples > 0 else 0.0

        if valid_samples > 0:
            mae = mean_absolute_error(y_t, y_p)
            log_msg = f"   🎯 {prop} MAE: {mae:.4f} | Valid: {valid_ratio:.1f}% ({valid_samples}/{total_samples})"
            logger.info(log_msg)
            eval_result_str += log_msg + "\n"
        else:
            log_msg = f"   ❌ {prop} MAE: N/A | Valid: 0.0% (0/{total_samples})"
            logger.info(log_msg)
            eval_result_str += log_msg + "\n"

    with open(os.path.join(output_dir, "eval_results.txt"), "a", encoding="utf-8") as f:
        f.write(eval_result_str + "-" * 40 + "\n")

    if tag != "periodic":
        tsv_path = os.path.join(output_dir, f"predictions_{tag}_{global_step}.tsv")
        try:
            with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['Property', 'Ground Truth', 'Prediction (Float)'])
                for prop, data in prop_metrics.items():
                    for gt, pred in zip(data['y_true'], data['y_pred']):
                        writer.writerow([prop, gt, pred])
            logger.info(f"📄 详细预测结果已保存至: {tsv_path}")
        except Exception as e:
            logger.error(f"保存 TSV 失败: {e}")

    # ================== 🚀 加强版：全自动硬盘保护清理机制 ==================
    save_path = os.path.join(output_dir, f"checkpoint-{tag}-{global_step}")
    model.save_pretrained(save_path, safe_serialization=False)
    logger.info(f"💾 模型已保存至: {save_path}")

    def clean_old_checkpoints(prefix_pattern, keep_n):
        checkpoints = glob.glob(os.path.join(output_dir, prefix_pattern))
        checkpoints_with_step = []
        for cp in checkpoints:
            try:
                step = int(cp.split("-")[-1])
                checkpoints_with_step.append((step, cp))
            except ValueError:
                continue
        checkpoints_with_step.sort(key=lambda x: x[0])
        if len(checkpoints_with_step) > keep_n:
            for step, cp_to_delete in checkpoints_with_step[:-keep_n]:
                shutil.rmtree(cp_to_delete, ignore_errors=True)

    if tag == "periodic":
        clean_old_checkpoints("checkpoint-periodic-*", keep_n=2)
    elif tag.startswith("epoch_"):
        clean_old_checkpoints("checkpoint-epoch_*_end-*", keep_n=2)

    model.train()


# ================= 原生主循环 (完全对齐 3D-MolT5 SOTA 参数) =================
def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    # 🚀 对齐官方 SOTA 超参：lr 降至 3e-4，并增加超参选项
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()
    set_seed(42)

    output_dir = "./checkpoints/gen_prop_Native_BF16"
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 128
    accum_steps = 8  # 实际等效 batch size = 128 * 8 = 1024 (非常稳健)
    num_train_epochs = 85
    eval_steps = 1000

    AutoConfig.register("most-t5", MoStT5Config)
    text_tokenizer = TextTokenizer("google/t5-v1_1-base", max_len=768)
    motif_tokenizer = MotifTokenizer(vocab_file="asset/mol_vocabs/vocab_phase2_25k.txt",
                                     model_name="google/t5-v1_1-base", max_len=768)
    e3fp_tokenizer = E3FPTokenizer(fp_level=3, fp_bits=4096)

    logger.info("⚙️ 正在加载模型...")
    config = MoStT5Config.from_pretrained(args.model_path, e3fp_num_levels=4, e3fp_vocab_size=4096)
    model = MoStT5ForConditionalGeneration.from_pretrained(args.model_path, config=config, ignore_mismatched_sizes=True)
    model.resize_token_embeddings(len(motif_tokenizer.tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = GenerativeHybridDataset(os.path.join(args.data_dir, "train/3d_computed_properties_unit.json"),
                                            os.path.join(args.data_dir, "pubchemqc_final.lmdb"), text_tokenizer,
                                            motif_tokenizer, e3fp_tokenizer)
    eval_dataset = GenerativeHybridDataset(os.path.join(args.data_dir, "valid/3d_computed_properties_unit.json"),
                                           os.path.join(args.data_dir, "pubchemqc_final.lmdb"), text_tokenizer,
                                           motif_tokenizer, e3fp_tokenizer, is_eval=True)

    collator = GenerativeCollator(text_pad_id=text_tokenizer.tokenizer.pad_token_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True, num_workers=8,
                              pin_memory=True, prefetch_factor=4, persistent_workers=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size * 2, collate_fn=collator, shuffle=True, num_workers=4,
                             pin_memory=True)

    # 🚀 替换优化器配置：引入 Weight Decay 防过拟合
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 🚀 替换调度器配置：使用官方验证的 Linear Scheduler + 1万步超长 Warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps
    )

    logger.info("=" * 60)
    logger.info(f"🚀 [Phase 3] 启动 SOTA 级属性预测微调 (已开启梯度裁剪与万步预热)!")
    logger.info(
        f"🔧 参数配置: LR={args.lr}, Warmup={args.warmup_steps}, GradClip={args.grad_clip}, Total Steps={args.total_steps}")
    logger.info("=" * 60)

    global_step = 0
    valid_steps = 0
    model.train()
    optimizer.zero_grad()

    for epoch in range(num_train_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_train_epochs}")

        for step, batch in enumerate(progress_bar):
            if global_step >= args.total_steps: break
            if not batch: continue

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss

            if torch.isnan(loss) or loss.item() > 50.0: continue

            loss = loss / accum_steps
            loss.backward()

            epoch_loss += loss.item() * accum_steps
            valid_steps += 1

            if valid_steps == accum_steps:
                # 🚀 极其关键：启用梯度裁剪 max_norm=1.0 防御梯度爆炸！
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip).item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                valid_steps = 0

                progress_bar.set_postfix(
                    {'loss': f"{epoch_loss / accum_steps:.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                     'grad': f"{grad_norm:.2f}"})
                epoch_loss = 0.0

                if global_step % eval_steps == 0:
                    evaluate_and_save(model, eval_loader, text_tokenizer, device, global_step, output_dir,
                                      tag="periodic")

        if global_step >= args.total_steps: break
        logger.info(f"🎉 Epoch {epoch + 1} 结束！进行大考评估...")
        evaluate_and_save(model, eval_loader, text_tokenizer, device, global_step, output_dir,
                          tag=f"epoch_{epoch + 1}_end")

    logger.info(f"🏁 训练完成！进行 Final 终极评估...")
    evaluate_and_save(model, eval_loader, text_tokenizer, device, global_step, output_dir, tag="final")


if __name__ == "__main__":
    main()