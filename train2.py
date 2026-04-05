import logging
import os
import sys
import re
import torch
import numpy as np
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoConfig
)

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# ================= 自定义模块导入 =================
from tokenization.text_tokenizer import TextTokenizer
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
from dataset.dataset2 import GSMATDataset, GSMATPhase2Collator
from model.configuration import MoStT5Config
from model.modeling import MoStT5ForConditionalGeneration
from arguments import ModelArguments, DataArguments

# ================= 日志配置 =================
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =========================================================================
# 🛠️ 核心外挂 1：极致鲁棒的 Motif 清洗器 (解决 SMARTS 贪婪陷阱与嵌套括号误杀)
# =========================================================================
def robust_motif_to_mol(motif_str):
    """验证一个字符串是否是合法的、具有三维潜力的化学基团"""
    # 1. 拦截特殊占位符
    if motif_str in ['<unk>', '<pad>', '<bom>', '<eom>', '[.]']:
        return "SPECIAL"
    if re.match(r'^<\d+\*>$', motif_str):
        return "ANCHOR"
    if re.match(r'^\[.*?\]:$', motif_str):
        return "TASK_PROMPT"

    # 2. 清理多余连接符
    clean_smiles = re.sub(r'\[.*?\]:', '', motif_str)
    clean_smiles = re.sub(r'<\d+\*>', '', clean_smiles).replace("()", "").strip()

    if not clean_smiles:
        return None

    # 提取剥壳版 (去掉最外层的一对括号)
    stripped_smiles = clean_smiles
    if clean_smiles.startswith('[') and clean_smiles.endswith(']'):
        stripped_smiles = clean_smiles[1:-1]

    # ==============================================================
    # 🚀 优先级 1：最严格的 SMILES 解析 (优先尝试原版，再尝试剥壳版)
    # ==============================================================
    mol = Chem.MolFromSmiles(clean_smiles)
    if mol is not None: return mol

    if clean_smiles != stripped_smiles:
        mol = Chem.MolFromSmiles(stripped_smiles)
        if mol is not None: return mol

    # ==============================================================
    # 🚀 优先级 2：宽容的 SMARTS 解析 (作为最后的兜底手段，强迫验证环结构)
    # ==============================================================
    mol = Chem.MolFromSmarts(clean_smiles)
    if mol is not None:
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSSSR(mol)  # 强迫进行环检查，过滤掉畸形 Query 节点
            return mol
        except Exception:
            pass

    if clean_smiles != stripped_smiles:
        mol = Chem.MolFromSmarts(stripped_smiles)
        if mol is not None:
            try:
                mol.UpdatePropertyCache(strict=False)
                Chem.GetSSSR(mol)
                return mol
            except Exception:
                pass

    return None


# =========================================================================
# 🛠️ 核心外挂 2：基于 Morgan 指纹的 3D 权重智能继承 (绝对无遗漏版)
# =========================================================================
def smart_initialize_new_embeddings(model, motif_tokenizer, old_vocab_size):
    logger.info("=" * 60)
    logger.info(f"🧠 启动基于 Morgan 指纹的 3D 权重智能继承 (旧词表大小={old_vocab_size})")

    embeddings = model.get_input_embeddings().weight.data
    new_vocab_size = len(motif_tokenizer.tokenizer)

    if new_vocab_size <= old_vocab_size:
        logger.info("⏭️ 词表未发生扩容，跳过智能初始化。")
        logger.info("=" * 60)
        return

    old_fps, valid_old_indices = [], []
    id_to_token = {v: k for k, v in motif_tokenizer.tokenizer.get_vocab().items()}

    # 1. 扫描老词表构建指纹库
    for idx in range(old_vocab_size):
        token_str = id_to_token.get(idx, "")
        mol = robust_motif_to_mol(token_str)

        if mol is not None and not isinstance(mol, str):
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                old_fps.append(fp)
                valid_old_indices.append(idx)
            except Exception:
                continue

    logger.info(f"✅ 成功从 {old_vocab_size} 个老词中提取了 {len(valid_old_indices)} 个合法的化学指纹锚点。")

    if not old_fps:
        logger.error("❌ 严重错误：未能提取到任何老基团的有效指纹，初始化失败！")
        return

    success_count = 0
    skipped_tokens = []

    # 2. 为新基团寻找最近邻并克隆权重
    for new_idx in range(old_vocab_size, new_vocab_size):
        new_token_str = id_to_token.get(new_idx, "")
        mol = robust_motif_to_mol(new_token_str)

        if mol is not None and not isinstance(mol, str):
            try:
                new_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                similarities = DataStructs.BulkTanimotoSimilarity(new_fp, old_fps)
                best_match_idx = similarities.index(max(similarities))
                actual_old_idx = valid_old_indices[best_match_idx]

                # 将相似旧词的 3D 特征复制给新词，并加入极微小对称破缺噪声
                embeddings[new_idx] = embeddings[actual_old_idx].clone()
                embeddings[new_idx] += torch.randn_like(embeddings[new_idx]) * 0.01
                success_count += 1
            except Exception:
                skipped_tokens.append(new_token_str)
        else:
            pass  # 正常跳过文本特殊符和锚点

    logger.info(f"✅ 智能初始化大功告成！成功为 {success_count} 个全新基团注入了 Phase 1 的 3D 物理先验。")
    if skipped_tokens:
        logger.warning(f"⚠️ 以下 {len(skipped_tokens)} 个实体基团指纹计算失败 (将使用普通高斯噪声): {skipped_tokens}")

    logger.info("=" * 60)


# =========================================================================
# 🚀 主训练流程
# =========================================================================
def main():
    AutoConfig.register("most-t5", MoStT5Config)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 设定全局随机种子
    set_seed(training_args.seed)

    # 关键设置：禁止 HF Trainer 自动移除 Dataloader 输出的字段 (我们需要 E3FP 等特征)
    training_args.remove_unused_columns = False
    # DDP 环境下推荐关闭 SafeTensors 以防同步死锁
    training_args.save_safetensors = False

    # ---------------------------------------------------------
    # 1. 词表联合挂载 (解决 ID 物理撞车的核心代码)
    # ---------------------------------------------------------
    text_tokenizer = TextTokenizer(model_args.tokenizer_name, max_len=data_args.max_seq_length)
    motif_tokenizer = MotifTokenizer(
        vocab_file=data_args.vocab_file,
        base_tokenizer=text_tokenizer.tokenizer,  # 🚀 极其关键：共享同一个底层字典！
        max_len=data_args.max_seq_length
    )
    e3fp_tokenizer = E3FPTokenizer(fp_level=model_args.e3fp_num_levels - 1, fp_bits=model_args.e3fp_vocab_size)

    # ---------------------------------------------------------
    # 2. 模型配置与张量扩容
    # ---------------------------------------------------------
    config = MoStT5Config.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        vocab_size=len(motif_tokenizer.tokenizer),
        e3fp_vocab_size=model_args.e3fp_vocab_size,
        e3fp_num_levels=model_args.e3fp_num_levels,
    )
    config.lambda_3d = 0.2  # 3D 辅助 Loss 的权重

    model = MoStT5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        ignore_mismatched_sizes=True
    )

    # Phase 1 的真实词表大小 (确保这跟你检查点中的维度完全一致)
    REAL_OLD_VOCAB_SIZE = 52306

    # 扩充模型的 Embedding 矩阵
    model.resize_token_embeddings(len(motif_tokenizer.tokenizer))

    # 执行 3D 物理知识继承
    smart_initialize_new_embeddings(model, motif_tokenizer, old_vocab_size=REAL_OLD_VOCAB_SIZE)

    # ---------------------------------------------------------
    # 3. 数据集与 Collator
    # ---------------------------------------------------------
    c4_path = getattr(data_args, 'c4_file', "")
    if not c4_path or not os.path.exists(c4_path):
        logger.error("=" * 60)
        logger.error(f"❌ 致命错误：C4 纯文本数据集不存在或路径错误: '{c4_path}'")
        raise FileNotFoundError(f"Required C4 dataset not found at: {c4_path}")

    # 正式训练：严格执行 4 任务各 25% 的均等分布
    phase2_task_probs = {"mmm": 0.25, "caption": 0.25, "text2mol": 0.25, "denoise": 0.25}
    logger.info("✅ 已开启正式多任务训练路由 (任务比重: 25% x 4)")

    train_dataset = GSMATDataset(
        lmdb_path=data_args.train_file,
        text_tokenizer=text_tokenizer,
        motif_tokenizer=motif_tokenizer,
        e3fp_tokenizer=e3fp_tokenizer,
        c4_lmdb_path=c4_path,
        max_seq_length=data_args.max_seq_length,
        task_probs=phase2_task_probs
    )

    # 注意：确保你的 GSMATPhase2Collator 已经应用了我们之前讨论过的
    # "前缀隔离" (Denoise & MMM) 和 "绝对零度抽样补丁"
    data_collator = GSMATPhase2Collator(
        motif_tokenizer=motif_tokenizer,
        text_tokenizer=text_tokenizer,
        text_weight_path=data_args.text_weight_path,
        e3fp_pad_id=-1,
        mask_ratio=0.15,
        is_train=True
    )

    # ---------------------------------------------------------
    # 4. 启动正式 Trainer
    # ---------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    logger.info("🚀 所有系统检查完毕，正式启动 Phase 2 跨模态预训练...")

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=None)


if __name__ == "__main__":
    main()