import logging
import os
import sys
import torch
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoConfig
)

from tokenization.text_tokenizer import TextTokenizer
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
from dataset.dataset import GSMATDataset, GSMATPhase2Collator
from model.configuration import MoStT5Config
from model.modeling import MoStT5ForConditionalGeneration
from arguments import ModelArguments, DataArguments

logger = logging.getLogger(__name__)

def main():
    AutoConfig.register("most-t5", MoStT5Config)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    set_seed(training_args.seed)

    text_tokenizer = TextTokenizer(model_args.tokenizer_name, max_len=data_args.max_seq_length)
    # 确保 data_args.vocab_file 此时指向的是扩充后的 25k 词表
    motif_tokenizer = MotifTokenizer(data_args.vocab_file, model_name=model_args.tokenizer_name, max_len=data_args.max_seq_length)
    e3fp_tokenizer = E3FPTokenizer(fp_level=model_args.e3fp_num_levels - 1, fp_bits=model_args.e3fp_vocab_size)

    # =========================================================================
    # 🚀 Phase 2 专属策略：降级 3D 损失，给文本生成交叉熵让路！
    # =========================================================================
    config = MoStT5Config.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        vocab_size=len(motif_tokenizer.tokenizer),
        e3fp_vocab_size=model_args.e3fp_vocab_size,
        e3fp_num_levels=model_args.e3fp_num_levels,
    )
    config.lambda_3d = 0.2  # 核心：降级为正则化项，避免冲垮文本生成

    model = MoStT5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        ignore_mismatched_sizes=True
    )

    # =========================================================================
    # 🚀 致命一步：执行结构解冻与词表扩容 (例如从 20k 扩容到 25k)
    # =========================================================================
    logger.info(f"Resizing model embeddings to fit Phase 2 Vocab size: {len(motif_tokenizer.tokenizer)}")
    model.resize_token_embeddings(len(motif_tokenizer.tokenizer))

    train_dataset = GSMATDataset(
        lmdb_path=data_args.train_file,
        text_tokenizer=text_tokenizer,
        motif_tokenizer=motif_tokenizer,
        e3fp_tokenizer=e3fp_tokenizer,
        c4_lmdb_path=data_args.c4_train_file if hasattr(data_args, 'c4_train_file') else "",
        whitelist_path=data_args.whitelist_path if hasattr(data_args, 'whitelist_path') else "",
        max_seq_length=data_args.max_seq_length
    )

    eval_dataset = None
    if data_args.validation_file:
        eval_dataset = GSMATDataset(
            lmdb_path=data_args.validation_file,
            text_tokenizer=text_tokenizer,
            motif_tokenizer=motif_tokenizer,
            e3fp_tokenizer=e3fp_tokenizer,
            c4_lmdb_path="",
            whitelist_path="",
            max_seq_length=data_args.max_seq_length
        )

    # Phase 2 专属 Collator，带有跨模态拼接机制
    data_collator = GSMATPhase2Collator(
        motif_tokenizer=motif_tokenizer,
        text_tokenizer=text_tokenizer,
        text_weight_path=data_args.text_weight_path,
        e3fp_pad_id=-1,
        mask_ratio=0.15,
        is_train=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("=" * 60)
    logger.info("🔥 ALL SYSTEMS GO. Starting Phase 2: MoSt-T5 Cross-Modal Alignment...")
    logger.info("=" * 60)

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

if __name__ == "__main__":
    main()