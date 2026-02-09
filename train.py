import logging
import os
import sys
import torch
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    EarlyStoppingCallback
)

# ... (引入模块保持不变) ...
from tokenization.text_tokenizer import TextTokenizer
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
from dataset.dataset import GSMATDataset, GSMATCollator
from model.configuration import MoStT5Config
from model.modeling import MoStT5ForConditionalGeneration
from arguments import ModelArguments, DataArguments

logger = logging.getLogger(__name__)


def main():
    # 1. 参数解析
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. 初始化日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info(f"Training parameters: {training_args}")
    set_seed(training_args.seed)

    # 3. 加载 Tokenizers
    logger.info("Loading Tokenizers...")
    text_tokenizer = TextTokenizer(model_name=model_args.model_name_or_path)
    motif_tokenizer = MotifTokenizer(vocab_file=model_args.vocab_path, model_name=model_args.model_name_or_path)
    e3fp_tokenizer = E3FPTokenizer(padding_idx=-1)

    # 4. 准备数据集 (Train + Valid)
    logger.info("Loading Datasets...")
    train_dataset = GSMATDataset(
        lmdb_path=data_args.train_file,
        text_tokenizer=text_tokenizer,
        motif_tokenizer=motif_tokenizer,
        e3fp_tokenizer=e3fp_tokenizer
    )
    logger.info(f"Train Dataset Size: {len(train_dataset)}")

    eval_dataset = None
    if data_args.validation_file:
        logger.info(f"Loading Validation Dataset from {data_args.validation_file}...")
        eval_dataset = GSMATDataset(
            lmdb_path=data_args.validation_file,
            text_tokenizer=text_tokenizer,
            motif_tokenizer=motif_tokenizer,
            e3fp_tokenizer=e3fp_tokenizer
        )
        logger.info(f"Eval Dataset Size: {len(eval_dataset)}")

    # 5. 初始化模型
    logger.info("Initializing MoSt-T5 Model...")
    config = MoStT5Config.from_pretrained(model_args.model_name_or_path)

    # [关键修复] 注入所有参数，确保与 arguments.py 一致
    config.update({
        'e3fp_num_levels': model_args.e3fp_num_levels,
        'e3fp_vocab_size': model_args.e3fp_vocab_size,
        'vocab_size': motif_tokenizer.vocab_size,  # 动态更新
        'fusion_type': model_args.fusion_type,
        'dropout_rate': model_args.dropout_rate
    })

    model = MoStT5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(motif_tokenizer.tokenizer))

    # 6. Collator
    data_collator = GSMATCollator(
        motif_pad_id=motif_tokenizer.pad_id,
        text_pad_id=text_tokenizer.pad_token_id,
        e3fp_pad_id=-1,
        ignore_index=-100
    )

    # 7. Trainer (加入 EarlyStopping)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=text_tokenizer.tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else None
    )

    # 8. 训练流程
    if training_args.do_train:
        logger.info("*** Starting Training ***")
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

    # 9. 评估流程
    if training_args.do_eval and eval_dataset:
        logger.info("*** Starting Evaluation ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()