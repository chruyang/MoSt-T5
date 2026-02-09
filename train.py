import logging
import os
import sys
import torch
import numpy as np
import evaluate
import nltk
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    EarlyStoppingCallback
)

from tokenization.text_tokenizer import TextTokenizer
from tokenization.motif_tokenizer import MotifTokenizer
from tokenization.e3fp_tokenizer import E3FPTokenizer
from dataset.dataset import GSMATDataset, GSMATCollator
from model.configuration import MoStT5Config
from model.modeling import MoStT5ForConditionalGeneration
from arguments import ModelArguments, DataArguments

logger = logging.getLogger(__name__)


def main():
    # 确保 nltk 资源存在
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    set_seed(training_args.seed)

    # --- Tokenizers ---
    logger.info("Loading Tokenizers...")
    text_tokenizer = TextTokenizer(model_name=model_args.model_name_or_path, max_len=data_args.max_len)
    motif_tokenizer = MotifTokenizer(vocab_file=model_args.vocab_path, model_name=model_args.model_name_or_path)
    e3fp_tokenizer = E3FPTokenizer(padding_idx=-1)

    # --- Datasets ---
    logger.info("Loading Datasets...")
    train_dataset = GSMATDataset(
        lmdb_path=data_args.train_file,
        text_tokenizer=text_tokenizer,
        motif_tokenizer=motif_tokenizer,
        e3fp_tokenizer=e3fp_tokenizer
    )

    eval_dataset = None
    if data_args.validation_file:
        logger.info(f"Loading Validation Dataset...")
        eval_dataset = GSMATDataset(
            lmdb_path=data_args.validation_file,
            text_tokenizer=text_tokenizer,
            motif_tokenizer=motif_tokenizer,
            e3fp_tokenizer=e3fp_tokenizer
        )

    # --- Model Config & Init ---
    logger.info("Initializing MoSt-T5 Model...")
    config = MoStT5Config.from_pretrained(model_args.model_name_or_path)
    config.update({
        'e3fp_num_levels': model_args.e3fp_num_levels,
        'e3fp_vocab_size': model_args.e3fp_vocab_size,
        'vocab_size': motif_tokenizer.vocab_size,
        'fusion_type': model_args.fusion_type,
        'dropout_rate': model_args.dropout_rate
    })

    model = MoStT5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True
    )
    # ================== ✅ NEW FIX (增强版) ==================
    # 强制补全所有必要的 Token ID，防止 evaluate 报错
    print("🔧 Applying config fix for generation...")
    pad_token_id = text_tokenizer.tokenizer.pad_token_id

    # 1. 修复 decoder_start_token_id (T5 必需)
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = pad_token_id

    # 2. 修复 bos_token_id (新版 Transformers 检查必需)
    # T5 没有 BOS，我们将其指向 PAD，骗过检查
    if model.config.bos_token_id is None:
        model.config.bos_token_id = pad_token_id

    # 3. 同步更新 generation_config
    if hasattr(model, "generation_config"):
        model.generation_config.decoder_start_token_id = pad_token_id
        model.generation_config.pad_token_id = pad_token_id
        model.generation_config.bos_token_id = pad_token_id
    # ================== ✅ FIX END ===========================

    # [改进] 显式修复生成配置
    model.config.decoder_start_token_id = text_tokenizer.tokenizer.pad_token_id
    model.config.eos_token_id = text_tokenizer.tokenizer.eos_token_id
    model.config.pad_token_id = text_tokenizer.tokenizer.pad_token_id
    model.generation_config.max_length = data_args.max_len
    model.generation_config.num_beams = 4
    model.generation_config.repetition_penalty = 1.2

    # --- Metrics ---
    metric_bleu = evaluate.load("sacrebleu")
    metric_rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple): preds = preds[0]

        decoded_preds = text_tokenizer.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, text_tokenizer.pad_token_id)
        decoded_labels = text_tokenizer.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 文本处理: 小写 + NLTK 分词
        decoded_preds = [pred.strip().lower() for pred in decoded_preds]
        decoded_labels = [label.strip().lower() for label in decoded_labels]

        decoded_preds_tok = [" ".join(nltk.word_tokenize(pred)) for pred in decoded_preds]
        decoded_labels_tok = [" ".join(nltk.word_tokenize(label)) for label in decoded_labels]

        # 打印 Sample 监控生成质量
        logger.info("\n" + "=" * 20 + " Sample Generation " + "=" * 20)
        logger.info(f"Pred : {decoded_preds[0]}")
        logger.info(f"Label: {decoded_labels[0]}")
        logger.info("=" * 60)

        result = {}
        # BLEU (sacrebleu 需要 list of list references)
        decoded_labels_bleu = [[l] for l in decoded_labels_tok]
        result["bleu"] = metric_bleu.compute(predictions=decoded_preds_tok, references=decoded_labels_bleu)["score"]

        # ROUGE
        rouge_score = metric_rouge.compute(predictions=decoded_preds_tok, references=decoded_labels_tok)
        result["rouge1"] = rouge_score["rouge1"]
        result["rougeL"] = rouge_score["rougeL"]

        return {k: round(v, 4) for k, v in result.items()}

    # --- Trainer ---
    data_collator = GSMATCollator(
        motif_pad_id=motif_tokenizer.pad_id,
        text_pad_id=text_tokenizer.pad_token_id,
        e3fp_pad_id=-1,
        ignore_index=-100
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=text_tokenizer.tokenizer,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] if eval_dataset else None
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

    if training_args.do_eval and eval_dataset:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()