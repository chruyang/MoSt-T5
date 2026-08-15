"""Run one BF16 forward/backward pass for each task on real cached data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoTokenizer

from most_t5_next.configuration import load_pretraining_config
from most_t5_next.modeling.loading import load_model_from_config
from most_t5_next.training.curriculum import CurriculumSchedule
from most_t5_next.training.data_provider import CurriculumDataLoaderProvider
from most_t5_next.training.engine import forward_task
from most_t5_next.training.runtime import TrainingRuntimeConfig


def _move(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value.to("cuda:0", non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in batch.items()
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--pcqm-cache", type=Path, required=True)
    parser.add_argument("--pubchem-cache", type=Path, required=True)
    parser.add_argument("--paired-text-cache", type=Path, required=True)
    parser.add_argument("--pubmed-cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the smoke requires one BF16-capable CUDA device")
    config = load_pretraining_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_root / "tokenizer_snapshot", use_fast=False, local_files_only=True
    )
    sentinels = tuple(
        int(tokenizer.convert_tokens_to_ids(f"<extra_id_{index}>"))
        for index in range(100)
    )
    runtime = TrainingRuntimeConfig(
        seed=42,
        precision="bf16",
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        num_workers=args.workers,
        prefetch_factor=2,
        persistent_workers=bool(args.workers),
    )
    common = {
        "pcqm_cache": args.pcqm_cache,
        "pubchem_cache": args.pubchem_cache,
        "paired_text_cache": args.paired_text_cache,
        "pubmed_cache": args.pubmed_cache,
        "pad_token_id": int(tokenizer.pad_token_id),
        "sentinel_token_ids": sentinels,
        "eos_token_id": int(tokenizer.eos_token_id),
        "runtime": runtime,
    }
    model = load_model_from_config(args.checkpoint, config).to("cuda:0")
    model.train()
    report: dict[str, object] = {}
    torch.cuda.reset_peak_memory_stats()
    for phase, total, tasks in ((1, 2, ("M", "MG")), (2, 4, ("SYN", "TXT", "CAP", "T2M"))):
        provider = CurriculumDataLoaderProvider(
            phase=phase,
            total_updates=total,
            populations={task: range(32) for task in tasks},
            **common,
        )
        schedule = CurriculumSchedule(phase, total)
        try:
            for update in range(total):
                task = schedule.task_at(update)
                batch = _move(provider(task, update)[0])
                model.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = forward_task(model, task.name, batch)
                    loss = output.loss
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError(f"{task.name} returned a non-finite loss")
                loss.backward()
                gradient = sum(
                    int(parameter.grad is not None and torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                )
                if not gradient:
                    raise RuntimeError(f"{task.name} produced no finite gradient")
                report[task.name] = {
                    "loss": float(loss.detach()),
                    "finite_gradient_tensors": gradient,
                    "input_shape": list(batch["input_ids"].shape),
                    "label_shape": list(batch["labels"].shape),
                }
        finally:
            provider.close()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "status": "pass",
                "gpu": torch.cuda.get_device_name(0),
                "peak_memory_mib": torch.cuda.max_memory_allocated() // (1024 * 1024),
                "tasks": report,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
