"""Read one real multi-worker batch from each formal pretraining task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer

from most_t5_next.training.curriculum import CurriculumSchedule
from most_t5_next.training.data_provider import CurriculumDataLoaderProvider
from most_t5_next.training.runtime import TrainingRuntimeConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--pcqm-cache", type=Path, required=True)
    parser.add_argument("--pubchem-cache", type=Path, required=True)
    parser.add_argument("--paired-text-cache", type=Path, required=True)
    parser.add_argument("--pubmed-cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_root / "tokenizer_snapshot",
        use_fast=False,
        local_files_only=True,
    )
    sentinels = tuple(
        int(tokenizer.convert_tokens_to_ids(f"<extra_id_{index}>"))
        for index in range(100)
    )
    runtime = TrainingRuntimeConfig(
        seed=42,
        precision="bf16",
        micro_batch_size=2,
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
    result: dict[str, object] = {}
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
                batch = provider(task, update)[0]
                result[task.name] = {
                    "input": list(batch["input_ids"].shape),
                    "labels": list(batch["labels"].shape),
                    "geometry_payload": "e3fp_ids" in batch,
                    "geometry_available": bool(
                        batch.get("e3fp_ids") is not None
                        and batch["e3fp_ids"].ge(0).any().item()
                    ),
                }
        finally:
            provider.close()
    print(json.dumps({"status": "pass", "tasks": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
