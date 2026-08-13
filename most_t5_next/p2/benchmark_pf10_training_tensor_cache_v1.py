"""CPU throughput benchmark for the derived PF-10 mmap training cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import torch

from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)

from .pf10_training_tensor_cache_v1 import build_v3_cache_dataloader


def benchmark_cache_loader(
    *,
    cache_root: Path,
    base_tokenizer_snapshot: Path,
    union_tokenizer_dir: Path,
    workers: Sequence[int],
    batches: int,
    micro_batch_size: int = 32,
    prefetch_factor: int = 4,
    cell: str = "F3D",
    seed: int = 20260807,
) -> dict[str, object]:
    if batches <= 0 or micro_batch_size <= 0:
        raise ValueError("batches and micro_batch_size must be positive")
    tokenizer = load_verified_canary_union_tokenizer(
        base_snapshot=Path(base_tokenizer_snapshot),
        output_dir=Path(union_tokenizer_dir),
    )
    rows = []
    for num_workers in workers:
        if num_workers < 0:
            raise ValueError("workers must be nonnegative")
        loader = build_v3_cache_dataloader(
            cache_root=cache_root,
            tokenizer=tokenizer.runtime,
            cell=cell,
            seed=seed,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=1,
            total_updates=batches,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        started = time.perf_counter()
        first_batch_seconds = None
        member_count = 0
        tensor_bytes = 0
        iterator = iter(loader)
        try:
            for batch_index, batch in enumerate(iterator, start=1):
                if first_batch_seconds is None:
                    first_batch_seconds = time.perf_counter() - started
                member_count += len(batch.record_ids)
                tensor_bytes += sum(
                    value.numel() * value.element_size()
                    for value in batch.inputs.values()
                    if isinstance(value, torch.Tensor)
                )
                if batch_index == batches:
                    break
        finally:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
            loader.dataset.close()
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "workers": num_workers,
                "batches": batches,
                "members": member_count,
                "seconds": elapsed,
                "first_batch_seconds": first_batch_seconds,
                "members_per_second_including_startup": member_count / elapsed,
                "collated_tensor_bytes": tensor_bytes,
            }
        )
    return {
        "scope": "CPU mmap decode plus dynamic V3 corruption/padding",
        "cell": cell,
        "micro_batch_size": micro_batch_size,
        "prefetch_factor": prefetch_factor,
        "results": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--union-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--workers", default="0,4,8,16")
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--cell", choices=("B0", "B2D", "F3D"), default="F3D")
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workers = tuple(int(value) for value in args.workers.split(","))
    report = benchmark_cache_loader(
        cache_root=args.cache_root,
        base_tokenizer_snapshot=args.base_tokenizer_snapshot,
        union_tokenizer_dir=args.union_tokenizer_dir,
        workers=workers,
        batches=args.batches,
        micro_batch_size=args.micro_batch_size,
        prefetch_factor=args.prefetch_factor,
        cell=args.cell,
    )
    output = Path(args.output_report).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
