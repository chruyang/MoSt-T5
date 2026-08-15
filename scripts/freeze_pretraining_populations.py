"""Build finite-budget MoSt-T5 task populations and the length-action ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer

from most_t5_next.configuration import load_pretraining_config
from most_t5_next.training.freeze_populations import freeze_populations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--pcqm-cache", type=Path, required=True)
    parser.add_argument("--pubchem-cache", type=Path, required=True)
    parser.add_argument("--paired-text-cache", type=Path, required=True)
    parser.add_argument("--pubmed-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunksize", type=int, default=64)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_pretraining_config(
        args.config, overrides=args.overrides, require_launch_values=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_root / "tokenizer_snapshot",
        use_fast=False,
        local_files_only=True,
    )
    sentinels = tuple(
        int(tokenizer.convert_tokens_to_ids(f"<extra_id_{index}>"))
        for index in range(100)
    )
    manifest = freeze_populations(
        pcqm_cache=args.pcqm_cache,
        pubchem_cache=args.pubchem_cache,
        paired_text_cache=args.paired_text_cache,
        pubmed_cache=args.pubmed_cache,
        output_dir=args.output_dir,
        sentinels=sentinels,
        eos=int(tokenizer.eos_token_id),
        seed=int(config["seed"]),
        phase_one_updates=int(config["curriculum"]["phase_one"]["total_updates"]),
        phase_two_updates=int(config["curriculum"]["phase_two"]["total_updates"]),
        micro_batch_size=int(config["batching"]["micro_batch_size"]),
        accumulation_steps=int(config["batching"]["gradient_accumulation_steps"]),
        workers=args.workers,
        chunksize=args.chunksize,
    )
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
