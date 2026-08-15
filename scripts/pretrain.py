"""Launch the two-phase MoSt-T5 pretraining protocol on one GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from most_t5_next.configuration import load_pretraining_config
from most_t5_next.modeling.loading import load_model_from_config
from most_t5_next.training.data_provider import CurriculumDataLoaderProvider
from most_t5_next.training.runner import run_two_phase_pretraining
from most_t5_next.training.runtime import runtime_from_config


POPULATION_SCHEMA = "most-t5/pretraining-populations/v1"


def _tokenizer_contract(root: Path) -> tuple[int, tuple[int, ...], int]:
    tokenizer = AutoTokenizer.from_pretrained(
        root / "tokenizer_snapshot", use_fast=False, local_files_only=True
    )
    pad = int(tokenizer.pad_token_id)
    eos = int(tokenizer.eos_token_id)
    sentinels = tuple(
        int(tokenizer.convert_tokens_to_ids(f"<extra_id_{index}>"))
        for index in range(100)
    )
    if pad < 0 or eos < 0 or min(sentinels) < 0 or len(set(sentinels)) != 100:
        raise ValueError("tokenizer PAD/EOS/sentinel contract failed")
    return pad, sentinels, eos


def _load_populations(root: Path) -> dict[str, np.ndarray]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != POPULATION_SCHEMA
        or manifest.get("status") != "pass"
        or manifest.get("training_admission") is not True
    ):
        raise ValueError("pretraining population manifest is not admitted")
    populations: dict[str, np.ndarray] = {}
    for task in ("M", "MG", "SYN", "CAP", "T2M", "TXT"):
        descriptor = manifest["arrays"][task]
        path = root / descriptor["file"]
        values = np.memmap(path, mode="r", dtype=descriptor["dtype"])
        if values.shape != tuple(descriptor["shape"]) or not len(values):
            raise ValueError(f"population array is invalid: {task}")
        populations[task] = values
    return populations


def _phase_populations(
    populations: Mapping[str, Sequence[int]], tasks: Sequence[str]
) -> dict[str, Sequence[int]]:
    return {task: populations[task] for task in tasks}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--pcqm-cache", type=Path, required=True)
    parser.add_argument("--pubchem-cache", type=Path, required=True)
    parser.add_argument("--paired-text-cache", type=Path, required=True)
    parser.add_argument("--pubmed-cache", type=Path, required=True)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    config = load_pretraining_config(
        args.config, overrides=args.overrides, require_launch_values=True
    )
    runtime = runtime_from_config(config)
    pad, sentinels, eos = _tokenizer_contract(args.tokenizer_root)
    populations = _load_populations(args.population_root)
    provider_kwargs = {
        "pcqm_cache": args.pcqm_cache,
        "pubchem_cache": args.pubchem_cache,
        "paired_text_cache": args.paired_text_cache,
        "pubmed_cache": args.pubmed_cache,
        "pad_token_id": pad,
        "sentinel_token_ids": sentinels,
        "eos_token_id": eos,
        "runtime": runtime,
    }
    phase_one = CurriculumDataLoaderProvider(
        phase=1,
        total_updates=int(config["curriculum"]["phase_one"]["total_updates"]),
        populations=_phase_populations(populations, ("M", "MG")),
        **provider_kwargs,
    )
    phase_one_closed = False
    phase_two_holder: list[CurriculumDataLoaderProvider] = []

    def phase_two_factory() -> CurriculumDataLoaderProvider:
        nonlocal phase_one_closed
        phase_one.close()
        phase_one_closed = True
        provider = CurriculumDataLoaderProvider(
            phase=2,
            total_updates=int(config["curriculum"]["phase_two"]["total_updates"]),
            populations=_phase_populations(
                populations, ("SYN", "TXT", "CAP", "T2M")
            ),
            **provider_kwargs,
        )
        phase_two_holder.append(provider)
        return provider
    model = load_model_from_config(args.checkpoint, config)
    tensorboard = Path(config["monitoring"]["tensorboard_root"]) / args.output_dir.name
    writer = SummaryWriter(tensorboard)
    try:
        report = run_two_phase_pretraining(
            model=model,
            phase_one_batch_provider=phase_one,
            phase_two_batch_provider_factory=phase_two_factory,
            config=config,
            output_dir=args.output_dir,
            device=args.device,
            writer=writer,
        )
    finally:
        writer.close()
        if not phase_one_closed:
            phase_one.close()
        for provider in phase_two_holder:
            provider.close()
    print(json.dumps({"status": report["status"], "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
