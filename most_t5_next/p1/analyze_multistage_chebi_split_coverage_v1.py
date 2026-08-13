#!/usr/bin/env python3
"""Replay Phase-I/II joint registries on ChEBI-20 train/validation/test.

Only ChEBI train counts may add task-aware macro rows.  Validation and test are
read solely after each registry is fixed and are never used for ranking.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
from typing import Sequence

from most_t5_next.p1.analyze_chebi20_task_aware_vocab_v1 import _metrics
from most_t5_next.p1.analyze_multistage_anchored_vocab_v1 import (
    _equal_stage_ranking,
    _load_chebi,
    _load_registry,
    _pooled_ranking,
)
from most_t5_next.p1.build_registered_downstream_pure_motif_census_v1 import (
    _project_smiles,
)


SCHEMA_VERSION = "most-t5-next/multistage-chebi-split-coverage/v1"


class MultistageChEBICoverageError(RuntimeError):
    """A train-selected registry cannot be replayed on all declared splits."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_parquet(
    path: Path,
    split: str,
    id_column: str,
    smiles_column: str,
    workers: int,
    chunksize: int,
) -> tuple[tuple[tuple[str, ...], ...], Counter[str]]:
    import pandas as pd

    frame = pd.read_parquet(path, columns=[id_column, smiles_column])
    tasks = [
        (int(index), f"{split}:{row[id_column]}", str(row[smiles_column]).strip())
        for index, row in frame.iterrows()
    ]
    if not tasks:
        raise MultistageChEBICoverageError(f"{split} split is empty")
    if workers == 1:
        results = map(_project_smiles, tasks)
        executor = None
    else:
        context = multiprocessing.get_context("spawn")
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        )
        results = executor.map(_project_smiles, tasks, chunksize=chunksize)
    sequences: list[tuple[str, ...]] = []
    counts: Counter[str] = Counter()
    try:
        for result in results:
            if result["error"] is not None:
                raise MultistageChEBICoverageError(
                    f"{split} projection rejected {result['record_id']}: {result['error']}"
                )
            motifs = result["pure_motifs"]
            if not isinstance(motifs, tuple) or not motifs:
                raise MultistageChEBICoverageError(f"{split} projection returned no motifs")
            sequences.append(motifs)
            counts.update(motifs)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return tuple(sequences), counts


def run(args: argparse.Namespace) -> dict[str, object]:
    p1_path = Path(args.phase1_registry).expanduser().resolve()
    p2_path = Path(args.phase2_registry).expanduser().resolve()
    train_census_path = Path(args.chebi_train_census).expanduser().resolve()
    split_paths = {
        "train": Path(args.train_parquet).expanduser().resolve(),
        "validation": Path(args.validation_parquet).expanduser().resolve(),
        "test": Path(args.test_parquet).expanduser().resolve(),
    }
    output = Path(args.output_report).expanduser().resolve()
    if output.exists():
        raise MultistageChEBICoverageError("output report must be absent")
    if args.base_budget <= 0 or args.workers <= 0:
        raise MultistageChEBICoverageError("budget and workers must be positive")
    additions = tuple(sorted({int(value) for value in args.additions.split(",")}))
    if not additions or additions[0] < 0:
        raise MultistageChEBICoverageError("additions must be nonnegative")
    phase1, _ = _load_registry(p1_path)
    phase2, _ = _load_registry(p2_path)
    chebi_train = _load_chebi(train_census_path)
    train_ranked = tuple(
        sorted(chebi_train, key=lambda pure: (-chebi_train[pure], pure.encode("utf-8")))
    )
    rankings = {
        "phase1_phase2_raw_pooled_frequency": _pooled_ranking(phase1, phase2),
        "phase1_phase2_equal_stage_mass": _equal_stage_ranking(phase1, phase2),
    }
    split_data = {
        split: _project_parquet(
            path,
            split,
            args.id_column,
            args.smiles_column,
            args.workers,
            args.chunksize,
        )
        for split, path in split_paths.items()
    }
    rows: list[dict[str, object]] = []
    for policy, ranking in rankings.items():
        base = set(ranking[: min(args.base_budget, len(ranking))])
        absent_train = tuple(pure for pure in train_ranked if pure not in base)
        for requested in additions:
            selected = base | set(absent_train[:requested])
            rows.append(
                {
                    "ranking_policy": policy,
                    "base_budget": args.base_budget,
                    "requested_chebi_train_additions": requested,
                    "selected_chebi_train_additions": len(selected - base),
                    "selected_macro_count": len(selected),
                    "train": _metrics(*split_data["train"], selected),
                    "validation": _metrics(*split_data["validation"], selected),
                    "test": _metrics(*split_data["test"], selected),
                }
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "inputs": {
            "phase1_registry": {"path": str(p1_path), "sha256": _sha256_file(p1_path)},
            "phase2_registry": {"path": str(p2_path), "sha256": _sha256_file(p2_path)},
            "chebi_train_census": {
                "path": str(train_census_path),
                "sha256": _sha256_file(train_census_path),
            },
            "splits": {
                split: {"path": str(path), "sha256": _sha256_file(path)}
                for split, path in split_paths.items()
            },
        },
        "split_counts": {
            split: {
                "molecules": len(sequences),
                "motif_occurrences": sum(counts.values()),
                "unique_motifs": len(counts),
            }
            for split, (sequences, counts) in split_data.items()
        },
        "budget_rows": rows,
        "contracts": {
            "macro_selection_uses_phase1_train": True,
            "macro_selection_uses_phase2_train": True,
            "chebi_addition_selection_uses_chebi_train_only": True,
            "validation_or_test_used_for_selection": False,
            "lossless_fallback_remains_mandatory": True,
            "training_admission": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-registry", required=True)
    parser.add_argument("--phase2-registry", required=True)
    parser.add_argument("--chebi-train-census", required=True)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--validation-parquet", required=True)
    parser.add_argument("--test-parquet", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--base-budget", type=int, default=16000)
    parser.add_argument("--additions", default="0,256,512,1024,2048")
    parser.add_argument("--id-column", default="cid")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunksize", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(_parser().parse_args(argv))
    except Exception as exc:
        print(f"multistage ChEBI split coverage failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
