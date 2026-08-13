#!/usr/bin/env python3
"""Replay a joint Phase-I/Phase-II motif registry on downstream populations."""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
import json
import multiprocessing
from pathlib import Path
import sys
import time
from typing import Sequence

from most_t5_next.p1.analyze_downstream_motif_coverage_v1 import (
    DownstreamMotifCoverageError,
    _load_dataset,
    _metrics,
    _project_smiles,
    _sha256_file,
)
from most_t5_next.p1.analyze_multistage_anchored_vocab_v1 import (
    _equal_stage_ranking,
    _load_chebi,
    _load_registry,
    _pooled_ranking,
)


SCHEMA_VERSION = "most-t5-next/multistage-downstream-motif-coverage/v1"
RANKING_POLICIES = {
    "phase1_phase2_equal_stage_mass": _equal_stage_ranking,
    "phase1_phase2_raw_pooled_frequency": _pooled_ranking,
}


def _selected_registries(args: argparse.Namespace):
    phase1, _phase1_ranking = _load_registry(
        Path(args.phase1_registry).expanduser().resolve()
    )
    phase2, _phase2_ranking = _load_registry(
        Path(args.phase2_registry).expanduser().resolve()
    )
    chebi = _load_chebi(Path(args.chebi_train_census).expanduser().resolve())
    ranker = RANKING_POLICIES.get(args.ranking_policy)
    if ranker is None:
        raise DownstreamMotifCoverageError(
            f"unsupported ranking policy: {args.ranking_policy}"
        )
    ranking = ranker(phase1, phase2)
    if args.base_budget <= 0 or args.base_budget > len(ranking):
        raise DownstreamMotifCoverageError("base budget is outside registry domain")
    base = set(ranking[: args.base_budget])
    all_chebi = base | set(chebi)
    return phase1, phase2, chebi, base, all_chebi


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_report).expanduser().resolve()
    if output.exists():
        raise DownstreamMotifCoverageError("output report must be absent")
    config_path = Path(args.dataset_config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise DownstreamMotifCoverageError("dataset config has no datasets")
    phase1, phase2, chebi, base, all_chebi = _selected_registries(args)
    policies = {
        "joint_pretraining_base": base,
        "joint_pretraining_base_plus_all_chebi20_train": all_chebi,
    }

    started = time.perf_counter()
    result_rows: list[dict[str, object]] = []
    context = multiprocessing.get_context("spawn")
    for dataset in datasets:
        tasks, dataset_manifest = _load_dataset(dataset)
        if not tasks:
            raise DownstreamMotifCoverageError(
                f"dataset {dataset_manifest['name']} has no unique molecules"
            )
        sequences: list[tuple[str, ...]] = []
        counts: Counter[str] = Counter()
        rejects: Counter[str] = Counter()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context
        ) as executor:
            for result in executor.map(
                _project_smiles, tasks, chunksize=args.chunksize
            ):
                if result["error"] is not None:
                    rejects[str(result["error"]).split(":", 1)[0]] += 1
                    continue
                motifs = result["pure_motifs"]
                if not isinstance(motifs, tuple) or not motifs:
                    rejects["EMPTY_MOTIF_SEQUENCE"] += 1
                    continue
                sequences.append(motifs)
                counts.update(motifs)
        if not sequences:
            raise DownstreamMotifCoverageError(
                f"dataset {dataset_manifest['name']} has no accepted molecules"
            )
        result_rows.append(
            {
                "dataset": dataset_manifest,
                "scientific_role": dataset.get("scientific_role"),
                "molecular_output_task": bool(dataset.get("molecular_output_task")),
                "accepted_molecules": len(sequences),
                "rejected_molecules": sum(rejects.values()),
                "rejects_by_error_prefix": dict(sorted(rejects.items())),
                "unique_motif_types": len(counts),
                "motif_occurrences": sum(counts.values()),
                "policies": {
                    name: {
                        "selected_macro_count": len(selected),
                        **_metrics(sequences, counts, selected),
                    }
                    for name, selected in policies.items()
                },
            }
        )

    p1_path = Path(args.phase1_registry).expanduser().resolve()
    p2_path = Path(args.phase2_registry).expanduser().resolve()
    chebi_path = Path(args.chebi_train_census).expanduser().resolve()
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "inputs": {
            "dataset_config": {
                "path": str(config_path),
                "sha256": _sha256_file(config_path),
            },
            "phase1_registry": {
                "path": str(p1_path),
                "sha256": _sha256_file(p1_path),
            },
            "phase2_registry": {
                "path": str(p2_path),
                "sha256": _sha256_file(p2_path),
            },
            "chebi_train_census": {
                "path": str(chebi_path),
                "sha256": _sha256_file(chebi_path),
            },
        },
        "selection": {
            "ranking_policy": args.ranking_policy,
            "base_budget": args.base_budget,
            "base_macro_count": len(base),
            "chebi_train_unique_motifs": len(chebi),
            "chebi_train_additions_outside_base": len(all_chebi - base),
            "all_chebi_macro_count": len(all_chebi),
            "phase1_unique_motifs": len(phase1),
            "phase2_unique_motifs": len(phase2),
        },
        "datasets": result_rows,
        "runtime": {
            "workers": args.workers,
            "wall_seconds": time.perf_counter() - started,
        },
        "contracts": {
            "phase1_and_phase2_train_select_general_base": True,
            "all_chebi20_train_motifs_are_registered_before_pretraining": True,
            "other_downstream_populations_are_evaluation_only": True,
            "validation_or_test_selects_tokens": False,
            "lossless_chemical_lexer_remains_mandatory": True,
            "coverage_does_not_imply_training_sufficiency": True,
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
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--phase1-registry", required=True)
    parser.add_argument("--phase2-registry", required=True)
    parser.add_argument("--chebi-train-census", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--base-budget", type=int, default=18000)
    parser.add_argument(
        "--ranking-policy",
        choices=tuple(RANKING_POLICIES),
        default="phase1_phase2_equal_stage_mass",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunksize", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(_parser().parse_args(argv))
    except Exception as exc:
        print(
            f"multistage downstream coverage failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
