#!/usr/bin/env python3
"""Replay candidate anchored-motif registries on registered downstream molecules."""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
from typing import Iterable, Mapping, Sequence

from most_t5_next.p1.build_registered_downstream_pure_motif_census_v1 import (
    _project_smiles,
)


SCHEMA_VERSION = "most-t5-next/downstream-motif-coverage/v1"
DEFAULT_DOWNSTREAM_MIN_COUNTS = "2,5,8,16,32"


class DownstreamMotifCoverageError(RuntimeError):
    """The registered dataset source or motif registry is inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pretrain_registry(path: Path) -> tuple[dict[str, int], tuple[str, ...]]:
    counts: dict[str, int] = {}
    ranking: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            pure = row.get("pure_motif")
            rank = row.get("rank")
            count = row.get("occurrences")
            if (
                not isinstance(pure, str)
                or rank != len(ranking)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                or pure in counts
            ):
                raise DownstreamMotifCoverageError(
                    f"invalid pretraining registry row {line_number}"
                )
            ranking.append(pure)
            counts[pure] = count
    if not ranking:
        raise DownstreamMotifCoverageError("pretraining registry is empty")
    return counts, tuple(ranking)


def _load_chebi_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            pure = row.get("pure_motif")
            count = row.get("train_occurrences")
            if (
                not isinstance(pure, str)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                or pure in counts
            ):
                raise DownstreamMotifCoverageError(
                    f"invalid ChEBI census row {line_number}"
                )
            counts[pure] = count
    if not counts:
        raise DownstreamMotifCoverageError("ChEBI census is empty")
    return counts


def _jsonl_smiles(path: Path, field: str) -> Iterable[tuple[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise DownstreamMotifCoverageError(
                    f"{path}: invalid {field!r} at row {line_number}"
                )
            record_id = row.get("member_id", row.get("record_id", line_number - 1))
            yield str(record_id), value.strip()


def _parquet_smiles(path: Path, field: str) -> Iterable[tuple[str, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DownstreamMotifCoverageError(
            "pyarrow is required for parquet sources"
        ) from exc
    table = pq.read_table(path, columns=[field])
    for index, value in enumerate(table.column(field).to_pylist()):
        if not isinstance(value, str) or not value.strip():
            raise DownstreamMotifCoverageError(
                f"{path}: invalid {field!r} at row {index + 1}"
            )
        yield str(index), value.strip()


def _csv_smiles(path: Path, field: str) -> Iterable[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or field not in reader.fieldnames:
            raise DownstreamMotifCoverageError(
                f"{path}: CSV has no {field!r} column"
            )
        for index, row in enumerate(reader):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise DownstreamMotifCoverageError(
                    f"{path}: invalid {field!r} at row {index + 2}"
                )
            yield str(index), value.strip()


def _source_smiles(source: Mapping[str, object]) -> Iterable[tuple[str, str]]:
    path_value = source.get("path")
    fmt = source.get("format")
    field = source.get("smiles_field")
    if not isinstance(path_value, str) or fmt not in {"csv", "jsonl", "parquet"}:
        raise DownstreamMotifCoverageError("source path/format is invalid")
    if not isinstance(field, str) or not field:
        raise DownstreamMotifCoverageError("source smiles_field is invalid")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise DownstreamMotifCoverageError(f"source is absent: {path}")
    if fmt == "jsonl":
        yield from _jsonl_smiles(path, field)
    elif fmt == "parquet":
        yield from _parquet_smiles(path, field)
    else:
        yield from _csv_smiles(path, field)


def _load_dataset(dataset: Mapping[str, object]) -> tuple[list[tuple[int, str, str]], dict]:
    name = dataset.get("name")
    sources = dataset.get("sources")
    if not isinstance(name, str) or not name or not isinstance(sources, list) or not sources:
        raise DownstreamMotifCoverageError("dataset name/sources is invalid")
    seen: set[str] = set()
    rows: list[tuple[int, str, str]] = []
    source_manifest = []
    input_rows = 0
    duplicates = 0
    for source in sources:
        if not isinstance(source, dict):
            raise DownstreamMotifCoverageError("dataset source must be an object")
        path = Path(str(source["path"])).expanduser().resolve()
        source_rows = 0
        for record_id, smiles in _source_smiles(source):
            source_rows += 1
            input_rows += 1
            if smiles in seen:
                duplicates += 1
                continue
            seen.add(smiles)
            rows.append((len(rows), f"{name}:{record_id}", smiles))
        source_manifest.append(
            {
                "path": str(path),
                "format": source["format"],
                "smiles_field": source["smiles_field"],
                "rows": source_rows,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows, {
        "name": name,
        "sources": source_manifest,
        "input_rows": input_rows,
        "unique_smiles": len(rows),
        "duplicate_smiles_rows": duplicates,
    }


def _metrics(
    sequences: Sequence[Sequence[str]], counts: Mapping[str, int], selected: set[str]
) -> dict[str, object]:
    total = sum(counts.values())
    covered = sum(value for pure, value in counts.items() if pure in selected)
    fallback_counts = [sum(pure not in selected for pure in row) for row in sequences]
    return {
        "macro_occurrence_coverage": covered / total,
        "fully_macro_tokenized_molecules": sum(value == 0 for value in fallback_counts),
        "fully_macro_tokenized_molecule_rate": sum(
            value == 0 for value in fallback_counts
        )
        / len(sequences),
        "molecules_with_at_most_1_fallback_rate": sum(
            value <= 1 for value in fallback_counts
        )
        / len(sequences),
        "mean_fallback_motifs_per_molecule": sum(fallback_counts) / len(sequences),
        "uncovered_motif_types": sum(pure not in selected for pure in counts),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    config_path = Path(args.dataset_config).expanduser().resolve()
    registry_path = Path(args.pretrain_registry).expanduser().resolve()
    chebi_path = Path(args.chebi_train_census).expanduser().resolve()
    output = Path(args.output_report).expanduser().resolve()
    if output.exists():
        raise DownstreamMotifCoverageError("output report must be absent")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise DownstreamMotifCoverageError("dataset config has no datasets")
    pretrain_counts, pretrain_ranking = _load_pretrain_registry(registry_path)
    chebi_counts = _load_chebi_counts(chebi_path)
    top_k = set(pretrain_ranking[: args.pretrain_budget])
    if len(top_k) != args.pretrain_budget:
        raise DownstreamMotifCoverageError("pretraining registry is smaller than budget")
    minimum_counts = sorted(
        {int(value) for value in args.downstream_min_counts.split(",")}
    )
    policies = {
        "pretrain_top_k": top_k,
        "pretrain_top_k_plus_all_chebi20_train_types": top_k | set(chebi_counts),
    }
    policies.update(
        {
            f"pretrain_top_k_plus_chebi20_train_count_ge_{minimum}": top_k
            | {pure for pure, count in chebi_counts.items() if count >= minimum}
            for minimum in minimum_counts
        }
    )
    started = time.perf_counter()
    result_rows = []
    context = multiprocessing.get_context("spawn")
    for dataset in datasets:
        tasks, dataset_manifest = _load_dataset(dataset)
        if not tasks:
            raise DownstreamMotifCoverageError(
                f"dataset {dataset_manifest['name']} has no unique molecules"
            )
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context
        )
        sequences: list[tuple[str, ...]] = []
        counts: Counter[str] = Counter()
        rejects: Counter[str] = Counter()
        try:
            for result in executor.map(_project_smiles, tasks, chunksize=args.chunksize):
                if result["error"] is not None:
                    rejects[str(result["error"]).split(":", 1)[0]] += 1
                    continue
                motifs = result["pure_motifs"]
                if not isinstance(motifs, tuple) or not motifs:
                    rejects["EMPTY_MOTIF_SEQUENCE"] += 1
                    continue
                sequences.append(motifs)
                counts.update(motifs)
        finally:
            executor.shutdown(wait=True)
        if not sequences:
            raise DownstreamMotifCoverageError(
                f"dataset {dataset_manifest['name']} has no accepted molecules"
            )
        result_rows.append(
            {
                "dataset": dataset_manifest,
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
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "inputs": {
            "dataset_config": {
                "path": str(config_path),
                "sha256": _sha256_file(config_path),
            },
            "pretrain_registry": {
                "path": str(registry_path),
                "sha256": _sha256_file(registry_path),
            },
            "chebi_train_census": {
                "path": str(chebi_path),
                "sha256": _sha256_file(chebi_path),
            },
        },
        "pretrain_budget": args.pretrain_budget,
        "datasets": result_rows,
        "runtime": {
            "workers": args.workers,
            "wall_seconds": time.perf_counter() - started,
        },
        "contracts": {
            "only_chebi20_train_influences_task_aware_registry": True,
            "other_downstream_splits_are_evaluation_only": True,
            "duplicate_smiles_are_evaluated_once_per_dataset": True,
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
    parser.add_argument("--pretrain-registry", required=True)
    parser.add_argument("--chebi-train-census", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--pretrain-budget", type=int, default=16000)
    parser.add_argument(
        "--downstream-min-counts", default=DEFAULT_DOWNSTREAM_MIN_COUNTS
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunksize", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(_parser().parse_args(argv))
    except Exception as exc:
        print(
            f"downstream motif coverage failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
