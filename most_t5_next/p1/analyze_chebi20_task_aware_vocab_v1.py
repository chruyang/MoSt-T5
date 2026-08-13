#!/usr/bin/env python3
"""Compare full-pretraining and ChEBI-20-train-aware motif registries."""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

from most_t5_next.p1.build_registered_downstream_pure_motif_census_v1 import (
    _project_smiles,
)


SCHEMA_VERSION = "most-t5-next/chebi20-task-aware-vocab-analysis/v1"
DEFAULT_BUDGETS = "512,2048,4096,8192,12000,16000,24735,30080,32768"
DEFAULT_DOWNSTREAM_MIN_COUNTS = "2,5,8,16,32"


class ChEBITaskAwareVocabError(RuntimeError):
    """The registered train split cannot support the comparison."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pretrain_registry(path: Path) -> tuple[dict[str, int], tuple[str, ...]]:
    counts: dict[str, int] = {}
    ranked: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            pure = row.get("pure_motif")
            rank = row.get("rank")
            count = row.get("occurrences")
            if (
                not isinstance(pure, str)
                or rank != len(ranked)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                or pure in counts
            ):
                raise ChEBITaskAwareVocabError(
                    f"invalid pretraining registry row {line_number}"
                )
            ranked.append(pure)
            counts[pure] = count
    if not ranked:
        raise ChEBITaskAwareVocabError("pretraining registry is empty")
    return counts, tuple(ranked)


def _metrics(
    sequences: Sequence[Sequence[str]],
    counts: Mapping[str, int],
    selected: set[str],
) -> dict[str, object]:
    total = sum(counts.values())
    covered = sum(value for pure, value in counts.items() if pure in selected)
    fallback_counts = [sum(pure not in selected for pure in row) for row in sequences]
    fully = sum(value == 0 for value in fallback_counts)
    return {
        "macro_occurrence_coverage": covered / total,
        "fully_macro_tokenized_molecules": fully,
        "fully_macro_tokenized_molecule_rate": fully / len(sequences),
        "molecules_with_at_most_1_fallback_rate": sum(
            value <= 1 for value in fallback_counts
        )
        / len(sequences),
        "molecules_with_at_most_2_fallback_rate": sum(
            value <= 2 for value in fallback_counts
        )
        / len(sequences),
        "mean_fallback_motifs_per_molecule": sum(fallback_counts)
        / len(sequences),
    }


def _balanced_ranking(
    pretrain: Mapping[str, int], downstream: Mapping[str, int]
) -> tuple[str, ...]:
    pretrain_total = sum(pretrain.values())
    downstream_total = sum(downstream.values())
    return tuple(
        sorted(
            set(pretrain) | set(downstream),
            key=lambda pure: (
                -(
                    pretrain.get(pure, 0) * downstream_total
                    + downstream.get(pure, 0) * pretrain_total
                ),
                pure.encode("utf-8"),
            ),
        )
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ChEBITaskAwareVocabError("pandas and pyarrow are required") from exc

    source = Path(args.source_parquet).expanduser().resolve()
    registry_path = Path(args.pretrain_registry).expanduser().resolve()
    output = Path(args.output_report).expanduser().resolve()
    if output.exists():
        raise ChEBITaskAwareVocabError("output report must be absent")
    if args.split != "train":
        raise ChEBITaskAwareVocabError("only ChEBI-20 train may influence vocabulary")
    pretrain_counts, pretrain_ranked = _load_pretrain_registry(registry_path)
    frame = pd.read_parquet(source, columns=[args.id_column, args.smiles_column])
    tasks = [
        (int(index), str(row[args.id_column]), str(row[args.smiles_column]).strip())
        for index, row in frame.iterrows()
    ]
    if not tasks:
        raise ChEBITaskAwareVocabError("ChEBI-20 train is empty")
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    if args.workers == 1:
        results = map(_project_smiles, tasks)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context
        )
        results = executor.map(_project_smiles, tasks, chunksize=args.chunksize)
    sequences: list[tuple[str, ...]] = []
    downstream_counts: Counter[str] = Counter()
    try:
        for result in results:
            if result["error"] is not None:
                raise ChEBITaskAwareVocabError(
                    f"ChEBI projection rejected {result['record_id']}: {result['error']}"
                )
            motifs = result["pure_motifs"]
            if not isinstance(motifs, tuple) or not motifs:
                raise ChEBITaskAwareVocabError("ChEBI projection returned no motifs")
            sequences.append(motifs)
            downstream_counts.update(motifs)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    budgets = sorted({int(value) for value in args.budgets.split(",")})
    if not budgets or budgets[0] <= 0:
        raise ChEBITaskAwareVocabError("budgets must be positive")
    balanced = _balanced_ranking(pretrain_counts, downstream_counts)
    downstream_min_counts = sorted(
        {int(value) for value in args.downstream_min_counts.split(",")}
    )
    if not downstream_min_counts or downstream_min_counts[0] <= 1:
        raise ChEBITaskAwareVocabError(
            "downstream minimum counts must all be greater than one"
        )
    rows = []
    for requested in budgets:
        budget = min(requested, len(pretrain_ranked))
        pretrain_set = set(pretrain_ranked[:budget])
        balanced_set = set(balanced[: min(requested, len(balanced))])
        all_chebi_union = pretrain_set | set(downstream_counts)
        policies = [
            ("pretrain_only_top_k", pretrain_set),
            ("equal_corpus_mass_balanced_fixed_k", balanced_set),
            ("pretrain_top_k_plus_all_chebi20_train_types", all_chebi_union),
        ]
        policies.extend(
            (
                f"pretrain_top_k_plus_chebi20_train_count_ge_{minimum}",
                pretrain_set
                | {
                    pure
                    for pure, count in downstream_counts.items()
                    if count >= minimum
                },
            )
            for minimum in downstream_min_counts
        )
        for policy, selected in policies:
            rows.append(
                {
                    "policy": policy,
                    "requested_pretrain_budget": requested,
                    "selected_macro_count": len(selected),
                    "additional_rows_beyond_pretrain_top_k": len(selected - pretrain_set),
                    "selected_chebi_only_types": len(selected - set(pretrain_counts)),
                    "additional_untied_vocab_parameters": len(selected)
                    * args.hidden_size
                    * (1 if args.tie_word_embeddings else 2),
                    "chebi20_train": _metrics(
                        sequences, downstream_counts, selected
                    ),
                }
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "dataset": "ChEBI-20",
        "split": "train",
        "inputs": {
            "source_parquet": {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": _sha256_file(source),
            },
            "pretrain_registry": {
                "path": str(registry_path),
                "bytes": registry_path.stat().st_size,
                "sha256": _sha256_file(registry_path),
            },
        },
        "counts": {
            "molecules": len(sequences),
            "motif_occurrences": sum(downstream_counts.values()),
            "unique_motifs": len(downstream_counts),
            "types_seen_in_pretraining": len(set(downstream_counts) & set(pretrain_counts)),
            "types_absent_from_pretraining": len(set(downstream_counts) - set(pretrain_counts)),
            "occurrence_coverage_by_any_pretraining_type": sum(
                value
                for pure, value in downstream_counts.items()
                if pure in pretrain_counts
            )
            / sum(downstream_counts.values()),
            "motif_type_frequency": {
                "singletons": sum(value == 1 for value in downstream_counts.values()),
                "count_le_2": sum(value <= 2 for value in downstream_counts.values()),
                "count_le_5": sum(value <= 5 for value in downstream_counts.values()),
                "count_ge_8": sum(value >= 8 for value in downstream_counts.values()),
                "count_ge_16": sum(value >= 16 for value in downstream_counts.values()),
                "count_ge_32": sum(value >= 32 for value in downstream_counts.values()),
            },
        },
        "budget_rows": rows,
        "runtime": {
            "workers": args.workers,
            "wall_seconds": time.perf_counter() - started,
        },
        "contracts": {
            "validation_or_test_used_for_vocabulary": False,
            "all_chebi_union_is_task_aware_specialist": True,
            "all_chebi_union_requires_real_phase_i_training_exposure": True,
            "tokens_may_not_be_added_after_phase_i_starts": True,
            "lossless_lexer_remains_available_for_every_unselected_type": True,
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
    parser.add_argument("--source-parquet", required=True)
    parser.add_argument("--pretrain-registry", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--id-column", default="cid")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument(
        "--downstream-min-counts", default=DEFAULT_DOWNSTREAM_MIN_COUNTS
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--tie-word-embeddings", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(_parser().parse_args(argv))
    except Exception as exc:
        print(f"ChEBI task-aware analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
