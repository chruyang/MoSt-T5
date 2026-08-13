#!/usr/bin/env python3
"""Replay task-aware motif registries on the full pretraining molecule cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence


SCHEMA_VERSION = "most-t5-next/task-aware-registry-full-cache-analysis/v1"
DEFAULT_BUDGETS = "512,2048,4096,8192,12000,16000,24735,30080,32768"
DEFAULT_DOWNSTREAM_MIN_COUNTS = "2,5,8,16,32"


class TaskAwareRegistryAnalysisError(RuntimeError):
    """The cache or one of the train-only registries is inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pretrain(path: Path) -> tuple[dict[str, tuple[int, int]], tuple[str, ...]]:
    rows: dict[str, tuple[int, int]] = {}
    ranking: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            pure = row.get("pure_motif")
            pure_id = row.get("pure_motif_id")
            rank = row.get("rank")
            count = row.get("occurrences")
            if (
                not isinstance(pure, str)
                or rank != len(ranking)
                or isinstance(pure_id, bool)
                or not isinstance(pure_id, int)
                or pure_id < 0
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                or pure in rows
            ):
                raise TaskAwareRegistryAnalysisError(
                    f"invalid pretrain registry row {line_number}"
                )
            ranking.append(pure)
            rows[pure] = (pure_id, count)
    if not ranking:
        raise TaskAwareRegistryAnalysisError("pretrain registry is empty")
    return rows, tuple(ranking)


def _load_downstream(path: Path) -> dict[str, int]:
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
                raise TaskAwareRegistryAnalysisError(
                    f"invalid downstream census row {line_number}"
                )
            counts[pure] = count
    if not counts:
        raise TaskAwareRegistryAnalysisError("downstream census is empty")
    return counts


def _balanced_ranking(
    pretrain: Mapping[str, tuple[int, int]], downstream: Mapping[str, int]
) -> tuple[str, ...]:
    pretrain_total = sum(count for _pure_id, count in pretrain.values())
    downstream_total = sum(downstream.values())
    return tuple(
        sorted(
            set(pretrain) | set(downstream),
            key=lambda pure: (
                -(
                    pretrain.get(pure, (-1, 0))[1] * downstream_total
                    + downstream.get(pure, 0) * pretrain_total
                ),
                pure.encode("utf-8"),
            ),
        )
    )


def _evaluate(cache: Path, selected_ids: set[int], pure_id_domain: int) -> dict[str, object]:
    import numpy as np

    selected = np.zeros(pure_id_domain, dtype=np.bool_)
    if selected_ids:
        selected[np.asarray(sorted(selected_ids), dtype=np.uint32)] = True
    records = 0
    motifs = 0
    covered = 0
    fully = 0
    fallback_le = {1: 0, 2: 0, 5: 0}
    fallback_total = 0
    for motif_path in sorted(cache.glob("shard-*.motif_ids.u32")):
        stem = motif_path.name[: -len(".motif_ids.u32")]
        ids = np.fromfile(motif_path, dtype="<u4")
        offsets = np.fromfile(cache / f"{stem}.offsets.u64", dtype="<u8")
        if len(offsets) == 1:
            continue
        starts = offsets[:-1].astype(np.int64, copy=False)
        fallback = ~selected[ids]
        fallback_counts = np.add.reduceat(fallback.astype(np.uint32), starts)
        records += len(fallback_counts)
        motifs += len(ids)
        covered += int(np.count_nonzero(~fallback))
        fully += int(np.count_nonzero(fallback_counts == 0))
        fallback_total += int(fallback_counts.sum())
        for threshold in fallback_le:
            fallback_le[threshold] += int(np.count_nonzero(fallback_counts <= threshold))
    if records <= 0 or motifs <= 0:
        raise TaskAwareRegistryAnalysisError("full cache is empty")
    return {
        "records": records,
        "motif_occurrences": motifs,
        "macro_occurrence_coverage": covered / motifs,
        "fully_macro_tokenized_molecules": fully,
        "fully_macro_tokenized_molecule_rate": fully / records,
        "molecules_with_at_most_1_fallback_rate": fallback_le[1] / records,
        "molecules_with_at_most_2_fallback_rate": fallback_le[2] / records,
        "molecules_with_at_most_5_fallback_rate": fallback_le[5] / records,
        "mean_fallback_motifs_per_molecule": fallback_total / records,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    cache = Path(args.cache_dir).expanduser().resolve()
    pretrain_path = Path(args.pretrain_registry).expanduser().resolve()
    downstream_path = Path(args.downstream_train_census).expanduser().resolve()
    output = Path(args.output_report).expanduser().resolve()
    if output.exists():
        raise TaskAwareRegistryAnalysisError("output report must be absent")
    pretrain, pretrain_ranking = _load_pretrain(pretrain_path)
    downstream = _load_downstream(downstream_path)
    balanced = _balanced_ranking(pretrain, downstream)
    pure_id_domain = max(pure_id for pure_id, _count in pretrain.values()) + 1
    budgets = sorted({int(value) for value in args.budgets.split(",")})
    if not budgets or budgets[0] <= 0:
        raise TaskAwareRegistryAnalysisError("budgets must be positive")
    downstream_min_counts = sorted(
        {int(value) for value in args.downstream_min_counts.split(",")}
    )
    if not downstream_min_counts or downstream_min_counts[0] <= 1:
        raise TaskAwareRegistryAnalysisError(
            "downstream minimum counts must all be greater than one"
        )
    rows = []
    for requested in budgets:
        pretrain_set = set(pretrain_ranking[: min(requested, len(pretrain_ranking))])
        balanced_set = set(balanced[: min(requested, len(balanced))])
        union_set = pretrain_set | set(downstream)
        policies = [
            ("pretrain_only_top_k", pretrain_set),
            ("equal_corpus_mass_balanced_fixed_k", balanced_set),
            ("pretrain_top_k_plus_all_downstream_train_types", union_set),
        ]
        policies.extend(
            (
                f"pretrain_top_k_plus_downstream_train_count_ge_{minimum}",
                pretrain_set
                | {pure for pure, count in downstream.items() if count >= minimum},
            )
            for minimum in downstream_min_counts
        )
        for policy, selected_values in policies:
            selected_ids = {
                pretrain[pure][0] for pure in selected_values if pure in pretrain
            }
            rows.append(
                {
                    "policy": policy,
                    "requested_pretrain_budget": requested,
                    "selected_macro_count": len(selected_values),
                    "selected_pretrain_types": len(selected_ids),
                    "selected_downstream_only_types": len(
                        selected_values - set(pretrain)
                    ),
                    "displaced_pretrain_top_k_types": len(
                        pretrain_set - selected_values
                    ),
                    "additional_untied_vocab_parameters": len(selected_values)
                    * args.hidden_size
                    * (1 if args.tie_word_embeddings else 2),
                    "pretraining": _evaluate(
                        cache, selected_ids, pure_id_domain
                    ),
                }
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "inputs": {
            "cache_dir": str(cache),
            "pretrain_registry": {
                "path": str(pretrain_path),
                "sha256": _sha256_file(pretrain_path),
            },
            "downstream_train_census": {
                "path": str(downstream_path),
                "sha256": _sha256_file(downstream_path),
            },
        },
        "budget_rows": rows,
        "contracts": {
            "downstream_validation_or_test_used": False,
            "balanced_policy_is_task_aware_specialist": True,
            "full_pretraining_cache_replayed_for_every_policy": True,
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
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pretrain-registry", required=True)
    parser.add_argument("--downstream-train-census", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument(
        "--downstream-min-counts", default=DEFAULT_DOWNSTREAM_MIN_COUNTS
    )
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--tie-word-embeddings", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(_parser().parse_args(argv))
    except Exception as exc:
        print(f"task-aware full-cache analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
