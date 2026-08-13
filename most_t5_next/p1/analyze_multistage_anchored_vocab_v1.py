#!/usr/bin/env python3
"""Compare Phase-I/Phase-II joint motif registries and ChEBI-train additions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence


SCHEMA_VERSION = "most-t5-next/multistage-anchored-vocab-analysis/v1"
DEFAULT_BUDGETS = "12000,16000,18048,24735"
DEFAULT_CHEBI_ADDITIONS = "0,1024,2048"


class MultistageVocabError(RuntimeError):
    """The staged vocabulary evidence is incomplete or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_registry(path: Path) -> tuple[dict[str, tuple[int, int]], tuple[str, ...]]:
    rows: dict[str, tuple[int, int]] = {}
    ranking: list[str] = []
    ids: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            pure = row.get("pure_motif")
            pure_id = row.get("pure_motif_id")
            count = row.get("occurrences")
            rank = row.get("rank")
            if (
                not isinstance(pure, str)
                or not pure
                or isinstance(pure_id, bool)
                or not isinstance(pure_id, int)
                or pure_id < 0
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                or rank != len(ranking)
                or pure in rows
                or pure_id in ids
            ):
                raise MultistageVocabError(f"invalid registry row {line_number}: {path}")
            rows[pure] = (pure_id, count)
            ids.add(pure_id)
            ranking.append(pure)
    if not rows:
        raise MultistageVocabError(f"registry is empty: {path}")
    return rows, tuple(ranking)


def _load_chebi(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            pure = row.get("pure_motif")
            count = row.get("train_occurrences")
            if (
                not isinstance(pure, str)
                or not pure
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                or pure in counts
            ):
                raise MultistageVocabError(f"invalid ChEBI census row {line_number}")
            counts[pure] = count
    return counts


def _pooled_ranking(
    phase1: Mapping[str, tuple[int, int]],
    phase2: Mapping[str, tuple[int, int]],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(phase1) | set(phase2),
            key=lambda pure: (
                -(phase1.get(pure, (-1, 0))[1] + phase2.get(pure, (-1, 0))[1]),
                pure.encode("utf-8"),
            ),
        )
    )


def _equal_stage_ranking(
    phase1: Mapping[str, tuple[int, int]],
    phase2: Mapping[str, tuple[int, int]],
) -> tuple[str, ...]:
    """Rank by count/P1_total + count/P2_total without float arithmetic."""

    total1 = sum(count for _pure_id, count in phase1.values())
    total2 = sum(count for _pure_id, count in phase2.values())
    return tuple(
        sorted(
            set(phase1) | set(phase2),
            key=lambda pure: (
                -(
                    phase1.get(pure, (-1, 0))[1] * total2
                    + phase2.get(pure, (-1, 0))[1] * total1
                ),
                pure.encode("utf-8"),
            ),
        )
    )


def _cache_pairs(cache: Path):
    import numpy as np

    shard_ids = sorted(cache.glob("shard-*.motif_ids.u32"))
    if shard_ids:
        for ids_path in shard_ids:
            stem = ids_path.name[: -len(".motif_ids.u32")]
            yield (
                np.fromfile(ids_path, dtype="<u4"),
                np.fromfile(cache / f"{stem}.offsets.u64", dtype="<u8"),
            )
        return
    ids_path = cache / "motif_ids.u32"
    offsets_path = cache / "offsets.u64"
    if ids_path.is_file() and offsets_path.is_file():
        yield np.fromfile(ids_path, dtype="<u4"), np.fromfile(offsets_path, dtype="<u8")
        return
    raise MultistageVocabError(f"compact cache is absent: {cache}")


def _evaluate_cache(
    cache: Path,
    registry: Mapping[str, tuple[int, int]],
    selected: set[str],
) -> dict[str, object]:
    import numpy as np

    domain = max(pure_id for pure_id, _count in registry.values()) + 1
    selected_mask = np.zeros(domain, dtype=np.bool_)
    local_ids = [registry[pure][0] for pure in selected if pure in registry]
    if local_ids:
        selected_mask[np.asarray(local_ids, dtype=np.uint32)] = True
    records = motifs = covered = fully = fallback_total = 0
    fallback_le = {1: 0, 2: 0, 5: 0}
    for ids, offsets in _cache_pairs(cache):
        if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(ids):
            raise MultistageVocabError(f"cache offsets are inconsistent: {cache}")
        if len(ids) and int(ids.max()) >= domain:
            raise MultistageVocabError(f"cache ID exceeds registry domain: {cache}")
        starts = offsets[:-1].astype(np.int64, copy=False)
        fallback = ~selected_mask[ids]
        fallback_counts = np.add.reduceat(fallback.astype(np.uint32), starts)
        records += len(fallback_counts)
        motifs += len(ids)
        covered += int(np.count_nonzero(~fallback))
        fully += int(np.count_nonzero(fallback_counts == 0))
        fallback_total += int(fallback_counts.sum())
        for threshold in fallback_le:
            fallback_le[threshold] += int(np.count_nonzero(fallback_counts <= threshold))
    if records <= 0 or motifs <= 0:
        raise MultistageVocabError(f"cache contains no records: {cache}")
    return {
        "records": records,
        "motif_occurrences": motifs,
        "macro_occurrence_coverage": covered / motifs,
        "fully_macro_tokenized_molecule_rate": fully / records,
        "molecules_with_at_most_1_fallback_rate": fallback_le[1] / records,
        "molecules_with_at_most_2_fallback_rate": fallback_le[2] / records,
        "molecules_with_at_most_5_fallback_rate": fallback_le[5] / records,
        "mean_fallback_motifs_per_molecule": fallback_total / records,
    }


def _count_metrics(counts: Mapping[str, int], selected: set[str]) -> dict[str, object]:
    total = sum(counts.values())
    if total <= 0:
        return {"motif_occurrences": 0, "macro_occurrence_coverage": None}
    return {
        "motif_occurrences": total,
        "macro_occurrence_coverage": sum(
            count for pure, count in counts.items() if pure in selected
        )
        / total,
        "selected_types": len(set(counts) & selected),
        "total_types": len(counts),
    }


def _parse_positive_csv(value: str, *, allow_zero: bool = False) -> tuple[int, ...]:
    values = tuple(sorted({int(item) for item in value.split(",")}))
    minimum = 0 if allow_zero else 1
    if not values or values[0] < minimum or (not allow_zero and values[0] == 0):
        raise MultistageVocabError("invalid numeric candidate list")
    return values


def run(args: argparse.Namespace) -> dict[str, object]:
    p1_registry_path = Path(args.phase1_registry).expanduser().resolve()
    p2_registry_path = Path(args.phase2_registry).expanduser().resolve()
    p1_cache = Path(args.phase1_cache).expanduser().resolve()
    p2_cache = Path(args.phase2_cache).expanduser().resolve()
    chebi_path = (
        Path(args.chebi_train_census).expanduser().resolve()
        if args.chebi_train_census
        else None
    )
    output = Path(args.output_report).expanduser().resolve()
    if output.exists():
        raise MultistageVocabError("output report must be absent")
    phase1, phase1_ranking = _load_registry(p1_registry_path)
    phase2, _phase2_ranking = _load_registry(p2_registry_path)
    chebi = _load_chebi(chebi_path)
    rankings = {
        "phase1_only_frequency": phase1_ranking,
        "phase1_phase2_raw_pooled_frequency": _pooled_ranking(phase1, phase2),
        "phase1_phase2_equal_stage_mass": _equal_stage_ranking(phase1, phase2),
    }
    budgets = _parse_positive_csv(args.budgets)
    additions = _parse_positive_csv(args.chebi_additions, allow_zero=True)
    chebi_ranked = tuple(
        sorted(chebi, key=lambda pure: (-chebi[pure], pure.encode("utf-8")))
    )
    rows: list[dict[str, object]] = []
    for ranking_policy, ranking in rankings.items():
        for requested in budgets:
            base = set(ranking[: min(requested, len(ranking))])
            absent_chebi = tuple(pure for pure in chebi_ranked if pure not in base)
            for addition in additions:
                selected = base | set(absent_chebi[:addition])
                selected_additions = selected - base
                pretraining_exposures = sorted(
                    phase1.get(pure, (-1, 0))[1] + phase2.get(pure, (-1, 0))[1]
                    for pure in selected
                )
                chebi_addition_exposures = sorted(
                    chebi[pure] for pure in selected_additions
                )
                rows.append(
                    {
                        "ranking_policy": ranking_policy,
                        "requested_base_budget": requested,
                        "selected_base_count": len(base),
                        "requested_chebi_train_additions": addition,
                        "selected_chebi_train_additions": len(selected - base),
                        "selected_macro_count": len(selected),
                        "selected_pretraining_exposure": {
                            "minimum": min(pretraining_exposures),
                            "median": pretraining_exposures[
                                len(pretraining_exposures) // 2
                            ],
                            "types_below_8": sum(
                                value < 8 for value in pretraining_exposures
                            ),
                            "types_absent_from_both_pretraining_stages": sum(
                                value == 0 for value in pretraining_exposures
                            ),
                        },
                        "chebi_addition_train_exposure": (
                            {
                                "minimum": min(chebi_addition_exposures),
                                "median": chebi_addition_exposures[
                                    len(chebi_addition_exposures) // 2
                                ],
                                "types_below_2": sum(
                                    value < 2 for value in chebi_addition_exposures
                                ),
                                "types_below_8": sum(
                                    value < 8 for value in chebi_addition_exposures
                                ),
                            }
                            if chebi_addition_exposures
                            else None
                        ),
                        "additional_untied_vocab_parameters": len(selected)
                        * args.hidden_size
                        * (1 if args.tie_word_embeddings else 2),
                        "phase1": _evaluate_cache(p1_cache, phase1, selected),
                        "phase2": _evaluate_cache(p2_cache, phase2, selected),
                        "chebi20_train": _count_metrics(chebi, selected),
                    }
                )
    overlap = set(phase1) & set(phase2)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "phase1_phase2_train_joint_general_registry_with_optional_chebi_train_additions",
        "inputs": {
            "phase1_registry": {
                "path": str(p1_registry_path),
                "sha256": _sha256_file(p1_registry_path),
            },
            "phase2_registry": {
                "path": str(p2_registry_path),
                "sha256": _sha256_file(p2_registry_path),
            },
            "phase1_cache": str(p1_cache),
            "phase2_cache": str(p2_cache),
            "chebi20_train_census": (
                {"path": str(chebi_path), "sha256": _sha256_file(chebi_path)}
                if chebi_path is not None
                else None
            ),
        },
        "corpus_counts": {
            "phase1_motif_occurrences": sum(count for _id, count in phase1.values()),
            "phase2_motif_occurrences": sum(count for _id, count in phase2.values()),
            "phase1_unique_motifs": len(phase1),
            "phase2_unique_motifs": len(phase2),
            "shared_motif_types": len(overlap),
            "phase2_only_motif_types": len(set(phase2) - set(phase1)),
            "chebi_train_unique_motifs": len(chebi),
        },
        "budget_rows": rows,
        "contracts": {
            "phase1_and_phase2_train_both_influence_general_registry": True,
            "phase2_text_content_influences_motif_ranking": False,
            "validation_or_test_influences_registry": False,
            "equal_stage_mass_is_schedule_neutral_default_candidate": True,
            "raw_pooled_frequency_is_reported_as_sensitivity_analysis": True,
            "chebi_additions_are_task_aware_specialist_rows": True,
            "lossless_chemical_lexer_remains_mandatory": True,
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
    parser.add_argument("--phase1-cache", required=True)
    parser.add_argument("--phase2-registry", required=True)
    parser.add_argument("--phase2-cache", required=True)
    parser.add_argument("--chebi-train-census")
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--chebi-additions", default=DEFAULT_CHEBI_ADDITIONS)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--tie-word-embeddings", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(_parser().parse_args(argv))
    except Exception as exc:
        print(f"multistage vocabulary analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
