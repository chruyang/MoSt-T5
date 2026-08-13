#!/usr/bin/env python3
"""Compare pretrain-only and registered-downstream-aware motif macro policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from most_t5_next.r1.tokenizer.stereo_free_motif_chemical_lexer_v1 import (
    lex_pure_motif,
)


SCHEMA_VERSION = "most-t5-next/anchored-macro-vocab-analysis/v2"


class AnchoredMacroVocabAnalysisError(RuntimeError):
    """The declared motif censuses cannot support a deterministic comparison."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_census(path: Path, fields: Sequence[str]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            pure = row.get("pure_motif")
            digest = row.get("pure_motif_sha256")
            if (
                not isinstance(pure, str)
                or not isinstance(digest, str)
                or hashlib.sha256(pure.encode("utf-8")).hexdigest() != digest
                or pure in result
            ):
                raise AnchoredMacroVocabAnalysisError(
                    f"invalid or duplicate census row at {path.name}:{line_number}"
                )
            counts = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise AnchoredMacroVocabAnalysisError(
                        f"invalid {field} at {path.name}:{line_number}"
                    )
                counts[field] = value
            result[pure] = counts
    if not result:
        raise AnchoredMacroVocabAnalysisError("motif census is empty")
    return result


def _rank_pretrain(
    pretrain: Mapping[str, Mapping[str, int]]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            pretrain,
            key=lambda pure: (
                -pretrain[pure]["train_occurrences"],
                pure.encode("utf-8"),
            ),
        )
    )


def _rank_balanced(
    pretrain: Mapping[str, Mapping[str, int]],
    downstream: Mapping[str, Mapping[str, int]],
) -> tuple[str, ...]:
    pretrain_total = sum(row["train_occurrences"] for row in pretrain.values())
    downstream_total = sum(row["train_occurrences"] for row in downstream.values())
    universe = set(pretrain) | set(downstream)
    # Integer cross multiplication gives the sum of corpus-normalized
    # frequencies without floating-point ranking drift.
    return tuple(
        sorted(
            universe,
            key=lambda pure: (
                -(
                    pretrain.get(pure, {}).get("train_occurrences", 0)
                    * downstream_total
                    + downstream.get(pure, {}).get("train_occurrences", 0)
                    * pretrain_total
                ),
                pure.encode("utf-8"),
            ),
        )
    )


def _corpus_metrics(
    corpus: Mapping[str, Mapping[str, int]], field: str, macro_set: set[str]
) -> dict[str, float | int]:
    total = sum(row[field] for row in corpus.values())
    macro_occurrences = sum(
        row[field] for pure, row in corpus.items() if pure in macro_set
    )
    lexical_tokens = sum(
        row[field] * (1 if pure in macro_set else len(lex_pure_motif(pure).tokens))
        for pure, row in corpus.items()
    )
    covered_types = sum(pure in macro_set for pure in corpus)
    return {
        "occurrences": total,
        "macro_occurrences": macro_occurrences,
        "macro_occurrence_coverage": macro_occurrences / total,
        "covered_types": covered_types,
        "type_coverage": covered_types / len(corpus),
        "mean_identity_tokens_per_motif": lexical_tokens / total,
    }


def analyze(args: argparse.Namespace) -> dict[str, object]:
    pretrain_path = Path(args.pretrain_census).expanduser().resolve()
    downstream_path = Path(args.downstream_train_census).expanduser().resolve()
    output = Path(args.output_report).expanduser().resolve()
    if output.exists():
        raise AnchoredMacroVocabAnalysisError("output report must be absent")
    pretrain = _load_census(
        pretrain_path, ("train_occurrences", "dev_occurrences")
    )
    downstream = _load_census(downstream_path, ("train_occurrences",))
    pretrain_rank = _rank_pretrain(pretrain)
    balanced_rank = _rank_balanced(pretrain, downstream)
    budgets = tuple(sorted(set(int(value) for value in args.budgets.split(","))))
    if not budgets or budgets[0] <= 0:
        raise AnchoredMacroVocabAnalysisError("macro budgets must be positive")

    rows = []
    for budget in budgets:
        for policy, ranking in (
            ("pretrain_train_only", pretrain_rank),
            ("balanced_pretrain_plus_registered_downstream_train", balanced_rank),
        ):
            selected = set(ranking[: min(budget, len(ranking))])
            rows.append(
                {
                    "policy": policy,
                    "requested_budget": budget,
                    "selected_macro_count": len(selected),
                    "selected_downstream_only_types": len(selected - set(pretrain)),
                    "additional_input_embedding_parameters": len(selected)
                    * args.hidden_size,
                    "additional_output_projection_parameters": (
                        0 if args.tie_word_embeddings else len(selected) * args.hidden_size
                    ),
                    "additional_total_vocab_parameters": len(selected)
                    * args.hidden_size
                    * (1 if args.tie_word_embeddings else 2),
                    "pretrain_train": _corpus_metrics(
                        pretrain, "train_occurrences", selected
                    ),
                    "pretrain_dev": _corpus_metrics(
                        pretrain, "dev_occurrences", selected
                    ),
                    "registered_downstream_train": _corpus_metrics(
                        downstream, "train_occurrences", selected
                    ),
                }
            )

    intersection = set(pretrain) & set(downstream)
    policy_comparisons = []
    for budget in budgets:
        pretrain_selected = pretrain_rank[: min(budget, len(pretrain_rank))]
        balanced_selected = balanced_rank[: min(budget, len(balanced_rank))]
        policy_comparisons.append(
            {
                "requested_budget": budget,
                "identity_set_intersection": len(
                    set(pretrain_selected) & set(balanced_selected)
                ),
                "same_identity_at_rank": sum(
                    left == right
                    for left, right in zip(pretrain_selected, balanced_selected)
                ),
                "macro_surface_sequence_is_identical": True,
                "checkpoint_may_switch_semantic_registry": False,
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "inputs": {
            "pretrain_census": {
                "path": str(pretrain_path),
                "sha256": _sha256_file(pretrain_path),
            },
            "registered_downstream_train_census": {
                "path": str(downstream_path),
                "sha256": _sha256_file(downstream_path),
            },
        },
        "overlap": {
            "pretrain_unique": len(pretrain),
            "downstream_train_unique": len(downstream),
            "intersection": len(intersection),
            "downstream_train_only": len(set(downstream) - set(pretrain)),
            "downstream_occurrence_coverage_by_any_pretrain_type": (
                sum(
                    row["train_occurrences"]
                    for pure, row in downstream.items()
                    if pure in pretrain
                )
                / sum(row["train_occurrences"] for row in downstream.values())
            ),
        },
        "macro_budget_rows": rows,
        "policy_comparisons": policy_comparisons,
        "model_vocabulary_cost": {
            "hidden_size": args.hidden_size,
            "tie_word_embeddings": args.tie_word_embeddings,
            "untied_lm_head_counted": not args.tie_word_embeddings,
        },
        "contracts": {
            "validation_or_test_influences_ranking": False,
            "pretrain_policy_uses_only_pretrain_train_frequency": True,
            "balanced_policy_uses_equal_corpus_mass_not_raw_count_pooling": True,
            "chemical_lexer_remains_lossless_for_every_unselected_type": True,
            "registered_downstream_policy_is_task_aware_specialist": True,
            "no_tokens_may_be_added_after_phase_i_starts": True,
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
    parser.add_argument("--pretrain-census", required=True)
    parser.add_argument("--downstream-train-census", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--budgets", default="64,128,256,512,1024,2048")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument(
        "--tie-word-embeddings",
        action="store_true",
        help="Set only for backbones whose input embedding and LM head are tied.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = analyze(_parser().parse_args(argv))
    except Exception as exc:
        print(f"macro vocabulary analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
