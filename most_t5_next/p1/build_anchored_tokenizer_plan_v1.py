#!/usr/bin/env python3
"""Build deterministic candidate token registries for anchored motif text.

This utility does not mutate or save a Hugging Face tokenizer.  It freezes the
exact ordinary-token additions that a later phrase-boundary experiment may
instantiate, while keeping the scientific choices (macro source policy and
boundary mode) explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from most_t5_next.r1.tokenizer.stereo_free_anchored_motif_surface_v1 import (
    anchor_token,
)
from most_t5_next.r1.tokenizer.anchored_motif_model_surface_v1 import (
    FALLBACK_MOTIF_SUFFIX,
    FORMAL_BOUNDARY_MODES,
    FROZEN_GENERATIVE_BOUNDARY_MODE,
    frozen_grammar_contract,
)
from most_t5_next.r1.tokenizer.stereo_free_motif_chemical_lexer_v1 import (
    LEXER_SCHEMA_VERSION,
    opaque_chemical_token_map,
)


SCHEMA_VERSION = "most-t5-next/anchored-tokenizer-plan/v2"
MACRO_POLICIES = (
    "pretrain_train_only",
    "balanced_pretrain_plus_registered_downstream_train",
)
BOUNDARY_MODES = FORMAL_BOUNDARY_MODES


class AnchoredTokenizerPlanError(RuntimeError):
    """Candidate tokenizer inputs or registries are inconsistent."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_census(path: Path, required_fields: Sequence[str]) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            pure = row.get("pure_motif")
            digest = row.get("pure_motif_sha256")
            if (
                not isinstance(pure, str)
                or not isinstance(digest, str)
                or _sha256_bytes(pure.encode("utf-8")) != digest
                or pure in rows
            ):
                raise AnchoredTokenizerPlanError(
                    f"invalid or duplicate census row at {path.name}:{line_number}"
                )
            counts: dict[str, int] = {}
            for field in required_fields:
                value = row.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise AnchoredTokenizerPlanError(
                        f"invalid {field} at {path.name}:{line_number}"
                    )
                counts[field] = value
            rows[pure] = counts
    if not rows:
        raise AnchoredTokenizerPlanError("motif census is empty")
    return rows


def _rank_pretrain(pretrain: Mapping[str, Mapping[str, int]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            pretrain,
            key=lambda pure: (-pretrain[pure]["train_occurrences"], pure.encode("utf-8")),
        )
    )


def _rank_balanced(
    pretrain: Mapping[str, Mapping[str, int]],
    downstream: Mapping[str, Mapping[str, int]],
) -> tuple[str, ...]:
    pretrain_total = sum(row["train_occurrences"] for row in pretrain.values())
    downstream_total = sum(row["train_occurrences"] for row in downstream.values())
    return tuple(
        sorted(
            set(pretrain) | set(downstream),
            key=lambda pure: (
                -(
                    pretrain.get(pure, {}).get("train_occurrences", 0) * downstream_total
                    + downstream.get(pure, {}).get("train_occurrences", 0) * pretrain_total
                ),
                pure.encode("utf-8"),
            ),
        )
    )


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def build(args: argparse.Namespace) -> dict[str, object]:
    pretrain_path = Path(args.pretrain_census).expanduser().resolve()
    downstream_path = Path(args.downstream_train_census).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise AnchoredTokenizerPlanError("output directory must be absent")
    if args.macro_budget <= 0 or args.max_anchor_id < 0 or args.base_vocab_size <= 0:
        raise AnchoredTokenizerPlanError("numeric plan parameters must be positive")

    pretrain = _load_census(pretrain_path, ("train_occurrences", "dev_occurrences"))
    downstream = _load_census(downstream_path, ("train_occurrences",))
    rankings = {
        "pretrain_train_only": _rank_pretrain(pretrain),
        "balanced_pretrain_plus_registered_downstream_train": _rank_balanced(
            pretrain, downstream
        ),
    }
    output.mkdir(parents=True)

    chemical_rows = tuple(
        {
            "rank": rank,
            "logical_token": logical,
            "surface_token": surface,
        }
        for rank, (logical, surface) in enumerate(opaque_chemical_token_map())
    )
    anchor_rows = tuple(
        {
            "anchor_id": anchor_id,
            "surface_token": anchor_token(anchor_id),
        }
        for anchor_id in range(args.max_anchor_id + 1)
    )
    _write_jsonl(output / "chemical_registry.jsonl", chemical_rows)
    _write_jsonl(output / "anchor_registry.jsonl", anchor_rows)

    macro_rows_by_policy: dict[str, tuple[dict[str, object], ...]] = {}
    for policy in MACRO_POLICIES:
        selected = rankings[policy][: min(args.macro_budget, len(rankings[policy]))]
        rows = tuple(
            {
                "rank": rank,
                "pure_motif": pure,
                "pure_motif_sha256": _sha256_bytes(pure.encode("utf-8")),
                "surface_token": f"<MOST:MACRO:{rank:06d}>",
                "pretrain_train_occurrences": pretrain.get(pure, {}).get(
                    "train_occurrences", 0
                ),
                "registered_downstream_train_occurrences": downstream.get(
                    pure, {}
                ).get("train_occurrences", 0),
            }
            for rank, pure in enumerate(selected)
        )
        macro_rows_by_policy[policy] = rows
        _write_jsonl(output / f"macro_registry.{policy}.jsonl", rows)

    plans = []
    chemical_surfaces = [row["surface_token"] for row in chemical_rows]
    anchor_surfaces = [row["surface_token"] for row in anchor_rows]
    for policy in MACRO_POLICIES:
        macro_surfaces = [row["surface_token"] for row in macro_rows_by_policy[policy]]
        for boundary_mode in BOUNDARY_MODES:
            controls = [FALLBACK_MOTIF_SUFFIX]
            additions = controls + anchor_surfaces + chemical_surfaces + macro_surfaces
            if len(additions) != len(set(additions)):
                raise AnchoredTokenizerPlanError("candidate token namespaces overlap")
            declared = tuple(
                {
                    "token_id": args.base_vocab_size + offset,
                    "surface_token": token,
                }
                for offset, token in enumerate(additions)
            )
            plan_core = {
                "schema_version": SCHEMA_VERSION,
                "macro_policy": policy,
                "boundary_mode": boundary_mode,
                "grammar_contract": frozen_grammar_contract(),
                "base_vocab_size": args.base_vocab_size,
                "final_vocab_size": args.base_vocab_size + len(declared),
                "ordinary_token_additions_only": True,
                "declared_added_tokens": declared,
            }
            plan = dict(plan_core)
            plan["plan_sha256"] = _sha256_bytes(_canonical_json(plan_core).encode("utf-8"))
            name = f"plan.{policy}.{boundary_mode}.json"
            _write_json(output / name, plan)
            plans.append(
                {
                    "path": name,
                    "sha256": _sha256_file(output / name),
                    "plan_sha256": plan["plan_sha256"],
                    "macro_policy": policy,
                    "boundary_mode": boundary_mode,
                    "added_tokens": len(declared),
                    "final_vocab_size": plan["final_vocab_size"],
                }
            )

    artifact_names = [
        "chemical_registry.jsonl",
        "anchor_registry.jsonl",
        *(f"macro_registry.{policy}.jsonl" for policy in MACRO_POLICIES),
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate",
        "inputs": {
            "pretrain_census": {
                "path": str(pretrain_path),
                "sha256": _sha256_file(pretrain_path),
            },
            "registered_downstream_train_census": {
                "path": str(downstream_path),
                "sha256": _sha256_file(downstream_path),
                "validation_or_test_used": False,
            },
        },
        "parameters": {
            "base_vocab_size": args.base_vocab_size,
            "macro_budget": args.macro_budget,
            "max_anchor_id": args.max_anchor_id,
            "chemical_lexer_schema": LEXER_SCHEMA_VERSION,
        },
        "registries": {
            name: {"sha256": _sha256_file(output / name)} for name in artifact_names
        },
        "plans": plans,
        "grammar_decision": {
            "status": "frozen",
            "boundary_mode": FROZEN_GENERATIVE_BOUNDARY_MODE,
            "contract": frozen_grammar_contract(),
            "macro_policy_status": "candidate",
        },
        "contracts": {
            "tokenizer_snapshot_created": False,
            "model_embeddings_resized": False,
            "raw_chemical_punctuation_registered_as_added_tokens": False,
            "fallback_suffix_tokens_per_fallback_motif": 1,
            "double_boundary_candidate_eliminated": True,
            "all_motif_prefix_candidate_eliminated": True,
            "implicit_sidecar_is_encoder_only_length_control": True,
            "implicit_sidecar_plan_emitted": False,
            "only_frozen_generative_boundary_mode_emitted": True,
            "chemical_lexer_is_lossless_floor": True,
            "macro_registry_is_optional_compression": True,
            "no_post_phase_i_token_additions": True,
            "training_admission": False,
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-census", required=True)
    parser.add_argument("--downstream-train-census", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-vocab-size", type=int, default=32100)
    parser.add_argument("--macro-budget", type=int, default=512)
    parser.add_argument("--max-anchor-id", type=int, default=511)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        manifest = build(_parser().parse_args(argv))
    except Exception as exc:
        print(f"tokenizer plan failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
