#!/usr/bin/env python3
"""Replay final macro+lexer anchored surfaces and report exact token lengths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Sequence

from most_t5_next.r1.tokenizer.anchored_motif_model_surface_v1 import (
    FROZEN_GENERATIVE_BOUNDARY_MODE,
    decode_explicit_sequence,
    decode_fallback_prefixed_sequence,
    decode_fallback_suffixed_sequence,
    decode_implicit_with_sidecar,
    encode_frozen_phrases,
    encode_phrases,
    frozen_grammar_contract,
)


SCHEMA_VERSION = "most-t5-next/anchored-model-surface-analysis/v2"
POLICIES = (
    "pretrain_train_only",
    "balanced_pretrain_plus_registered_downstream_train",
)


class AnchoredModelSurfaceAnalysisError(RuntimeError):
    """The published logical surface cannot be replayed exactly."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise AnchoredModelSurfaceAnalysisError(
                    f"non-object row at {path.name}:{line_number}"
                )
            rows.append(row)
    if not rows:
        raise AnchoredModelSurfaceAnalysisError(f"empty registry: {path}")
    return rows


def _distribution(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        raise AnchoredModelSurfaceAnalysisError("cannot summarize an empty distribution")

    def percentile(fraction: float) -> int:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def analyze(args: argparse.Namespace) -> dict[str, object]:
    records_path = Path(args.surface_records).expanduser().resolve()
    bundle = Path(args.plan_bundle).expanduser().resolve()
    output = Path(args.output_report).expanduser().resolve()
    if output.exists():
        raise AnchoredModelSurfaceAnalysisError("output report must be absent")
    bundle_manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    grammar_decision = bundle_manifest.get("grammar_decision")
    if (
        not isinstance(grammar_decision, dict)
        or grammar_decision.get("status") != "frozen"
        or grammar_decision.get("boundary_mode") != FROZEN_GENERATIVE_BOUNDARY_MODE
        or grammar_decision.get("contract") != frozen_grammar_contract()
    ):
        raise AnchoredModelSurfaceAnalysisError(
            "plan bundle does not bind the frozen fallback-suffix grammar"
        )
    registry_contract = bundle_manifest.get("registries")
    if not isinstance(registry_contract, dict):
        raise AnchoredModelSurfaceAnalysisError("plan bundle has no registry contract")
    macro_rows = {}
    for policy in POLICIES:
        name = f"macro_registry.{policy}.jsonl"
        path = bundle / name
        descriptor = registry_contract.get(name)
        if not isinstance(descriptor, dict) or descriptor.get("sha256") != _sha256_file(path):
            raise AnchoredModelSurfaceAnalysisError(f"macro registry drift: {policy}")
        macro_rows[policy] = _load_jsonl(path)

    states = {
        policy: {
            "explicit_lengths": [],
            "implicit_lengths": [],
            "fallback_prefix_lengths": [],
            "fallback_suffix_lengths": [],
            "motif_counts": [],
            "anchor_occurrences": [],
            "macro_occurrences": 0,
            "fallback_occurrences": 0,
            "over_512_explicit": 0,
            "over_512_implicit": 0,
            "over_512_fallback_prefix": 0,
            "over_512_fallback_suffix": 0,
        }
        for policy in POLICIES
    }
    records = 0
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            try:
                phrases = row["surface"]["phrases"]
            except (KeyError, TypeError) as exc:
                raise AnchoredModelSurfaceAnalysisError(
                    f"missing phrase surface at record {line_number}"
                ) from exc
            expected = tuple(
                (
                    phrase["pure_motif"],
                    tuple(anchor["anchor_id"] for anchor in phrase["anchors"]),
                )
                for phrase in phrases
            )
            for policy in POLICIES:
                macros = macro_rows[policy]
                explicit = encode_phrases(
                    phrases, macros, boundary_mode="explicit_single_prefix"
                )
                implicit = encode_phrases(
                    phrases, macros, boundary_mode="implicit_sidecar"
                )
                fallback_prefix = encode_phrases(
                    phrases, macros, boundary_mode="fallback_single_prefix"
                )
                fallback_suffix = encode_frozen_phrases(phrases, macros)
                if decode_explicit_sequence(explicit.tokens, macros) != expected:
                    raise AnchoredModelSurfaceAnalysisError(
                        f"explicit round trip drift at record {line_number}"
                    )
                if (
                    decode_implicit_with_sidecar(
                        implicit.tokens, implicit.phrase_spans, macros
                    )
                    != expected
                ):
                    raise AnchoredModelSurfaceAnalysisError(
                        f"implicit sidecar round trip drift at record {line_number}"
                    )
                if (
                    decode_fallback_prefixed_sequence(fallback_prefix.tokens, macros)
                    != expected
                ):
                    raise AnchoredModelSurfaceAnalysisError(
                        f"fallback-prefix round trip drift at record {line_number}"
                    )
                if (
                    decode_fallback_suffixed_sequence(fallback_suffix.tokens, macros)
                    != expected
                ):
                    raise AnchoredModelSurfaceAnalysisError(
                        f"fallback-suffix round trip drift at record {line_number}"
                    )
                if len(fallback_suffix.tokens) != len(fallback_prefix.tokens):
                    raise AnchoredModelSurfaceAnalysisError(
                        "fallback prefix/suffix candidates differ in token count"
                    )
                state = states[policy]
                state["explicit_lengths"].append(len(explicit.tokens))
                state["implicit_lengths"].append(len(implicit.tokens))
                state["fallback_prefix_lengths"].append(len(fallback_prefix.tokens))
                state["fallback_suffix_lengths"].append(len(fallback_suffix.tokens))
                state["motif_counts"].append(len(phrases))
                state["anchor_occurrences"].append(
                    sum(len(anchor_ids) for _pure, anchor_ids in expected)
                )
                state["macro_occurrences"] += sum(explicit.macro_used)
                state["fallback_occurrences"] += len(phrases) - sum(explicit.macro_used)
                state["over_512_explicit"] += len(explicit.tokens) > 512
                state["over_512_implicit"] += len(implicit.tokens) > 512
                state["over_512_fallback_prefix"] += len(fallback_prefix.tokens) > 512
                state["over_512_fallback_suffix"] += len(fallback_suffix.tokens) > 512
                if len(fallback_suffix.tokens) - len(implicit.tokens) != sum(
                    not used for used in fallback_suffix.macro_used
                ):
                    raise AnchoredModelSurfaceAnalysisError(
                        "single fallback suffix cost is not exactly one per fallback motif"
                    )
                if len(explicit.tokens) - len(implicit.tokens) != len(phrases):
                    raise AnchoredModelSurfaceAnalysisError(
                        "single-prefix length delta is not exactly one per motif"
                    )
            records += 1
    if records == 0:
        raise AnchoredModelSurfaceAnalysisError("surface record input is empty")

    policies = {}
    for policy, state in states.items():
        motif_occurrences = state["macro_occurrences"] + state["fallback_occurrences"]
        policies[policy] = {
            "records": records,
            "motif_occurrences": motif_occurrences,
            "macro_occurrences": state["macro_occurrences"],
            "fallback_occurrences": state["fallback_occurrences"],
            "macro_occurrence_coverage": state["macro_occurrences"] / motif_occurrences,
            "explicit_single_prefix_token_lengths": _distribution(state["explicit_lengths"]),
            "implicit_sidecar_token_lengths": _distribution(state["implicit_lengths"]),
            "fallback_single_prefix_token_lengths": _distribution(
                state["fallback_prefix_lengths"]
            ),
            "fallback_single_suffix_token_lengths": _distribution(
                state["fallback_suffix_lengths"]
            ),
            "motifs_per_record": _distribution(state["motif_counts"]),
            "anchor_occurrences_per_record": _distribution(state["anchor_occurrences"]),
            "over_512_explicit": state["over_512_explicit"],
            "over_512_implicit": state["over_512_implicit"],
            "over_512_fallback_prefix": state["over_512_fallback_prefix"],
            "over_512_fallback_suffix": state["over_512_fallback_suffix"],
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "inputs": {
            "surface_records": {
                "path": str(records_path),
                "sha256": _sha256_file(records_path),
            },
            "plan_bundle_manifest": {
                "path": str(bundle / "manifest.json"),
                "sha256": _sha256_file(bundle / "manifest.json"),
            },
        },
        "policies": policies,
        "grammar_decision": {
            "status": "frozen",
            "boundary_mode": FROZEN_GENERATIVE_BOUNDARY_MODE,
            "contract": frozen_grammar_contract(),
            "diagnostic_prefix_results_are_not_training_candidates": True,
        },
        "contracts": {
            "all_records_replayed": True,
            "explicit_standalone_decode_exact": True,
            "fallback_single_prefix_standalone_decode_exact": True,
            "fallback_single_suffix_standalone_decode_exact": True,
            "fallback_suffix_preserves_macro_carrier_position": True,
            "fallback_prefix_and_suffix_have_equal_token_count": True,
            "implicit_decode_requires_sidecar": True,
            "all_motif_prefix_cost_exactly_one_token_per_motif": True,
            "fallback_suffix_cost_exactly_one_token_per_fallback_motif": True,
            "double_boundary_eliminated": True,
            "only_fallback_single_suffix_is_model_facing": True,
            "training_admission": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-records", required=True)
    parser.add_argument("--plan-bundle", required=True)
    parser.add_argument("--output-report", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = analyze(_parser().parse_args(argv))
    except Exception as exc:
        print(f"model-surface analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
