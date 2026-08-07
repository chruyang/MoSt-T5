#!/usr/bin/env python3
"""Merge matched PF-2A M0-R/M1-F manifests and apply frozen practical gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from most_t5_next.p2.reference_geometry_fusion_v1 import FUSION_ID
from most_t5_next.p2.run_pf2_reference_fusion_v1 import (
    FUSION_CONTRACT,
    PF2_MANIFEST_NAME,
    REPORT_SCHEMA as CONDITION_SCHEMA,
)


REPORT_SCHEMA = "most-t5-p2/reference-fusion-paired-decision/v1"
FINAL_UPDATE = 1000
NLL_RATIO_LIMIT = 1.02
ACCURACY_DROP_LIMIT = 0.01
SENSITIVITY_ABSOLUTE_FLOOR = 0.01
SENSITIVITY_RETENTION_FRACTION = 0.10


class PF2MergeError(ValueError):
    """The two runs are not a matched PF-2A comparison."""


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PF2MergeError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise PF2MergeError("JSON artifact must be an object")
    return payload


def _manifest_path(path: Path) -> Path:
    path = Path(path)
    return path / PF2_MANIFEST_NAME if path.is_dir() else path


def _condition(
    manifest: Mapping[str, object],
    expected: str,
) -> dict[str, object]:
    if manifest.get("schema_version") != CONDITION_SCHEMA:
        raise PF2MergeError("condition manifest schema differs")
    if manifest.get("status") != "pass":
        raise PF2MergeError("condition training did not complete")
    if manifest.get("fusion_contract") != FUSION_CONTRACT:
        raise PF2MergeError("condition fusion contract differs")
    rows = manifest.get("conditions")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise PF2MergeError("condition manifest must contain exactly one cell")
    row = dict(rows[0])
    if row.get("condition") != expected:
        raise PF2MergeError(f"expected {expected} condition manifest")
    if row.get("optimizer_updates") != FINAL_UPDATE:
        raise PF2MergeError("condition did not complete the frozen update budget")
    return row


def _final_evaluation(row: Mapping[str, object]) -> dict[str, object]:
    evaluations = row.get("evaluations")
    if not isinstance(evaluations, list):
        raise PF2MergeError("condition evaluations are absent")
    final = [value for value in evaluations if value.get("update") == FINAL_UPDATE]
    if len(final) != 1 or not isinstance(final[0], dict):
        raise PF2MergeError("condition lacks one final evaluation")
    return dict(final[0])


def _finite_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PF2MergeError(f"{key} is not numeric")
    value = float(value)
    if not math.isfinite(value):
        raise PF2MergeError(f"{key} is not finite")
    return value


def merge_pf2_reference_fusion(
    *,
    m0_manifest: Mapping[str, object],
    m1_manifest: Mapping[str, object],
    initial_sensitivity: Mapping[str, object],
) -> dict[str, object]:
    """Validate the paired contract and return a non-overstated decision."""

    for key in (
        "data",
        "optimization",
        "precision",
        "evaluation_updates",
        "checkpoint_updates",
    ):
        if m0_manifest.get(key) != m1_manifest.get(key):
            raise PF2MergeError(f"matched condition contract differs at {key}")
    m0_execution = m0_manifest.get("execution")
    m1_execution = m1_manifest.get("execution")
    if not isinstance(m0_execution, Mapping) or not isinstance(m1_execution, Mapping):
        raise PF2MergeError("condition execution contract is absent")
    for key in (
        "forward_seed",
        "geometry_fusion_seed",
        "num_e3fp_embeddings",
        "expected_vocab_size",
        "base_model_snapshot",
        "base_tokenizer_snapshot",
        "union_tokenizer_dir",
        "union_init_dir",
    ):
        if m0_execution.get(key) != m1_execution.get(key):
            raise PF2MergeError(f"matched execution differs at {key}")

    m0 = _condition(m0_manifest, "M0")
    m1 = _condition(m1_manifest, "M1")
    for key in (
        "members_seen",
        "train_encoder_nonpadding_tokens",
        "train_supervised_target_tokens",
        "final_data_cursor",
    ):
        if m0.get(key) != m1.get(key):
            raise PF2MergeError(f"paired training exposure differs at {key}")
    m0_eval = _final_evaluation(m0)
    m1_eval = _final_evaluation(m1)
    for key in (
        "members",
        "encoder_nonpadding_tokens",
        "supervised_target_tokens",
    ):
        if m0_eval.get(key) != m1_eval.get(key):
            raise PF2MergeError(f"paired final evaluation differs at {key}")

    if initial_sensitivity.get("status") != "pass":
        raise PF2MergeError("initial sensitivity artifact did not pass")
    if initial_sensitivity.get("fusion_id") != FUSION_ID:
        raise PF2MergeError("initial sensitivity names a different fusion")
    if initial_sensitivity.get("optimizer_updates") != 0:
        raise PF2MergeError("initial sensitivity was not measured at update zero")
    initial = initial_sensitivity.get("initial_sensitivity")
    if not isinstance(initial, Mapping) or initial.get("update") != 0:
        raise PF2MergeError("initial sensitivity payload is not update zero")
    initial_delta = _finite_float(initial, "delta_nll")
    sensitivity_floor = max(
        SENSITIVITY_ABSOLUTE_FLOOR,
        SENSITIVITY_RETENTION_FRACTION * initial_delta,
    )
    declared_floor = _finite_float(initial_sensitivity, "practical_final_gate")
    if not math.isclose(declared_floor, sensitivity_floor, rel_tol=0.0, abs_tol=1e-12):
        raise PF2MergeError("initial sensitivity gate was not frozen correctly")

    final_sensitivity = m1.get("final_e3fp_shuffle_diagnostic")
    if not isinstance(final_sensitivity, Mapping):
        raise PF2MergeError("M1 lacks final geometry sensitivity")
    if final_sensitivity.get("update") != FINAL_UPDATE:
        raise PF2MergeError("M1 sensitivity is not from the final update")
    if m0.get("final_e3fp_shuffle_diagnostic") is not None:
        raise PF2MergeError("M0 unexpectedly reports a geometry diagnostic")

    m0_nll = _finite_float(m0_eval, "token_weighted_nll")
    m1_nll = _finite_float(m1_eval, "token_weighted_nll")
    m0_accuracy = _finite_float(m0_eval, "masked_token_accuracy")
    m1_accuracy = _finite_float(m1_eval, "masked_token_accuracy")
    final_delta = _finite_float(final_sensitivity, "delta_nll")
    nll_gate = m1_nll <= NLL_RATIO_LIMIT * m0_nll
    accuracy_gate = m1_accuracy >= m0_accuracy - ACCURACY_DROP_LIMIT
    sensitivity_gate = final_delta >= sensitivity_floor
    non_degradation = nll_gate and accuracy_gate
    if non_degradation and sensitivity_gate:
        decision = "retain_f_ref_and_proceed_to_3d_sensitive_probe"
    elif non_degradation:
        decision = "enter_pf2b_t3mi"
    else:
        decision = "test_independent_f_gate_before_pf2b"

    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "status_semantics": "artifact_and_pair_contract_pass_only",
        "scientific_gate_pass": non_degradation and sensitivity_gate,
        "decision": decision,
        "fusion_contract": FUSION_CONTRACT,
        "paired_contract": {
            "same_data_optimization_precision_and_schedule": True,
            "same_initialization_and_tokenizer_bindings": True,
            "same_train_and_final_dev_exposure": True,
            "single_seed_mechanism_screen": True,
            "statistical_noninferiority_claim": False,
        },
        "frozen_practical_gates": {
            "nll_ratio_limit": NLL_RATIO_LIMIT,
            "accuracy_drop_limit": ACCURACY_DROP_LIMIT,
            "initial_delta_nll": initial_delta,
            "sensitivity_absolute_floor": SENSITIVITY_ABSOLUTE_FLOOR,
            "sensitivity_retention_fraction": SENSITIVITY_RETENTION_FRACTION,
            "final_sensitivity_floor": sensitivity_floor,
        },
        "metrics": {
            "m0_final_nll": m0_nll,
            "m1_final_nll": m1_nll,
            "m1_to_m0_nll_ratio": m1_nll / m0_nll,
            "m0_final_accuracy": m0_accuracy,
            "m1_final_accuracy": m1_accuracy,
            "m1_minus_m0_accuracy": m1_accuracy - m0_accuracy,
            "m1_final_shuffled_minus_aligned_delta_nll": final_delta,
            "m1_sensitivity_retention_fraction": (
                final_delta / initial_delta if initial_delta != 0.0 else None
            ),
        },
        "gates": {
            "nll_practical_non_degradation": nll_gate,
            "accuracy_practical_non_degradation": accuracy_gate,
            "geometry_sensitivity_retained": sensitivity_gate,
        },
        "interpretation": {
            "architecture_superiority_claim": False,
            "statistical_significance_claim": False,
            "conformer_understanding_claim": False,
            "teacher_or_mse_justified_by_this_report_alone": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-run", type=Path, required=True)
    parser.add_argument("--m1-run", type=Path, required=True)
    parser.add_argument("--initial-sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise PF2MergeError("output must be a new path")
    report = merge_pf2_reference_fusion(
        m0_manifest=_read_json(_manifest_path(args.m0_run)),
        m1_manifest=_read_json(_manifest_path(args.m1_run)),
        initial_sensitivity=_read_json(args.initial_sensitivity),
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PF2MergeError",
    "REPORT_SCHEMA",
    "build_parser",
    "main",
    "merge_pf2_reference_fusion",
]
