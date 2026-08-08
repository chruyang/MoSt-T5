#!/usr/bin/env python3
"""Merge matched M0-T/M1-T manifests and apply the frozen T3MI gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from most_t5_next.p2.run_pf2_gated_fusion_v1 import FUSION_CONTRACT
from most_t5_next.p2.run_pf2_t3mi_v1 import (
    REPORT_SCHEMA as CONDITION_SCHEMA,
    T3MI_MANIFEST_NAME,
    T3MI_OBJECTIVE_CONTRACT,
    T3MI_PROTOCOL_ID,
)


REPORT_SCHEMA = "most-t5-p2/t3mi-paired-decision/v1"
FINAL_UPDATE = 1000
NLL_RATIO_LIMIT = 0.98
ACCURACY_DROP_LIMIT = 0.01
SENSITIVITY_ABSOLUTE_FLOOR = 0.01
EFFECTIVE_GATE_ABSOLUTE_FLOOR = 0.001


class T3MIMergeError(ValueError):
    """The two artifacts are not one matched T3MI mechanism screen."""


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise T3MIMergeError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise T3MIMergeError("JSON artifact must be an object")
    return payload


def _manifest_path(path: Path) -> Path:
    path = Path(path)
    return path / T3MI_MANIFEST_NAME if path.is_dir() else path


def _finite_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise T3MIMergeError(f"{key} is not numeric")
    value = float(value)
    if not math.isfinite(value):
        raise T3MIMergeError(f"{key} is not finite")
    return value


def _condition(
    manifest: Mapping[str, object],
    expected: str,
) -> dict[str, object]:
    if manifest.get("schema_version") != CONDITION_SCHEMA:
        raise T3MIMergeError("condition manifest schema differs")
    if manifest.get("status") != "pass":
        raise T3MIMergeError("condition training did not complete")
    if manifest.get("objective_contract") != T3MI_OBJECTIVE_CONTRACT:
        raise T3MIMergeError("condition objective contract differs")
    if manifest.get("fusion_contract") != FUSION_CONTRACT:
        raise T3MIMergeError("condition fusion contract differs")
    data = manifest.get("data")
    if not isinstance(data, Mapping) or data.get("mask_probability") != 1.0:
        raise T3MIMergeError("condition is not the all-identity T3MI view")
    optimization = manifest.get("optimization")
    if not isinstance(optimization, Mapping) or optimization.get(
        "protocol_id"
    ) != T3MI_PROTOCOL_ID:
        raise T3MIMergeError("condition optimization protocol differs")
    rows = manifest.get("conditions")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise T3MIMergeError("condition manifest must contain exactly one cell")
    row = dict(rows[0])
    if row.get("condition") != expected:
        raise T3MIMergeError(f"expected {expected} condition manifest")
    if row.get("optimizer_updates") != FINAL_UPDATE:
        raise T3MIMergeError("condition did not complete the frozen update budget")
    trajectory = row.get("geometry_gate_trajectory")
    if not isinstance(trajectory, list) or len(trajectory) != 2:
        raise T3MIMergeError("condition lacks the two-point gate trajectory")
    if any(not isinstance(value, Mapping) for value in trajectory):
        raise T3MIMergeError("condition gate trajectory row is invalid")
    if [value.get("update") for value in trajectory] != [500, 1000]:
        raise T3MIMergeError("condition gate trajectory updates differ")
    for value in trajectory:
        logit = _finite_float(value, "logit")
        effective = _finite_float(value, "effective_tanh_gate")
        if not math.isclose(
            effective,
            math.tanh(logit),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise T3MIMergeError("condition gate trajectory is inconsistent")
    if row.get("final_geometry_gate") != trajectory[-1]:
        raise T3MIMergeError("condition final gate differs from its trajectory")
    return row


def _final_evaluation(row: Mapping[str, object]) -> dict[str, object]:
    evaluations = row.get("evaluations")
    if not isinstance(evaluations, list):
        raise T3MIMergeError("condition evaluations are absent")
    final = [value for value in evaluations if value.get("update") == FINAL_UPDATE]
    if len(final) != 1 or not isinstance(final[0], dict):
        raise T3MIMergeError("condition lacks one final evaluation")
    return dict(final[0])


def merge_pf2_t3mi(
    *,
    m0_manifest: Mapping[str, object],
    m1_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Validate a matched T3MI pair and return the preregistered route."""

    for key in (
        "data",
        "optimization",
        "precision",
        "evaluation_updates",
        "checkpoint_updates",
        "objective_contract",
        "fusion_contract",
    ):
        if m0_manifest.get(key) != m1_manifest.get(key):
            raise T3MIMergeError(f"matched condition contract differs at {key}")
    m0_execution = m0_manifest.get("execution")
    m1_execution = m1_manifest.get("execution")
    if not isinstance(m0_execution, Mapping) or not isinstance(m1_execution, Mapping):
        raise T3MIMergeError("condition execution contract is absent")
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
            raise T3MIMergeError(f"matched execution differs at {key}")

    m0 = _condition(m0_manifest, "M0")
    m1 = _condition(m1_manifest, "M1")
    for key in (
        "members_seen",
        "train_encoder_nonpadding_tokens",
        "train_supervised_target_tokens",
        "final_data_cursor",
    ):
        if m0.get(key) != m1.get(key):
            raise T3MIMergeError(f"paired training exposure differs at {key}")
    m0_eval = _final_evaluation(m0)
    m1_eval = _final_evaluation(m1)
    for key in (
        "members",
        "encoder_nonpadding_tokens",
        "supervised_target_tokens",
    ):
        if m0_eval.get(key) != m1_eval.get(key):
            raise T3MIMergeError(f"paired final evaluation differs at {key}")

    m0_trajectory = m0["geometry_gate_trajectory"]
    if any(
        value["logit"] != 0.0 or value["effective_tanh_gate"] != 0.0
        for value in m0_trajectory
    ):
        raise T3MIMergeError("M0 geometry gate changed despite an inactive branch")
    sensitivity = m1.get("final_e3fp_shuffle_diagnostic")
    if not isinstance(sensitivity, Mapping) or sensitivity.get("update") != FINAL_UPDATE:
        raise T3MIMergeError("M1 lacks final-update geometry sensitivity")
    if m0.get("final_e3fp_shuffle_diagnostic") is not None:
        raise T3MIMergeError("M0 unexpectedly reports a geometry diagnostic")

    m0_nll = _finite_float(m0_eval, "token_weighted_nll")
    m1_nll = _finite_float(m1_eval, "token_weighted_nll")
    if m0_nll <= 0.0 or m1_nll < 0.0:
        raise T3MIMergeError("paired final NLL values are outside their domain")
    m0_accuracy = _finite_float(m0_eval, "masked_token_accuracy")
    m1_accuracy = _finite_float(m1_eval, "masked_token_accuracy")
    sensitivity_delta = _finite_float(sensitivity, "delta_nll")
    final_gate = m1["final_geometry_gate"]
    final_gate_logit = _finite_float(final_gate, "logit")
    final_effective_gate = _finite_float(final_gate, "effective_tanh_gate")

    efficacy_gate = m1_nll <= NLL_RATIO_LIMIT * m0_nll
    accuracy_gate = m1_accuracy >= m0_accuracy - ACCURACY_DROP_LIMIT
    sensitivity_gate = sensitivity_delta >= SENSITIVITY_ABSOLUTE_FLOOR
    opening_gate = abs(final_effective_gate) >= EFFECTIVE_GATE_ABSOLUTE_FLOOR
    scientific_gate = (
        efficacy_gate and accuracy_gate and sensitivity_gate and opening_gate
    )
    decision = (
        "retain_f_gate_plus_t3mi_and_proceed_to_same_2d_probe"
        if scientific_gate
        else "stop_t3mi_and_revisit_geometry_state"
    )

    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "status_semantics": "artifact_and_pair_contract_pass_only",
        "scientific_gate_pass": scientific_gate,
        "decision": decision,
        "objective_contract": T3MI_OBJECTIVE_CONTRACT,
        "fusion_contract": FUSION_CONTRACT,
        "paired_contract": {
            "same_data_optimization_precision_and_schedule": True,
            "same_initialization_and_tokenizer_bindings": True,
            "same_train_and_final_dev_exposure": True,
            "all_logical_motif_identities_masked": True,
            "single_seed_mechanism_screen": True,
            "statistical_superiority_claim": False,
        },
        "frozen_practical_gates": {
            "m1_to_m0_nll_ratio_limit": NLL_RATIO_LIMIT,
            "accuracy_drop_limit": ACCURACY_DROP_LIMIT,
            "final_geometry_sensitivity_floor": SENSITIVITY_ABSOLUTE_FLOOR,
            "absolute_effective_gate_floor": EFFECTIVE_GATE_ABSOLUTE_FLOOR,
        },
        "metrics": {
            "m0_final_nll": m0_nll,
            "m1_final_nll": m1_nll,
            "m1_to_m0_nll_ratio": m1_nll / m0_nll,
            "m0_final_accuracy": m0_accuracy,
            "m1_final_accuracy": m1_accuracy,
            "m1_minus_m0_accuracy": m1_accuracy - m0_accuracy,
            "m1_final_shuffled_minus_aligned_delta_nll": sensitivity_delta,
            "m1_final_gate_logit": final_gate_logit,
            "m1_final_effective_tanh_gate": final_effective_gate,
        },
        "gates": {
            "geometry_improves_t3mi_nll": efficacy_gate,
            "accuracy_practical_non_degradation": accuracy_gate,
            "geometry_sensitivity_retained": sensitivity_gate,
            "geometry_gate_opened": opening_gate,
        },
        "interpretation": {
            "tests_geometry_required_identity_recovery": True,
            "topology_conditioned_not_pure_3d": True,
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
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise T3MIMergeError("output must be a new path")
    report = merge_pf2_t3mi(
        m0_manifest=_read_json(_manifest_path(args.m0_run)),
        m1_manifest=_read_json(_manifest_path(args.m1_run)),
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
    "T3MIMergeError",
    "REPORT_SCHEMA",
    "build_parser",
    "main",
    "merge_pf2_t3mi",
]
