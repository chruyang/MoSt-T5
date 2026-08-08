#!/usr/bin/env python3
"""Merge G2-C/G2-G condition manifests and apply the frozen bridge gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from most_t5_next.p2.g1_deep_sets_geometry_fusion_v1 import FUSION_ID
from most_t5_next.p2.run_g2_geometry_to_motif_ce_v1 import (
    G2_MANIFEST_NAME,
    G2_OBJECTIVE_CONTRACT,
    G2_PROTOCOL_ID,
    REPORT_SCHEMA as CONDITION_SCHEMA,
)


REPORT_SCHEMA = "most-t5-p2/g2-geometry-to-motif-paired-decision/v1"
FINAL_UPDATE = 1000
NLL_RATIO_LIMIT = 0.98
ACCURACY_DROP_LIMIT = 0.01
SENSITIVITY_FLOOR = 0.01


class G2MergeError(ValueError):
    pass


def _finite(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise G2MergeError(key + " is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise G2MergeError(key + " is not finite")
    return result


def _cell(manifest: Mapping[str, object], expected: str) -> dict[str, object]:
    if (
        manifest.get("schema_version") != CONDITION_SCHEMA
        or manifest.get("status") != "pass"
        or manifest.get("objective_contract") != G2_OBJECTIVE_CONTRACT
        or manifest.get("fusion_id") != FUSION_ID
    ):
        raise G2MergeError("G2 condition contract differs")
    rows = manifest.get("conditions")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise G2MergeError("G2 condition manifest must contain one cell")
    row = dict(rows[0])
    if row.get("condition") != expected or row.get("optimizer_updates") != FINAL_UPDATE:
        raise G2MergeError("G2 cell identity or update budget differs")
    return row


def _final(row: Mapping[str, object]) -> Mapping[str, object]:
    evaluations = row.get("evaluations")
    if not isinstance(evaluations, list):
        raise G2MergeError("G2 evaluations are absent")
    values = [value for value in evaluations if value.get("update") == FINAL_UPDATE]
    if len(values) != 1 or not isinstance(values[0], Mapping):
        raise G2MergeError("G2 final evaluation is absent")
    return values[0]


def merge_g2(
    *, control_manifest: Mapping[str, object], geometry_manifest: Mapping[str, object]
) -> dict[str, object]:
    for key in (
        "data", "optimization", "precision", "evaluation_updates",
        "checkpoint_updates", "objective_contract", "fusion_id",
    ):
        if control_manifest.get(key) != geometry_manifest.get(key):
            raise G2MergeError("paired G2 contract differs at " + key)
    control = _cell(control_manifest, "M0")
    geometry = _cell(geometry_manifest, "M1")
    optimization = control_manifest.get("optimization")
    if not isinstance(optimization, Mapping) or optimization.get("protocol_id") != G2_PROTOCOL_ID:
        raise G2MergeError("G2 optimization protocol differs")
    for key in (
        "members_seen", "train_encoder_nonpadding_tokens",
        "train_supervised_target_tokens", "final_data_cursor",
    ):
        if control.get(key) != geometry.get(key):
            raise G2MergeError("paired G2 exposure differs at " + key)
    control_final = _final(control)
    geometry_final = _final(geometry)
    control_nll = _finite(control_final, "token_weighted_nll")
    geometry_nll = _finite(geometry_final, "token_weighted_nll")
    control_accuracy = _finite(control_final, "masked_token_accuracy")
    geometry_accuracy = _finite(geometry_final, "masked_token_accuracy")
    diagnostic = geometry.get("final_e3fp_shuffle_diagnostic")
    if not isinstance(diagnostic, Mapping) or diagnostic.get("update") != FINAL_UPDATE:
        raise G2MergeError("G2-G lacks the final aligned-versus-shuffled diagnostic")
    sensitivity = _finite(diagnostic, "delta_nll")
    if control.get("final_e3fp_shuffle_diagnostic") is not None:
        raise G2MergeError("G2-C unexpectedly reports geometry sensitivity")

    nll_gate = geometry_nll <= NLL_RATIO_LIMIT * control_nll
    accuracy_gate = geometry_accuracy >= control_accuracy - ACCURACY_DROP_LIMIT
    sensitivity_gate = sensitivity >= SENSITIVITY_FLOOR
    passed = nll_gate and accuracy_gate and sensitivity_gate
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "status_semantics": "paired_artifact_contract_pass_only",
        "scientific_gate_pass": passed,
        "decision": (
            "accept_frozen_g1b_bridge_for_stage1_pretraining"
            if passed
            else "do_not_start_full_pretraining_revise_geometry_bridge"
        ),
        "objective_contract": G2_OBJECTIVE_CONTRACT,
        "fusion_id": FUSION_ID,
        "frozen_gates": {
            "g2g_to_g2c_nll_ratio_limit": NLL_RATIO_LIMIT,
            "accuracy_drop_limit": ACCURACY_DROP_LIMIT,
            "shuffled_minus_aligned_delta_nll_floor": SENSITIVITY_FLOOR,
        },
        "metrics": {
            "g2c_final_nll": control_nll,
            "g2g_final_nll": geometry_nll,
            "g2g_to_g2c_nll_ratio": geometry_nll / control_nll,
            "g2c_final_accuracy": control_accuracy,
            "g2g_final_accuracy": geometry_accuracy,
            "g2g_minus_g2c_accuracy": geometry_accuracy - control_accuracy,
            "g2g_shuffled_minus_aligned_delta_nll": sensitivity,
        },
        "gates": {
            "geometry_improves_identity_recovery_nll": nll_gate,
            "accuracy_practical_non_degradation": accuracy_gate,
            "aligned_geometry_sensitivity_retained": sensitivity_gate,
        },
        "interpretation": {
            "single_seed_mechanism_screen": True,
            "topology_conditioned_not_pure_3d": True,
            "downstream_or_statistical_superiority_claim": False,
        },
    }


def _read(path: Path) -> dict[str, object]:
    target = path / G2_MANIFEST_NAME if path.is_dir() else path
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise G2MergeError("G2 manifest must be one object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise G2MergeError("output must be a new path")
    report = merge_g2(
        control_manifest=_read(args.control_run),
        geometry_manifest=_read(args.geometry_run),
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["G2MergeError", "REPORT_SCHEMA", "merge_g2", "main"]
