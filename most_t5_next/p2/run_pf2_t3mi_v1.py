#!/usr/bin/env python3
"""Run the paired PF-2B topology-conditioned 3D identity mechanism screen.

T3MI changes one training view only: every logical-motif identity span is
replaced by a T5 sentinel while the GraphPorts connection skeleton and motif
order remain visible.  M0-T and M1-T share this view, initialization, CE loss,
optimizer, update budget and data order; only M1-T executes the zero-init
gated E3FP path.  No vocabulary, prediction head, teacher or auxiliary loss is
introduced.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from most_t5_next.p1.pf1_optimization import (
    FROZEN_PF1_PROTOCOL,
    PF1_SCREEN_PROTOCOL_ID,
    PF1OptimizationProtocol,
)
from most_t5_next.p1.run_pf1_four_grid_v1 import (
    CONDITION_MANIFEST_NAME,
    PF1TrainingError,
    execute_pf1_four_grid,
    run as run_pf1,
    write_pf1_checkpoint,
)
from most_t5_next.p2.gated_reference_geometry_fusion_v1 import (
    ZeroInitGatedE3FPCarrierFusion,
    load_verified_gated_four_grid_wrapper,
)
from most_t5_next.p2.run_pf2_gated_fusion_v1 import (
    FUSION_CONTRACT,
    build_f_gate_optimizer,
    paired_release_binding,
)


REPORT_SCHEMA = "most-t5-p2/t3mi-mechanism-screen/v1"
T3MI_MANIFEST_NAME = "pf2_t3mi_manifest.json"
T3MI_CHECKPOINT_CONTRACT_NAME = "pf2_t3mi_contract.json"
T3MI_PROTOCOL_ID = "pf2b-t3mi-all-identity-32x4-v1"
T3MI_MASK_PROBABILITY = 1.0
T3MI_PROTOCOL = replace(
    FROZEN_PF1_PROTOCOL,
    micro_batch_size=32,
    gradient_accumulation_steps=4,
)
T3MI_OBJECTIVE_CONTRACT = {
    "objective_id": "topology-conditioned-e3fp-to-motif-identity-reconstruction-v1",
    "corruption_unit": "complete_logical_motif_identity_span",
    "mask_probability": T3MI_MASK_PROBABILITY,
    "all_logical_motif_identity_spans_selected": True,
    "identity_macro_or_fallback_tokens_visible_in_encoder": False,
    "graphports_connection_skeleton_visible": True,
    "motif_order_visible": True,
    "geometry_visible_only_in_m1": True,
    "decoder_target": "standard_t5_sentinel_ce_full_motif_identity",
    "ordinary_identity_ce_mixture_fraction_in_this_screen": 0.0,
    "t3mi_fraction_in_this_screen": 1.0,
    "new_vocabulary": False,
    "new_prediction_head": False,
    "teacher": False,
    "mse_or_auxiliary_loss": False,
    "pure_3d_claim": False,
}


def _gate_state(model: Any) -> dict[str, float]:
    fusion = getattr(model, "geometry_fusion", None)
    if not isinstance(fusion, ZeroInitGatedE3FPCarrierFusion):
        raise PF1TrainingError("T3MI checkpoint has a different fusion module")
    value = fusion.geometry_gate_logit.detach().to(device="cpu").reshape(-1)
    if value.numel() != 1:
        raise PF1TrainingError("T3MI gate logit is not scalar")
    logit = float(value.item())
    effective = math.tanh(logit)
    if not math.isfinite(logit) or not math.isfinite(effective):
        raise PF1TrainingError("T3MI gate state is non-finite")
    return {"logit": logit, "effective_tanh_gate": effective}


def write_t3mi_checkpoint(**kwargs: Any) -> str:
    checkpoint_dir = Path(write_pf1_checkpoint(**kwargs))
    contract = {
        "schema_version": "most-t5-p2/t3mi-checkpoint-contract/v1",
        "condition_id": kwargs["condition_id"],
        "completed_updates": kwargs["update"],
        "objective_contract": T3MI_OBJECTIVE_CONTRACT,
        "fusion_contract": FUSION_CONTRACT,
        "optimization_protocol": asdict(T3MI_PROTOCOL),
        "optimization_protocol_id": T3MI_PROTOCOL_ID,
        "gate_state": _gate_state(kwargs["model"]),
    }
    (checkpoint_dir / T3MI_CHECKPOINT_CONTRACT_NAME).write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(checkpoint_dir)


def validate_t3mi_checkpoint_contract(
    checkpoint_dir: Path,
    *,
    condition_id: str,
) -> dict[str, object]:
    path = Path(checkpoint_dir) / T3MI_CHECKPOINT_CONTRACT_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PF1TrainingError("T3MI resume checkpoint lacks a valid contract") from exc
    if not isinstance(payload, dict):
        raise PF1TrainingError("T3MI checkpoint contract must be an object")
    expected_static = {
        "schema_version": "most-t5-p2/t3mi-checkpoint-contract/v1",
        "condition_id": condition_id,
        "objective_contract": T3MI_OBJECTIVE_CONTRACT,
        "fusion_contract": FUSION_CONTRACT,
        "optimization_protocol": asdict(T3MI_PROTOCOL),
        "optimization_protocol_id": T3MI_PROTOCOL_ID,
    }
    if any(payload.get(key) != value for key, value in expected_static.items()):
        raise PF1TrainingError("T3MI checkpoint static contract differs")
    if payload.get("completed_updates") not in (500, 1000):
        raise PF1TrainingError("T3MI checkpoint update is invalid")
    state = payload.get("gate_state")
    if not isinstance(state, Mapping):
        raise PF1TrainingError("T3MI checkpoint lacks gate state")
    logit = state.get("logit")
    effective = state.get("effective_tanh_gate")
    if (
        isinstance(logit, bool)
        or not isinstance(logit, (int, float))
        or isinstance(effective, bool)
        or not isinstance(effective, (int, float))
        or not math.isfinite(float(logit))
        or not math.isfinite(float(effective))
        or not math.isclose(
            float(effective),
            math.tanh(float(logit)),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise PF1TrainingError("T3MI checkpoint gate state is invalid")
    return payload


def _attach_gate_trajectory(report: dict[str, object]) -> None:
    rows = report.get("conditions")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise PF1TrainingError("T3MI engine must return one condition")
    row = rows[0]
    condition_id = row.get("condition")
    checkpoints = row.get("checkpoints")
    if condition_id not in ("M0", "M1") or not isinstance(checkpoints, list):
        raise PF1TrainingError("T3MI condition checkpoint list is invalid")
    trajectory = []
    for checkpoint in checkpoints:
        contract = validate_t3mi_checkpoint_contract(
            Path(str(checkpoint)),
            condition_id=str(condition_id),
        )
        trajectory.append(
            {
                "update": contract["completed_updates"],
                **dict(contract["gate_state"]),
            }
        )
    if [row["update"] for row in trajectory] != [500, 1000]:
        raise PF1TrainingError("T3MI gate trajectory must contain steps 500 and 1000")
    row["geometry_gate_trajectory"] = trajectory
    row["final_geometry_gate"] = dict(trajectory[-1])


def execute_pf2_t3mi(
    *,
    engine: Callable[..., dict[str, object]] = execute_pf1_four_grid,
    **kwargs: Any,
) -> dict[str, object]:
    requested = tuple(kwargs.get("condition_ids", ()))
    if len(requested) != 1 or requested[0] not in ("M0", "M1"):
        raise PF1TrainingError("T3MI executor requires exactly M0 or M1")
    incoming_protocol = kwargs.pop("protocol", None)
    if incoming_protocol != T3MI_PROTOCOL:
        raise PF1TrainingError("T3MI requires the frozen 32x4 protocol")
    report = engine(
        **kwargs,
        wrapper_loader=load_verified_gated_four_grid_wrapper,
        protocol=T3MI_PROTOCOL,
        mask_probability=T3MI_MASK_PROBABILITY,
        checkpoint_writer=write_t3mi_checkpoint,
        optimizer_builder=build_f_gate_optimizer,
    )
    if not isinstance(report, dict) or report.get("status") != "pass":
        raise PF1TrainingError("T3MI engine did not return a passing report")
    data = report.get("data")
    if not isinstance(data, dict):
        raise PF1TrainingError("T3MI engine lacks its data contract")
    if data.get("mask_probability") != T3MI_MASK_PROBABILITY:
        raise PF1TrainingError("T3MI engine did not use the all-identity view")
    _attach_gate_trajectory(report)

    engine_schema = report.get("schema_version")
    report["schema_version"] = REPORT_SCHEMA
    report["training_engine_schema_version"] = engine_schema
    report["scope"] = "pf2b_t3mi_mechanism_screen_only"
    report["objective_contract"] = T3MI_OBJECTIVE_CONTRACT
    report["fusion_contract"] = FUSION_CONTRACT
    data.update(paired_release_binding(kwargs.get("reader")))
    optimization = report.get("optimization")
    if not isinstance(optimization, dict):
        raise PF1TrainingError("T3MI engine lacks optimization contract")
    optimization["protocol_id"] = T3MI_PROTOCOL_ID
    optimization["geometry_gate_parameter_rms_scaling"] = False
    optimization["all_other_parameter_rms_scaling"] = True
    interpretation = report.get("interpretation")
    if not isinstance(interpretation, Mapping):
        interpretation = {}
    report["interpretation"] = {
        **dict(interpretation),
        "tests_geometry_required_identity_recovery_not_conformer_understanding": True,
        "topology_remains_visible": True,
        "pure_3d_claim": False,
        "teacher_or_mse_decision_allowed_from_this_run_alone": False,
    }

    output_dir = Path(kwargs["output_dir"])
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output_dir / CONDITION_MANIFEST_NAME).write_text(payload, encoding="utf-8")
    (output_dir / T3MI_MANIFEST_NAME).write_text(payload, encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", required=True)
    parser.add_argument("--base-model-snapshot", required=True)
    parser.add_argument("--base-tokenizer-snapshot", required=True)
    parser.add_argument("--union-init-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--geometry-fusion-seed", type=int, required=True)
    parser.add_argument("--num-e3fp-embeddings", type=int, default=4096)
    parser.add_argument("--condition-id", choices=("M0", "M1"), required=True)
    parser.add_argument("--resume-condition", choices=("M0", "M1"))
    parser.add_argument("--resume-checkpoint")
    parser.set_defaults(protocol_id=PF1_SCREEN_PROTOCOL_ID)
    return parser


def run(args: Any, **kwargs: Any) -> dict[str, object]:
    if getattr(args, "condition_id", None) not in ("M0", "M1"):
        raise PF1TrainingError("T3MI requires exactly one motif condition")
    if getattr(args, "protocol_id", None) != PF1_SCREEN_PROTOCOL_ID:
        raise PF1TrainingError("T3MI CLI protocol binding differs")
    if getattr(args, "resume_checkpoint", None) is not None:
        validate_t3mi_checkpoint_contract(
            Path(args.resume_checkpoint).expanduser().resolve(),
            condition_id=args.resume_condition,
        )
    return run_pf1(args, executor=execute_pf2_t3mi, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (PF1TrainingError, RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "REPORT_SCHEMA",
    "T3MI_CHECKPOINT_CONTRACT_NAME",
    "T3MI_MANIFEST_NAME",
    "T3MI_MASK_PROBABILITY",
    "T3MI_OBJECTIVE_CONTRACT",
    "T3MI_PROTOCOL",
    "T3MI_PROTOCOL_ID",
    "build_parser",
    "execute_pf2_t3mi",
    "main",
    "run",
    "validate_t3mi_checkpoint_contract",
    "write_t3mi_checkpoint",
]
