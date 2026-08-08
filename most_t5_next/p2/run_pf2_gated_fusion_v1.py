#!/usr/bin/env python3
"""Run the independent PF-2 F-Gate screen on GraphPorts-v1 records.

M0-G and M1-G share the PF-1 run3 data, masks, union initialization,
optimizer, variable-tail 64x2 protocol, and update budget.  Their only model
difference is whether the zero-initialized gated geometry path is executed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from most_t5_next.p1.build_pf1_paired_release_v1 import MANIFEST_NAME
from most_t5_next.p1.pf1_optimization import (
    G_CODEC_PF1_PROTOCOL,
    G_CODEC_PROTOCOL_ID,
)
from most_t5_next.p1.run_pf1_four_grid_v1 import (
    CONDITION_MANIFEST_NAME,
    FOUR_GRID_MANIFEST_NAME,
    PF1TrainingError,
    execute_pf1_four_grid,
    run as run_pf1,
    write_pf1_checkpoint,
)
from most_t5_next.p2.gated_reference_geometry_fusion_v1 import (
    FUSION_ID,
    ZeroInitGatedE3FPCarrierFusion,
    load_verified_gated_four_grid_wrapper,
)


REPORT_SCHEMA = "most-t5-p2/zero-init-gated-fusion-screen/v1"
F_GATE_MANIFEST_NAME = "pf2_gated_fusion_manifest.json"
F_GATE_CHECKPOINT_CONTRACT_NAME = "pf2_gated_fusion_contract.json"
F_GATE_PROTOCOL_ID = "pf2-f-gate-64x2-v1"
F_GATE_PROTOCOL = G_CODEC_PF1_PROTOCOL
FUSION_CONTRACT = {
    "fusion_id": FUSION_ID,
    "shared_e3fp_embedding_table": True,
    "fixed_shell_slot_count": 4,
    "shell_reduction": "mean_over_four_slots_including_zero_padding",
    "carrier_atom_reduction": "arithmetic_mean",
    "carrier_injection": "identity_plus_tanh_scalar_gate_times_geometry",
    "gate_parameter_shape": [1],
    "gate_initial_logit": 0.0,
    "initial_function_equals_geometry_free_model": True,
    "noncarrier_identity_unchanged": True,
    "projection": False,
    "normalization": False,
    "teacher": False,
    "auxiliary_loss": False,
    "matched_motif_pair_required": True,
}


def _paired_release_binding(reader: Any) -> dict[str, object]:
    """Bind the matched runs to one published paired release, not just counts."""

    release_root = getattr(reader, "release_root", None)
    manifest = getattr(reader, "manifest", None)
    if release_root is None or not isinstance(manifest, Mapping):
        raise PF1TrainingError("F-Gate reader lacks a published-release binding")
    manifest_path = Path(release_root) / MANIFEST_NAME
    try:
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PF1TrainingError("F-Gate paired manifest cannot be read") from exc
    return {
        "paired_release_root": str(Path(release_root).resolve()),
        "paired_release_manifest_sha256": manifest_sha256,
        "paired_release_schema_version": manifest.get("schema_version"),
        "paired_release_counts": manifest.get("counts"),
    }


def _gate_state(model: Any) -> dict[str, float]:
    fusion = getattr(model, "geometry_fusion", None)
    if not isinstance(fusion, ZeroInitGatedE3FPCarrierFusion):
        raise PF1TrainingError("F-Gate checkpoint has a different fusion module")
    value = fusion.geometry_gate_logit.detach().to(device="cpu").reshape(-1)
    if value.numel() != 1:
        raise PF1TrainingError("F-Gate logit is not one scalar")
    logit = float(value.item())
    effective = math.tanh(logit)
    if not math.isfinite(logit) or not math.isfinite(effective):
        raise PF1TrainingError("F-Gate state is not finite")
    return {"logit": logit, "effective_tanh_gate": effective}


def write_f_gate_checkpoint(**kwargs: Any) -> str:
    """Write the full PF-1 state plus the F-Gate method and scalar state."""

    checkpoint_dir = Path(write_pf1_checkpoint(**kwargs))
    contract = {
        "schema_version": "most-t5-p2/gated-fusion-checkpoint-contract/v1",
        "fusion_contract": FUSION_CONTRACT,
        "condition_id": kwargs["condition_id"],
        "completed_updates": kwargs["update"],
        "optimization_protocol": asdict(F_GATE_PROTOCOL),
        "optimization_protocol_id": F_GATE_PROTOCOL_ID,
        "gate_state": _gate_state(kwargs["model"]),
    }
    (checkpoint_dir / F_GATE_CHECKPOINT_CONTRACT_NAME).write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(checkpoint_dir)


def validate_f_gate_checkpoint_contract(
    checkpoint_dir: Path,
    *,
    condition_id: str,
) -> dict[str, object]:
    """Validate the static method binding and the serialized scalar gate."""

    path = Path(checkpoint_dir) / F_GATE_CHECKPOINT_CONTRACT_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PF1TrainingError("F-Gate resume checkpoint lacks a valid contract") from exc
    if not isinstance(payload, dict):
        raise PF1TrainingError("F-Gate checkpoint contract must be an object")
    if payload.get("schema_version") != (
        "most-t5-p2/gated-fusion-checkpoint-contract/v1"
    ):
        raise PF1TrainingError("F-Gate checkpoint schema differs")
    if payload.get("fusion_contract") != FUSION_CONTRACT:
        raise PF1TrainingError("F-Gate checkpoint fusion contract differs")
    if payload.get("condition_id") != condition_id:
        raise PF1TrainingError("F-Gate checkpoint condition differs")
    if payload.get("completed_updates") not in (500, 1000):
        raise PF1TrainingError("F-Gate checkpoint update is invalid")
    if payload.get("optimization_protocol") != asdict(F_GATE_PROTOCOL):
        raise PF1TrainingError("F-Gate checkpoint optimization differs")
    if payload.get("optimization_protocol_id") != F_GATE_PROTOCOL_ID:
        raise PF1TrainingError("F-Gate checkpoint protocol id differs")
    state = payload.get("gate_state")
    if not isinstance(state, Mapping):
        raise PF1TrainingError("F-Gate checkpoint lacks gate state")
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
        raise PF1TrainingError("F-Gate checkpoint gate state is invalid")
    return payload


def _attach_gate_trajectory(report: dict[str, object]) -> None:
    rows = report.get("conditions")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise PF1TrainingError("F-Gate engine must return exactly one condition")
    row = rows[0]
    condition_id = row.get("condition")
    checkpoints = row.get("checkpoints")
    if condition_id not in ("M0", "M1") or not isinstance(checkpoints, list):
        raise PF1TrainingError("F-Gate condition checkpoint list is invalid")
    trajectory = []
    for checkpoint in checkpoints:
        contract = validate_f_gate_checkpoint_contract(
            Path(str(checkpoint)),
            condition_id=str(condition_id),
        )
        trajectory.append(
            {
                "update": contract["completed_updates"],
                **dict(contract["gate_state"]),
            }
        )
    if [value["update"] for value in trajectory] != [500, 1000]:
        raise PF1TrainingError("F-Gate trajectory must contain steps 500 and 1000")
    row["geometry_gate_trajectory"] = trajectory
    row["final_geometry_gate"] = dict(trajectory[-1])


def execute_pf2_gated_fusion(
    *,
    engine: Callable[..., dict[str, object]] = execute_pf1_four_grid,
    **kwargs: Any,
) -> dict[str, object]:
    """Execute the PF-1 engine while changing only the fusion module."""

    requested = tuple(kwargs.get("condition_ids", ()))
    if len(requested) != 1 or requested[0] not in ("M0", "M1"):
        raise PF1TrainingError(
            "F-Gate executor requires exactly one motif condition, M0 or M1"
        )
    incoming_protocol = kwargs.pop("protocol", None)
    if incoming_protocol != F_GATE_PROTOCOL:
        raise PF1TrainingError("F-Gate requires the frozen variable-tail 64x2 protocol")
    report = engine(
        **kwargs,
        wrapper_loader=load_verified_gated_four_grid_wrapper,
        protocol=F_GATE_PROTOCOL,
        checkpoint_writer=write_f_gate_checkpoint,
    )
    if not isinstance(report, dict) or report.get("status") != "pass":
        raise PF1TrainingError("F-Gate engine did not return a passing report")
    _attach_gate_trajectory(report)

    engine_schema = report.get("schema_version")
    report["schema_version"] = REPORT_SCHEMA
    report["training_engine_schema_version"] = engine_schema
    report["scope"] = "pf2_zero_init_gated_fusion_mechanism_screen_only"
    report["fusion_contract"] = FUSION_CONTRACT
    data = report.get("data")
    if not isinstance(data, dict):
        raise PF1TrainingError("F-Gate engine lacks its data contract")
    data.update(_paired_release_binding(kwargs.get("reader")))
    optimization = report.get("optimization")
    if not isinstance(optimization, dict):
        raise PF1TrainingError("F-Gate engine lacks optimization contract")
    optimization["protocol_id"] = F_GATE_PROTOCOL_ID
    interpretation = report.get("interpretation")
    if not isinstance(interpretation, Mapping):
        interpretation = {}
    report["interpretation"] = {
        **dict(interpretation),
        "tests_zero_init_safe_injection_not_e3fp_or_motif_validity": True,
        "tests_e3fp_use_not_conformer_understanding": True,
        "fresh_matched_m0_is_primary_baseline": True,
        "teacher_or_mse_decision_allowed_from_this_run_alone": False,
    }

    output_dir = Path(kwargs["output_dir"])
    engine_manifest_name = (
        FOUR_GRID_MANIFEST_NAME
        if len(requested) != 1
        else CONDITION_MANIFEST_NAME
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output_dir / engine_manifest_name).write_text(payload, encoding="utf-8")
    (output_dir / F_GATE_MANIFEST_NAME).write_text(payload, encoding="utf-8")
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
    parser.set_defaults(protocol_id=G_CODEC_PROTOCOL_ID)
    return parser


def run(args: Any, **kwargs: Any) -> dict[str, object]:
    if getattr(args, "condition_id", None) not in ("M0", "M1"):
        raise PF1TrainingError("F-Gate requires exactly one motif condition")
    if getattr(args, "protocol_id", None) != G_CODEC_PROTOCOL_ID:
        raise PF1TrainingError("F-Gate CLI protocol binding differs")
    if getattr(args, "resume_checkpoint", None) is not None:
        validate_f_gate_checkpoint_contract(
            Path(args.resume_checkpoint).expanduser().resolve(),
            condition_id=args.resume_condition,
        )
    return run_pf1(args, executor=execute_pf2_gated_fusion, **kwargs)


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
    "FUSION_CONTRACT",
    "F_GATE_CHECKPOINT_CONTRACT_NAME",
    "F_GATE_MANIFEST_NAME",
    "F_GATE_PROTOCOL",
    "F_GATE_PROTOCOL_ID",
    "REPORT_SCHEMA",
    "build_parser",
    "execute_pf2_gated_fusion",
    "main",
    "run",
    "validate_f_gate_checkpoint_contract",
    "write_f_gate_checkpoint",
]
