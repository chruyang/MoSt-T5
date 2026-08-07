#!/usr/bin/env python3
"""Run the PF-2A reference-fusion mechanism screen on the PF-1 release.

This is intentionally a thin specialization of the frozen PF-1 training
engine.  Data order, corruption, optimizer, effective batch, update budget,
evaluation, resume, and checkpoint behavior remain unchanged.  Both matched
motif cells use the previously probed 64 x 2 realization of effective batch
128; their only architectural difference is the geometry path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from most_t5_next.p1.pf1_optimization import FROZEN_PF1_PROTOCOL
from most_t5_next.p1.run_pf1_four_grid_v1 import (
    CONDITION_MANIFEST_NAME,
    FOUR_GRID_MANIFEST_NAME,
    PF1TrainingError,
    execute_pf1_four_grid,
    run as run_pf1,
    write_pf1_checkpoint,
)
from most_t5_next.p2.reference_geometry_fusion_v1 import (
    FUSION_ID,
    load_verified_reference_four_grid_wrapper,
)


REPORT_SCHEMA = "most-t5-p2/reference-fusion-screen/v1"
PF2_MANIFEST_NAME = "pf2_reference_fusion_manifest.json"
PF2_CHECKPOINT_CONTRACT_NAME = "pf2_reference_fusion_contract.json"
PF2_PROTOCOL = replace(
    FROZEN_PF1_PROTOCOL,
    micro_batch_size=64,
    gradient_accumulation_steps=2,
)
FUSION_CONTRACT = {
    "fusion_id": FUSION_ID,
    "shared_e3fp_embedding_table": True,
    "fixed_shell_slot_count": 4,
    "shell_reduction": "mean_over_four_slots_including_zero_padding",
    "carrier_atom_reduction": "arithmetic_mean",
    "carrier_identity_coefficient": 0.5,
    "carrier_geometry_coefficient": 0.5,
    "noncarrier_identity_unchanged": True,
    "gate": False,
    "projection": False,
    "teacher": False,
    "auxiliary_loss": False,
    "matched_motif_pair_required": True,
}


def write_pf2_checkpoint(**kwargs: Any) -> str:
    """Write the PF-1 recovery state plus an explicit PF-2 fusion binding."""

    checkpoint_dir = Path(write_pf1_checkpoint(**kwargs))
    contract = {
        "schema_version": "most-t5-p2/reference-fusion-checkpoint-contract/v1",
        "fusion_contract": FUSION_CONTRACT,
        "condition_id": kwargs["condition_id"],
        "completed_updates": kwargs["update"],
        "optimization_protocol": asdict(PF2_PROTOCOL),
    }
    (checkpoint_dir / PF2_CHECKPOINT_CONTRACT_NAME).write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(checkpoint_dir)


def validate_pf2_checkpoint_contract(
    checkpoint_dir: Path,
    *,
    condition_id: str,
) -> dict[str, object]:
    """Reject a resume checkpoint whose declared PF-2 method has drifted."""

    path = Path(checkpoint_dir) / PF2_CHECKPOINT_CONTRACT_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PF1TrainingError("PF-2 resume checkpoint lacks a valid contract") from exc
    expected = {
        "schema_version": "most-t5-p2/reference-fusion-checkpoint-contract/v1",
        "fusion_contract": FUSION_CONTRACT,
        "condition_id": condition_id,
        "completed_updates": payload.get("completed_updates"),
        "optimization_protocol": asdict(PF2_PROTOCOL),
    }
    if payload != expected or payload.get("completed_updates") not in (500, 1000):
        raise PF1TrainingError("PF-2 resume checkpoint contract differs")
    return payload


def execute_pf2_reference_fusion(
    *,
    engine: Callable[..., dict[str, object]] = execute_pf1_four_grid,
    **kwargs: Any,
) -> dict[str, object]:
    """Execute the unchanged PF-1 loop with only PF-2A fusion substituted."""

    requested = tuple(kwargs.get("condition_ids", ()))
    if len(requested) != 1 or requested[0] not in ("M0", "M1"):
        raise PF1TrainingError(
            "PF-2A executor requires exactly one motif condition, M0 or M1"
        )
    report = engine(
        **kwargs,
        wrapper_loader=load_verified_reference_four_grid_wrapper,
        protocol=PF2_PROTOCOL,
        checkpoint_writer=write_pf2_checkpoint,
    )
    if not isinstance(report, dict) or report.get("status") != "pass":
        raise PF1TrainingError("PF-2A engine did not return a passing report")

    engine_schema = report.get("schema_version")
    report["schema_version"] = REPORT_SCHEMA
    report["training_engine_schema_version"] = engine_schema
    report["scope"] = "pf2_reference_fusion_mechanism_screen_only"
    report["fusion_contract"] = FUSION_CONTRACT
    interpretation = report.get("interpretation")
    if not isinstance(interpretation, Mapping):
        interpretation = {}
    report["interpretation"] = {
        **dict(interpretation),
        "tests_fusion_bundle_not_individual_component_causality": True,
        "tests_e3fp_use_not_conformer_understanding": True,
        "fresh_matched_m0_is_primary_baseline": True,
        "historical_pf1_m0_is_supportive_only": True,
        "teacher_or_mse_decision_allowed_from_this_run_alone": False,
    }

    output_dir = Path(kwargs["output_dir"])
    engine_manifest_name = (
        FOUR_GRID_MANIFEST_NAME
        if len(requested) != 1
        else CONDITION_MANIFEST_NAME
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    # Keep the training engine's conventional manifest path truthful for
    # resume/audit tooling, and publish one explicitly named PF-2 artifact.
    (output_dir / engine_manifest_name).write_text(payload, encoding="utf-8")
    (output_dir / PF2_MANIFEST_NAME).write_text(payload, encoding="utf-8")
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
    return parser


def run(args: Any, **kwargs: Any) -> dict[str, object]:
    """Bind the verified release and execute the PF-2A specialization."""

    if getattr(args, "condition_id", None) not in ("M0", "M1"):
        raise PF1TrainingError(
            "PF-2A requires exactly one --condition-id, M0 or M1"
        )
    if getattr(args, "resume_checkpoint", None) is not None:
        validate_pf2_checkpoint_contract(
            Path(args.resume_checkpoint).expanduser().resolve(),
            condition_id=args.resume_condition,
        )
    return run_pf1(args, executor=execute_pf2_reference_fusion, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (PF1TrainingError, RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "PF2_MANIFEST_NAME",
    "PF2_CHECKPOINT_CONTRACT_NAME",
    "PF2_PROTOCOL",
    "REPORT_SCHEMA",
    "build_parser",
    "execute_pf2_reference_fusion",
    "main",
    "run",
    "validate_pf2_checkpoint_contract",
    "write_pf2_checkpoint",
]
