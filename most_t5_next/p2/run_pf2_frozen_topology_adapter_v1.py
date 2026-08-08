#!/usr/bin/env python3
"""Train only the E3FP adapter on a frozen, trained M0-T backbone.

This screen removes the training-trajectory confound exposed by T3MI: the
topology-language T5 parameters are loaded from the completed M0-T checkpoint
and frozen.  Only the shared E3FP table and its zero-initialized scalar gate
can update under the same all-motif-identity CE objective.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from most_t5_next.p1.pf1_optimization import AdamWScale, G_CODEC_PROTOCOL_ID
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
    paired_release_binding,
)
from most_t5_next.p2.run_pf2_t3mi_v1 import (
    T3MI_MASK_PROBABILITY,
    T3MI_OBJECTIVE_CONTRACT,
    T3MI_PROTOCOL,
    validate_t3mi_checkpoint_contract,
)


REPORT_SCHEMA = "most-t5-p2/frozen-topology-e3fp-adapter-screen/v1"
MANIFEST_NAME = "pf2_frozen_topology_adapter_manifest.json"
CHECKPOINT_CONTRACT_NAME = "pf2_frozen_topology_adapter_contract.json"
PROTOCOL_ID = "pf2c-frozen-m0t-e3fp-adapter-64x2-v1"
NLL_RATIO_LIMIT = 0.98
ACCURACY_DROP_LIMIT = 0.01
SENSITIVITY_FLOOR = 0.01
GATE_FLOOR = 0.001


def _load_source_manifest(path: Path) -> tuple[dict[str, object], dict[str, float]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PF1TrainingError("frozen-adapter source manifest is invalid") from exc
    rows = payload.get("conditions") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("condition") != "M0":
        raise PF1TrainingError("frozen-adapter source must be one completed M0-T run")
    evaluations = rows[0].get("evaluations")
    if not isinstance(evaluations, list) or not evaluations or evaluations[-1].get("update") != 1000:
        raise PF1TrainingError("frozen-adapter source lacks step-1000 evaluation")
    final = evaluations[-1]
    return payload, {
        "token_weighted_nll": float(final["token_weighted_nll"]),
        "masked_token_accuracy": float(final["masked_token_accuracy"]),
    }


def load_frozen_topology_adapter_wrapper(
    *,
    topology_checkpoint_dir: Path,
    **kwargs: Any,
) -> Any:
    import torch

    if kwargs.get("condition_id") != "M1":
        raise PF1TrainingError("frozen topology adapter requires M1")
    contract = validate_t3mi_checkpoint_contract(
        topology_checkpoint_dir,
        condition_id="M0",
    )
    if contract.get("completed_updates") != 1000:
        raise PF1TrainingError("frozen topology adapter requires M0-T step1000")
    wrapper = load_verified_gated_four_grid_wrapper(**kwargs)
    payload = torch.load(
        topology_checkpoint_dir / "training_state.pt",
        map_location="cpu",
    )
    if payload.get("condition_id") != "M0" or payload.get("completed_updates") != 1000:
        raise PF1TrainingError("M0-T checkpoint training state differs")
    wrapper.load_state_dict(payload["wrapper_state_dict"], strict=True)
    return wrapper


def build_frozen_adapter_optimizer(model: Any, protocol: Any) -> AdamWScale:
    fusion = getattr(model, "geometry_fusion", None)
    if not isinstance(fusion, ZeroInitGatedE3FPCarrierFusion):
        raise PF1TrainingError("frozen adapter received a different fusion")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fusion.shared_embedding.weight.requires_grad_(True)
    fusion.geometry_gate_logit.requires_grad_(True)
    return AdamWScale(
        [
            {"params": [fusion.shared_embedding.weight], "scale_parameter": True},
            {"params": [fusion.geometry_gate_logit], "scale_parameter": False},
        ],
        lr=protocol.base_learning_rate,
        betas=(protocol.beta1, protocol.beta2),
        eps=protocol.epsilon,
        weight_decay=protocol.weight_decay,
    )


def _gate_state(model: Any) -> dict[str, float]:
    fusion = model.geometry_fusion
    logit = float(fusion.geometry_gate_logit.detach().cpu().item())
    return {"logit": logit, "effective_tanh_gate": math.tanh(logit)}


def write_frozen_adapter_checkpoint(
    *, topology_checkpoint_dir: Path, **kwargs: Any
) -> str:
    checkpoint_dir = Path(write_pf1_checkpoint(**kwargs))
    contract = {
        "schema_version": "most-t5-p2/frozen-topology-adapter-checkpoint/v1",
        "completed_updates": kwargs["update"],
        "condition_id": kwargs["condition_id"],
        "source_m0_t_checkpoint": str(topology_checkpoint_dir.resolve()),
        "optimization_protocol": asdict(T3MI_PROTOCOL),
        "optimization_protocol_id": PROTOCOL_ID,
        "trainable_modules": [
            "geometry_fusion.shared_embedding.weight",
            "geometry_fusion.geometry_gate_logit",
        ],
        "frozen_backbone": True,
        "gate_state": _gate_state(kwargs["model"]),
    }
    (checkpoint_dir / CHECKPOINT_CONTRACT_NAME).write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(checkpoint_dir)


def execute_frozen_topology_adapter(
    *,
    topology_checkpoint_dir: Path,
    source_manifest_path: Path,
    engine: Callable[..., dict[str, object]] = execute_pf1_four_grid,
    **kwargs: Any,
) -> dict[str, object]:
    requested = tuple(kwargs.get("condition_ids", ()))
    if requested != ("M1",):
        raise PF1TrainingError("frozen topology adapter runs only M1")
    if kwargs.pop("protocol", None) != T3MI_PROTOCOL:
        raise PF1TrainingError("frozen topology adapter requires 64x2")
    _source_payload, baseline = _load_source_manifest(source_manifest_path)
    report = engine(
        **kwargs,
        wrapper_loader=partial(
            load_frozen_topology_adapter_wrapper,
            topology_checkpoint_dir=topology_checkpoint_dir,
        ),
        protocol=T3MI_PROTOCOL,
        mask_probability=T3MI_MASK_PROBABILITY,
        checkpoint_writer=partial(
            write_frozen_adapter_checkpoint,
            topology_checkpoint_dir=topology_checkpoint_dir,
        ),
        optimizer_builder=build_frozen_adapter_optimizer,
    )
    rows = report.get("conditions")
    if not isinstance(rows, list) or len(rows) != 1:
        raise PF1TrainingError("frozen adapter engine returned invalid conditions")
    row = rows[0]
    evaluations = row.get("evaluations")
    sensitivity = row.get("final_e3fp_shuffle_diagnostic")
    if not isinstance(evaluations, list) or not isinstance(sensitivity, Mapping):
        raise PF1TrainingError("frozen adapter lacks final diagnostics")
    final = evaluations[-1]
    final_nll = float(final["token_weighted_nll"])
    final_accuracy = float(final["masked_token_accuracy"])
    delta = float(sensitivity["delta_nll"])
    gate = _gate_state_from_checkpoints(row)
    gates = {
        "aligned_nll_improves_frozen_m0t": final_nll / baseline["token_weighted_nll"] <= NLL_RATIO_LIMIT,
        "accuracy_practical_non_degradation": final_accuracy >= baseline["masked_token_accuracy"] - ACCURACY_DROP_LIMIT,
        "geometry_sensitivity_retained": delta >= SENSITIVITY_FLOOR,
        "geometry_gate_opened": abs(gate["effective_tanh_gate"]) >= GATE_FLOOR,
    }
    report.update(
        {
            "schema_version": REPORT_SCHEMA,
            "scope": "frozen_topology_backbone_e3fp_adapter_mechanism_screen",
            "source_m0_t_checkpoint": str(topology_checkpoint_dir.resolve()),
            "source_m0_t_manifest": str(source_manifest_path.resolve()),
            "objective_contract": T3MI_OBJECTIVE_CONTRACT,
            "fusion_contract": FUSION_CONTRACT,
            "baseline": baseline,
            "adapter_metrics": {
                "final_nll": final_nll,
                "final_accuracy": final_accuracy,
                "nll_ratio_to_frozen_m0t": final_nll / baseline["token_weighted_nll"],
                "shuffled_minus_aligned_delta_nll": delta,
                "final_gate": gate,
            },
            "gates": gates,
            "scientific_gate_pass": all(gates.values()),
            "decision": (
                "proceed_to_same_2d_conformer_probe"
                if all(gates.values())
                else "revisit_e3fp_state_or_carrier_reduction"
            ),
            "teacher": False,
            "mse_or_auxiliary_loss": False,
        }
    )
    data = report.get("data")
    if isinstance(data, dict):
        data.update(paired_release_binding(kwargs.get("reader")))
    optimization = report.get("optimization")
    if isinstance(optimization, dict):
        optimization.update(
            {
                "protocol_id": PROTOCOL_ID,
                "frozen_t5_backbone": True,
                "trainable_parameter_surfaces": ["e3fp_table", "scalar_gate"],
            }
        )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output_dir = Path(kwargs["output_dir"])
    (output_dir / CONDITION_MANIFEST_NAME).write_text(payload, encoding="utf-8")
    (output_dir / MANIFEST_NAME).write_text(payload, encoding="utf-8")
    return report


def _gate_state_from_checkpoints(row: Mapping[str, object]) -> dict[str, float]:
    checkpoints = row.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 2:
        raise PF1TrainingError("frozen adapter checkpoint trajectory differs")
    path = Path(str(checkpoints[-1])) / CHECKPOINT_CONTRACT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["gate_state"].items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", required=True)
    parser.add_argument("--base-model-snapshot", required=True)
    parser.add_argument("--base-tokenizer-snapshot", required=True)
    parser.add_argument("--union-init-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--geometry-fusion-seed", type=int, required=True)
    parser.add_argument("--num-e3fp-embeddings", type=int, default=4096)
    parser.add_argument("--topology-checkpoint", required=True)
    parser.add_argument("--source-m0-manifest", required=True)
    parser.set_defaults(
        condition_id="M1",
        protocol_id=G_CODEC_PROTOCOL_ID,
        resume_condition=None,
        resume_checkpoint=None,
    )
    return parser


def run(args: Any) -> dict[str, object]:
    return run_pf1(
        args,
        executor=partial(
            execute_frozen_topology_adapter,
            topology_checkpoint_dir=Path(args.topology_checkpoint).resolve(),
            source_manifest_path=Path(args.source_m0_manifest).resolve(),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (PF1TrainingError, RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
