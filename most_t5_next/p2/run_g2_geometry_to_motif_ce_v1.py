#!/usr/bin/env python3
"""Run the paired G2 frozen-state geometry-to-motif CE bridge.

Every motif identity span is replaced by a T5 sentinel.  G2-C reconstructs
the identities from the visible GraphPorts topology alone; G2-G receives the
same input plus a frozen G1b Deep Sets motif state projected to the sentinel
carrier.  Both cells retain the ordinary T5 cross-entropy objective.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from most_t5_next.p1.pf1_optimization import (
    G_CODEC_PF1_PROTOCOL,
    G_CODEC_PROTOCOL_ID,
)
from most_t5_next.p1.run_pf1_four_grid_v1 import (
    CONDITION_MANIFEST_NAME,
    PF1TrainingError,
    execute_pf1_four_grid,
    run as run_pf1,
)
from most_t5_next.p2.g1_deep_sets_geometry_fusion_v1 import (
    FUSION_ID,
    load_verified_g1_bridge_wrapper,
)


REPORT_SCHEMA = "most-t5-p2/g2-geometry-to-motif-ce/v1"
G2_MANIFEST_NAME = "g2_geometry_to_motif_ce_manifest.json"
G2_PROTOCOL = G_CODEC_PF1_PROTOCOL
G2_PROTOCOL_ID = "g2-frozen-g1b-all-identity-64x2-v1"
G2_MASK_PROBABILITY = 1.0
G2_OBJECTIVE_CONTRACT = {
    "objective_id": "frozen-g1b-geometry-to-motif-identity-ce-v1",
    "mask_probability": 1.0,
    "all_logical_motif_identity_spans_selected": True,
    "graphports_topology_visible": True,
    "g2_control": "topology_only",
    "g2_geometry": "topology_plus_frozen_g1b_deep_sets_carrier",
    "frozen_g1_encoder": True,
    "trainable_bridge": "layer_norm_plus_linear_128_to_t5_hidden",
    "loss": "standard_t5_sentinel_cross_entropy_only",
    "teacher": False,
    "mse_or_auxiliary_loss": False,
    "pure_3d_claim": False,
}


def _attach_bridge_summary(report: dict[str, object], g1_checkpoint: Path) -> None:
    rows = report.get("conditions")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise PF1TrainingError("G2 engine must return exactly one condition")
    condition = rows[0]
    condition["g2_cell"] = "G2-C" if condition.get("condition") == "M0" else "G2-G"
    condition["g1_encoder_frozen"] = True
    condition["fusion_id"] = FUSION_ID
    condition["g1_checkpoint"] = str(Path(g1_checkpoint).resolve())


def execute_g2_geometry_to_motif_ce(
    *,
    g1_checkpoint: Path,
    engine: Callable[..., dict[str, object]] = execute_pf1_four_grid,
    **kwargs: Any,
) -> dict[str, object]:
    requested = tuple(kwargs.get("condition_ids", ()))
    if len(requested) != 1 or requested[0] not in ("M0", "M1"):
        raise PF1TrainingError("G2 requires exactly one topology or geometry cell")
    incoming_protocol = kwargs.pop("protocol", None)
    if incoming_protocol != G2_PROTOCOL:
        raise PF1TrainingError("G2 requires the frozen 64x2 motif protocol")
    checkpoint = Path(g1_checkpoint).expanduser().resolve()
    wrapper_loader = partial(
        load_verified_g1_bridge_wrapper,
        g1_checkpoint=checkpoint,
    )
    report = engine(
        **kwargs,
        wrapper_loader=wrapper_loader,
        protocol=G2_PROTOCOL,
        mask_probability=G2_MASK_PROBABILITY,
    )
    if not isinstance(report, dict) or report.get("status") != "pass":
        raise PF1TrainingError("G2 engine did not return a passing report")
    data = report.get("data")
    if not isinstance(data, dict) or data.get("mask_probability") != 1.0:
        raise PF1TrainingError("G2 did not use the all-identity view")
    _attach_bridge_summary(report, checkpoint)
    report["training_engine_schema_version"] = report.get("schema_version")
    report["schema_version"] = REPORT_SCHEMA
    report["scope"] = "g2_frozen_state_geometry_to_motif_bridge_screen"
    report["objective_contract"] = G2_OBJECTIVE_CONTRACT
    report["fusion_id"] = FUSION_ID
    optimization = report.get("optimization")
    if not isinstance(optimization, dict):
        raise PF1TrainingError("G2 engine lacks its optimization contract")
    optimization["protocol_id"] = G2_PROTOCOL_ID
    report["interpretation"] = {
        **dict(report.get("interpretation", {})),
        "paired_cells_must_be_merged_before_scientific_decision": True,
        "tests_incremental_geometry_value_given_visible_topology": True,
        "not_a_pure_3d_or_downstream_performance_claim": True,
    }
    output_dir = Path(kwargs["output_dir"])
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output_dir / CONDITION_MANIFEST_NAME).write_text(payload, encoding="utf-8")
    (output_dir / G2_MANIFEST_NAME).write_text(payload, encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", required=True)
    parser.add_argument("--base-model-snapshot", required=True)
    parser.add_argument("--base-tokenizer-snapshot", required=True)
    parser.add_argument("--union-init-dir", required=True)
    parser.add_argument("--g1-checkpoint", required=True)
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
        raise PF1TrainingError("G2 requires exactly one motif condition")
    if getattr(args, "protocol_id", None) != G_CODEC_PROTOCOL_ID:
        raise PF1TrainingError("G2 CLI protocol binding differs")
    checkpoint = Path(args.g1_checkpoint).expanduser().resolve()

    def executor(**engine_kwargs: Any) -> dict[str, object]:
        return execute_g2_geometry_to_motif_ce(
            g1_checkpoint=checkpoint,
            **engine_kwargs,
        )

    return run_pf1(args, executor=executor, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (PF1TrainingError, RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "G2_MANIFEST_NAME",
    "G2_MASK_PROBABILITY",
    "G2_OBJECTIVE_CONTRACT",
    "G2_PROTOCOL",
    "G2_PROTOCOL_ID",
    "REPORT_SCHEMA",
    "build_parser",
    "execute_g2_geometry_to_motif_ce",
    "main",
    "run",
]
