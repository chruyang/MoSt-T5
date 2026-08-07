#!/usr/bin/env python3
"""Adjudicate the paired PF-1 M0 GraphPorts-v1/v2 GPU screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from most_t5_next.p1.run_pf1_four_grid_v1 import REPORT_SCHEMA
from most_t5_next.p1.validate_pf1_graph_ports_codec_pair_v1 import (
    REPORT_SCHEMA as PAIR_REPORT_SCHEMA,
)


ADJUDICATION_SCHEMA = "most-t5-p1/pf1-graphports-codec-adjudication/v1"
MAX_NLL_RATIO = 1.02
MAX_ACCURACY_DROP = 0.01
MAX_ENCODER_TOKEN_RATIO = 0.70
MIN_MEMBER_THROUGHPUT_RATIO = 0.95
MAX_PEAK_MEMORY_RATIO = 1.05
MAX_LATE_NLL_RATIO = 1.01
MAX_LATE_ACCURACY_DROP = 0.005
RETAIN_NLL_RATIO = 1.05
RETAIN_ACCURACY_DROP = 0.02


class PF1GraphPortsAdjudicationError(RuntimeError):
    """The two GPU reports do not form the preregistered codec gate."""


def _load(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise PF1GraphPortsAdjudicationError("required report is absent: " + str(path))
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PF1GraphPortsAdjudicationError("report root must be an object")
    return payload


def _condition(report: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = report.get("conditions")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise PF1GraphPortsAdjudicationError("each GPU report must contain one condition")
    if rows[0].get("condition") != "M0":
        raise PF1GraphPortsAdjudicationError("codec gate accepts M0 only")
    return rows[0]


def _evaluations(condition: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = condition.get("evaluations")
    if not isinstance(rows, list):
        raise PF1GraphPortsAdjudicationError("condition evaluations are absent")
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("update"), int):
            raise PF1GraphPortsAdjudicationError("evaluation row is invalid")
        result[int(row["update"])] = row
    if set(result) != {0, 250, 500, 750, 1000}:
        raise PF1GraphPortsAdjudicationError("evaluation update set changed")
    return result


def _require_shared_contract(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    common_fields = (
        "schema_version",
        "status",
        "scope",
        "interpretation",
        "comparison_contract",
        "data",
        "optimization",
        "precision",
        "evaluation_updates",
        "checkpoint_updates",
    )
    if left.get("schema_version") != REPORT_SCHEMA or right.get("schema_version") != REPORT_SCHEMA:
        raise PF1GraphPortsAdjudicationError("training report schema is invalid")
    if any(left.get(field) != right.get(field) for field in common_fields):
        raise PF1GraphPortsAdjudicationError("GPU reports differ in shared training contract")
    if left.get("status") != "pass" or right.get("status") != "pass":
        raise PF1GraphPortsAdjudicationError("both GPU reports must pass")

    left_execution = left.get("execution")
    right_execution = right.get("execution")
    if not isinstance(left_execution, dict) or not isinstance(right_execution, dict):
        raise PF1GraphPortsAdjudicationError("execution contracts are absent")
    path_specific = {"union_tokenizer_dir", "requested_conditions", "complete_four_grid"}
    left_shared = {key: value for key, value in left_execution.items() if key not in path_specific}
    right_shared = {key: value for key, value in right_execution.items() if key not in path_specific}
    if left_shared != right_shared:
        raise PF1GraphPortsAdjudicationError("GPU execution contracts differ")
    if left_execution.get("requested_conditions") != ["M0"] or right_execution.get(
        "requested_conditions"
    ) != ["M0"]:
        raise PF1GraphPortsAdjudicationError("both runs must select M0 exactly")


def _require_exposure_parity(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> None:
    fields = (
        "optimizer_updates",
        "members_seen",
        "nominal_effective_batch_size",
        "short_microbatches",
        "min_microbatch_members",
        "max_microbatch_members",
        "mean_microbatch_members",
        "min_members_per_update",
        "max_members_per_update",
        "mean_members_per_update",
        "train_supervised_target_tokens",
        "final_data_cursor",
        "last_update_learning_rate",
    )
    if any(source.get(field) != target.get(field) for field in fields):
        raise PF1GraphPortsAdjudicationError(
            "member exposure, CE targets, schedule, or cursor differs"
        )
    if source.get("short_microbatches") is None:
        raise PF1GraphPortsAdjudicationError(
            "reports predate the standard tail-microbatch contract"
        )
    source_eval = _evaluations(source)
    target_eval = _evaluations(target)
    for update in source_eval:
        for field in ("members", "supervised_target_tokens"):
            if source_eval[update].get(field) != target_eval[update].get(field):
                raise PF1GraphPortsAdjudicationError(
                    "dev exposure or CE target count differs"
                )
    if source.get("final_e3fp_shuffle_diagnostic") is not None or target.get(
        "final_e3fp_shuffle_diagnostic"
    ) is not None:
        raise PF1GraphPortsAdjudicationError("M0 codec gate must not run geometry shuffle")


def adjudicate(
    *,
    pair_report: Path,
    source_manifest: Path,
    target_manifest: Path,
    output_report: Path,
) -> dict[str, Any]:
    pair = _load(pair_report)
    source_report = _load(source_manifest)
    target_report = _load(target_manifest)
    output_report = Path(output_report).expanduser().resolve()
    if output_report.exists():
        raise PF1GraphPortsAdjudicationError("output_report must be a new path")
    if not (
        pair.get("schema_version") == PAIR_REPORT_SCHEMA
        and pair.get("status") == "pass"
        and pair.get("decision_boundary", {}).get(
            "eligible_for_paired_m0_codec_screen"
        )
        is True
    ):
        raise PF1GraphPortsAdjudicationError("CPU codec-pair preflight did not pass")
    _require_shared_contract(source_report, target_report)
    source = _condition(source_report)
    target = _condition(target_report)
    _require_exposure_parity(source, target)

    source_eval = _evaluations(source)
    target_eval = _evaluations(target)
    source_final = source_eval[1000]
    target_final = target_eval[1000]
    target_late = target_eval[750]
    source_nll = float(source_final["token_weighted_nll"])
    target_nll = float(target_final["token_weighted_nll"])
    source_accuracy = float(source_final["masked_token_accuracy"])
    target_accuracy = float(target_final["masked_token_accuracy"])
    nll_ratio = target_nll / source_nll
    accuracy_drop = source_accuracy - target_accuracy
    encoder_token_ratio = float(target_final["encoder_nonpadding_tokens"]) / float(
        source_final["encoder_nonpadding_tokens"]
    )
    throughput_ratio = float(target["members_per_second"]) / float(
        source["members_per_second"]
    )
    peak_memory_ratio = float(target["peak_gpu_memory_bytes"]) / float(
        source["peak_gpu_memory_bytes"]
    )
    late_nll_ratio = target_nll / float(target_late["token_weighted_nll"])
    late_accuracy_drop = float(target_late["masked_token_accuracy"]) - target_accuracy

    gates = {
        "final_nll_ratio_at_most_1_02": nll_ratio <= MAX_NLL_RATIO,
        "final_accuracy_drop_at_most_0_01": accuracy_drop <= MAX_ACCURACY_DROP,
        "late_nll_not_worse_by_more_than_1_percent": late_nll_ratio <= MAX_LATE_NLL_RATIO,
        "late_accuracy_not_worse_by_more_than_0_005": late_accuracy_drop
        <= MAX_LATE_ACCURACY_DROP,
        "encoder_token_ratio_at_most_0_70": encoder_token_ratio
        <= MAX_ENCODER_TOKEN_RATIO,
        "member_throughput_ratio_at_least_0_95": throughput_ratio
        >= MIN_MEMBER_THROUGHPUT_RATIO,
        "peak_memory_ratio_at_most_1_05": peak_memory_ratio <= MAX_PEAK_MEMORY_RATIO,
    }
    severe_failure = (
        nll_ratio > RETAIN_NLL_RATIO
        or accuracy_drop > RETAIN_ACCURACY_DROP
        or (
            target_eval[750]["token_weighted_nll"]
            > source_eval[750]["token_weighted_nll"] * RETAIN_NLL_RATIO
            and target_eval[1000]["token_weighted_nll"]
            > source_eval[1000]["token_weighted_nll"] * RETAIN_NLL_RATIO
        )
    )
    if all(gates.values()):
        decision = "promote_graphports_v2"
    elif severe_failure:
        decision = "retain_graphports_v1"
    else:
        decision = "run_one_additional_paired_seed"

    report = {
        "schema_version": ADJUDICATION_SCHEMA,
        "status": "pass",
        "scope": "pf1_graphports_v1_v2_m0_codec_gate",
        "decision": decision,
        "metrics": {
            "source_final_nll": source_nll,
            "target_final_nll": target_nll,
            "final_nll_ratio": nll_ratio,
            "source_final_accuracy": source_accuracy,
            "target_final_accuracy": target_accuracy,
            "final_accuracy_drop": accuracy_drop,
            "encoder_token_ratio": encoder_token_ratio,
            "member_throughput_ratio": throughput_ratio,
            "peak_memory_ratio": peak_memory_ratio,
            "target_late_nll_ratio_1000_over_750": late_nll_ratio,
            "target_late_accuracy_drop_750_to_1000": late_accuracy_drop,
        },
        "promotion_gates": gates,
        "contracts": {
            "complete_cpu_pair_preflight": True,
            "same_m0_training_contract": True,
            "same_member_and_target_exposure": True,
            "same_final_cursor": True,
            "geometry_shuffle_absent": True,
            "old_pre_tail_batch_report_forbidden": True,
        },
        "interpretation": {
            "codec_only": True,
            "automatic_architecture_superiority_claim": False,
            "atom_vs_motif_raw_ce_comparison": False,
            "grey_zone_allows_exactly_one_additional_paired_seed": True,
        },
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    staging = output_report.with_name(output_report.name + ".staging")
    if staging.exists():
        raise PF1GraphPortsAdjudicationError("output report staging path exists")
    with staging.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    staging.rename(output_report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-report", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--output-report", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = adjudicate(
        pair_report=Path(args.pair_report),
        source_manifest=Path(args.source_manifest),
        target_manifest=Path(args.target_manifest),
        output_report=Path(args.output_report),
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PF1GraphPortsAdjudicationError", "adjudicate"]
