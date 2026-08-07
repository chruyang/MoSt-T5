#!/usr/bin/env python3
"""Merge four independently trained PF-1 condition manifests.

Each input must come from ``run_pf1_four_grid_v1 --condition-id``.  The merger
does not average or reinterpret results: it verifies one shared run contract,
orders A0/A1/M0/M1, and publishes the same four-grid manifest surface used by
the sequential runner.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Mapping, Sequence

from most_t5_next.p1.run_pf1_four_grid_v1 import (
    CONDITION_MANIFEST_NAME,
    CONDITION_ORDER,
    FOUR_GRID_MANIFEST_NAME,
    PF1TrainingError,
    REPORT_SCHEMA,
)


_COMMON_KEYS = (
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


def _load_manifest(path: Path) -> dict[str, object]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise PF1TrainingError("PF-1 condition manifest is absent: " + str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PF1TrainingError("PF-1 condition manifest is invalid: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise PF1TrainingError("PF-1 condition manifest root must be an object")
    return payload


def _shared_execution_contract(execution: Mapping[str, object]) -> dict[str, object]:
    return {
        key: copy.deepcopy(value)
        for key, value in execution.items()
        if key not in {"requested_conditions", "complete_four_grid"}
    }


def merge_pf1_condition_manifests(
    *, condition_manifests: Sequence[Path], output_dir: Path
) -> dict[str, object]:
    if len(condition_manifests) != len(CONDITION_ORDER):
        raise PF1TrainingError("exactly four PF-1 condition manifests are required")

    source_paths = tuple(Path(path).expanduser().resolve() for path in condition_manifests)
    reports = tuple(_load_manifest(path) for path in source_paths)
    reference = reports[0]
    if reference.get("schema_version") != REPORT_SCHEMA:
        raise PF1TrainingError("PF-1 condition report schema is invalid")

    by_condition: dict[str, dict[str, object]] = {}
    common_execution: dict[str, object] | None = None
    resumed_conditions: list[str] = []
    for path, report in zip(source_paths, reports):
        for key in _COMMON_KEYS:
            if report.get(key) != reference.get(key):
                raise PF1TrainingError(
                    "PF-1 condition manifests differ at shared field " + key
                )
        if report.get("status") != "pass":
            raise PF1TrainingError("PF-1 condition manifest is not passed")
        execution = report.get("execution")
        conditions = report.get("conditions")
        if not isinstance(execution, dict) or not isinstance(conditions, list):
            raise PF1TrainingError("PF-1 condition manifest structure is invalid")
        requested = execution.get("requested_conditions")
        if execution.get("complete_four_grid") is not False or not (
            isinstance(requested, list) and len(requested) == 1
        ):
            raise PF1TrainingError(
                "PF-1 merge inputs must each contain exactly one condition"
            )
        condition_id = requested[0]
        if condition_id not in CONDITION_ORDER or condition_id in by_condition:
            raise PF1TrainingError("PF-1 condition set is duplicated or unknown")
        if len(conditions) != 1 or not isinstance(conditions[0], dict):
            raise PF1TrainingError("PF-1 condition result cardinality is invalid")
        if conditions[0].get("condition") != condition_id:
            raise PF1TrainingError("PF-1 condition result label is inconsistent")
        current_execution = _shared_execution_contract(execution)
        if common_execution is None:
            common_execution = current_execution
        elif current_execution != common_execution:
            raise PF1TrainingError("PF-1 condition execution contracts differ")
        resumed = report.get("resumed_condition")
        if resumed is not None:
            if resumed != condition_id:
                raise PF1TrainingError("PF-1 resumed condition label is inconsistent")
            resumed_conditions.append(condition_id)
        result = copy.deepcopy(conditions[0])
        result["condition_manifest"] = str(path)
        by_condition[condition_id] = result

    if set(by_condition) != set(CONDITION_ORDER) or common_execution is None:
        raise PF1TrainingError("PF-1 merge requires A0, A1, M0 and M1 exactly once")

    merged = copy.deepcopy(reference)
    merged["conditions"] = [by_condition[name] for name in CONDITION_ORDER]
    merged["execution"] = {
        **common_execution,
        "requested_conditions": list(CONDITION_ORDER),
        "complete_four_grid": True,
        "merged_from_independent_condition_processes": True,
        "source_condition_manifests": [str(path) for path in source_paths],
    }
    merged["resumed_condition"] = (
        resumed_conditions[0] if len(resumed_conditions) == 1 else None
    )
    merged["resumed_conditions"] = resumed_conditions

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / FOUR_GRID_MANIFEST_NAME).write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition-manifest",
        action="append",
        required=True,
        help="repeat exactly four times for A0, A1, M0 and M1",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = merge_pf1_condition_manifests(
            condition_manifests=tuple(Path(path) for path in args.condition_manifest),
            output_dir=Path(args.output_dir),
        )
    except (PF1TrainingError, OSError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

