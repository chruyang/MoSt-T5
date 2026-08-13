"""Freeze 128 real PF-10 records for the factorized GPU wiring smoke.

Selection is deterministic and coverage-oriented: 128 evenly spaced rows from
the published train common-state membership, including both endpoints.  This
is a runtime smoke sample, not a training/evaluation subset or a model-selection
sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .build_pf10_morgan_overlay_v1 import (
    COMMON_TRAIN_MEMBERSHIP,
    MANIFEST_NAME as OVERLAY_MANIFEST_NAME,
    SCHEMA_VERSION as OVERLAY_SCHEMA_VERSION,
)


SCHEMA_VERSION = "most-t5-p2/pf10-factorized-smoke-membership/v1"
SMOKE_COUNT = 128
MEMBERSHIP_NAME = "membership.jsonl"
MANIFEST_NAME = "manifest.json"


class PF10SmokeMembershipError(ValueError):
    """The common state domain cannot provide the frozen real smoke sample."""


def evenly_spaced_indices(total: int, count: int = SMOKE_COUNT) -> tuple[int, ...]:
    if isinstance(total, bool) or not isinstance(total, int) or total < count:
        raise PF10SmokeMembershipError("total must cover the requested smoke count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise PF10SmokeMembershipError("count must be an integer of at least two")
    indices = tuple((index * (total - 1)) // (count - 1) for index in range(count))
    if len(set(indices)) != count or indices[0] != 0 or indices[-1] != total - 1:
        raise PF10SmokeMembershipError("evenly spaced schedule is not unique or closed")
    return indices


def freeze_pf10_factorized_smoke128(
    *,
    morgan_overlay: Path,
    output_dir: Path,
) -> dict[str, object]:
    overlay = Path(morgan_overlay).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise PF10SmokeMembershipError("smoke membership output must be a new path")
    manifest = json.loads(
        (overlay / OVERLAY_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != OVERLAY_SCHEMA_VERSION or manifest.get("status") != "pass":
        raise PF10SmokeMembershipError("Morgan overlay is not a passed release")
    common_count = int(manifest["counts"]["common_train_records"])
    selected_indices = evenly_spaced_indices(common_count)
    selected_lookup = {source_index: smoke_index for smoke_index, source_index in enumerate(selected_indices)}
    selected: list[dict[str, object] | None] = [None] * SMOKE_COUNT
    source_path = overlay / COMMON_TRAIN_MEMBERSHIP
    observed = 0
    with source_path.open("r", encoding="utf-8") as handle:
        for common_index, line in enumerate(handle):
            if common_index in selected_lookup:
                row = json.loads(line)
                smoke_index = selected_lookup[common_index]
                selected[smoke_index] = {
                    "schema_version": SCHEMA_VERSION,
                    "smoke_index": smoke_index,
                    "common_membership_index": common_index,
                    "split": "train",
                    "split_index": int(row["split_index"]),
                    "selection_index": int(row["selection_index"]),
                    "record_id": str(row["record_id"]),
                    "storage_key": str(row["storage_key"]),
                    "eligible_motifs": row["eligible_motifs"],
                }
            observed += 1
    if observed != common_count or any(row is None for row in selected):
        raise PF10SmokeMembershipError("common membership count or selection changed")
    output.mkdir(parents=True)
    with (output / MEMBERSHIP_NAME).open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    output_manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "source": {
            "morgan_overlay": str(overlay),
            "common_train_records": common_count,
        },
        "counts": {"smoke_records": SMOKE_COUNT},
        "selection": {
            "method": "closed_evenly_spaced_common_train_indices",
            "first_common_index": selected_indices[0],
            "last_common_index": selected_indices[-1],
            "unique": True,
            "model_or_label_signal_used": False,
        },
        "scope": {
            "runtime_wiring_smoke_only": True,
            "effect_size_or_model_selection": False,
        },
    }
    (output / MANIFEST_NAME).write_text(
        json.dumps(output_manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morgan-overlay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    freeze_pf10_factorized_smoke128(
        morgan_overlay=args.morgan_overlay,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MANIFEST_NAME",
    "MEMBERSHIP_NAME",
    "PF10SmokeMembershipError",
    "SCHEMA_VERSION",
    "SMOKE_COUNT",
    "evenly_spaced_indices",
    "freeze_pf10_factorized_smoke128",
]
