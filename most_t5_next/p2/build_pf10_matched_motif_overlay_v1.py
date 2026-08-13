"""Build a split-scoped matched-motif state overlay used by PF-10 F3D.

The publisher does not alter the paired release.  It replaces E3FP rows only
inside motif occurrences that have a cross-molecule donor with the same
frozen motif identity, atom count, canonical-local port pattern and incident
bond types.  Unmatched occurrences remain aligned and their coverage is
reported explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from most_t5_next.p1.build_pf1_paired_release_v1 import PF1PairedReleaseReader

from .matched_motif_state_donor_v1 import (
    DONOR_PLAN_ID,
    build_matched_motif_donor_plan,
    materialize_matched_state_overlay,
)


SCHEMA_VERSION = "most-t5-p2/pf10-matched-motif-state-overlay/v1"
ROW_SCHEMA = "most-t5-p2/pf10-matched-motif-state-row/v1"
MANIFEST_NAME = "manifest.json"
ROWS_NAME = "matched_state_rows.jsonl"


class PF10MatchedMotifOverlayError(RuntimeError):
    """The matched-motif diagnostic overlay could not be built or loaded."""


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _load_manifest(root: Path) -> dict[str, object]:
    path = Path(root) / MANIFEST_NAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PF10MatchedMotifOverlayError("matched overlay manifest is absent or invalid") from exc
    if not isinstance(value, dict) or not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("status") == "pass"
    ):
        raise PF10MatchedMotifOverlayError("matched overlay is not a passed release")
    return value


def build_pf10_matched_motif_overlay(
    *,
    paired_release: Path,
    output_dir: Path,
    split: str = "dev",
    lmdb_module: Any | None = None,
) -> dict[str, object]:
    """Plan and publish one deterministic overlay over a complete frozen split."""

    paired_release = Path(paired_release).expanduser().absolute()
    output_dir = Path(output_dir).expanduser().absolute()
    staging = output_dir.with_name(output_dir.name + ".staging")
    if output_dir.exists() or staging.exists():
        raise PF10MatchedMotifOverlayError("output or sibling staging path already exists")
    if split not in {"train", "dev"}:
        raise PF10MatchedMotifOverlayError("matched overlay split must be train or dev")

    reader = PF1PairedReleaseReader(paired_release, lmdb_module=lmdb_module)
    expected_records = (
        reader.train_member_count if split == "train" else reader.dev_member_count
    )
    sidecars = iter(reader.iter_donor_atom_maps(split=split))

    documents: list[dict[str, object]] = []
    record_ids: list[str] = []
    for split_index, (membership, raw_document) in enumerate(
        reader.iter_raw_motif_documents(split=split)
    ):
        try:
            sidecar = next(sidecars)
        except StopIteration as exc:
            raise PF10MatchedMotifOverlayError(
                f"{split} donor-map stream ended before paired membership"
            ) from exc
        if not (
            sidecar["member_id"] == membership["member_id"]
            and sidecar["storage_key"] == membership["storage_key"]
        ):
            raise PF10MatchedMotifOverlayError(
                f"{split} donor-map order differs from paired membership"
            )
        document = dict(raw_document)
        document["overlay_planning_sidecar"] = sidecar["overlay_planning_sidecar"]
        documents.append(document)
        record_ids.append(str(membership["member_id"]))
    if len(documents) != expected_records:
        raise PF10MatchedMotifOverlayError(
            f"raw {split} replay did not exhaust membership"
        )
    try:
        next(sidecars)
    except StopIteration:
        pass
    else:
        raise PF10MatchedMotifOverlayError(
            f"{split} donor-map stream contains rows beyond paired membership"
        )

    plan = build_matched_motif_donor_plan(documents, strict_neighbors=False)
    overlay = materialize_matched_state_overlay(documents, plan)
    staging.mkdir(parents=True, exist_ok=False)
    row_count = 0
    changed_records = 0
    with (staging / ROWS_NAME).open("w", encoding="utf-8", newline="\n") as handle:
        for split_index, record_id in enumerate(record_ids):
            state = overlay.state_by_record_id[record_id]
            changed_motifs = overlay.changed_motifs_by_record_id.get(record_id, ())
            if changed_motifs:
                changed_records += 1
            handle.write(
                _json_text(
                    {
                        "schema_version": ROW_SCHEMA,
                        "split_index": split_index,
                        "record_id": record_id,
                        "changed_motif_indices": list(changed_motifs),
                        "state_ids": [list(row) for row in state],
                    }
                )
                + "\n"
            )
            row_count += 1
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": f"pf10_{split}_matched_motif_state_overlay",
        "split": split,
        "state_kind": "matched_motif_e3fp",
        "source_paired_release": str(paired_release),
        "rows_file": ROWS_NAME,
        "donor_plan_id": DONOR_PLAN_ID,
        "matching_signature": [
            "exact_motif_identity",
            "atom_count",
            "canonical_local_port_degree_pattern",
            "incident_bond_types",
        ],
        "strict_neighbor_match": False,
        "counts": {
            f"{split}_records": expected_records,
            "published_rows": row_count,
            "records_with_any_matched_motif": changed_records,
            "changed_state_slots": overlay.changed_state_slot_count,
        },
        "coverage": dict(plan.coverage),
        "semantics": {
            "cross_molecule_donors_only": True,
            "canonical_local_atom_correspondence": True,
            "unmatched_motifs_keep_aligned_state": True,
            "paired_release_unchanged": True,
            "dev_diagnostic_only": split == "dev",
            "training_counterfactual_overlay": split == "train",
        },
    }
    with (staging / MANIFEST_NAME).open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(_json_text(manifest) + "\n")
    staging.rename(output_dir)
    return manifest


class MatchedMotifStateProvider:
    """Small eager dev provider implementing the factorized collator protocol."""

    state_kind = "matched_motif_e3fp"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().absolute()
        self.manifest = _load_manifest(self.root)
        rows_path = self.root / str(self.manifest["rows_file"])
        state_by_id: dict[str, tuple[tuple[int, int, int, int], ...]] = {}
        changed_motifs_by_id: dict[str, tuple[int, ...]] = {}
        with rows_path.open("r", encoding="utf-8") as handle:
            for split_index, line in enumerate(handle):
                raw = json.loads(line)
                if not isinstance(raw, dict) or not (
                    raw.get("schema_version") == ROW_SCHEMA
                    and raw.get("split_index") == split_index
                    and isinstance(raw.get("record_id"), str)
                    and raw["record_id"] not in state_by_id
                ):
                    raise PF10MatchedMotifOverlayError("matched overlay row order is invalid")
                try:
                    state = tuple(tuple(int(value) for value in row) for row in raw["state_ids"])
                except (TypeError, ValueError) as exc:
                    raise PF10MatchedMotifOverlayError("matched overlay state is malformed") from exc
                if not state or any(len(row) != 4 for row in state):
                    raise PF10MatchedMotifOverlayError("matched overlay state width differs")
                try:
                    changed_motifs = tuple(int(value) for value in raw["changed_motif_indices"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise PF10MatchedMotifOverlayError(
                        "matched overlay changed-motif indices are malformed"
                    ) from exc
                if (
                    tuple(sorted(set(changed_motifs))) != changed_motifs
                    or (changed_motifs and changed_motifs[0] < 0)
                ):
                    raise PF10MatchedMotifOverlayError(
                        "matched overlay changed-motif indices are invalid"
                    )
                record_id = str(raw["record_id"])
                state_by_id[record_id] = state
                changed_motifs_by_id[record_id] = changed_motifs
        expected = int(self.manifest["counts"]["published_rows"])  # type: ignore[index]
        if len(state_by_id) != expected:
            raise PF10MatchedMotifOverlayError("matched overlay row count differs")
        self._state_by_id = state_by_id
        self._changed_motifs_by_id = changed_motifs_by_id

    def get(self, record_id: str) -> Sequence[Sequence[int]]:
        try:
            return self._state_by_id[record_id]
        except KeyError as exc:
            raise PF10MatchedMotifOverlayError("record is absent from matched overlay") from exc

    def changed_motif_indices(self, record_id: str) -> tuple[int, ...]:
        """Return the exact logical motifs whose atom-state rows were replaced."""

        try:
            return self._changed_motifs_by_id[record_id]
        except KeyError as exc:
            raise PF10MatchedMotifOverlayError(
                "record is absent from matched overlay"
            ) from exc

    def close(self) -> None:
        self._state_by_id.clear()
        self._changed_motifs_by_id.clear()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_pf10_matched_motif_overlay(
        paired_release=args.paired_release,
        output_dir=args.output_dir,
        split=args.split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MatchedMotifStateProvider",
    "PF10MatchedMotifOverlayError",
    "SCHEMA_VERSION",
    "build_pf10_matched_motif_overlay",
]
