"""Streaming GraphPorts canonical-local atom maps for PF-10 donor planning.

The paired training wire intentionally omits ``MotifRecord.source_atom_map``
because the model never consumes it.  F3D matched-state corruption does need
that map to establish an atom correspondence across different molecules.  This
module publishes the map as a separate JSONL planning artifact: one bounded row
is created beside each paired record, and readers consume it one row at a time.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Iterator, Mapping


SIDECAR_SCHEMA = "most-t5-p2/graphports-donor-atom-map-sidecar/v1"
ROW_SCHEMA = "most-t5-p1/graphports-donor-atom-map-row/v1"
DEFAULT_BENCHMARK_ROWS = 1024


class GraphPortsDonorAtomMapError(ValueError):
    """A GraphPorts encoding or persisted planning row is malformed."""


def _canonical_json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_graphports_donor_atom_map_sidecar(
    graph_encoding: object,
) -> dict[str, object]:
    """Serialize canonical local atom IDs to projected model atom rows."""

    try:
        motifs = tuple(graph_encoding.motifs)  # type: ignore[attr-defined]
        format_version = str(graph_encoding.format_version)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise GraphPortsDonorAtomMapError(
            "GraphPorts encoding fields are malformed"
        ) from exc
    try:
        motif_ids = tuple(int(motif.motif_id) for motif in motifs)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as exc:
        raise GraphPortsDonorAtomMapError("GraphPorts motif IDs are malformed") from exc
    if not motifs or motif_ids != tuple(range(len(motifs))):
        raise GraphPortsDonorAtomMapError(
            "GraphPorts motifs must be in contiguous frozen-logical order"
        )

    rows: list[list[list[int]]] = []
    for motif in motifs:
        try:
            source_map = tuple(
                (int(local_id), int(model_atom))
                for local_id, model_atom in motif.source_atom_map  # type: ignore[attr-defined]
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise GraphPortsDonorAtomMapError(
                "GraphPorts motif source_atom_map is malformed"
            ) from exc
        if tuple(local_id for local_id, _model_atom in source_map) != tuple(
            range(1, len(source_map) + 1)
        ):
            raise GraphPortsDonorAtomMapError(
                "GraphPorts canonical local atom IDs must be contiguous"
            )
        model_atoms = tuple(model_atom for _local_id, model_atom in source_map)
        if (
            not model_atoms
            or any(model_atom < 0 for model_atom in model_atoms)
            or len(set(model_atoms)) != len(model_atoms)
        ):
            raise GraphPortsDonorAtomMapError(
                "GraphPorts source_atom_map has an invalid model atom axis"
            )
        rows.append([[local_id, model_atom] for local_id, model_atom in source_map])
    return {
        "schema_version": SIDECAR_SCHEMA,
        "source_codec_format_version": format_version,
        "canonical_local_atom_to_model_atom": rows,
    }


def build_release_row(
    *,
    selection_index: int,
    member_id: str,
    sdf_record_index: int,
    split: str,
    storage_key: str,
    graph_encoding: object,
) -> dict[str, object]:
    """Build one selection-ordered row without retaining a molecule."""

    if (
        isinstance(selection_index, bool)
        or not isinstance(selection_index, int)
        or selection_index < 0
        or isinstance(sdf_record_index, bool)
        or not isinstance(sdf_record_index, int)
        or sdf_record_index < 0
        or not isinstance(member_id, str)
        or not member_id
        or not isinstance(storage_key, str)
        or not storage_key
        or split not in {"train", "dev"}
    ):
        raise GraphPortsDonorAtomMapError("planning row lineage is malformed")
    sidecar = build_graphports_donor_atom_map_sidecar(graph_encoding)
    return {
        "schema_version": ROW_SCHEMA,
        "selection_index": selection_index,
        "member_id": member_id,
        "sdf_record_index": sdf_record_index,
        "split": split,
        "storage_key": storage_key,
        "motif_count": len(sidecar["canonical_local_atom_to_model_atom"]),  # type: ignore[arg-type]
        "overlay_planning_sidecar": sidecar,
    }


def validate_release_row(raw: object) -> dict[str, object]:
    """Validate one decoded JSONL row and return an ordinary mapping."""

    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "selection_index",
        "member_id",
        "sdf_record_index",
        "split",
        "storage_key",
        "motif_count",
        "overlay_planning_sidecar",
    }:
        raise GraphPortsDonorAtomMapError("planning row fields differ")
    if raw.get("schema_version") != ROW_SCHEMA:
        raise GraphPortsDonorAtomMapError("planning row schema differs")
    selection_index = raw.get("selection_index")
    sdf_record_index = raw.get("sdf_record_index")
    motif_count = raw.get("motif_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (selection_index, sdf_record_index)
    ):
        raise GraphPortsDonorAtomMapError("planning row integer lineage differs")
    if (
        isinstance(motif_count, bool)
        or not isinstance(motif_count, int)
        or motif_count <= 0
        or not isinstance(raw.get("member_id"), str)
        or not raw["member_id"]
        or not isinstance(raw.get("storage_key"), str)
        or not raw["storage_key"]
        or raw.get("split") not in {"train", "dev"}
    ):
        raise GraphPortsDonorAtomMapError("planning row lineage differs")
    sidecar = raw.get("overlay_planning_sidecar")
    if not isinstance(sidecar, dict) or set(sidecar) != {
        "schema_version",
        "source_codec_format_version",
        "canonical_local_atom_to_model_atom",
    }:
        raise GraphPortsDonorAtomMapError("planning sidecar fields differ")
    if sidecar.get("schema_version") != SIDECAR_SCHEMA:
        raise GraphPortsDonorAtomMapError("planning sidecar schema differs")
    maps = sidecar.get("canonical_local_atom_to_model_atom")
    if not isinstance(maps, list) or len(maps) != motif_count:
        raise GraphPortsDonorAtomMapError("planning sidecar motif count differs")
    seen_model_atoms: set[int] = set()
    for atom_map in maps:
        if not isinstance(atom_map, list) or not atom_map:
            raise GraphPortsDonorAtomMapError("planning atom map is empty or malformed")
        expected_local_id = 1
        for pair in atom_map:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or pair[0] != expected_local_id
                or isinstance(pair[1], bool)
                or not isinstance(pair[1], int)
                or pair[1] < 0
                or pair[1] in seen_model_atoms
            ):
                raise GraphPortsDonorAtomMapError(
                    "planning canonical/model atom mapping differs"
                )
            seen_model_atoms.add(pair[1])
            expected_local_id += 1
    if seen_model_atoms != set(range(len(seen_model_atoms))):
        raise GraphPortsDonorAtomMapError(
            "planning model atom maps do not form the complete row axis"
        )
    return raw


def write_release_row(handle: object, row: Mapping[str, object]) -> int:
    """Validate and append one canonical JSONL row; return encoded bytes."""

    validated = validate_release_row(dict(row))
    text = _canonical_json_text(validated) + "\n"
    try:
        handle.write(text)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise GraphPortsDonorAtomMapError("planning sidecar handle is not writable") from exc
    return len(text.encode("utf-8"))


def iter_release_rows(path: Path) -> Iterator[dict[str, object]]:
    """Stream validated planning rows without materializing the release."""

    with Path(path).open("r", encoding="utf-8") as handle:
        previous_index = -1
        for line_number, line in enumerate(handle, 1):
            try:
                raw = json.loads(line)
            except (TypeError, ValueError) as exc:
                raise GraphPortsDonorAtomMapError(
                    f"planning sidecar line {line_number} is invalid JSON"
                ) from exc
            row = validate_release_row(raw)
            selection_index = int(row["selection_index"])
            if selection_index != previous_index + 1:
                raise GraphPortsDonorAtomMapError(
                    "planning sidecar selection order is not dense"
                )
            previous_index = selection_index
            yield row


def benchmark_release_prefix(
    path: Path, *, max_rows: int = DEFAULT_BENCHMARK_ROWS
) -> dict[str, int | float | bool]:
    """Replay a bounded prefix to validate the 1,024-row streaming boundary."""

    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
        raise GraphPortsDonorAtomMapError("max_rows must be positive")
    started = time.perf_counter()
    rows = 0
    motifs = 0
    atom_mappings = 0
    for row in iter_release_rows(path):
        rows += 1
        motifs += int(row["motif_count"])
        sidecar = row["overlay_planning_sidecar"]
        atom_mappings += sum(
            len(atom_map)
            for atom_map in sidecar["canonical_local_atom_to_model_atom"]  # type: ignore[index]
        )
        if rows == max_rows:
            break
    elapsed = time.perf_counter() - started
    return {
        "requested_max_rows": max_rows,
        "rows_replayed": rows,
        "motifs_replayed": motifs,
        "atom_mappings_replayed": atom_mappings,
        "bounded_prefix_complete": rows == max_rows,
        "seconds": elapsed,
        "rows_per_second": rows / elapsed if elapsed else 0.0,
    }


__all__ = [
    "DEFAULT_BENCHMARK_ROWS",
    "GraphPortsDonorAtomMapError",
    "ROW_SCHEMA",
    "SIDECAR_SCHEMA",
    "benchmark_release_prefix",
    "build_graphports_donor_atom_map_sidecar",
    "build_release_row",
    "iter_release_rows",
    "validate_release_row",
    "write_release_row",
]
