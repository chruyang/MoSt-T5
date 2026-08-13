"""Select canonical-local atom maps for one frozen anchored surface release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from most_t5_next.r1.adapter.graphports_donor_atom_map_sidecar_v1 import (
    iter_release_rows,
    write_release_row,
)


SCHEMA_VERSION = "most-t5-p2/anchored-donor-atom-map-subset/v1"
ROWS_NAME = "donor_atom_maps.jsonl"
MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_anchored_donor_atom_maps(
    *,
    surface_records: Path,
    source_donor_atom_maps: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir = Path(output_dir).expanduser().resolve()
    staging = output_dir.with_name(output_dir.name + ".staging")
    if output_dir.exists() or staging.exists():
        raise FileExistsError("anchored donor output or staging already exists")
    ordered: list[Mapping[str, object]] = []
    wanted: dict[str, Mapping[str, object]] = {}
    with Path(surface_records).open(encoding="utf-8") as handle:
        for expected_index, line in enumerate(handle):
            row = json.loads(line)
            key = row.get("storage_key") if isinstance(row, Mapping) else None
            surface = row.get("surface") if isinstance(row, Mapping) else None
            if (
                not isinstance(key, str)
                or not isinstance(surface, Mapping)
                or row.get("selection_index") != expected_index
                or row.get("split") not in {"train", "dev"}
                or key in wanted
            ):
                raise ValueError("anchored surface lineage is malformed")
            wanted[key] = row
            ordered.append(row)

    selected: dict[str, dict[str, object]] = {}
    scanned = 0
    for source in iter_release_rows(Path(source_donor_atom_maps)):
        scanned += 1
        key = str(source["storage_key"])
        target = wanted.get(key)
        if target is None:
            continue
        if source.get("member_id") != target["surface"]["member_id"]:  # type: ignore[index]
            raise ValueError("source donor member differs from anchored surface")
        rewritten = dict(source)
        rewritten["selection_index"] = target["selection_index"]
        rewritten["sdf_record_index"] = target["sdf_record_index"]
        rewritten["split"] = target["split"]
        selected[key] = rewritten
    if set(selected) != set(wanted):
        missing = len(set(wanted).difference(selected))
        raise ValueError(f"source donor maps miss {missing} anchored records")

    staging.mkdir(parents=True)
    rows_path = staging / ROWS_NAME
    train = dev = motifs = atoms = 0
    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for target in ordered:
            row = selected[str(target["storage_key"])]
            write_release_row(handle, row)
            train += row["split"] == "train"
            dev += row["split"] == "dev"
            motifs += int(row["motif_count"])
            sidecar = row["overlay_planning_sidecar"]
            atoms += sum(
                len(mapping)
                for mapping in sidecar["canonical_local_atom_to_model_atom"]  # type: ignore[index]
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "source": {
            "surface_records_sha256": _sha256(Path(surface_records)),
            "donor_atom_maps_sha256": _sha256(Path(source_donor_atom_maps)),
            "source_rows_scanned": scanned,
        },
        "counts": {
            "records": len(ordered),
            "train": train,
            "dev": dev,
            "motifs": motifs,
            "atoms": atoms,
        },
        "artifact": {
            "file": ROWS_NAME,
            "bytes": rows_path.stat().st_size,
            "sha256": _sha256(rows_path),
        },
        "contracts": {
            "selection_order_rewritten_to_anchored_surface": True,
            "canonical_local_atom_maps_unchanged": True,
            "chemistry_or_geometry_recomputed": False,
        },
    }
    (staging / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.rename(output_dir)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-records", type=Path, required=True)
    parser.add_argument("--source-donor-atom-maps", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            derive_anchored_donor_atom_maps(
                surface_records=args.surface_records,
                source_donor_atom_maps=args.source_donor_atom_maps,
                output_dir=args.output_dir,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["SCHEMA_VERSION", "derive_anchored_donor_atom_maps"]
