"""Join standard QM9 properties onto the frozen anchored probe records.

The anchored records remain immutable.  This module emits a compact overlay
keyed by ``record_id`` after a canonical-isomeric-SMILES join against the
official QM9 SDF/CSV pair.  Ambiguous or missing identities are reported and
never guessed from row order alone.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from rdkit import Chem


SCHEMA_VERSION = "most-t5-p2/qm9-standard-target-overlay/v1"
TARGETS = (
    ("mu", "D"),
    ("alpha", "bohr^3"),
    ("r2", "bohr^2"),
    ("u0", "hartree"),
    ("u0_atom", "kcal/mol"),
)


class QM9StandardTargetOverlayError(ValueError):
    """The standard QM9 source cannot be joined without ambiguity."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_smiles_from_text(smiles: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise QM9StandardTargetOverlayError("probe SMILES must be non-empty")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise QM9StandardTargetOverlayError("probe SMILES is not RDKit-readable")
    return Chem.MolToSmiles(
        Chem.RemoveHs(mol), canonical=True, isomericSmiles=True
    )


def _canonical_smiles_from_mol(mol: Chem.Mol) -> str:
    if mol is None:
        raise QM9StandardTargetOverlayError("standard QM9 SDF contains null molecule")
    normalized = Chem.Mol(mol)
    sanitize_result = Chem.SanitizeMol(normalized, catchErrors=True)
    if sanitize_result == Chem.SanitizeFlags.SANITIZE_NONE:
        heavy = Chem.RemoveHs(normalized)
    else:
        # A handful of published QM9 rows fail current RDKit sanitization.
        # Preserve their stored graph for explicit downstream coverage
        # accounting instead of making the whole official SDF unreadable.
        heavy = Chem.RemoveHs(mol, sanitize=False)
    return Chem.MolToSmiles(
        heavy, canonical=True, isomericSmiles=True
    )


def _read_probe_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise QM9StandardTargetOverlayError("probe row must be an object")
            record = row.get("record")
            if not isinstance(record, dict):
                raise QM9StandardTargetOverlayError("probe row lacks record document")
            record_id = record.get("record_id")
            if not isinstance(record_id, str) or record_id in seen_ids:
                raise QM9StandardTargetOverlayError("probe record_id is invalid or duplicated")
            seen_ids.add(record_id)
            records.append({
                "line_index": line_index,
                "record_id": record_id,
                "storage_key": record.get("storage_key"),
                "split": row.get("split"),
                "source_smiles": row.get("smiles"),
                "canonical_smiles": _canonical_smiles_from_text(str(row.get("smiles", ""))),
            })
    if not records:
        raise QM9StandardTargetOverlayError("probe cache is empty")
    return records


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"mol_id", *(name for name, _ in TARGETS)}
    if not rows or not required.issubset(rows[0]):
        raise QM9StandardTargetOverlayError("QM9 CSV header is incomplete")
    return rows


def build_overlay(
    *,
    probe_records_path: Path,
    qm9_sdf_path: Path,
    qm9_csv_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    probe_records = _read_probe_records(probe_records_path)
    wanted = {str(row["canonical_smiles"]) for row in probe_records}
    csv_rows = _read_csv_rows(qm9_csv_path)
    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)

    observed_sdf = 0
    with qm9_sdf_path.open("rb") as sdf_handle:
        supplier = Chem.ForwardSDMolSupplier(
            sdf_handle, sanitize=False, removeHs=False, strictParsing=True
        )
        for index, mol in enumerate(supplier):
            if index >= len(csv_rows):
                raise QM9StandardTargetOverlayError("SDF has more rows than CSV")
            if mol is None:
                raise QM9StandardTargetOverlayError(
                    f"standard QM9 SDF row {index} is unreadable"
                )
            csv_row = csv_rows[index]
            mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            if mol_id != csv_row["mol_id"]:
                raise QM9StandardTargetOverlayError("SDF/CSV mol_id order differs")
            canonical = _canonical_smiles_from_mol(mol)
            if canonical in wanted:
                targets = {name: float(csv_row[name]) for name, _ in TARGETS}
                candidates[canonical].append({
                    "qm9_mol_id": mol_id,
                    "qm9_row_index": index,
                    "targets": targets,
                })
            observed_sdf += 1
    if observed_sdf != len(csv_rows):
        raise QM9StandardTargetOverlayError("SDF and CSV row counts differ")

    overlays: list[dict[str, object]] = []
    rejects: list[dict[str, object]] = []
    split_joined: Counter[str] = Counter()
    for probe in probe_records:
        matches = candidates.get(str(probe["canonical_smiles"]), [])
        if len(matches) != 1:
            rejects.append({
                "line_index": probe["line_index"],
                "record_id": probe["record_id"],
                "split": probe["split"],
                "canonical_smiles": probe["canonical_smiles"],
                "reason": "MISSING_STANDARD_IDENTITY" if not matches else "AMBIGUOUS_STANDARD_IDENTITY",
                "candidate_count": len(matches),
            })
            continue
        match = matches[0]
        overlay = {
            "schema_version": SCHEMA_VERSION,
            "record_id": probe["record_id"],
            "storage_key": probe["storage_key"],
            "split": probe["split"],
            "canonical_smiles": probe["canonical_smiles"],
            "qm9_mol_id": match["qm9_mol_id"],
            "qm9_row_index": match["qm9_row_index"],
            "targets": match["targets"],
        }
        overlay["overlay_sha256"] = hashlib.sha256(
            _canonical_json_bytes(overlay)
        ).hexdigest()
        overlays.append(overlay)
        split_joined[str(probe["split"])] += 1

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if not rejects else "partial",
        "probe_records": len(probe_records),
        "joined_records": len(overlays),
        "rejected_records": len(rejects),
        "joined_by_split": dict(sorted(split_joined.items())),
        "standard_qm9_rows": observed_sdf,
        "wanted_canonical_identities": len(wanted),
        "matched_canonical_identities": sum(
            1 for identity in wanted if len(candidates.get(identity, [])) == 1
        ),
        "ambiguous_canonical_identities": sum(
            1 for identity in wanted if len(candidates.get(identity, [])) > 1
        ),
        "target_units": {name: unit for name, unit in TARGETS},
    }
    return overlays, rejects, report


def write_overlay(
    *,
    probe_cache_root: Path,
    qm9_sdf_path: Path,
    qm9_csv_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    probe_records_path = probe_cache_root / "records.jsonl"
    if output_dir.exists():
        raise QM9StandardTargetOverlayError("output directory must be new")
    output_dir.mkdir(parents=True)
    overlays, rejects, report = build_overlay(
        probe_records_path=probe_records_path,
        qm9_sdf_path=qm9_sdf_path,
        qm9_csv_path=qm9_csv_path,
    )
    overlay_path = output_dir / "targets.jsonl"
    reject_path = output_dir / "rejects.jsonl"
    with overlay_path.open("wb") as handle:
        for row in overlays:
            handle.write(_canonical_json_bytes(row) + b"\n")
    with reject_path.open("wb") as handle:
        for row in rejects:
            handle.write(_canonical_json_bytes(row) + b"\n")
    report["inputs"] = {
        "probe_records_sha256": _sha256_file(probe_records_path),
        "qm9_sdf_sha256": _sha256_file(qm9_sdf_path),
        "qm9_csv_sha256": _sha256_file(qm9_csv_path),
    }
    report["artifacts"] = {
        "targets_file": overlay_path.name,
        "targets_sha256": _sha256_file(overlay_path),
        "rejects_file": reject_path.name,
        "rejects_sha256": _sha256_file(reject_path),
    }
    (output_dir / "manifest.json").write_bytes(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-cache-root", required=True, type=Path)
    parser.add_argument("--qm9-sdf", required=True, type=Path)
    parser.add_argument("--qm9-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = write_overlay(
        probe_cache_root=args.probe_cache_root.resolve(),
        qm9_sdf_path=args.qm9_sdf.resolve(),
        qm9_csv_path=args.qm9_csv.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
