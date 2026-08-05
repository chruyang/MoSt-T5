#!/usr/bin/env python3
"""Bounded, metadata-only diagnosis for a failed PCQM4Mv2 identity smoke.

This helper is intentionally *not* a data adapter.  It streams a bounded
prefix from the frozen OGB train-3D archive, compares it to the frozen
official CSV companion, and writes only mismatch indices, graph summaries and
hashes.  It never extracts the SDF, writes an LMDB, or emits raw molecular
strings.  It is useful for distinguishing an ordinal-mapping error from a
representation or source inconsistency before any full-pass is designed.
"""

from __future__ import print_function

import argparse
import collections
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

from pcqm_identity_smoke import (
    MAX_SMOKE_RECORDS,
    canonical_forms,
    find_sdf_member,
    load_split_dict,
    regular_file,
    scalar_to_int,
    sha256_text,
    utc_now,
    write_json_atomic,
)


def graph_summary(Chem, mol, forms):
    """Return comparison-relevant graph statistics without raw molecule text."""
    normalized = Chem.RemoveHs(Chem.Mol(mol))
    Chem.SanitizeMol(normalized)
    atom_histogram = collections.Counter(
        "{}:{}".format(atom.GetAtomicNum(), atom.GetFormalCharge())
        for atom in normalized.GetAtoms()
    )
    bond_histogram = collections.Counter(
        "{}".format(bond.GetBondType()) for bond in normalized.GetBonds()
    )
    return {
        "atom_count": int(normalized.GetNumAtoms()),
        "heavy_atom_count": int(normalized.GetNumHeavyAtoms()),
        "bond_count": int(normalized.GetNumBonds()),
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in normalized.GetAtoms())),
        "atom_number_charge_histogram": dict(sorted(atom_histogram.items())),
        "bond_type_histogram": dict(sorted(bond_histogram.items())),
        "strict_smiles_sha256": sha256_text(forms["strict"]),
        "connectivity_smiles_sha256": sha256_text(forms["connectivity"]),
    }


def heavy_atom_projection_forms(Chem, mol):
    """Canonical forms after an explicit, broad RDKit hydrogen projection.

    Default ``Chem.RemoveHs`` deliberately retains wedged/stereo-defining or
    mapped H atoms.  That is right for preserving a drawing, but it can make
    otherwise identical heavy-atom graphs appear different when compared with
    a CSV SMILES whose hydrogen is implicit.  We use RDKit's own broad removal
    parameters instead of manually deleting atoms: the latter can invalidate
    aromatic bookkeeping before sanitization.  This is diagnostic only; it
    does not transform a source record or define the future feature adapter.
    """
    initial_hydrogens = sum(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
    parameters = Chem.RemoveHsParameters()
    for option in (
        "removeAndTrackIsotopes",
        "removeDefiningBondStereo",
        "removeDegreeZero",
        "removeHigherDegrees",
        "removeHydrides",
        "removeInSGroups",
        "removeIsotopes",
        "removeMapped",
        "removeNonimplicit",
        "removeNontetrahedralNeighbors",
        "removeWithQuery",
        "removeWithWedgedBond",
    ):
        if hasattr(parameters, option):
            setattr(parameters, option, True)
    projected = Chem.RemoveHs(Chem.Mol(mol), parameters, sanitize=True)
    residual_hydrogens = sum(atom.GetAtomicNum() == 1 for atom in projected.GetAtoms())
    Chem.AssignStereochemistry(projected, cleanIt=True, force=True)
    return {
        "strict": Chem.MolToSmiles(projected, canonical=True, isomericSmiles=True),
        "connectivity": Chem.MolToSmiles(projected, canonical=True, isomericSmiles=False),
        "input_explicit_hydrogens": int(initial_hydrogens),
        "residual_explicit_hydrogens": int(residual_hydrogens),
    }


def load_sdf_prefix(Chem, archive_path, max_records):
    import tarfile

    records = []
    with tarfile.open(str(archive_path), mode="r|gz") as archive:
        member = find_sdf_member(archive)
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError("cannot open SDF member")
        try:
            supplier = Chem.ForwardSDMolSupplier(stream, sanitize=True, removeHs=False)
            for ordinal, mol in enumerate(supplier):
                if mol is None:
                    records.append({"ordinal": int(ordinal), "status": "sdf_rdkit_none"})
                else:
                    try:
                        forms = canonical_forms(Chem, mol)
                        heavy_projection = heavy_atom_projection_forms(Chem, mol)
                        records.append(
                            {
                                "ordinal": int(ordinal),
                                "status": "ok",
                                "forms": forms,
                                "heavy_projection": heavy_projection,
                                "summary": graph_summary(Chem, mol, forms),
                            }
                        )
                    except Exception as exc:
                        records.append(
                            {
                                "ordinal": int(ordinal),
                                "status": "sdf_canonicalization_error:{}".format(type(exc).__name__),
                            }
                        )
                if ordinal + 1 >= max_records:
                    break
        finally:
            stream.close()
    return records


def load_csv_rows(Chem, data_csv_path, wanted_rows):
    """Sequentially resolve a small, bounded set of companion rows."""
    wanted = set(wanted_rows)
    if not wanted:
        return {}
    resolved = {}
    with gzip.open(str(data_csv_path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "smiles" not in (reader.fieldnames or []):
            raise RuntimeError("official CSV does not expose a smiles column")
        for row_index, row in enumerate(reader):
            if row_index > max(wanted):
                break
            if row_index not in wanted:
                continue
            try:
                mol = Chem.MolFromSmiles(row.get("smiles", ""))
                if mol is None:
                    raise ValueError("MolFromSmiles returned None")
                forms = canonical_forms(Chem, mol)
                heavy_projection = heavy_atom_projection_forms(Chem, mol)
                resolved[row_index] = {
                    "status": "ok",
                    "forms": forms,
                    "heavy_projection": heavy_projection,
                    "summary": graph_summary(Chem, mol, forms),
                }
            except Exception as exc:
                resolved[row_index] = {
                    "status": "csv_canonicalization_error:{}".format(type(exc).__name__)
                }
            if len(resolved) == len(wanted):
                break
    return resolved


def classify(sdf_record, csv_record):
    if sdf_record["status"] != "ok":
        return sdf_record["status"]
    if csv_record is None:
        return "missing_official_csv_row"
    if csv_record["status"] != "ok":
        return csv_record["status"]
    if sdf_record["forms"]["strict"] == csv_record["forms"]["strict"]:
        return "strict_isomeric_match"
    if sdf_record["forms"]["connectivity"] == csv_record["forms"]["connectivity"]:
        return "stereo_or_isomeric_representation_mismatch"
    return "connectivity_graph_mismatch"


def property_deltas(sdf_summary, csv_summary):
    return {
        key: sdf_summary[key] == csv_summary[key]
        for key in (
            "atom_count",
            "heavy_atom_count",
            "bond_count",
            "formal_charge",
            "atom_number_charge_histogram",
            "bond_type_histogram",
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--split-dict", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--neighbor-window", type=int, default=4,
                        help="bounded companion-row window used only to detect an ordinal swap")
    parser.add_argument("--max-mismatch-details", type=int, default=64)
    parser.add_argument("--allow-unsafe-legacy-torch-load", action="store_true")
    args = parser.parse_args()

    if args.max_records < 1 or args.max_records > MAX_SMOKE_RECORDS:
        parser.error("--max-records must be within [1, {}]".format(MAX_SMOKE_RECORDS))
    if args.neighbor_window < 0 or args.neighbor_window > 128:
        parser.error("--neighbor-window must be within [0, 128]")
    if args.max_mismatch_details < 0 or args.max_mismatch_details > 256:
        parser.error("--max-mismatch-details must be within [0, 256]")

    archive_path = regular_file(args.archive, "archive")
    data_csv_path = regular_file(args.data_csv, "data CSV")
    split_dict_path = regular_file(args.split_dict, "split dict")
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required") from exc

    split_dict, split_loading_method = load_split_dict(
        split_dict_path, args.allow_unsafe_legacy_torch_load
    )
    train = split_dict.get("train") if isinstance(split_dict, dict) else None
    if train is None or len(train) < args.max_records:
        raise RuntimeError("official train split is missing or too short")
    train_rows = [scalar_to_int(train[index]) for index in range(args.max_records)]
    row_index_rows = list(range(args.max_records))
    train_equals_row_index = train_rows == row_index_rows

    sdf_records = load_sdf_prefix(Chem, archive_path, args.max_records)
    wanted_rows = set(train_rows)
    # A neighbourhood lookup is diagnostic only: it tells us if a mismatched
    # SDF graph is an exact graph match for a nearby companion row.
    for row in train_rows:
        wanted_rows.update(range(max(0, row - args.neighbor_window), row + args.neighbor_window + 1))
    csv_rows = load_csv_rows(Chem, data_csv_path, wanted_rows)

    counts = collections.Counter()
    heavy_projection_counts = collections.Counter()
    mismatch_details = []
    for sdf_record in sdf_records:
        ordinal = sdf_record["ordinal"]
        companion_row = train_rows[ordinal]
        official = csv_rows.get(companion_row)
        classification = classify(sdf_record, official)
        counts[classification] += 1
        if sdf_record["status"] == "ok" and official is not None and official["status"] == "ok":
            if sdf_record["heavy_projection"]["strict"] == official["heavy_projection"]["strict"]:
                heavy_projection_counts["strict_isomeric_match"] += 1
            elif sdf_record["heavy_projection"]["connectivity"] == official["heavy_projection"]["connectivity"]:
                heavy_projection_counts["stereo_or_isomeric_representation_mismatch"] += 1
            else:
                heavy_projection_counts["connectivity_graph_mismatch"] += 1
        if classification in ("strict_isomeric_match",) or len(mismatch_details) >= args.max_mismatch_details:
            continue
        detail = {
            "sdf_record_ordinal": ordinal,
            "train_split_companion_row": companion_row,
            "classification": classification,
        }
        if sdf_record["status"] == "ok" and official is not None and official["status"] == "ok":
            detail["same_basic_graph_statistics"] = property_deltas(
                sdf_record["summary"], official["summary"]
            )
            detail["heavy_atom_projection"] = {
                "sdf_input_explicit_hydrogens": sdf_record["heavy_projection"]["input_explicit_hydrogens"],
                "sdf_residual_explicit_hydrogens": sdf_record["heavy_projection"]["residual_explicit_hydrogens"],
                "official_input_explicit_hydrogens": official["heavy_projection"]["input_explicit_hydrogens"],
                "official_residual_explicit_hydrogens": official["heavy_projection"]["residual_explicit_hydrogens"],
                "strict_match": (
                    sdf_record["heavy_projection"]["strict"] == official["heavy_projection"]["strict"]
                ),
                "connectivity_match": (
                    sdf_record["heavy_projection"]["connectivity"] == official["heavy_projection"]["connectivity"]
                ),
                "sdf_connectivity_smiles_sha256": sha256_text(
                    sdf_record["heavy_projection"]["connectivity"]
                ),
                "official_connectivity_smiles_sha256": sha256_text(
                    official["heavy_projection"]["connectivity"]
                ),
            }
            nearby_connectivity_matches = []
            nearby_strict_matches = []
            for candidate_row in range(max(0, companion_row - args.neighbor_window), companion_row + args.neighbor_window + 1):
                candidate = csv_rows.get(candidate_row)
                if candidate is None or candidate["status"] != "ok":
                    continue
                if candidate["forms"]["connectivity"] == sdf_record["forms"]["connectivity"]:
                    nearby_connectivity_matches.append(candidate_row)
                if candidate["forms"]["strict"] == sdf_record["forms"]["strict"]:
                    nearby_strict_matches.append(candidate_row)
            detail["nearby_companion_connectivity_match_rows"] = nearby_connectivity_matches
            detail["nearby_companion_strict_match_rows"] = nearby_strict_matches
            detail["sdf"] = sdf_record["summary"]
            detail["official"] = official["summary"]
        mismatch_details.append(detail)

    report = {
        "schema_version": "most-t5-r1/pcqm-identity-diagnosis/v1",
        "created_utc": utc_now(),
        "scope": {
            "archive_streamed_not_extracted": True,
            "sdf_prefix_records": args.max_records,
            "csv_materialized": False,
            "lmdb_records_written": 0,
            "local_data_transfer": False,
            "raw_smiles_emitted": False,
        },
        "inputs": {
            "archive": str(archive_path.resolve()),
            "data_csv": str(data_csv_path.resolve()),
            "split_dict": str(split_dict_path.resolve()),
        },
        "mapping_check": {
            "train_prefix_equals_row_index_prefix": train_equals_row_index,
            "train_prefix_first_row": train_rows[0],
            "train_prefix_last_row": train_rows[-1],
            "interpretation": (
                "For this bounded prefix, row-index and train-split diagnose the same mapping. "
                "A difference between their outcomes cannot explain a mismatch here."
                if train_equals_row_index else
                "The train split differs from row-index; only the train-split mapping is authoritative."
            ),
        },
        "split_loading_method": split_loading_method,
        "classification_counts": dict(sorted(counts.items())),
        "all_explicit_hydrogen_projection_counts": dict(sorted(heavy_projection_counts.items())),
        "mismatch_details": mismatch_details,
        "recommendation": (
            "Do not use raw ordinal equality as release admission. Quarantine connectivity mismatches; "
            "retain strict matches; and define a separate, documented policy for stereo-only mismatches. "
            "A full-pass reject ledger is required before any PCQM-derived P1 release."
        ),
    }
    write_json_atomic(args.output, report)
    if counts.get("connectivity_graph_mismatch", 0):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
