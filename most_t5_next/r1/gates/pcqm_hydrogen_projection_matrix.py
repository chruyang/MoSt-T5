#!/usr/bin/env python3
"""Compare bounded RDKit explicit-hydrogen projection settings for PCQM4Mv2.

This R1 diagnostic is deliberately separate from the release gate.  It uses a
small SDF prefix and the corresponding official CSV rows, reports aggregate
identity counts for a fixed set of ``Chem.RemoveHsParameters`` profiles, and
does not write molecular records, raw SMILES, or an LMDB.
"""

from __future__ import print_function

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path

from pcqm_identity_smoke import (
    MAX_SMOKE_RECORDS,
    find_sdf_member,
    regular_file,
    utc_now,
    write_json_atomic,
)


PROFILES = {
    "default": (),
    "remove_with_wedged_bond": ("removeWithWedgedBond",),
    "remove_defining_bond_stereo": ("removeDefiningBondStereo",),
    "wedged_and_defining_bond_stereo": (
        "removeWithWedgedBond",
        "removeDefiningBondStereo",
    ),
    "wedged_stereo_and_mapped": (
        "removeWithWedgedBond",
        "removeDefiningBondStereo",
        "removeMapped",
    ),
    "broad_identity_projection": (
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
    ),
}


def forms(Chem, mol, enabled_options):
    if enabled_options:
        parameters = Chem.RemoveHsParameters()
        for option in enabled_options:
            if not hasattr(parameters, option):
                raise RuntimeError("installed RDKit lacks RemoveHsParameters.{}".format(option))
            setattr(parameters, option, True)
        normalized = Chem.RemoveHs(Chem.Mol(mol), parameters, sanitize=True)
    else:
        normalized = Chem.RemoveHs(Chem.Mol(mol), sanitize=True)
    Chem.SanitizeMol(normalized)
    Chem.AssignStereochemistry(normalized, cleanIt=True, force=True)
    return {
        "strict": Chem.MolToSmiles(normalized, canonical=True, isomericSmiles=True),
        "connectivity": Chem.MolToSmiles(normalized, canonical=True, isomericSmiles=False),
        "residual_explicit_hydrogens": sum(atom.GetAtomicNum() == 1 for atom in normalized.GetAtoms()),
    }


def read_sdf_prefix(Chem, archive_path, max_records):
    molecules = []
    with tarfile.open(str(archive_path), mode="r|gz") as archive:
        member = find_sdf_member(archive)
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError("cannot open SDF member")
        try:
            supplier = Chem.ForwardSDMolSupplier(stream, sanitize=True, removeHs=False)
            for ordinal, mol in enumerate(supplier):
                if mol is None:
                    molecules.append((ordinal, None))
                else:
                    molecules.append((ordinal, Chem.Mol(mol)))
                if ordinal + 1 >= max_records:
                    break
        finally:
            stream.close()
    return molecules


def read_csv_prefix(Chem, data_csv_path, max_records):
    molecules = []
    with gzip.open(str(data_csv_path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "smiles" not in (reader.fieldnames or []):
            raise RuntimeError("official CSV has no smiles column")
        for ordinal, row in enumerate(reader):
            if ordinal >= max_records:
                break
            molecules.append((ordinal, Chem.MolFromSmiles(row.get("smiles", ""))))
    return molecules


def evaluate_profile(Chem, sdf_molecules, csv_molecules, enabled_options):
    if len(sdf_molecules) != len(csv_molecules):
        raise RuntimeError("bounded SDF and CSV prefixes have different lengths")
    classifications = Counter()
    residual_hydrogens = Counter()
    for (sdf_ordinal, sdf_mol), (csv_ordinal, csv_mol) in zip(sdf_molecules, csv_molecules):
        if sdf_ordinal != csv_ordinal:
            raise RuntimeError("unexpected ordinal mapping inside bounded diagnostic")
        if sdf_mol is None or csv_mol is None:
            classifications["parse_error"] += 1
            continue
        try:
            sdf = forms(Chem, sdf_mol, enabled_options)
            csv_forms = forms(Chem, csv_mol, enabled_options)
        except Exception as exc:
            classifications["canonicalization_error:{}".format(type(exc).__name__)] += 1
            continue
        residual_hydrogens[str(sdf["residual_explicit_hydrogens"])] += 1
        if sdf["strict"] == csv_forms["strict"]:
            classifications["strict_isomeric_match"] += 1
        elif sdf["connectivity"] == csv_forms["connectivity"]:
            classifications["stereo_or_isomeric_representation_mismatch"] += 1
        else:
            classifications["connectivity_graph_mismatch"] += 1
    return {
        "enabled_remove_hs_options": list(enabled_options),
        "classification_counts": dict(sorted(classifications.items())),
        "sdf_residual_explicit_hydrogen_count_histogram": dict(sorted(residual_hydrogens.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=1000)
    args = parser.parse_args()
    if args.max_records < 1 or args.max_records > MAX_SMOKE_RECORDS:
        parser.error("--max-records must be within [1, {}]".format(MAX_SMOKE_RECORDS))

    archive_path = regular_file(args.archive, "archive")
    data_csv_path = regular_file(args.data_csv, "data CSV")
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required") from exc

    sdf_molecules = read_sdf_prefix(Chem, archive_path, args.max_records)
    csv_molecules = read_csv_prefix(Chem, data_csv_path, args.max_records)
    if len(sdf_molecules) != args.max_records or len(csv_molecules) != args.max_records:
        raise RuntimeError("the requested bounded prefix was not fully available")

    report = {
        "schema_version": "most-t5-r1/pcqm-hydrogen-projection-matrix/v1",
        "created_utc": utc_now(),
        "scope": {
            "archive_streamed_not_extracted": True,
            "sdf_records_requested": args.max_records,
            "csv_materialized": False,
            "lmdb_records_written": 0,
            "local_data_transfer": False,
            "raw_smiles_emitted": False,
        },
        "inputs": {
            "archive": str(archive_path.resolve()),
            "data_csv": str(data_csv_path.resolve()),
        },
        "profiles": {
            name: evaluate_profile(Chem, sdf_molecules, csv_molecules, options)
            for name, options in PROFILES.items()
        },
        "interpretation": (
            "This matrix is a bounded representation diagnostic. It does not replace a full-release "
            "identity/reject ledger and it does not authorize using companion-SMILES atom order."
        ),
    }
    write_json_atomic(args.output, report)


if __name__ == "__main__":
    main()
