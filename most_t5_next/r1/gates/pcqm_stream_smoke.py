#!/usr/bin/env python3
"""Run a bounded, no-extraction PCQM4Mv2 SDF stream smoke test.

The gate opens the compressed tar archive in streaming mode, finds its SDF
member, and asks RDKit to parse only the first ``--max-records`` entries.  It
emits aggregate counters and hashed samples only.  It does not extract the SDF
to disk, calculate E3FP, construct a vocabulary, or write an LMDB.
"""

from __future__ import print_function

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(str(temporary), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def find_sdf_member(archive):
    for member in archive:
        if member.isfile() and member.name.lower().endswith(".sdf"):
            return member
    raise RuntimeError("no regular .sdf member found in archive")


def finite_coordinates(mol):
    if mol.GetNumConformers() < 1:
        return False, "no_conformer"
    try:
        positions = mol.GetConformer().GetPositions()
    except Exception:
        return False, "conformer_positions_error"
    if len(positions) != mol.GetNumAtoms():
        return False, "atom_coordinate_count_mismatch"
    for row in positions:
        for value in row:
            if not math.isfinite(float(value)):
                return False, "non_finite_coordinates"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True, help="new sidecar JSON report path")
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--sample-limit", type=int, default=5)
    args = parser.parse_args()
    if args.max_records < 1:
        parser.error("--max-records must be positive")
    if args.sample_limit < 0:
        parser.error("--sample-limit must be non-negative")

    archive_path = Path(args.archive)
    if not archive_path.is_file():
        raise FileNotFoundError("archive does not exist: {}".format(archive_path))

    # RDKit is imported after basic argument validation so an environment
    # failure is explicit and cannot be confused with a dataset rejection.
    from rdkit import Chem

    reasons = Counter()
    samples = []
    records_seen = 0
    valid_structural_records = 0
    member_name = None
    member_size = None
    with tarfile.open(str(archive_path), mode="r|gz") as archive:
        member = find_sdf_member(archive)
        member_name = member.name
        member_size = int(member.size)
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError("cannot open SDF tar member: {}".format(member.name))
        supplier = Chem.ForwardSDMolSupplier(stream, sanitize=True, removeHs=False)
        try:
            for row_index, mol in enumerate(supplier):
                records_seen += 1
                if mol is None:
                    reasons["rdkit_none"] += 1
                else:
                    coordinate_ok, reason = finite_coordinates(mol)
                    if not coordinate_ok:
                        reasons[reason] += 1
                    else:
                        try:
                            canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                        except Exception:
                            reasons["canonical_smiles_error"] += 1
                        else:
                            valid_structural_records += 1
                            reasons["structural_ok"] += 1
                            if len(samples) < args.sample_limit:
                                samples.append(
                                    {
                                        "ogb_row_index": row_index,
                                        "atom_count": int(mol.GetNumAtoms()),
                                        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
                                        "canonical_smiles_sha256": sha256_text(canonical_smiles),
                                    }
                                )
                if records_seen >= args.max_records:
                    break
        finally:
            stream.close()

    report = {
        "schema_version": "most-t5-r1/pcqm-stream-smoke/v1",
        "created_utc": utc_now(),
        "scope": {
            "archive_streamed_not_extracted": True,
            "max_records_requested": args.max_records,
            "records_seen": records_seen,
            "complete_archive_scan": False,
            "dataset_records_written": 0,
            "local_data_transfer": False,
        },
        "archive": {
            "path": str(archive_path.resolve()),
            "sdf_member": member_name,
            "sdf_member_uncompressed_bytes": member_size,
        },
        "structural_integrity": {
            "valid_structural_records": valid_structural_records,
            "reason_counts": dict(sorted(reasons.items())),
            "hashed_samples": samples,
        },
        "not_yet_evaluated": [
            "official PCQM SMILES/split identity comparison",
            "E3FP generation and singleton policy",
            "SELFIES atom alignment",
            "atom-to-motif mapping",
            "downstream identity exclusion",
            "membership/reject-ledger release",
        ],
        "pass": records_seen == args.max_records and valid_structural_records > 0,
        "next_gate": "Obtain and freeze official PCQM v2 SMILES/split metadata before any full-pass adapter run.",
    }
    write_json_atomic(args.output, report)
    print(json.dumps({"pass": report["pass"], "records_seen": records_seen, "output": str(Path(args.output).resolve())}, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
