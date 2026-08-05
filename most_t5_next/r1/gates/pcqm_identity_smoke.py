#!/usr/bin/env python3
"""Compare a bounded PCQM4Mv2 SDF prefix with the official SMILES companion.

This is an R1 *identity smoke*, not a dataset adapter.  It streams at most
``--max-records`` structures from the compressed OGB train-3D archive and
keeps only that bounded prefix in memory.  The official ``data.csv.gz`` is
read sequentially only far enough to resolve those companion row IDs.  No SDF
is extracted, no LMDB is written, and the JSON output contains only aggregate
counts and SHA-256 hashes of canonical SMILES.

The SDF is deliberately parsed with explicit hydrogens intact, then projected
using a *minimal*, version-recorded RDKit rule before graph comparison:
``RemoveHsParameters.removeDefiningBondStereo=True``.  Default ``RemoveHs``
retains a hydrogen that defines bond stereochemistry; that otherwise harmless
representation difference produced false connectivity mismatches against the
implicit-H CSV SMILES in the initial R1 probe.  This script does not use a
broad hydrogen-removal profile, does not alter isotope handling, and never
uses CSV atom order as an alignment signal.

The normal OGB relationship is selected with ``--mapping-mode train-split``:
SDF record *i* is compared with the CSV row referenced by
``split_dict.pt['train'][i]``.  ``--mapping-mode row-index`` is intentionally
available only as a diagnostic alternative; it must not be used to silently
repair a failed train-split comparison.

``split_dict.pt`` is a PyTorch serialization.  The script tries
``weights_only=True`` first.  Some official OGB releases store NumPy objects
that this safe loader rejects; the legacy fallback is available only after the
caller explicitly acknowledges ``--allow-unsafe-legacy-torch-load`` and only
for a frozen, official companion artifact.
"""

from __future__ import print_function

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path


MAX_SMOKE_RECORDS = 100000
IDENTITY_NORMALIZATION_PROFILE = {
    "name": "rdkit_remove_hs_minimal_stereo_defining_h/v1",
    "sdf_parser": {
        "sanitize": True,
        "remove_hs": False,
    },
    "post_parse_projection": {
        "operation": "Chem.RemoveHs",
        "removeDefiningBondStereo": True,
        "all_other_RemoveHsParameters": "RDKit_defaults_pinned_by_reported_rdkit_version",
    },
    "prohibited": [
        "manual_atom_deletion",
        "broad_RemoveHs_parameter_override",
        "csv_atom_order_as_alignment",
    ],
}


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


def regular_file(path, label):
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError("{} is not a regular file: {}".format(label, result))
    return result


def import_source_integrity():
    """Load the stdlib-only verifier before any parser or torch deserializer."""
    path = Path(__file__).resolve().parents[1] / "adapter" / "pcqm_source_integrity.py"
    if not path.is_file():
        raise FileNotFoundError("PCQM source-integrity helper is missing: {}".format(path))
    spec = importlib.util.spec_from_file_location("r1_pcqm_source_integrity_identity", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct PCQM source-integrity module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scalar_to_int(value):
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def select_companion_indices(split_dict, split_key, mapping_mode, max_records):
    """Return only the bounded selected IDs, not a materialized train split."""
    if mapping_mode == "row-index":
        return list(range(max_records)), {
            "mapping_mode": mapping_mode,
            "split_key": None,
            "split_entries": None,
            "selected_from_split": False,
        }

    if not isinstance(split_dict, dict):
        raise RuntimeError("split_dict.pt did not deserialize to a dictionary")
    if split_key not in split_dict:
        raise KeyError("split key {!r} is absent; found {}".format(split_key, sorted(split_dict.keys())))
    split_values = split_dict[split_key]
    try:
        split_entries = len(split_values)
    except TypeError as exc:
        raise RuntimeError("selected split does not expose a length") from exc
    if split_entries < max_records:
        raise RuntimeError(
            "selected split contains {} entries, smaller than requested {}".format(split_entries, max_records)
        )

    selected = [scalar_to_int(split_values[index]) for index in range(max_records)]
    if len(set(selected)) != len(selected):
        raise RuntimeError("duplicate companion row IDs occur in the selected split prefix")
    if min(selected) < 0:
        raise RuntimeError("selected split contains a negative companion row ID")
    return selected, {
        "mapping_mode": mapping_mode,
        "split_key": split_key,
        "split_entries": int(split_entries),
        "selected_from_split": True,
        "selected_row_min": min(selected),
        "selected_row_max": max(selected),
    }


def normalized_comparison_mol(Chem, mol):
    """Return the locked minimal explicit-H projection for graph comparison.

    A default ``Chem.RemoveHs`` intentionally preserves stereo-defining H
    atoms.  The frozen OGB SDF carries such atoms explicitly while the OGB CSV
    normally represents them implicitly.  Setting *only*
    ``removeDefiningBondStereo`` removes this representation mismatch without
    silently dropping isotope, query, mapped, or other chemically meaningful
    distinctions.  The RDKit version is emitted in every report and is part of
    the later release source lock.
    """
    parameters = Chem.RemoveHsParameters()
    if not hasattr(parameters, "removeDefiningBondStereo"):
        raise RuntimeError("installed RDKit lacks RemoveHsParameters.removeDefiningBondStereo")
    parameters.removeDefiningBondStereo = True
    return Chem.RemoveHs(Chem.Mol(mol), parameters, sanitize=True)


def canonical_forms(Chem, mol):
    """Return strict and connectivity keys under the locked projection."""
    normalized = normalized_comparison_mol(Chem, mol)
    Chem.SanitizeMol(normalized)
    Chem.AssignStereochemistry(normalized, cleanIt=True, force=True)
    return {
        "strict": Chem.MolToSmiles(normalized, canonical=True, isomericSmiles=True),
        "connectivity": Chem.MolToSmiles(normalized, canonical=True, isomericSmiles=False),
        "atom_count": int(normalized.GetNumAtoms()),
        "heavy_atom_count": int(normalized.GetNumHeavyAtoms()),
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in normalized.GetAtoms())),
        "residual_explicit_hydrogen_count": int(
            sum(atom.GetAtomicNum() == 1 for atom in normalized.GetAtoms())
        ),
    }


def stream_sdf_prefix(Chem, archive_path, companion_indices, max_records):
    """Stream a bounded SDF prefix and retain canonical forms only for that prefix."""
    records = []
    records_seen = 0
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
            for sdf_record_index, mol in enumerate(supplier):
                records_seen += 1
                record = {
                    "sdf_record_index": int(sdf_record_index),
                    "companion_row_index": int(companion_indices[sdf_record_index]),
                    "sdf_status": "ok",
                }
                if mol is None:
                    record["sdf_status"] = "sdf_rdkit_none"
                else:
                    try:
                        record["sdf"] = canonical_forms(Chem, mol)
                    except Exception as exc:  # Report the class, never raw molecule content.
                        record["sdf_status"] = "sdf_canonicalization_error:{}".format(type(exc).__name__)
                records.append(record)
                if records_seen >= max_records:
                    break
        finally:
            stream.close()
    return records, {
        "sdf_member": member_name,
        "sdf_member_uncompressed_bytes": member_size,
        "records_seen": int(records_seen),
    }


def resolve_csv_smiles(data_csv_path, companion_indices, max_csv_rows):
    """Resolve just the selected CSV rows in a sequential, non-materializing pass."""
    wanted = set(companion_indices)
    resolved = {}
    rows_scanned = 0
    stopped_by_limit = False
    with gzip.open(str(data_csv_path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "smiles" not in fieldnames:
            raise RuntimeError("official CSV has no 'smiles' column; found {}".format(fieldnames))
        for row_index, row in enumerate(reader):
            if max_csv_rows and row_index >= max_csv_rows:
                stopped_by_limit = True
                break
            rows_scanned += 1
            if row_index in wanted:
                resolved[row_index] = row.get("smiles", "")
                if len(resolved) == len(wanted):
                    break
    return resolved, {
        "header": fieldnames,
        "rows_scanned": int(rows_scanned),
        "stopped_by_max_csv_rows": stopped_by_limit,
        "selected_rows_resolved": int(len(resolved)),
        "selected_rows_requested": int(len(wanted)),
    }


def compare_records(Chem, sdf_records, official_smiles_by_row, sample_limit):
    reasons = Counter()
    samples = []
    mismatch_samples = []
    residual_hydrogens = Counter()
    compared_records = 0
    strict_matches = 0
    connectivity_matches = 0
    missing_csv_rows = 0
    for record in sdf_records:
        sample = {
            "sdf_record_index": record["sdf_record_index"],
            "companion_row_index": record["companion_row_index"],
            "classification": None,
        }
        if record["sdf_status"] != "ok":
            classification = record["sdf_status"]
            reasons[classification] += 1
        elif record["companion_row_index"] not in official_smiles_by_row:
            classification = "missing_official_csv_row"
            reasons[classification] += 1
            missing_csv_rows += 1
        else:
            raw_smiles = official_smiles_by_row[record["companion_row_index"]]
            try:
                official_mol = Chem.MolFromSmiles(raw_smiles)
                if official_mol is None:
                    raise ValueError("MolFromSmiles returned None")
                official = canonical_forms(Chem, official_mol)
            except Exception as exc:
                classification = "official_smiles_parse_or_canonicalization_error:{}".format(type(exc).__name__)
                reasons[classification] += 1
            else:
                compared_records += 1
                sdf = record["sdf"]
                sample.update(
                    {
                        "sdf_strict_smiles_sha256": sha256_text(sdf["strict"]),
                        "official_strict_smiles_sha256": sha256_text(official["strict"]),
                        "sdf_connectivity_smiles_sha256": sha256_text(sdf["connectivity"]),
                        "official_connectivity_smiles_sha256": sha256_text(official["connectivity"]),
                        "sdf_atom_count": sdf["atom_count"],
                        "official_atom_count": official["atom_count"],
                        "sdf_heavy_atom_count": sdf["heavy_atom_count"],
                        "official_heavy_atom_count": official["heavy_atom_count"],
                        "sdf_formal_charge": sdf["formal_charge"],
                        "official_formal_charge": official["formal_charge"],
                        "sdf_residual_explicit_hydrogen_count": sdf["residual_explicit_hydrogen_count"],
                        "official_residual_explicit_hydrogen_count": official["residual_explicit_hydrogen_count"],
                    }
                )
                residual_hydrogens[str(sdf["residual_explicit_hydrogen_count"])] += 1
                if sdf["strict"] == official["strict"]:
                    classification = "strict_isomeric_match"
                    strict_matches += 1
                    connectivity_matches += 1
                elif sdf["connectivity"] == official["connectivity"]:
                    classification = "stereo_or_isomeric_representation_mismatch"
                    connectivity_matches += 1
                else:
                    classification = "connectivity_graph_mismatch"
                reasons[classification] += 1
        sample["classification"] = classification
        if len(samples) < sample_limit:
            samples.append(sample)
        if classification != "strict_isomeric_match" and len(mismatch_samples) < sample_limit:
            mismatch_samples.append(sample)
    return {
        "records_selected": int(len(sdf_records)),
        "records_compared": int(compared_records),
        "strict_isomeric_matches": int(strict_matches),
        "connectivity_matches": int(connectivity_matches),
        "missing_official_csv_rows": int(missing_csv_rows),
        "reason_counts": dict(sorted(reasons.items())),
        "sdf_residual_explicit_hydrogen_count_histogram": dict(sorted(residual_hydrogens.items())),
        "hashed_samples": samples,
        "hashed_non_strict_samples": mismatch_samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="frozen OGB train-3D .tar.gz archive")
    parser.add_argument("--data-csv", required=True, help="official pcqm4m-v2/raw/data.csv.gz")
    parser.add_argument("--split-dict", required=True, help="official pcqm4m-v2/split_dict.pt")
    parser.add_argument("--output", required=True, help="new sidecar JSON report path")
    parser.add_argument("--source-contract", required=True, help="locked R1 PCQM v2 source contract JSON")
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--max-csv-rows", type=int, default=0,
                        help="0 means stream until all selected CSV rows are found; never materializes the CSV")
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--split-key", default="train")
    parser.add_argument("--mapping-mode", choices=("train-split", "row-index"), default="train-split")
    parser.add_argument(
        "--require-strict-isomeric",
        action="store_true",
        help=(
            "also fail this bounded gate on strict isomeric differences. Without this flag, "
            "the gate admits only the next full graph-ledger design step; every non-strict "
            "record remains explicitly reported and cannot enter a strict aligned release."
        ),
    )
    parser.add_argument("--allow-unsafe-legacy-torch-load", action="store_true",
                        help="required only when safe weights-only loading is unavailable or the official split contains unsupported NumPy objects; use only for frozen official OGB split_dict.pt")
    args = parser.parse_args()

    if args.max_records < 1 or args.max_records > MAX_SMOKE_RECORDS:
        parser.error("--max-records must be within [1, {}] for an identity smoke".format(MAX_SMOKE_RECORDS))
    if args.max_csv_rows < 0:
        parser.error("--max-csv-rows must be non-negative")
    if args.sample_limit < 0:
        parser.error("--sample-limit must be non-negative")

    archive_path = regular_file(args.archive, "archive")
    data_csv_path = regular_file(args.data_csv, "data CSV")
    split_dict_path = regular_file(args.split_dict, "split dict")
    source_integrity = import_source_integrity()
    # Verify archive, companion ZIP, CSV, split, and supporting manifests
    # before RDKit starts parsing or torch is allowed to deserialize the split.
    verified_inputs = source_integrity.verify_pcqm_inputs(
        args.source_contract, archive_path, data_csv_path, split_dict_path
    )

    # Delay expensive imports until all paths and bounds are validated.
    try:
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise RuntimeError("RDKit is required for canonical graph identity comparison") from exc

    split_dict, split_load_mode = source_integrity.load_verified_split_dict(
        verified_inputs, args.allow_unsafe_legacy_torch_load
    )
    companion_indices, mapping = select_companion_indices(
        split_dict, args.split_key, args.mapping_mode, args.max_records
    )
    if args.mapping_mode == "train-split" and mapping["split_entries"] != verified_inputs.source_record_count:
        raise RuntimeError(
            "official train split has {} entries but the verified source contract expects {}".format(
                mapping["split_entries"], verified_inputs.source_record_count
            )
        )

    sdf_records, archive_observed = stream_sdf_prefix(Chem, archive_path, companion_indices, args.max_records)
    official_smiles_by_row, csv_observed = resolve_csv_smiles(
        data_csv_path, companion_indices, args.max_csv_rows
    )
    identity = compare_records(Chem, sdf_records, official_smiles_by_row, args.sample_limit)

    errors = []
    if archive_observed["records_seen"] != args.max_records:
        errors.append("SDF ended before the requested bounded prefix")
    if csv_observed["selected_rows_resolved"] != args.max_records:
        errors.append("not every selected companion row was resolved from data.csv.gz")
    if identity["records_compared"] != args.max_records:
        errors.append("not every selected SDF record could be canonically compared")
    graph_identity_pass = identity["connectivity_matches"] == args.max_records
    strict_alignment_pass = identity["strict_isomeric_matches"] == args.max_records
    warnings = []
    if not graph_identity_pass:
        errors.append("normalized connectivity graph mismatch; quarantine affected records before adapter admission")
    if not strict_alignment_pass:
        message = (
            "strict isomeric canonical identity mismatch; records are explicitly reported and "
            "must be quarantined from a strict 2D/text/3D aligned release"
        )
        if args.require_strict_isomeric:
            errors.append(message)
        else:
            warnings.append(message)

    report = {
        "schema_version": "most-t5-r1/pcqm-identity-smoke/v2",
        "created_utc": utc_now(),
        "scope": {
            "archive_streamed_not_extracted": True,
            "sdf_records_requested": int(args.max_records),
            "sdf_records_buffered_in_memory_at_most": int(args.max_records),
            "csv_materialized": False,
            "lmdb_records_written": 0,
            "local_data_transfer": False,
            "full_release_identity_claim": False,
        },
        "inputs": {
            "archive": str(archive_path.resolve()),
            "data_csv": str(data_csv_path.resolve()),
            "split_dict": str(split_dict_path.resolve()),
            "source_contract": str(Path(args.source_contract).resolve()),
            "data_csv_bytes": int(data_csv_path.stat().st_size),
            "split_dict_bytes": int(split_dict_path.stat().st_size),
            "rdkit_version": rdBase.rdkitVersion,
            "verified_input_lock": verified_inputs.report(),
        },
        "split_loading": {
            "method": split_load_mode,
            **mapping
        },
        "archive_observed": archive_observed,
        "csv_observed": csv_observed,
        "identity": identity,
        "normalization": IDENTITY_NORMALIZATION_PROFILE,
        "graph_identity_pass": graph_identity_pass,
        "strict_alignment_pass": strict_alignment_pass,
        "strict_isomeric_required": bool(args.require_strict_isomeric),
        "pass": not errors,
        "errors": errors,
        "warnings": warnings,
        "interpretation": {
            "strict_match_definition": "canonical isomeric SMILES match after the locked minimal explicit-H projection",
            "connectivity_match_definition": "canonical non-isomeric SMILES match after the locked minimal explicit-H projection",
            "graph_gate_policy": "A passing graph gate only permits the design of one remote full-pass identity/reject ledger. It never admits PCQM4Mv2 to P1.",
            "stereo_only_result_policy": "A stereo/isomeric representation mismatch is never silently accepted. It is recorded in hashed_non_strict_samples and must be quarantined from a strict 2D/text/3D aligned release unless a later, separately evaluated SDF-self-consistent ablation is approved.",
        },
        "next_gate": "A graph-identity pass permits a single remote full-pass identity/reject-ledger design review; it does not admit PCQM4Mv2 to P1.",
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "records_seen": archive_observed["records_seen"],
                "connectivity_matches": identity["connectivity_matches"],
                "strict_matches": identity["strict_isomeric_matches"],
                "strict_alignment_pass": report["strict_alignment_pass"],
                "output": str(Path(args.output).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
