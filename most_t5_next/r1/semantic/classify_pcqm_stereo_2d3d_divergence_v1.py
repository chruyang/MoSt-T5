#!/usr/bin/env python3
"""Classify the frozen PCQM 2D/3D strict-isomeric reject population.

The production release deliberately quarantined every record whose official
CSV SMILES and SDF-derived molecule had equal non-isomeric connectivity but
different strict-isomeric canonical SMILES.  That closed reject code is broad:
it does not say whether the official side omitted stereo, the SDF side omitted
stereo, both sides specified different stereo, or a non-stereo isomeric field
(for example an isotope) caused the difference.

This utility replays only the frozen reject ordinals.  It scans the SDF archive
once without parsing non-selected records, streams the official CSV once, and
emits a metadata-only row for every reject plus a closed summary manifest.  It
does not mutate or republish the production release.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path


REASON = "PCQM_STEREO_2D3D_DIVERGENCE"
SCHEMA = "most-t5-r1/pcqm-stereo-2d3d-divergence-classification/v3"
RECOVERY_ACTIONS = frozenset((
    "candidate_representation_normalization",
    "candidate_stereo_free_identity_plus_sdf_state",
))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_line(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _load_rejects(release_root: Path, expected_count: int) -> dict[int, dict]:
    selected = {}
    for path in sorted(release_root.glob("shard-*/reject_ledger.jsonl")):
        with path.open("rb") as handle:
            for raw in handle:
                row = json.loads(raw)
                if row.get("reason_code") != REASON:
                    continue
                ordinal = int(row["sdf_record_index"])
                if ordinal in selected:
                    raise RuntimeError("duplicate reject ordinal: {}".format(ordinal))
                if int(row["official_csv_row_index"]) != ordinal:
                    raise RuntimeError("reject ordinal/CSV row mismatch: {}".format(ordinal))
                selected[ordinal] = {
                    "member_id": row["member_id"],
                    "source_address_sha256": row["source_address_sha256"],
                    "source_mol_identity_sha256": row["source_mol_identity_sha256"],
                    "geometry_mol_identity_sha256": row["geometry_mol_identity_sha256"],
                }
    if len(selected) != expected_count:
        raise RuntimeError("expected {} rejects, found {}".format(expected_count, len(selected)))
    return selected


def _load_official_smiles(data_csv: Path, selected: set[int]) -> dict[int, str]:
    resolved = {}
    largest = max(selected)
    with gzip.open(str(data_csv), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "smiles" not in (reader.fieldnames or []):
            raise RuntimeError("official CSV lacks smiles column")
        for row_index, row in enumerate(reader):
            if row_index in selected:
                resolved[row_index] = row["smiles"]
            if row_index >= largest:
                break
    missing = selected.difference(resolved)
    if missing:
        raise RuntimeError("official CSV missing {} selected rows".format(len(missing)))
    return resolved


def _iter_selected_sdf_blocks(archive_path: Path, selected: set[int]):
    largest = max(selected)
    with tarfile.open(str(archive_path), mode="r|gz") as archive:
        sdf_member = None
        for member in archive:
            if member.isfile() and member.name.lower().endswith((".sdf", ".sd")):
                sdf_member = member
                break
        if sdf_member is None:
            raise RuntimeError("SDF member not found in archive")
        stream = archive.extractfile(sdf_member)
        if stream is None:
            raise RuntimeError("cannot extract SDF stream")
        ordinal = 0
        capture = ordinal in selected
        chunks = []
        try:
            for line in stream:
                if capture:
                    chunks.append(line)
                if line.strip() != b"$$$$":
                    continue
                if capture:
                    yield ordinal, b"".join(chunks)
                if ordinal >= largest:
                    return
                ordinal += 1
                capture = ordinal in selected
                chunks = []
        finally:
            stream.close()


def _normalized_geometry(Chem, mol):
    params = Chem.RemoveHsParameters()
    if not hasattr(params, "removeDefiningBondStereo"):
        raise RuntimeError("RDKit lacks removeDefiningBondStereo")
    params.removeDefiningBondStereo = True
    normalized = Chem.RemoveHs(Chem.Mol(mol), params, sanitize=True)
    Chem.SanitizeMol(normalized)
    Chem.AssignStereochemistry(normalized, cleanIt=True, force=True)
    return normalized


def _forms(Chem, mol) -> dict:
    normalized = _normalized_geometry(Chem, mol)
    strict = Chem.MolToSmiles(normalized, canonical=True, isomericSmiles=True)
    connectivity = Chem.MolToSmiles(normalized, canonical=True, isomericSmiles=False)
    return {
        "mol": normalized,
        "strict_sha256": _sha256_bytes(strict.encode("utf-8")),
        "connectivity_sha256": _sha256_bytes(connectivity.encode("utf-8")),
    }


def _stereo_profile(Chem, mol) -> dict:
    atom_tags = collections.Counter()
    cip_codes = collections.Counter()
    isotopes = collections.Counter()
    atom_maps = 0
    for atom in mol.GetAtoms():
        tag = str(atom.GetChiralTag())
        if tag != "CHI_UNSPECIFIED":
            atom_tags[tag] += 1
        if atom.HasProp("_CIPCode"):
            cip_codes[atom.GetProp("_CIPCode")] += 1
        if atom.GetIsotope():
            isotopes[str(int(atom.GetIsotope()))] += 1
        if atom.GetAtomMapNum():
            atom_maps += 1
    bond_stereo = collections.Counter()
    for bond in mol.GetBonds():
        stereo = str(bond.GetStereo())
        if stereo != "STEREONONE":
            bond_stereo[stereo] += 1
    stereo_count = sum(atom_tags.values()) + sum(bond_stereo.values())
    return {
        "atom_chiral_tags": dict(sorted(atom_tags.items())),
        "cip_codes": dict(sorted(cip_codes.items())),
        "bond_stereo": dict(sorted(bond_stereo.items())),
        "stereo_feature_count": int(stereo_count),
        "isotope_histogram": dict(sorted(isotopes.items())),
        "mapped_atom_count": int(atom_maps),
    }


def _chirality_relation(Chem, official_mol, sdf_mol) -> dict:
    # In ``target.HasSubstructMatch(query)``, an unspecified query stereo
    # feature may match a specified target feature.  Computing both directions
    # therefore distinguishes equivalence from a one-sided refinement without
    # depending on atom order.
    official_query_matches_sdf = bool(
        sdf_mol.HasSubstructMatch(official_mol, useChirality=True)
    )
    sdf_query_matches_official = bool(
        official_mol.HasSubstructMatch(sdf_mol, useChirality=True)
    )
    result = {
        "official_query_matches_sdf": official_query_matches_sdf,
        "sdf_query_matches_official": sdf_query_matches_official,
        "mutual_chiral_match": official_query_matches_sdf and sdf_query_matches_official,
    }
    try:
        from rdkit.Chem import inchi

        official_key = inchi.MolToInchiKey(official_mol)
        sdf_key = inchi.MolToInchiKey(sdf_mol)
        result.update({
            "inchi_available": bool(official_key and sdf_key),
            "inchi_connectivity_block_equal": bool(
                official_key and sdf_key and official_key.split("-", 1)[0] == sdf_key.split("-", 1)[0]
            ),
            "inchi_full_key_equal": bool(official_key and sdf_key and official_key == sdf_key),
        })
    except Exception:
        result.update({
            "inchi_available": False,
            "inchi_connectivity_block_equal": False,
            "inchi_full_key_equal": False,
        })
    return result


def _classify(official: dict, sdf: dict, relation: dict) -> tuple[str, str]:
    o_stereo = official["stereo_feature_count"] > 0
    s_stereo = sdf["stereo_feature_count"] > 0
    non_stereo_field_divergence = (
        official["isotope_histogram"] != sdf["isotope_histogram"]
        or official["mapped_atom_count"] != sdf["mapped_atom_count"]
    )
    if non_stereo_field_divergence:
        return "non_stereo_isomeric_field_divergence", "retain_quarantine"
    if not o_stereo and s_stereo:
        return "official_unstated_sdf_defined", "candidate_stereo_free_identity_plus_sdf_state"
    if o_stereo and not s_stereo:
        return "official_defined_sdf_unstated", "retain_quarantine"
    if o_stereo and s_stereo:
        if relation["mutual_chiral_match"] and relation["inchi_full_key_equal"]:
            return "both_defined_chemically_equivalent_surface_mismatch", "candidate_representation_normalization"
        if relation["official_query_matches_sdf"] and not relation["sdf_query_matches_official"]:
            return "both_defined_sdf_stereo_refines_official", "candidate_stereo_free_identity_plus_sdf_state"
        if relation["sdf_query_matches_official"] and not relation["official_query_matches_sdf"]:
            return "both_defined_official_stereo_refines_sdf", "retain_quarantine"
        return "both_defined_stereo_conflict_or_unsupported", "retain_quarantine"
    return "no_detected_stereo_feature_strict_mismatch", "retain_quarantine"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=12978)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    release_root = Path(args.release_root).resolve()
    source_archive = Path(args.source_archive).resolve()
    data_csv = Path(args.data_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise RuntimeError("output directory already exists: {}".format(output_dir))
    output_dir.mkdir(parents=True)

    from rdkit import Chem, rdBase

    started = time.time()
    rejects = _load_rejects(release_root, args.expected_count)
    selected = set(rejects)
    official_smiles = _load_official_smiles(data_csv, selected)
    rows_path = output_dir / "classification_rows.jsonl"
    recovery_path = output_dir / "recovery_membership.jsonl"
    quarantine_path = output_dir / "quarantine_membership.jsonl"
    class_counts = collections.Counter()
    action_counts = collections.Counter()
    relation_counts = collections.Counter()
    profile_pair_counts = collections.Counter()
    observed = set()

    recovery_count = 0
    quarantine_count = 0
    with rows_path.open("wb") as output, recovery_path.open("wb") as recovery_output, quarantine_path.open("wb") as quarantine_output:
        for completed, (ordinal, block) in enumerate(
            _iter_selected_sdf_blocks(source_archive, selected), start=1
        ):
            supplier = Chem.ForwardSDMolSupplier(io.BytesIO(block), sanitize=True, removeHs=False)
            source_mol = next(iter(supplier), None)
            if source_mol is None:
                raise RuntimeError("selected SDF record failed to parse: {}".format(ordinal))
            official_mol = Chem.MolFromSmiles(official_smiles[ordinal])
            if official_mol is None:
                raise RuntimeError("selected official SMILES failed to parse: {}".format(ordinal))
            sdf_forms = _forms(Chem, source_mol)
            official_forms = _forms(Chem, official_mol)
            if sdf_forms["connectivity_sha256"] != official_forms["connectivity_sha256"]:
                raise RuntimeError("frozen stereo reject lost connectivity equality: {}".format(ordinal))
            if sdf_forms["strict_sha256"] == official_forms["strict_sha256"]:
                raise RuntimeError("frozen stereo reject became strict-equal: {}".format(ordinal))
            sdf_profile = _stereo_profile(Chem, sdf_forms["mol"])
            official_profile = _stereo_profile(Chem, official_forms["mol"])
            relation = _chirality_relation(Chem, official_forms["mol"], sdf_forms["mol"])
            classification, action = _classify(official_profile, sdf_profile, relation)
            class_counts[classification] += 1
            action_counts[action] += 1
            relation_counts[(
                relation["official_query_matches_sdf"],
                relation["sdf_query_matches_official"],
                relation["inchi_full_key_equal"],
            )] += 1
            profile_pair_counts[(
                official_profile["stereo_feature_count"],
                sdf_profile["stereo_feature_count"],
            )] += 1
            witness = rejects[ordinal]
            classification_row = {
                "schema": SCHEMA,
                "sdf_record_index": ordinal,
                "official_csv_row_index": ordinal,
                "member_id": witness["member_id"],
                "source_address_sha256": witness["source_address_sha256"],
                "source_mol_identity_sha256": witness["source_mol_identity_sha256"],
                "geometry_mol_identity_sha256": witness["geometry_mol_identity_sha256"],
                "connectivity_sha256": sdf_forms["connectivity_sha256"],
                "official_strict_sha256": official_forms["strict_sha256"],
                "sdf_strict_sha256": sdf_forms["strict_sha256"],
                "official_profile": official_profile,
                "sdf_profile": sdf_profile,
                "chirality_relation": relation,
                "classification": classification,
                "policy_action": action,
            }
            output.write(_json_line(classification_row))
            membership_row = {
                "schema": "most-t5-r1/pcqm-stereo-recovery-membership/v1",
                "member_id": witness["member_id"],
                "sdf_record_index": ordinal,
                "official_csv_row_index": ordinal,
                "source_address_sha256": witness["source_address_sha256"],
                "source_mol_identity_sha256": witness["source_mol_identity_sha256"],
                "geometry_mol_identity_sha256": witness["geometry_mol_identity_sha256"],
                "connectivity_sha256": sdf_forms["connectivity_sha256"],
                "official_strict_sha256": official_forms["strict_sha256"],
                "sdf_strict_sha256": sdf_forms["strict_sha256"],
                "classification": classification,
                "policy_action": action,
                "requires_payload_reprocessing_from_source_sdf": action in RECOVERY_ACTIONS,
            }
            if action in RECOVERY_ACTIONS:
                membership_row["selection_index"] = recovery_count
                recovery_output.write(_json_line(membership_row))
                recovery_count += 1
            else:
                membership_row["selection_index"] = quarantine_count
                quarantine_output.write(_json_line(membership_row))
                quarantine_count += 1
            observed.add(ordinal)
            if args.progress_every and completed % args.progress_every == 0:
                print("classified {}/{}".format(completed, len(selected)), flush=True)

    if observed != selected:
        raise RuntimeError("SDF scan missed {} selected records".format(len(selected - observed)))
    rows_sha256 = _sha256_file(rows_path)
    if recovery_count + quarantine_count != len(observed):
        raise RuntimeError("recovery/quarantine partition does not close")
    manifest = {
        "schema": SCHEMA,
        "status": "pass",
        "scope": "metadata_only_replay_of_frozen_stereo_2d3d_rejects",
        "mutates_production_release": False,
        "inputs": {
            "release_root": str(release_root),
            "source_archive": str(source_archive),
            "data_csv": str(data_csv),
            "frozen_reason_code": REASON,
        },
        "runtime": {
            "rdkit_version": rdBase.rdkitVersion,
            "wall_seconds": time.time() - started,
            "sdf_scan": "single_sequential_archive_pass_parse_selected_only",
            "csv_scan": "single_sequential_gzip_pass_selected_only",
        },
        "counts": {
            "expected": args.expected_count,
            "classified": len(observed),
            "classification": dict(sorted(class_counts.items())),
            "policy_action": dict(sorted(action_counts.items())),
            "chirality_relation": {
                "official_query_matches_sdf_{}_sdf_query_matches_official_{}_inchi_full_equal_{}".format(
                    str(a).lower(), str(b).lower(), str(c).lower()
                ): count
                for (a, b, c), count in sorted(relation_counts.items())
            },
            "stereo_feature_count_pairs": {
                "official_{}_sdf_{}".format(a, b): count
                for (a, b), count in sorted(profile_pair_counts.items())
            },
        },
        "policy_boundary": {
            "candidate_stereo_free_identity_plus_sdf_state": (
                "Official 2D identity has no detected stereo feature, the SDF does, connectivity is equal, "
                "and no isotope/atom-map divergence was detected. Candidate only for a separately frozen "
                "stereo-free-identity plus SDF/E3FP-state release."
            ),
            "candidate_requires_explicit_policy": (
                "Both sides carry stereo but strict identities differ. Do not silently admit; an explicit "
                "SDF-authoritative policy and comparative evaluation are required."
            ),
            "candidate_representation_normalization": (
                "Bidirectional chirality-aware graph matching and full InChIKey agree. Candidate for a "
                "version-locked representation-normalization repair without changing chemical identity."
            ),
            "retain_quarantine": "Not eligible for automatic recovery under this diagnostic.",
        },
        "artifacts": {
            "classification_rows": {
                "path": rows_path.name,
                "rows": len(observed),
                "bytes": rows_path.stat().st_size,
                "sha256": rows_sha256,
            },
            "recovery_membership": {
                "path": recovery_path.name,
                "rows": recovery_count,
                "bytes": recovery_path.stat().st_size,
                "sha256": _sha256_file(recovery_path),
                "payload_status": "requires_reprocessing_from_source_sdf",
            },
            "quarantine_membership": {
                "path": quarantine_path.name,
                "rows": quarantine_count,
                "bytes": quarantine_path.stat().st_size,
                "sha256": _sha256_file(quarantine_path),
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    (output_dir / "COMPLETED").write_text(_sha256_file(manifest_path) + "\n", encoding="ascii")
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
