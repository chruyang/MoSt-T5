#!/usr/bin/env python3
"""Compile the frozen 3D-MolT5 QM9 subset into anchored motif records.

The source parquet stores E3FP on the SELFIES token axis.  This builder removes
structural SELFIES rows, maps the remaining rows to the stereo-free heavy-atom
axis used by the anchored motif linearizer, and performs all deterministic
chemistry before training.  Property masks remain explicit because the public
instruction dataset does not contain every property for every exact state.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from most_t5_next.p1.bound_record import Span
from most_t5_next.p1.production_bridge import ProductionMotifRecord
from most_t5_next.p2.anchored_training_record_v1 import (
    AnchoredTokenizerBinding,
    bind_anchored_training_record,
    tokenizer_binding_from_candidate_manifest,
)
from most_t5_next.p2.freeze_qm9_3dmolt5_probe_subset_v1 import (
    PROPERTY_ORDER,
    state_sha256,
)
from most_t5_next.r1.adapter.mol_linearizer import linearize_mol
from most_t5_next.r1.adapter.p1_topology_augmentation_v1 import (
    build_topology_augmentation,
)
from most_t5_next.r1.tokenizer.stereo_free_anchored_motif_surface_v1 import (
    build_stereo_free_anchored_surface,
    surface_document,
)


SCHEMA_VERSION = "most-t5-p2/qm9-anchored-probe-cache/v1"
RECORD_SCHEMA_VERSION = "most-t5-p2/qm9-anchored-probe-record/v1"
PROPERTY_NAMES = ("homo", "lumo", "gap")


class QM9AnchoredProbeCacheError(RuntimeError):
    """The public QM9 state cannot be projected onto the anchored atom axis."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise QM9AnchoredProbeCacheError(
                    f"blank JSONL row at line {line_number}"
                )
            row = json.loads(line)
            if not isinstance(row, dict):
                raise QM9AnchoredProbeCacheError("JSONL row must be an object")
            rows.append(row)
    if not rows:
        raise QM9AnchoredProbeCacheError("JSONL input is empty")
    return rows


def _stereo_free_heavy_mol(Chem, smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() <= 0:
        raise QM9AnchoredProbeCacheError("QM9 SMILES is not an RDKit molecule")
    # The anchored identity language deliberately delegates stereochemical
    # state to the geometry channel.  Removing stereo before H removal also
    # lets RDKit remove explicit hydrogens that only support slash notation.
    Chem.RemoveStereochemistry(molecule)
    molecule = Chem.RemoveHs(molecule)
    Chem.SanitizeMol(molecule)
    if molecule.GetNumAtoms() <= 0 or any(
        atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()
    ):
        raise QM9AnchoredProbeCacheError("stereo-free heavy-atom projection failed")
    return molecule


def _morgan_rows(rdFingerprintGenerator, molecule) -> tuple[tuple[int, ...], ...]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=3,
        fpSize=4096,
        includeChirality=False,
        useBondTypes=True,
        includeRedundantEnvironments=True,
    )
    additional = rdFingerprintGenerator.AdditionalOutput()
    additional.AllocateBitInfoMap()
    generator.GetFingerprint(molecule, additionalOutput=additional)
    rows = [[-1] * 4 for _ in range(molecule.GetNumAtoms())]
    for bit_id, occurrences in additional.GetBitInfoMap().items():
        for atom_index, radius in occurrences:
            if not 0 <= int(radius) <= 3:
                continue
            old = rows[int(atom_index)][int(radius)]
            if old not in (-1, int(bit_id)):
                raise QM9AnchoredProbeCacheError(
                    "one atom/radius maps to multiple folded Morgan bits"
                )
            rows[int(atom_index)][int(radius)] = int(bit_id)
    if any(row[0] < 0 for row in rows):
        raise QM9AnchoredProbeCacheError("Morgan radius-zero coverage is incomplete")
    return tuple(tuple(row) for row in rows)


def _atom_to_motif(surface: Mapping[str, object], atom_count: int) -> tuple[int, ...]:
    phrases = surface.get("phrases")
    if not isinstance(phrases, list) or not phrases:
        raise QM9AnchoredProbeCacheError("anchored surface has no motif phrases")
    owners = [-1] * atom_count
    for motif_id, phrase in enumerate(phrases):
        atoms = phrase.get("motif_atom_indices") if isinstance(phrase, Mapping) else None
        if not isinstance(atoms, list) or not atoms:
            raise QM9AnchoredProbeCacheError("motif atom row is malformed")
        for atom in atoms:
            if (
                isinstance(atom, bool)
                or not isinstance(atom, int)
                or not 0 <= atom < atom_count
                or owners[atom] != -1
            ):
                raise QM9AnchoredProbeCacheError("motifs do not partition atoms")
            owners[atom] = motif_id
    if -1 in owners:
        raise QM9AnchoredProbeCacheError("motif atom partition is incomplete")
    return tuple(owners)


def _renumber_anchors_by_edge_order(topology: dict[str, object]) -> None:
    """Apply the persisted-pair model-facing anchor policy in-place.

    ``build_topology_augmentation`` retains molecule-local source anchor IDs,
    while the published anchored surface renumbers them after its deterministic
    edge sort.  Direct downstream projection must perform the same join.
    """

    domain = topology.get("logical_motif_domain")
    if not isinstance(domain, dict):
        raise QM9AnchoredProbeCacheError("topology motif domain is malformed")
    bonds = domain.get("cross_motif_bonds")
    slot_rows = domain.get("motif_slot_anchor_ids")
    if not isinstance(bonds, list) or not isinstance(slot_rows, list):
        raise QM9AnchoredProbeCacheError("topology anchor rows are malformed")
    source_to_edge = {}
    for bond in bonds:
        if not isinstance(bond, dict):
            raise QM9AnchoredProbeCacheError("topology bond row is malformed")
        source = int(bond["source_anchor_id"])
        edge = int(bond["edge_id"])
        if source in source_to_edge or edge < 0:
            raise QM9AnchoredProbeCacheError("source anchor IDs are not unique")
        source_to_edge[source] = edge
        bond["source_anchor_id"] = edge
    if sorted(source_to_edge.values()) != list(range(len(bonds))):
        raise QM9AnchoredProbeCacheError("edge IDs are not dense")
    try:
        domain["motif_slot_anchor_ids"] = [
            [source_to_edge[int(anchor)] for anchor in row]
            for row in slot_rows
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise QM9AnchoredProbeCacheError(
            "motif slots reference an unknown source anchor"
        ) from exc


def _record_document(record: ProductionMotifRecord) -> dict[str, object]:
    return {
        "record_artifact_sha256": record.record_artifact_sha256,
        "record_id": record.record_id,
        "storage_key": record.storage_key,
        "release_id": record.release_id,
        "geometry_record_content_sha256": record.geometry_record_content_sha256,
        "tokenizer_contract_sha256": record.tokenizer_contract_sha256,
        "tokenizer_snapshot_sha256": record.tokenizer_snapshot_sha256,
        "input_ids": list(record.input_ids),
        "token_to_logical_motif": list(record.token_to_logical_motif),
        "token_role": list(record.token_role),
        "identity_spans": [[span.start, span.stop] for span in record.identity_spans],
        "connection_token_indices": [list(row) for row in record.connection_token_indices],
        "logical_to_carrier": list(record.logical_to_carrier),
        "exact_identity_sha256": list(record.exact_identity_sha256),
        "source_atom_count": record.source_atom_count,
        "full_e3fp_ids": [list(row) for row in record.full_e3fp_ids],
        "atom_valid_mask": list(record.atom_valid_mask),
        "model_to_source_atom_index": list(record.model_to_source_atom_index),
        "atom_to_logical_motif": list(record.atom_to_logical_motif),
        "atom_is_attachment": list(record.atom_is_attachment),
        "connection_token_to_atom": list(record.connection_token_to_atom),
    }


def record_from_document(document: Mapping[str, object]) -> ProductionMotifRecord:
    """Rebuild the immutable training row without any chemistry hot path."""

    return ProductionMotifRecord(
        record_artifact_sha256=str(document["record_artifact_sha256"]),
        record_id=str(document["record_id"]),
        storage_key=str(document["storage_key"]),
        release_id=str(document["release_id"]),
        geometry_record_content_sha256=str(document["geometry_record_content_sha256"]),
        tokenizer_contract_sha256=str(document["tokenizer_contract_sha256"]),
        tokenizer_snapshot_sha256=str(document["tokenizer_snapshot_sha256"]),
        input_ids=tuple(int(value) for value in document["input_ids"]),  # type: ignore[index]
        token_to_logical_motif=tuple(int(value) for value in document["token_to_logical_motif"]),  # type: ignore[index]
        token_role=tuple(str(value) for value in document["token_role"]),  # type: ignore[index]
        identity_spans=tuple(Span(int(row[0]), int(row[1])) for row in document["identity_spans"]),  # type: ignore[index]
        connection_token_indices=tuple(tuple(int(value) for value in row) for row in document["connection_token_indices"]),  # type: ignore[index]
        logical_to_carrier=tuple(int(value) for value in document["logical_to_carrier"]),  # type: ignore[index]
        exact_identity_sha256=tuple(str(value) for value in document["exact_identity_sha256"]),  # type: ignore[index]
        source_atom_count=int(document["source_atom_count"]),
        full_e3fp_ids=tuple(tuple(int(value) for value in row) for row in document["full_e3fp_ids"]),  # type: ignore[index]
        atom_valid_mask=tuple(bool(value) for value in document["atom_valid_mask"]),  # type: ignore[index]
        model_to_source_atom_index=tuple(int(value) for value in document["model_to_source_atom_index"]),  # type: ignore[index]
        atom_to_logical_motif=tuple(int(value) for value in document["atom_to_logical_motif"]),  # type: ignore[index]
        atom_is_attachment=tuple(bool(value) for value in document["atom_is_attachment"]),  # type: ignore[index]
        connection_token_to_atom=tuple(int(value) for value in document["connection_token_to_atom"]),  # type: ignore[index]
    )


def project_state(
    *,
    Chem,
    rdFingerprintGenerator,
    smiles: str,
    selfies: str,
    molecule_fp: Sequence[Sequence[int]],
    split: str,
    molecule_order_index: int,
    source_row_index: int,
    targets: Mapping[str, float],
    macro_rows: Sequence[Mapping[str, object]],
    tokenizer: AnchoredTokenizerBinding,
    linearizer_sha256: str,
) -> dict[str, object]:
    if split not in {"train", "dev", "test"}:
        raise QM9AnchoredProbeCacheError("QM9 split is invalid")
    state = state_sha256(selfies, molecule_fp)
    atom_rows = tuple(
        tuple(int(value) for value in row)
        for row in molecule_fp
        if len(row) == 4 and int(row[0]) >= 0
    )
    if not atom_rows:
        raise QM9AnchoredProbeCacheError("QM9 state has no usable E3FP atom row")
    molecule = _stereo_free_heavy_mol(Chem, smiles)
    if molecule.GetNumAtoms() != len(atom_rows):
        raise QM9AnchoredProbeCacheError(
            "SELFIES E3FP rows differ from the stereo-free heavy-atom axis"
        )

    linearization = linearize_mol(molecule)
    fragment_digests = tuple(
        hashlib.sha256(fragment.encode("utf-8")).hexdigest()
        for fragment in linearization.fragment_sequence
    )
    record_id = f"qm9:{molecule_order_index}:{state[:16]}"
    storage_key = hashlib.sha256(f"{smiles}\0{state}".encode("utf-8")).hexdigest()
    topology = build_topology_augmentation(
        linearization_result=linearization,
        member_id=record_id,
        base_record_content_sha256=state,
        linearizer_spec_sha256=linearizer_sha256,
        expected_motif_atom_indices=linearization.motif_atom_groups,
        expected_motif_lexeme_sha256=fragment_digests,
        source_atom_count=molecule.GetNumAtoms(),
        model_to_source_atom_index=tuple(range(molecule.GetNumAtoms())),
    )
    _renumber_anchors_by_edge_order(topology)
    surface = surface_document(build_stereo_free_anchored_surface(topology))
    owners = _atom_to_motif(surface, molecule.GetNumAtoms())
    attachment = tuple(bool(value) for value in surface["atom_is_attachment"])  # type: ignore[index]
    geometry = ProductionMotifRecord(
        record_artifact_sha256=state,
        record_id=record_id,
        storage_key=storage_key,
        release_id="qm9-3dmolt5-source",
        geometry_record_content_sha256=state,
        tokenizer_contract_sha256=tokenizer.tokenizer_contract_sha256,
        tokenizer_snapshot_sha256=tokenizer.tokenizer_snapshot_sha256,
        input_ids=(),
        token_to_logical_motif=(),
        token_role=(),
        identity_spans=(),
        connection_token_indices=(),
        logical_to_carrier=(),
        exact_identity_sha256=(),
        source_atom_count=molecule.GetNumAtoms(),
        full_e3fp_ids=atom_rows,
        atom_valid_mask=tuple(True for _ in atom_rows),
        model_to_source_atom_index=tuple(range(len(atom_rows))),
        atom_to_logical_motif=owners,
        atom_is_attachment=attachment,
    )
    anchored = bind_anchored_training_record(
        {"surface": surface, "storage_key": storage_key},
        geometry,
        macro_rows=macro_rows,
        tokenizer=tokenizer,
        release_id="qm9-anchored-probe-v1",
    ).as_factorized_record()
    morgan = _morgan_rows(rdFingerprintGenerator, molecule)
    values = [targets.get(name) for name in PROPERTY_NAMES]
    target_mask = [value is not None for value in values]
    if not any(target_mask):
        raise QM9AnchoredProbeCacheError("QM9 state has no registered target")
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "split": split,
        "molecule_order_index": int(molecule_order_index),
        "source_row_index": int(source_row_index),
        "smiles": smiles,
        "state_sha256": state,
        "targets_hartree": [float(value) if value is not None else None for value in values],
        "target_mask": target_mask,
        "morgan_state_ids": [list(row) for row in morgan],
        "record": _record_document(anchored),
        "state_coverage": {
            "atom_rows": len(atom_rows),
            "nonnegative_by_level": [
                sum(int(row[level] >= 0) for row in atom_rows) for level in range(4)
            ],
        },
    }


def _group_membership(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["smiles"]), str(row["state_sha256"]))
        group = groups.setdefault(
            key,
            {
                "smiles": key[0],
                "state_sha256": key[1],
                "split": str(row["split"]),
                "molecule_order_index": int(row["molecule_order_index"]),
                "source_row_indices": [],
                "targets": {},
            },
        )
        if (
            group["split"] != row["split"]
            or group["molecule_order_index"] != row["molecule_order_index"]
        ):
            raise QM9AnchoredProbeCacheError("one exact state crosses frozen splits")
        prop = str(row["property"])
        if prop not in PROPERTY_ORDER or prop in group["targets"]:  # type: ignore[operator]
            raise QM9AnchoredProbeCacheError("state/property membership is duplicated")
        group["targets"][prop] = float(row["target_hartree"])  # type: ignore[index]
        group["source_row_indices"].append(int(row["source_row_index"]))  # type: ignore[union-attr]
    result = list(groups.values())
    result.sort(key=lambda row: (int(row["molecule_order_index"]), str(row["state_sha256"])))
    return result


def build_cache(
    *,
    source_parquet: Path,
    membership_jsonl: Path,
    macro_registry: Path,
    tokenizer_manifest: Path,
    output_dir: Path,
) -> dict[str, object]:
    import pyarrow.parquet as pq
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    output_dir = Path(output_dir).expanduser().resolve()
    staging = output_dir.with_name(output_dir.name + ".staging")
    if output_dir.exists() or staging.exists():
        raise QM9AnchoredProbeCacheError("output and staging paths must be absent")
    membership = _read_jsonl(membership_jsonl)
    groups = _group_membership(membership)
    macro_rows = _read_jsonl(macro_registry)
    tokenizer_document = json.loads(Path(tokenizer_manifest).read_text(encoding="utf-8"))
    tokenizer = tokenizer_binding_from_candidate_manifest(tokenizer_document)
    table = pq.read_table(
        source_parquet, columns=["smiles", "selfies", "molecule_fp"]
    )
    linearizer_sha256 = _sha256_file(Path(__import__(
        "most_t5_next.r1.adapter.mol_linearizer", fromlist=["__file__"]
    ).__file__))
    staging.mkdir(parents=True)
    rejects = []
    records = []
    level_counts = [0, 0, 0, 0]
    split_counts = defaultdict(int)
    property_counts = {split: [0, 0, 0] for split in ("train", "dev", "test")}
    for group in groups:
        source_index = min(group["source_row_indices"])
        source = {
            name: table[name][source_index].as_py() for name in table.column_names
        }
        if source["smiles"] != group["smiles"]:
            raise QM9AnchoredProbeCacheError("membership/source SMILES differ")
        try:
            row = project_state(
                Chem=Chem,
                rdFingerprintGenerator=rdFingerprintGenerator,
                smiles=str(source["smiles"]),
                selfies=str(source["selfies"]),
                molecule_fp=source["molecule_fp"],
                split=str(group["split"]),
                molecule_order_index=int(group["molecule_order_index"]),
                source_row_index=source_index,
                targets=group["targets"],
                macro_rows=macro_rows,
                tokenizer=tokenizer,
                linearizer_sha256=linearizer_sha256,
            )
        except Exception as exc:
            rejects.append({
                "molecule_order_index": group["molecule_order_index"],
                "state_sha256": group["state_sha256"],
                "stage": type(exc).__name__,
                "error": str(exc),
            })
            continue
        records.append(row)
        split = str(row["split"])
        split_counts[split] += 1
        for index, present in enumerate(row["target_mask"]):
            property_counts[split][index] += int(present)
        for index, count in enumerate(row["state_coverage"]["nonnegative_by_level"]):
            level_counts[index] += int(count)

    with (staging / "records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (staging / "rejects.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rejects:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if records else "failed",
        "sources": {
            "parquet": str(Path(source_parquet).resolve()),
            "membership": str(Path(membership_jsonl).resolve()),
            "macro_registry": str(Path(macro_registry).resolve()),
            "tokenizer_manifest": str(Path(tokenizer_manifest).resolve()),
        },
        "counts": {
            "scheduled_exact_states": len(groups),
            "records": len(records),
            "rejects": len(rejects),
            "records_by_split": dict(split_counts),
            "property_targets_by_split": {
                split: dict(zip(PROPERTY_NAMES, values))
                for split, values in property_counts.items()
            },
            "nonnegative_e3fp_ids_by_level": level_counts,
        },
        "contracts": {
            "split_unit": "exact_smiles_molecular_identity",
            "stereo_free_motif_identity": True,
            "stereochemical_state_delegated_to_e3fp": True,
            "e3fp_selfies_axis_filtered_to_heavy_atoms": True,
            "morgan_include_chirality": False,
            "morgan_is_coordinate_blind_control": True,
            "random_corruption_cached": False,
            "graphports_exposed": False,
            "partial_property_targets_retained_with_mask": True,
        },
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.rename(output_dir)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--membership-jsonl", type=Path, required=True)
    parser.add_argument("--macro-registry", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_cache(
        source_parquet=args.source_parquet,
        membership_jsonl=args.membership_jsonl,
        macro_registry=args.macro_registry,
        tokenizer_manifest=args.tokenizer_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": report["status"], **report["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROPERTY_NAMES",
    "QM9AnchoredProbeCacheError",
    "SCHEMA_VERSION",
    "build_cache",
    "project_state",
    "record_from_document",
]
