#!/usr/bin/env python3
"""Freeze official KPGT scaffold memberships for BACE, BBBP, and ClinTox.

The KPGT release stores each scaffold replica as a NumPy object array.  Reading
such an array requires pickle.  This module deliberately keeps that trust
boundary narrow: ``allow_pickle=True`` is reached only after the caller has
asserted the exact official-source provenance token and the tool has hashed
the supplied official Figshare archive and proved that all 12 files under
``dataset_root`` are byte-identical to unique, safe regular members of that
archive. A caller-provided SHA-256 is optional integrity metadata, not a
scientific-admission condition. A merely layout-compatible directory is not
silently promoted to an official benchmark release.

The input tree is read only.  Output is written only to a new directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import stat
import tarfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from rdkit import Chem, rdBase
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError as exc:  # pragma: no cover - only a broken runtime reaches this.
    raise RuntimeError(
        "build_kpgt_scaffold_manifests.py requires RDKit"
    ) from exc


OFFICIAL_SOURCE_PROVENANCE = "verified_official_kpgt_figshare"
KPGT_REPOSITORY_URL = "https://github.com/lihan97/KPGT"
KPGT_PAPER_RELEASE_COMMIT = "390f29529dde268fed19203e7435307ae15dc083"
KPGT_INSPECTED_CURRENT_COMMIT = "47dc1646c70b2138a157de481d24a1ac35d174cd"
FIGSHARE_DOI = "10.6084/m9.figshare.19914811"
FIGSHARE_FILE_ID = "35391163"
SPLIT_REPLICAS = ("scaffold-0", "scaffold-1", "scaffold-2")
PARTITIONS = ("train", "validation", "test")

SOURCE_MANIFEST_FILENAME = "source_manifest.json"
PROTECTED_UNION_FILENAME = "protected_eval_union.jsonl"
SUMMARY_FILENAME = "summary.json"

SOURCE_SCHEMA = "most-t5-r1/kpgt-official-source-manifest/v1"
MEMBER_SCHEMA = "most-t5-r1/kpgt-scaffold-member/v1"
PROTECTED_SCHEMA = "most-t5-r1/kpgt-protected-connectivity/v1"
SUMMARY_SCHEMA = "most-t5-r1/kpgt-scaffold-manifest-summary/v1"
IDENTITY_COLLECTION_SCHEMA = "most-t5-r1/identity-collection-manifest/v1"
IDENTITY_ROW_SCHEMA = "most-t5-r1/molecule-identity-row/v1"
IDENTITY_NORMALIZATION_CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "contracts"
    / "pcqm4mv2_identity_normalization_contract.json"
)

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
MISSING_LABELS = frozenset(("", "na", "n/a", "nan", "none", "null"))


class KpgtManifestError(ValueError):
    """Raised when source provenance or benchmark membership is invalid."""


@dataclass(frozen=True)
class TaskSpec:
    task: str
    csv_name: str
    smiles_column: str
    label_columns: tuple[str, ...]


TASK_SPECS = (
    TaskSpec("bace", "bace.csv", "smiles", ("Class",)),
    TaskSpec("bbbp", "bbbp.csv", "smiles", ("p_np",)),
    TaskSpec(
        "clintox",
        "clintox.csv",
        "smiles",
        ("FDA_APPROVED", "CT_TOX"),
    ),
)


@dataclass(frozen=True)
class FileFact:
    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ArchiveMemberFact:
    logical_relative_path: str
    archive_member_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class MoleculeFact:
    source_row_index: int
    source_smiles: str
    canonical_isomeric_smiles: str
    canonical_connectivity_smiles: str
    canonical_isomeric_sha256: str
    canonical_connectivity_sha256: str
    murcko_scaffold_achiral: str
    murcko_scaffold_chiral: str
    labels: Mapping[str, int | None]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> FileFact:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return FileFact(relative_path="", bytes=size, sha256=digest.hexdigest())


def _validate_official_provenance(
    source_provenance: str, official_archive_sha256: str | None
) -> str | None:
    if source_provenance != OFFICIAL_SOURCE_PROVENANCE:
        raise KpgtManifestError(
            "source_provenance must be exactly " + OFFICIAL_SOURCE_PROVENANCE
        )
    if official_archive_sha256 is None:
        return None
    normalized = official_archive_sha256.lower()
    if not SHA256_RE.fullmatch(normalized):
        raise KpgtManifestError(
            "optional official_archive_sha256 must be exactly 64 hexadecimal characters"
        )
    return normalized


def _require_regular_nonsymlink(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise KpgtManifestError(label + " must be a regular non-symlink file: " + str(path))
    return path


def _require_source_file(root: Path, relative_path: str) -> Path:
    if root.is_symlink():
        raise KpgtManifestError("dataset_root must not be a symbolic link")
    path = root / Path(relative_path)
    current = root
    for component in Path(relative_path).parts:
        current = current / component
        if current.is_symlink():
            raise KpgtManifestError(
                "KPGT source path must not contain symbolic links: " + relative_path
            )
    if not path.is_file():
        raise KpgtManifestError("required KPGT source file is missing: " + relative_path)
    return path


def _file_fact(root: Path, relative_path: str) -> FileFact:
    path = _require_source_file(root, relative_path)
    fact = sha256_file(path)
    return FileFact(relative_path=relative_path, bytes=fact.bytes, sha256=fact.sha256)


def _required_source_paths() -> tuple[str, ...]:
    paths: list[str] = []
    for spec in TASK_SPECS:
        paths.append(f"{spec.task}/{spec.csv_name}")
        paths.extend(
            f"{spec.task}/splits/{split}.npy" for split in SPLIT_REPLICAS
        )
    return tuple(paths)


def _safe_archive_member_path(raw_name: str) -> str:
    """Return one strict POSIX member path, rejecting traversal spellings."""
    if not isinstance(raw_name, str) or not raw_name or "\x00" in raw_name:
        raise KpgtManifestError("archive contains an empty or NUL-bearing member name")
    if "\\" in raw_name or raw_name.startswith("/"):
        raise KpgtManifestError("archive contains an unsafe member path: " + raw_name)
    core = raw_name[:-1] if raw_name.endswith("/") else raw_name
    if not core:
        raise KpgtManifestError("archive contains an unsafe root member")
    parts = core.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise KpgtManifestError("archive contains path traversal: " + raw_name)
    if re.fullmatch(r"[A-Za-z]:.*", parts[0]):
        raise KpgtManifestError("archive contains a drive-qualified member: " + raw_name)
    normalized = PurePosixPath(*parts).as_posix()
    if normalized != core:
        raise KpgtManifestError("archive member path is not canonical: " + raw_name)
    return normalized


def _hash_stream(handle: Any) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        size += len(block)
        digest.update(block)
    return size, digest.hexdigest()


def _match_required_member_names(
    file_names: Sequence[str], required_paths: Sequence[str]
) -> dict[str, str]:
    matches: dict[str, str] = {}
    prefixes: set[str] = set()
    for relative_path in required_paths:
        suffix = "/" + relative_path
        candidates = [
            name for name in file_names if name == relative_path or name.endswith(suffix)
        ]
        if len(candidates) != 1:
            raise KpgtManifestError(
                "official archive must contain exactly one member ending in "
                + relative_path
                + f"; observed {len(candidates)}"
            )
        member_name = candidates[0]
        prefix = member_name[: -len(relative_path)].rstrip("/")
        prefixes.add(prefix)
        matches[relative_path] = member_name
    if len(prefixes) != 1:
        raise KpgtManifestError(
            "required KPGT archive members do not share one archive-root prefix"
        )
    return matches


def _bind_zip_members(
    archive_path: Path, required_paths: Sequence[str]
) -> tuple[str, dict[str, ArchiveMemberFact]]:
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        files: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            name = _safe_archive_member_path(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(mode)
            if file_type == stat.S_IFLNK:
                raise KpgtManifestError("official archive contains a symbolic link: " + name)
            if info.is_dir():
                continue
            if file_type not in (0, stat.S_IFREG):
                raise KpgtManifestError(
                    "official archive contains a non-regular member: " + name
                )
            if name in files:
                raise KpgtManifestError("official archive contains duplicate member: " + name)
            files[name] = info
        selected = _match_required_member_names(tuple(files), required_paths)
        facts: dict[str, ArchiveMemberFact] = {}
        for relative_path in required_paths:
            name = selected[relative_path]
            with archive.open(files[name], mode="r") as handle:
                size, digest = _hash_stream(handle)
            if size != files[name].file_size:
                raise KpgtManifestError(
                    "archive member decompressed size differs from ZIP metadata: " + name
                )
            facts[relative_path] = ArchiveMemberFact(
                logical_relative_path=relative_path,
                archive_member_path=name,
                bytes=size,
                sha256=digest,
            )
    return "zip", facts


def _bind_tar_members(
    archive_path: Path, required_paths: Sequence[str]
) -> tuple[str, dict[str, ArchiveMemberFact]]:
    with tarfile.open(archive_path, mode="r:*") as archive:
        files: dict[str, tarfile.TarInfo] = {}
        for info in archive.getmembers():
            name = _safe_archive_member_path(info.name)
            if info.issym() or info.islnk():
                raise KpgtManifestError("official archive contains a symbolic link: " + name)
            if info.isdir():
                continue
            if not info.isfile():
                raise KpgtManifestError(
                    "official archive contains a non-regular member: " + name
                )
            if name in files:
                raise KpgtManifestError("official archive contains duplicate member: " + name)
            files[name] = info
        selected = _match_required_member_names(tuple(files), required_paths)
        facts: dict[str, ArchiveMemberFact] = {}
        for relative_path in required_paths:
            name = selected[relative_path]
            handle = archive.extractfile(files[name])
            if handle is None:
                raise KpgtManifestError("cannot read archive member: " + name)
            with handle:
                size, digest = _hash_stream(handle)
            if size != files[name].size:
                raise KpgtManifestError(
                    "archive member size differs from TAR metadata: " + name
                )
            facts[relative_path] = ArchiveMemberFact(
                logical_relative_path=relative_path,
                archive_member_path=name,
                bytes=size,
                sha256=digest,
            )
    return "tar", facts


def _bind_archive_members(
    archive_path: Path, required_paths: Sequence[str]
) -> tuple[str, dict[str, ArchiveMemberFact]]:
    """Hash required members in-place; nothing is extracted to the filesystem."""
    try:
        if zipfile.is_zipfile(archive_path):
            return _bind_zip_members(archive_path, required_paths)
        if tarfile.is_tarfile(archive_path):
            return _bind_tar_members(archive_path, required_paths)
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
        raise KpgtManifestError("cannot inspect official archive: " + str(exc)) from exc
    raise KpgtManifestError("official archive must be a readable ZIP or TAR archive")


def _parse_binary_label(raw: object, *, task: str, column: str, row_index: int) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text.lower() in MISSING_LABELS:
        return None
    try:
        numeric = float(text)
    except ValueError as exc:
        raise KpgtManifestError(
            f"{task} row {row_index} label {column!r} is not binary: {text!r}"
        ) from exc
    if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise KpgtManifestError(
            f"{task} row {row_index} label {column!r} is not 0/1 or missing: {text!r}"
        )
    return int(numeric)


def _canonicalize_molecule(
    smiles: str,
    *,
    task: str,
    row_index: int,
    labels: Mapping[str, int | None],
) -> MoleculeFact:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise KpgtManifestError(f"{task} row {row_index} has invalid SMILES: {smiles!r}")
    parameters = Chem.RemoveHsParameters()
    if not hasattr(parameters, "removeDefiningBondStereo"):
        raise RuntimeError(
            "installed RDKit lacks RemoveHsParameters.removeDefiningBondStereo"
        )
    parameters.removeDefiningBondStereo = True
    normalized = Chem.RemoveHs(Chem.Mol(mol), parameters, sanitize=True)
    Chem.SanitizeMol(normalized)
    Chem.AssignStereochemistry(normalized, cleanIt=True, force=True)
    isomeric = Chem.MolToSmiles(
        normalized, canonical=True, isomericSmiles=True, kekuleSmiles=False
    )
    connectivity = Chem.MolToSmiles(
        normalized, canonical=True, isomericSmiles=False, kekuleSmiles=False
    )
    achiral = MurckoScaffold.MurckoScaffoldSmiles(
        mol=normalized, includeChirality=False
    )
    chiral = MurckoScaffold.MurckoScaffoldSmiles(
        mol=normalized, includeChirality=True
    )
    return MoleculeFact(
        source_row_index=row_index,
        source_smiles=smiles,
        canonical_isomeric_smiles=isomeric,
        canonical_connectivity_smiles=connectivity,
        canonical_isomeric_sha256=sha256_bytes(isomeric.encode("utf-8")),
        canonical_connectivity_sha256=sha256_bytes(connectivity.encode("utf-8")),
        murcko_scaffold_achiral=achiral,
        murcko_scaffold_chiral=chiral,
        labels=dict(labels),
    )


def _read_task_csv(path: Path, spec: TaskSpec) -> tuple[list[MoleculeFact], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or ())
        required = (spec.smiles_column,) + spec.label_columns
        missing = [name for name in required if name not in columns]
        if missing:
            raise KpgtManifestError(
                f"{spec.task} CSV is missing required columns: {', '.join(missing)}"
            )
        molecules: list[MoleculeFact] = []
        for row_index, row in enumerate(reader):
            if None in row:
                raise KpgtManifestError(
                    f"{spec.task} row {row_index} has fields beyond the CSV header"
                )
            smiles = (row.get(spec.smiles_column) or "").strip()
            if not smiles:
                raise KpgtManifestError(
                    f"{spec.task} row {row_index} has an empty SMILES value"
                )
            labels = {
                column: _parse_binary_label(
                    row.get(column), task=spec.task, column=column, row_index=row_index
                )
                for column in spec.label_columns
            }
            molecules.append(
                _canonicalize_molecule(
                    smiles,
                    task=spec.task,
                    row_index=row_index,
                    labels=labels,
                )
            )
    if not molecules:
        raise KpgtManifestError(spec.task + " CSV contains no molecule rows")
    return molecules, columns


def _integer_indices(raw: Any, *, split: str, partition: str) -> list[int]:
    array = np.asarray(raw)
    if array.ndim != 1:
        raise KpgtManifestError(f"{split} {partition} indices must be one-dimensional")
    indices: list[int] = []
    for position, value in enumerate(array.tolist()):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise KpgtManifestError(
                f"{split} {partition} index {position} is not an integer"
            )
        indices.append(int(value))
    return indices


def _load_split_membership(
    path: Path,
    *,
    split: str,
    row_count: int,
    provenance_already_verified: bool,
) -> dict[str, list[int]]:
    """Load one KPGT object array inside the explicit official-source boundary."""
    if not provenance_already_verified:
        raise KpgtManifestError("refusing allow_pickle outside verified official KPGT boundary")
    try:
        payload = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise KpgtManifestError(f"cannot read KPGT split file {path.name}: {exc}") from exc
    if not isinstance(payload, np.ndarray) or payload.ndim != 1 or len(payload) != 3:
        raise KpgtManifestError(
            f"{split} must be a one-dimensional three-part object array"
        )
    membership = {
        partition: _integer_indices(payload[index], split=split, partition=partition)
        for index, partition in enumerate(PARTITIONS)
    }
    for partition, indices in membership.items():
        if len(indices) != len(set(indices)):
            raise KpgtManifestError(f"{split} {partition} contains duplicate indices")
        invalid = [index for index in indices if index < 0 or index >= row_count]
        if invalid:
            raise KpgtManifestError(
                f"{split} {partition} contains out-of-range index {invalid[0]}"
            )
    sets = {partition: set(indices) for partition, indices in membership.items()}
    for left_index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[left_index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise KpgtManifestError(
                    f"{split} leaks {len(overlap)} row indices between {left} and {right}"
                )
    covered = set().union(*(sets[name] for name in PARTITIONS))
    expected = set(range(row_count))
    if covered != expected:
        missing = sorted(expected - covered)
        raise KpgtManifestError(
            f"{split} does not completely cover the CSV; first missing index is {missing[0]}"
        )
    return membership


def _validate_scaffolds_and_labels(
    *,
    spec: TaskSpec,
    split: str,
    membership: Mapping[str, Sequence[int]],
    molecules: Sequence[MoleculeFact],
) -> dict[str, object]:
    scaffold_sets: dict[str, dict[str, set[str]]] = {
        "achiral": {},
        "chiral": {},
    }
    label_counts: dict[str, dict[str, dict[str, int]]] = {}
    for partition in PARTITIONS:
        selected = [molecules[index] for index in membership[partition]]
        scaffold_sets["achiral"][partition] = {
            item.murcko_scaffold_achiral for item in selected
        }
        scaffold_sets["chiral"][partition] = {
            item.murcko_scaffold_chiral for item in selected
        }
        label_counts[partition] = {}
        for label in spec.label_columns:
            counts = Counter(item.labels[label] for item in selected)
            positives = counts[1]
            negatives = counts[0]
            missing = counts[None]
            if positives == 0 or negatives == 0:
                raise KpgtManifestError(
                    f"{spec.task} {split} {partition} label {label!r} has a single "
                    "evaluable class (both 0 and 1 are required)"
                )
            label_counts[partition][label] = {
                "negative": negatives,
                "positive": positives,
                "missing": missing,
                "evaluable": positives + negatives,
            }
    intersections: dict[str, dict[str, int]] = {}
    for flavor in ("achiral", "chiral"):
        pairs: dict[str, int] = {}
        for left_index, left in enumerate(PARTITIONS):
            for right in PARTITIONS[left_index + 1 :]:
                overlap = scaffold_sets[flavor][left] & scaffold_sets[flavor][right]
                pairs[left + "__" + right] = len(overlap)
                if overlap:
                    raise KpgtManifestError(
                        f"{spec.task} {split} has {len(overlap)} {flavor} Bemis-Murcko "
                        f"scaffolds shared by {left} and {right}"
                    )
        intersections[flavor] = pairs
    return {
        "partition_counts": {
            partition: len(membership[partition]) for partition in PARTITIONS
        },
        "label_counts": label_counts,
        "murcko_cross_partition_intersection_counts": intersections,
    }


def _member_row(
    *,
    spec: TaskSpec,
    split: str,
    partition: str,
    molecule: MoleculeFact,
    csv_sha256: str,
    split_sha256: str,
) -> dict[str, object]:
    return {
        "schema": MEMBER_SCHEMA,
        "task": spec.task,
        "split_replica": split,
        "partition": partition,
        "member_id": f"kpgt:{spec.task}:{molecule.source_row_index}",
        "source_csv_row_index": molecule.source_row_index,
        "source_csv_sha256": csv_sha256,
        "source_split_sha256": split_sha256,
        "source_smiles": molecule.source_smiles,
        "canonical_isomeric_smiles": molecule.canonical_isomeric_smiles,
        "canonical_isomeric_smiles_sha256": molecule.canonical_isomeric_sha256,
        "canonical_connectivity_smiles": molecule.canonical_connectivity_smiles,
        "canonical_connectivity_smiles_sha256": molecule.canonical_connectivity_sha256,
        "murcko_scaffold_achiral": molecule.murcko_scaffold_achiral,
        "murcko_scaffold_chiral": molecule.murcko_scaffold_chiral,
        "labels": dict(molecule.labels),
    }


def _write_bytes_new(path: Path, payload: bytes) -> dict[str, object]:
    if path.exists():
        raise KpgtManifestError("refusing to overwrite output artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "relative_path": path.as_posix(),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _identity_rows_payload(
    *, collection_id: str, molecules: Sequence[MoleculeFact]
) -> tuple[bytes, str]:
    rows = sorted(
        (
            {
                "schema_version": IDENTITY_ROW_SCHEMA,
                "collection_id": collection_id,
                "member_id": f"kpgt:{molecule.source_row_index}",
                "connectivity_identity_sha256": molecule.canonical_connectivity_sha256,
                "stereo_identity_sha256": molecule.canonical_isomeric_sha256,
                "conformer_identity_sha256": None,
            }
            for molecule in molecules
        ),
        key=lambda row: row["member_id"].encode("utf-8"),
    )
    key_digest = hashlib.sha256()
    for row in rows:
        key_digest.update(row["member_id"].encode("utf-8") + b"\n")
    return _json_lines(rows), key_digest.hexdigest()


def _write_identity_collection(
    *,
    output: Path,
    spec: TaskSpec,
    split_replica: str,
    partition: str,
    molecules: Sequence[MoleculeFact],
    source_manifest_sha256: str,
    identity_spec_sha256: str,
    extractor_sha256: str,
    excluded_columns: Sequence[str],
    archive_sha256: str,
) -> dict[str, object]:
    collection_id = f"kpgt-{spec.task}-{split_replica}-{partition}-official-v1"
    relative_directory = f"collections/{spec.task}/{split_replica}/{partition}"
    rows_payload, key_sha256 = _identity_rows_payload(
        collection_id=collection_id, molecules=molecules
    )
    rows_artifact = _write_bytes_new(
        output / relative_directory / "molecule_identity_rows.jsonl",
        rows_payload,
    )
    rows_artifact["relative_path"] = (
        relative_directory + "/molecule_identity_rows.jsonl"
    )
    rows_declaration = {
        "path": "molecule_identity_rows.jsonl",
        "bytes": rows_artifact["bytes"],
        "sha256": rows_artifact["sha256"],
        "row_count": len(molecules),
        "key_lf_sha256": key_sha256,
    }
    role = {
        "train": "downstream_train",
        "validation": "downstream_validation",
        "test": "downstream_test",
    }[partition]
    manifest = {
        "schema_version": IDENTITY_COLLECTION_SCHEMA,
        "collection_id": collection_id,
        "dataset_id": "kpgt-" + spec.task,
        "release_id": "kpgt-figshare-35391163-" + archive_sha256[:12],
        "phase": "downstream",
        "split": partition,
        "role": role,
        "task_family": "moleculenet_" + spec.task + "_classification",
        "identity_specs": {
            "connectivity_identity_spec_sha256": identity_spec_sha256,
            "stereo_identity_spec_sha256": identity_spec_sha256,
            "conformer_identity": {"status": "unavailable", "spec_sha256": None},
            "text_identity": {
                "status": "unavailable",
                "exact_spec_sha256": None,
                "normalized_spec_sha256": None,
            },
        },
        "molecule_rows": rows_declaration,
        "text_pair_rows": None,
        "provenance": {
            "source_identity_namespace": (
                f"kpgt_figshare_35391163:{spec.task}:{split_replica}:csv_row_index"
            ),
            "source_release_manifest_sha256": source_manifest_sha256,
            "extractor_sha256": extractor_sha256,
            "excluded_source_metadata_keys": sorted(set(excluded_columns)),
        },
    }
    manifest_payload = _json_document(manifest)
    manifest_artifact = _write_bytes_new(
        output / relative_directory / "collection_manifest.json",
        manifest_payload,
    )
    manifest_artifact["relative_path"] = relative_directory + "/collection_manifest.json"
    manifest_artifact.update(
        {
            "collection_id": collection_id,
            "task": spec.task,
            "split_replica": split_replica,
            "partition": partition,
            "role": role,
            "molecule_rows": rows_artifact,
        }
    )
    return manifest_artifact


def _json_document(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _json_lines(rows: Iterable[object]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def build_kpgt_scaffold_manifests(
    dataset_root: Path | str,
    output_dir: Path | str,
    *,
    official_archive_path: Path | str,
    official_archive_sha256: str | None = None,
    source_provenance: str,
) -> dict[str, object]:
    """Validate official KPGT files and write deterministic benchmark manifests."""
    recorded_archive_sha256 = _validate_official_provenance(
        source_provenance, official_archive_sha256
    )
    archive_path = _require_regular_nonsymlink(
        Path(official_archive_path), "official_archive_path"
    )
    archive_fact = sha256_file(archive_path)
    root = Path(dataset_root)
    output = Path(output_dir)
    if not root.is_dir():
        raise KpgtManifestError("dataset_root is not a directory: " + str(root))
    if output.exists():
        raise KpgtManifestError("output_dir must not already exist: " + str(output))

    required_paths = _required_source_paths()
    archive_format, archive_members = _bind_archive_members(
        archive_path, required_paths
    )
    source_file_by_path: dict[str, FileFact] = {}
    for relative_path in required_paths:
        source_fact = _file_fact(root, relative_path)
        member_fact = archive_members[relative_path]
        if (source_fact.bytes, source_fact.sha256) != (
            member_fact.bytes,
            member_fact.sha256,
        ):
            raise KpgtManifestError(
                "dataset_root file is not byte-identical to its official archive "
                "member: "
                + relative_path
            )
        source_file_by_path[relative_path] = source_fact

    if not IDENTITY_NORMALIZATION_CONTRACT.is_file():
        raise KpgtManifestError(
            "shared identity normalization contract is missing: "
            + str(IDENTITY_NORMALIZATION_CONTRACT)
        )
    identity_contract_fact = sha256_file(IDENTITY_NORMALIZATION_CONTRACT)
    extractor_fact = sha256_file(Path(__file__).resolve())

    task_payloads: dict[str, dict[str, Any]] = {}
    for spec in TASK_SPECS:
        csv_relative = f"{spec.task}/{spec.csv_name}"
        csv_fact = source_file_by_path[csv_relative]
        molecules, csv_columns = _read_task_csv(root / csv_relative, spec)
        split_payloads: dict[str, Any] = {}
        for split in SPLIT_REPLICAS:
            split_relative = f"{spec.task}/splits/{split}.npy"
            split_fact = source_file_by_path[split_relative]
            membership = _load_split_membership(
                root / split_relative,
                split=split,
                row_count=len(molecules),
                provenance_already_verified=True,
            )
            checks = _validate_scaffolds_and_labels(
                spec=spec,
                split=split,
                membership=membership,
                molecules=molecules,
            )
            split_payloads[split] = {
                "source_fact": split_fact,
                "membership": membership,
                "checks": checks,
            }
        task_payloads[spec.task] = {
            "spec": spec,
            "csv_fact": csv_fact,
            "csv_columns": csv_columns,
            "molecules": molecules,
            "splits": split_payloads,
        }

    source_manifest = {
        "schema": SOURCE_SCHEMA,
        "source_provenance": source_provenance,
        "official_archive": {
            "figshare_doi": FIGSHARE_DOI,
            "figshare_file_id": FIGSHARE_FILE_ID,
            "file_name": archive_path.name,
            "format": archive_format,
            "bytes": archive_fact.bytes,
            "sha256": archive_fact.sha256,
            "sha256_role": "optional_integrity_record_not_scientific_admission",
            "sha256_basis": "observed digest computed by this tool",
            "caller_recorded_sha256": recorded_archive_sha256,
            "caller_record_matches_observed": (
                None
                if recorded_archive_sha256 is None
                else recorded_archive_sha256 == archive_fact.sha256
            ),
        },
        "repository": {
            "url": KPGT_REPOSITORY_URL,
            "paper_release_commit": KPGT_PAPER_RELEASE_COMMIT,
            "inspected_current_commit": KPGT_INSPECTED_CURRENT_COMMIT,
        },
        "numpy_object_array_trust_boundary": {
            "allow_pickle": True,
            "scope": "only the nine exact task/splits/scaffold-{0,1,2}.npy paths listed below",
            "precondition": (
                "official-source provenance asserted; every required dataset_root file "
                "is byte-identical to one unique safe regular archive member before loading; "
                "SHA-256 is recorded but is not the scientific-admission criterion"
            ),
        },
        "canonicalization": {
            "library": "RDKit",
            "version": rdBase.rdkitVersion,
            "shared_identity_normalization_contract": {
                "relative_repository_path": "most_t5_next/r1/contracts/pcqm4mv2_identity_normalization_contract.json",
                "bytes": identity_contract_fact.bytes,
                "sha256": identity_contract_fact.sha256,
            },
            "isomeric_identity": {
                "canonical": True,
                "isomericSmiles": True,
                "kekuleSmiles": False,
            },
            "protected_connectivity_identity": {
                "canonical": True,
                "isomericSmiles": False,
                "kekuleSmiles": False,
            },
            "scaffold": "RDKit Bemis-Murcko with both includeChirality=false and true",
        },
        "tasks": [
            {
                "task": spec.task,
                "smiles_column": spec.smiles_column,
                "label_columns": list(spec.label_columns),
            }
            for spec in TASK_SPECS
        ],
        "source_files": [
            {
                "relative_path": relative_path,
                "bytes": source_file_by_path[relative_path].bytes,
                "sha256": source_file_by_path[relative_path].sha256,
                "archive_member_path": archive_members[relative_path].archive_member_path,
                "archive_member_bytes": archive_members[relative_path].bytes,
                "archive_member_sha256": archive_members[relative_path].sha256,
                "archive_member_byte_identity_verified": True,
            }
            for relative_path in required_paths
        ],
    }

    output.mkdir(parents=True, exist_ok=False)
    source_artifact = _write_bytes_new(
        output / SOURCE_MANIFEST_FILENAME,
        _json_document(source_manifest),
    )
    source_artifact["relative_path"] = SOURCE_MANIFEST_FILENAME

    member_artifacts: list[dict[str, object]] = []
    identity_collection_artifacts: list[dict[str, object]] = []
    protected: dict[str, dict[str, Any]] = {}
    task_summaries: dict[str, object] = {}
    for spec in TASK_SPECS:
        payload = task_payloads[spec.task]
        molecules: list[MoleculeFact] = payload["molecules"]
        csv_fact: FileFact = payload["csv_fact"]
        split_summaries: dict[str, object] = {}
        for split in SPLIT_REPLICAS:
            split_payload = payload["splits"][split]
            split_fact: FileFact = split_payload["source_fact"]
            membership: dict[str, list[int]] = split_payload["membership"]
            rows: list[dict[str, object]] = []
            for partition in PARTITIONS:
                selected_molecules = [molecules[index] for index in membership[partition]]
                identity_collection_artifacts.append(
                    _write_identity_collection(
                        output=output,
                        spec=spec,
                        split_replica=split,
                        partition=partition,
                        molecules=selected_molecules,
                        source_manifest_sha256=source_artifact["sha256"],
                        identity_spec_sha256=identity_contract_fact.sha256,
                        extractor_sha256=extractor_fact.sha256,
                        excluded_columns=[
                            column
                            for column in payload["csv_columns"]
                            if column != spec.smiles_column
                        ],
                        archive_sha256=archive_fact.sha256,
                    )
                )
                for row_index in membership[partition]:
                    molecule = molecules[row_index]
                    rows.append(
                        _member_row(
                            spec=spec,
                            split=split,
                            partition=partition,
                            molecule=molecule,
                            csv_sha256=csv_fact.sha256,
                            split_sha256=split_fact.sha256,
                        )
                    )
                    if partition in ("validation", "test"):
                        union_item = protected.setdefault(
                            molecule.canonical_connectivity_sha256,
                            {
                                "canonical_connectivity_smiles": molecule.canonical_connectivity_smiles,
                                "canonical_connectivity_smiles_sha256": molecule.canonical_connectivity_sha256,
                                "protected_by": [],
                            },
                        )
                        union_item["protected_by"].append(
                            {
                                "task": spec.task,
                                "split_replica": split,
                                "partition": partition,
                                "source_csv_row_index": row_index,
                            }
                        )
            relative_path = f"members/{spec.task}/{split}.jsonl"
            artifact = _write_bytes_new(output / relative_path, _json_lines(rows))
            artifact["relative_path"] = relative_path
            artifact["row_count"] = len(rows)
            artifact["task"] = spec.task
            artifact["split_replica"] = split
            member_artifacts.append(artifact)
            split_summaries[split] = {
                **split_payload["checks"],
                "member_manifest": artifact,
            }
        task_summaries[spec.task] = {
            "csv_relative_path": csv_fact.relative_path,
            "csv_sha256": csv_fact.sha256,
            "csv_columns": payload["csv_columns"],
            "row_count": len(molecules),
            "split_replicas": split_summaries,
        }

    protected_rows = []
    for identity_hash in sorted(protected):
        item = protected[identity_hash]
        item["protected_by"] = sorted(
            item["protected_by"],
            key=lambda value: (
                value["task"],
                value["split_replica"],
                value["partition"],
                value["source_csv_row_index"],
            ),
        )
        protected_rows.append({"schema": PROTECTED_SCHEMA, **item})
    protected_artifact = _write_bytes_new(
        output / PROTECTED_UNION_FILENAME,
        _json_lines(protected_rows),
    )
    protected_artifact["relative_path"] = PROTECTED_UNION_FILENAME
    protected_artifact["identity_count"] = len(protected_rows)

    summary = {
        "schema": SUMMARY_SCHEMA,
        "source_manifest": source_artifact,
        "task_scope": [spec.task for spec in TASK_SPECS],
        "split_replicas": list(SPLIT_REPLICAS),
        "partition_order": list(PARTITIONS),
        "tasks": task_summaries,
        "member_manifests": member_artifacts,
        "identity_collection_manifests": identity_collection_artifacts,
        "protected_eval_union": protected_artifact,
        "scientific_boundary": {
            "protected_partitions": ["validation", "test"],
            "protected_across": "union of scaffold-0, scaffold-1, and scaffold-2",
            "identity": "canonical non-isomeric molecular connectivity",
            "training_members": "not included in the protected union",
            "clean_membership_integration": (
                "Pass the validation/test collection_manifest.json paths directly to "
                "derive_clean_pretrain_membership_v1.py; observed digests are provenance "
                "records, not caller-supplied scientific-admission arguments"
            ),
        },
    }
    _write_bytes_new(output / SUMMARY_FILENAME, _json_document(summary))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-archive-path", type=Path, required=True)
    parser.add_argument(
        "--official-archive-sha256",
        default=None,
        help=(
            "Optional integrity record. The observed archive digest is always computed; "
            "this value does not control scientific admission."
        ),
    )
    parser.add_argument(
        "--source-provenance",
        required=True,
        choices=(OFFICIAL_SOURCE_PROVENANCE,),
        help="Explicit trust assertion required before KPGT object arrays are unpickled.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_kpgt_scaffold_manifests(
        args.dataset_root,
        args.output_dir,
        official_archive_path=args.official_archive_path,
        official_archive_sha256=args.official_archive_sha256,
        source_provenance=args.source_provenance,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
