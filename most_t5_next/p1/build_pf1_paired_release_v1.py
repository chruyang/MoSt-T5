#!/usr/bin/env python3
"""Materialize the frozen PF-1 paired A/M training release.

The builder is deliberately two-phase.  Phase A scans the locked PCQM SDF
member once, keeps at most ``max_pending`` molecule/production-record tasks in
flight, and spools tokenizer-independent paired surfaces to one temporary
SQLite file.  Phase B freezes the shared union tokenizer and motif macro
registry, then the parent process alone writes ``paired_records.lmdb``.

No molecule or decoded production record is retained for the full 33,600-row
run.  Every frozen member must be represented exactly once; a chemistry reject
is recorded and makes the release ineligible for publication.  There is no
replacement, truncation, or silent row filtering.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import concurrent.futures
from dataclasses import dataclass
import datetime as dt
from importlib import metadata as importlib_metadata
import json
import multiprocessing
from pathlib import Path
import pickle
import sqlite3
import sys
import tarfile
import hashlib
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from most_t5_next.p1 import freeze_pf1_connectivity_sample_v1 as selection
from most_t5_next.p1.runtime_bridge import P1ArtifactBindings, P1MemberRef
from most_t5_next.r1.adapter import build_p1_inherited_e3fp_overlay_v1 as overlay
from most_t5_next.r1.adapter import build_p1_paired_canary_v1 as canary
from most_t5_next.r1.adapter import build_pcqm_p1_geometry_production_v1 as production
from most_t5_next.r1.adapter import mol_linearizer
from most_t5_next.r1.adapter import paired_record_wire_v1 as paired_wire
from most_t5_next.r1.adapter import p1_topology_augmentation_v1 as topology
from most_t5_next.r1.adapter import production_paired_identity_records_v1 as paired
from most_t5_next.r1.adapter import run_p1_topology_canary_v1 as release_reader
from most_t5_next.r1.adapter import sidecar_v2_codec
from most_t5_next.r1.gates import pcqm_e3fp_preflight as projection
from most_t5_next.r1.semantic import e3fp_duplicate_inheritance_v1 as inheritance
from most_t5_next.r1.tokenizer import build_p1_canary_union_tokenizer_v1 as union_builder
from most_t5_next.r1.tokenizer import production_atom_selfies_codec_v1 as atom_codec
from most_t5_next.r1.tokenizer import production_graph_ports_codec_v1 as graph_codec


SCHEMA_VERSION = "most-t5-p1/pf1-paired-release/v1"
OUTPUT_MEMBERSHIP_SCHEMA = "most-t5-p1/pf1-paired-release-member/v1"
REJECT_SCHEMA = "most-t5-p1/pf1-paired-release-reject/v1"
MACRO_REGISTRY_SCHEMA = "most-t5-p1/pf1-macro-registry/v1"
EFFECTIVE_GEOMETRY_SCHEMA = "most-t5-p1/pf1-inherited-e3fp-effective/v1"
TRAIN_MEMBERSHIP_NAME = "train_membership.jsonl"
DEV_MEMBERSHIP_NAME = "dev_membership.jsonl"
REJECTS_NAME = "rejects.jsonl"
MACRO_REGISTRY_NAME = "macro_registry.json"
MANIFEST_NAME = "manifest.json"
TOKENIZER_DIRECTORY = "union_tokenizer"
LMDB_DIRECTORY = "paired_records.lmdb"
STAGING_SUFFIX = ".staging"
DEFAULT_WORKERS = 16
DEFAULT_MAX_PENDING = 24
DEFAULT_MAP_SIZE_GIB = 4
DEFAULT_COMMIT_EVERY = 512
MAX_SEQUENCE_LENGTH = 512

_WORKER_STATE: dict[str, Any] = {}


class PF1PairedReleaseError(RuntimeError):
    """The frozen PF-1 paired release could not be materialized exactly."""


def _decode_paired_wire_cache_worker(
    item: tuple[str, bytes],
) -> tuple[str, Any]:
    """Decode one canonical wire row in a bounded CPU worker.

    The parent process remains the only LMDB reader and cache writer.  Workers
    receive one immutable value and return one fully validated record, so the
    ordering and trust boundary are identical to ordinary reader replay.
    """

    storage_key, raw = item
    return storage_key, paired_wire.decode_paired_training_record(raw)


@dataclass(frozen=True)
class FrozenMember:
    selection_index: int
    group_order_index: int
    member_id: str
    sdf_record_index: int
    connectivity_identity_sha256: str
    split: str


@dataclass(frozen=True)
class PreparedMember:
    frozen: FrozenMember
    storage_key: str
    source_atom_count: int
    model_to_source_atom_index: tuple[int, ...]
    inherited_e3fp: tuple[tuple[int, ...], ...]
    base_record_content_sha256: str
    effective_geometry_content_sha256: str
    prepared_surfaces: paired.PreparedPairedIdentitySurfaces
    atom_count: int
    motif_count: int
    edge_count: int
    inheritance_summary: dict[str, object]


@dataclass(frozen=True)
class _ShardPlan:
    shard_index: int
    range_start: int
    range_end: int
    shard_dir: Path
    shard_manifest_sha256: str


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_row(handle: Any, value: Mapping[str, object]) -> None:
    handle.write(_canonical_json_bytes(value).decode("utf-8") + "\n")


def _tree_file_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())


def _value_distribution(values: Iterable[int]) -> dict[str, int]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"count": 0, "min": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}

    def percentile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def split_selfies_coverage(
    *,
    train_observed: Iterable[str],
    dev_observed: Iterable[str],
    robust_alphabet: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Describe SELFIES syntax support across the frozen unlabeled cohort.

    AtomSELFIES symbols are serialization grammar, not frequency-selected
    motif identities.  Their registry is therefore frozen before optimization
    from the complete unlabeled PF-1 cohort, while train/dev provenance remains
    explicit.  Motif macro frequency and rank stay train-only below.
    """

    train = set(train_observed)
    dev = set(dev_observed)
    robust = set(robust_alphabet)
    for label, values in (("train", train), ("dev", dev), ("robust", robust)):
        if any(not isinstance(value, str) or not value for value in values):
            raise PF1PairedReleaseError(
                "{} SELFIES symbols must be non-empty text".format(label)
            )
    order = lambda value: value.encode("utf-8")
    return {
        "cohort_observed": tuple(sorted(train | dev, key=order)),
        "train_observed": tuple(sorted(train, key=order)),
        "dev_observed": tuple(sorted(dev, key=order)),
        "train_nonrobust": tuple(sorted(train - robust, key=order)),
        "dev_nonrobust": tuple(sorted(dev - robust, key=order)),
        "dev_only_nonrobust": tuple(sorted(dev - robust - train, key=order)),
    }


def load_frozen_membership(
    path: Path, *, expected_members: int | None = None
) -> tuple[FrozenMember, ...]:
    """Load the frozen group-complete membership without changing its order."""

    rows: list[FrozenMember] = []
    seen_ordinals: set[int] = set()
    seen_members: set[str] = set()
    group_split: dict[str, str] = {}
    required = {
        "schema_version",
        "selection_index",
        "group_order_index",
        "member_id",
        "sdf_record_index",
        "connectivity_identity_sha256",
        "split",
    }
    for line_number, raw in selection._read_jsonl(Path(path)):
        if raw.get("schema_version") != selection.MEMBERSHIP_SCHEMA or set(raw) != required:
            raise PF1PairedReleaseError(
                "frozen membership line {} has an unexpected schema".format(line_number)
            )
        index = raw.get("selection_index")
        ordinal = raw.get("sdf_record_index")
        member_id = raw.get("member_id")
        group_index = raw.get("group_order_index")
        group_id = raw.get("connectivity_identity_sha256")
        split = raw.get("split")
        if index != len(rows):
            raise PF1PairedReleaseError("frozen selection indices must be dense and ordered")
        if (
            not isinstance(group_index, int)
            or isinstance(group_index, bool)
            or group_index < 0
            or not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or ordinal < 0
            or selection.member_ordinal(member_id) != ordinal
        ):
            raise PF1PairedReleaseError("frozen member ordinal or group index is invalid")
        if not isinstance(group_id, str) or not group_id or split not in {"train", "dev"}:
            raise PF1PairedReleaseError("frozen connectivity group or split is invalid")
        if ordinal in seen_ordinals or member_id in seen_members:
            raise PF1PairedReleaseError("frozen membership repeats a molecule")
        previous_split = group_split.setdefault(group_id, str(split))
        if previous_split != split:
            raise PF1PairedReleaseError("one connectivity group crosses train/dev")
        seen_ordinals.add(ordinal)
        seen_members.add(member_id)
        rows.append(
            FrozenMember(
                selection_index=int(index),
                group_order_index=int(group_index),
                member_id=str(member_id),
                sdf_record_index=int(ordinal),
                connectivity_identity_sha256=group_id,
                split=str(split),
            )
        )
    if not rows or {row.split for row in rows} != {"train", "dev"}:
        raise PF1PairedReleaseError("frozen membership must contain train and dev rows")
    if expected_members is not None and len(rows) != expected_members:
        raise PF1PairedReleaseError(
            "expected {:,} frozen members, observed {:,}".format(
                expected_members, len(rows)
            )
        )
    return tuple(rows)


class _PreparedSpool:
    """One-file, parent-written phase boundary keyed by selection index."""

    def __init__(self, path: Path, *, create: bool) -> None:
        self.path = Path(path)
        self.connection = sqlite3.connect(str(self.path))
        if create:
            self.connection.execute("PRAGMA journal_mode=OFF")
            self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute(
                "CREATE TABLE prepared (selection_index INTEGER PRIMARY KEY, payload BLOB NOT NULL)"
            )

    def put(self, member: PreparedMember) -> None:
        self.connection.execute(
            "INSERT INTO prepared(selection_index,payload) VALUES (?,?)",
            (
                member.frozen.selection_index,
                sqlite3.Binary(pickle.dumps(member, protocol=pickle.HIGHEST_PROTOCOL)),
            ),
        )

    def commit(self) -> None:
        self.connection.commit()

    def get(self, selection_index: int) -> PreparedMember:
        cursor = self.connection.execute(
            "SELECT payload FROM prepared WHERE selection_index=?", (selection_index,)
        )
        row = cursor.fetchone()
        if row is None:
            raise PF1PairedReleaseError(
                "prepared spool lacks selection index {}".format(selection_index)
            )
        value = pickle.loads(bytes(row[0]))
        if not isinstance(value, PreparedMember):
            raise PF1PairedReleaseError("prepared spool contains an unknown value")
        return value

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM prepared").fetchone()[0])

    def close(self) -> None:
        self.connection.close()


class _ProductionBindingStream:
    """Read only selected production rows, keeping at most one shard LMDB open."""

    def __init__(
        self,
        *,
        release_root: Path,
        members: Sequence[FrozenMember],
        np: Any,
        lmdb_module: Any,
    ) -> None:
        self.release_root = Path(release_root)
        self.np = np
        self.lmdb_module = lmdb_module
        full_manifest_path = self.release_root / "full_release_manifest.json"
        candidate = release_reader.load_json(
            full_manifest_path, "PF-1 production full manifest"
        )
        release_selection = {
            "release": {
                "release_id": candidate.get("release_id"),
                "full_release_manifest_sha256": release_reader.sha256_file(
                    full_manifest_path
                ),
                "logical_release_root_sha256": candidate.get(
                    "logical_release_root_sha256"
                ),
            }
        }
        self.manifest_path, self.manifest = release_reader.load_release_manifest(
            self.release_root, release_selection
        )
        by_shard: dict[int, list[int]] = defaultdict(list)
        top_entries: dict[int, dict[str, object]] = {}
        for member in members:
            shard = release_reader._shard_for_ordinal(
                self.manifest, member.sdf_record_index
            )
            shard_index = int(shard["shard_index"])
            by_shard[shard_index].append(member.sdf_record_index)
            top_entries[shard_index] = shard

        self.memberships: dict[int, dict[str, object]] = {}
        self.plan_by_ordinal: dict[int, _ShardPlan] = {}
        self.shard_receipts: list[dict[str, object]] = []
        for shard_index in sorted(by_shard):
            top = top_entries[shard_index]
            shard_dir = self.release_root / "shard-{:06d}".format(shard_index)
            shard_manifest_path = shard_dir / "shard_manifest.json"
            shard_manifest = release_reader.load_json(
                shard_manifest_path, "PF-1 production shard manifest"
            )
            if release_reader.sha256_file(shard_manifest_path) != top.get(
                "shard_manifest_sha256"
            ):
                raise PF1PairedReleaseError("production shard manifest binding differs")
            start = int(shard_manifest["range_start"])
            end = int(shard_manifest["range_end"])
            ordinals = sorted(by_shard[shard_index])
            selected = release_reader._read_selected_membership(
                shard_dir / "membership.jsonl", start, ordinals
            )
            plan = _ShardPlan(
                shard_index=shard_index,
                range_start=start,
                range_end=end,
                shard_dir=shard_dir,
                shard_manifest_sha256=str(top["shard_manifest_sha256"]),
            )
            for ordinal in ordinals:
                row = selected[ordinal]
                if row.get("disposition") != "admit":
                    raise PF1PairedReleaseError(
                        "frozen PF-1 member is rejected by production release"
                    )
                self.memberships[ordinal] = row
                self.plan_by_ordinal[ordinal] = plan
            self.shard_receipts.append(
                {
                    "shard_index": shard_index,
                    "range_start": start,
                    "range_end": end,
                    "selected_record_count": len(ordinals),
                    "shard_manifest_sha256": plan.shard_manifest_sha256,
                }
            )
        self._current_shard: int | None = None
        self._environment: Any = None

    def binding_for(self, ordinal: int) -> dict[str, object]:
        plan = self.plan_by_ordinal.get(ordinal)
        membership = self.memberships.get(ordinal)
        if plan is None or membership is None:
            raise PF1PairedReleaseError("ordinal is absent from the production stream plan")
        if self._current_shard != plan.shard_index:
            self.close_environment()
            self._environment = self.lmdb_module.open(
                str(plan.shard_dir / "geometry_records.lmdb"),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=8,
            )
            self._current_shard = plan.shard_index
        with self._environment.begin(write=False) as transaction:
            raw = transaction.get(str(membership["record_storage_key"]).encode("ascii"))
        if raw is None:
            raise PF1PairedReleaseError("selected production LMDB payload is absent")
        record, logical_hash = sidecar_v2_codec.decode_record(self.np, bytes(raw))
        if logical_hash != membership.get("record_content_sha256"):
            raise PF1PairedReleaseError("selected production logical content differs")
        overlay.validate_overlay_release_record(record, membership, ordinal)
        return {
            "record": record,
            "membership": membership,
            "shard_index": plan.shard_index,
        }

    def close_environment(self) -> None:
        if self._environment is not None:
            self._environment.close()
            self._environment = None
            self._current_shard = None

    def close(self) -> None:
        self.close_environment()


def _iter_selected_sdf(
    Chem: Any,
    *,
    archive_path: Path,
    locked_member: Mapping[str, object],
    selected_ordinals: Iterable[int],
    expected_record_count: int,
    observation: dict[str, object],
    progress_every: int,
) -> Iterator[tuple[int, bytes | None, str | None]]:
    """Yield selected molecule binaries in SDF order without retaining them."""

    targets = set(int(value) for value in selected_ordinals)
    if not targets:
        raise PF1PairedReleaseError("selected SDF ordinal set is empty")
    maximum = max(targets)
    seen: set[int] = set()
    digest = hashlib.sha256()
    byte_count = 0
    ordinal = 0
    record_has_content = False
    buffer: bytearray | None = bytearray() if 0 in targets else None
    member: Any = None
    with tarfile.open(str(archive_path), mode="r|gz") as archive:
        member = next(
            (
                candidate
                for candidate in archive
                if candidate.name == locked_member.get("tar_member_name")
            ),
            None,
        )
        if member is None or not member.isfile():
            raise PF1PairedReleaseError("locked SDF tar member is absent")
        if int(member.size) != locked_member.get("uncompressed_bytes"):
            raise PF1PairedReleaseError("locked SDF member size differs")
        stream = archive.extractfile(member)
        if stream is None:
            raise PF1PairedReleaseError("locked SDF member cannot be opened")
        try:
            for line in stream:
                digest.update(line)
                byte_count += len(line)
                if line.rstrip(b"\r\n") == b"$$$$":
                    if ordinal in targets:
                        try:
                            molecule = release_reader._parse_selected_mol(
                                Chem, bytes(buffer or b"")
                            )
                            yield ordinal, bytes(molecule.ToBinary()), None
                        except Exception as exc:
                            yield ordinal, None, "{}: {}".format(type(exc).__name__, exc)
                        seen.add(ordinal)
                    ordinal += 1
                    if progress_every > 0 and ordinal % progress_every == 0:
                        print(
                            "[pf1-materialize] scanned {:,}/{:,} SDF records".format(
                                ordinal, expected_record_count
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                    record_has_content = False
                    if ordinal > maximum:
                        break
                    buffer = bytearray() if ordinal in targets else None
                    continue
                record_has_content = True
                if buffer is not None:
                    buffer.extend(line)
            if record_has_content and ordinal in targets:
                try:
                    molecule = release_reader._parse_selected_mol(Chem, bytes(buffer or b""))
                    yield ordinal, bytes(molecule.ToBinary()), None
                except Exception as exc:
                    yield ordinal, None, "{}: {}".format(type(exc).__name__, exc)
                seen.add(ordinal)
                ordinal += 1
        finally:
            stream.close()
    if seen != targets:
        raise PF1PairedReleaseError("SDF stream did not resolve every frozen ordinal")
    complete = byte_count == int(member.size) and ordinal == expected_record_count
    if complete and digest.hexdigest() != locked_member.get("sha256"):
        raise PF1PairedReleaseError("complete SDF member digest differs from source lock")
    observation.update(
        {
            "sdf_records_scanned": ordinal,
            "selected_records_yielded": len(seen),
            "maximum_selected_ordinal": maximum,
            "uncompressed_bytes_scanned": byte_count,
            "complete_member_rehash": complete,
        }
    )


def _init_prepare_worker(e3fp_source: str, linearizer_sha256: str) -> None:
    import numpy as np
    import selfies as sf
    from rdkit import Chem

    import_root, package_root, _files = projection.resolve_e3fp_source(
        Path(e3fp_source)
    )
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "Chem": Chem,
            "np": np,
            "sf": sf,
            "e3fp_api": projection.import_locked_e3fp(import_root, package_root),
            "linearizer_sha256": linearizer_sha256,
        }
    )


def _reject_row(
    frozen: FrozenMember,
    *,
    stage: str,
    reason: str,
) -> dict[str, object]:
    return {
        "schema_version": REJECT_SCHEMA,
        "selection_index": frozen.selection_index,
        "member_id": frozen.member_id,
        "sdf_record_index": frozen.sdf_record_index,
        "split": frozen.split,
        "stage": stage,
        "reason": reason,
    }


def _prepare_one(
    task: tuple[FrozenMember, bytes | None, str | None, Mapping[str, object]]
) -> dict[str, object]:
    frozen, mol_binary, parse_error, binding = task
    if mol_binary is None:
        return {
            "status": "reject",
            "reject": _reject_row(
                frozen, stage="SDF_PARSE", reason=parse_error or "selected Mol is absent"
            ),
        }
    Chem = _WORKER_STATE["Chem"]
    np = _WORKER_STATE["np"]
    sf = _WORKER_STATE["sf"]
    stage = "SOURCE_MOL"
    try:
        source_mol = Chem.Mol(mol_binary)
        if source_mol is None:
            raise PF1PairedReleaseError("worker could not restore source Mol")
        ordinal = frozen.sdf_record_index
        base_record, base_membership = overlay.validate_base_binding(
            np, binding, ordinal
        )

        stage = "PROJECTION_PARITY"
        tagged, source_atom_count, _ = projection.tag_source_atoms(Chem, source_mol)
        projected_mol, model_to_source = projection.project_hydrogens(
            Chem, tagged, source_atom_count
        )
        atom_universe = base_record["atom_universe"]
        geometry = base_record["geometry"]
        mapping = np.ascontiguousarray(np.asarray(model_to_source, dtype=np.int32))
        coordinates = np.ascontiguousarray(
            np.asarray(projected_mol.GetConformer(0).GetPositions(), dtype=np.float32)
        )
        if not (
            source_atom_count == atom_universe["source_atom_count"]
            and int(projected_mol.GetNumAtoms()) == atom_universe["model_atom_count"]
            and bool(np.array_equal(mapping, atom_universe["model_to_source_atom_index"]))
            and bool(np.array_equal(coordinates, geometry["coordinates"]))
        ):
            raise PF1PairedReleaseError("projection differs from production geometry")

        stage = "INHERITED_E3FP"
        raw, inherited_ids, duplicate_mask, summary, resolved = (
            inheritance.generate_e3fp_projection_pair(
                np, _WORKER_STATE["e3fp_api"], projected_mol, ordinal
            )
        )
        if not bool(np.array_equal(raw, geometry["e3fp"])):
            raise PF1PairedReleaseError("raw E3FP differs from production geometry")
        inherited_ids = np.ascontiguousarray(inherited_ids, dtype=np.int32)
        effective_document = {
            "schema_version": EFFECTIVE_GEOMETRY_SCHEMA,
            "semantics_id": inheritance.SEMANTICS_ID,
            "member_id": frozen.member_id,
            "sdf_record_index": ordinal,
            "base_record_content_sha256": base_membership["record_content_sha256"],
            "base_e3fp_sha256": geometry["e3fp_sha256"],
            "resolved_e3fp_config_sha256": overlay._resolved_config_sha256(resolved),
            "inherited_e3fp": inherited_ids.tolist(),
            "duplicate_mask_sha256": overlay._array_sha256(
                np.ascontiguousarray(duplicate_mask, dtype=np.bool_)
            ),
        }
        effective_sha = _sha256_json(effective_document)

        stage = "PAIRED_SURFACE_DISCOVERY"
        linearization = mol_linearizer.linearize_mol(projected_mol)
        augmentation = topology.build_topology_augmentation(
            linearization_result=linearization,
            member_id=frozen.member_id,
            base_record_content_sha256=base_membership["record_content_sha256"],
            linearizer_spec_sha256=_WORKER_STATE["linearizer_sha256"],
            expected_motif_atom_indices=base_record["topology"]["motif_atom_indices"],
            expected_motif_lexeme_sha256=base_record["topology"]["motif_lexeme_sha256"],
            source_atom_count=source_atom_count,
            model_to_source_atom_index=model_to_source,
        )
        groups = tuple(
            tuple(row)
            for row in augmentation["logical_motif_domain"]["motif_atom_indices"]
        )
        cross_edges = canary.cross_edges_from_augmentation(augmentation)
        surfaces = paired.discover_production_paired_identity_surfaces(
            Chem, sf, projected_mol, groups, cross_edges
        )
        prepared = PreparedMember(
            frozen=frozen,
            storage_key=str(base_membership["record_storage_key"]),
            source_atom_count=int(source_atom_count),
            model_to_source_atom_index=tuple(int(value) for value in model_to_source),
            inherited_e3fp=tuple(
                tuple(int(value) for value in row) for row in inherited_ids
            ),
            base_record_content_sha256=str(base_membership["record_content_sha256"]),
            effective_geometry_content_sha256=effective_sha,
            prepared_surfaces=surfaces,
            atom_count=int(projected_mol.GetNumAtoms()),
            motif_count=len(groups),
            edge_count=len(cross_edges),
            inheritance_summary=dict(summary),
        )
        return {"status": "pass", "prepared": prepared}
    except Exception as exc:
        return {
            "status": "reject",
            "reject": _reject_row(
                frozen,
                stage=stage,
                reason="{}: {}".format(type(exc).__name__, exc),
            ),
        }


def _binding_base(
    *,
    production_manifest: Mapping[str, object],
    production_manifest_path: Path,
    frozen_membership: Path,
) -> dict[str, str]:
    graph_source = Path(graph_codec.__file__).resolve()
    atom_source = Path(atom_codec.__file__).resolve()
    paired_source = Path(paired.__file__).resolve()
    return {
        "release_id": "{}:pf1-inherited-e3fp-v1".format(
            production_manifest["release_id"]
        ),
        "data_release_manifest_sha256": release_reader.sha256_file(
            production_manifest_path
        ),
        "geometry_record_schema_sha256": _sha256_json(
            {
                "schema_version": EFFECTIVE_GEOMETRY_SCHEMA,
                "semantics_id": inheritance.SEMANTICS_ID,
            }
        ),
        "membership_manifest_sha256": release_reader.sha256_file(frozen_membership),
        "identity_codec_sha256": _sha256_json(
            {
                "atom_selfies": release_reader.sha256_file(atom_source),
                "graph_ports": release_reader.sha256_file(graph_source),
                "paired_producer": release_reader.sha256_file(paired_source),
            }
        ),
        "connection_codec_sha256": release_reader.sha256_file(graph_source),
    }


def _output_membership_row(
    prepared: PreparedMember,
    loaded: paired_wire.LoadedPairedTrainingRecord,
    payload: bytes,
    *,
    split_index: int,
) -> dict[str, object]:
    frozen = prepared.frozen
    return {
        "schema_version": OUTPUT_MEMBERSHIP_SCHEMA,
        "split": frozen.split,
        "split_index": split_index,
        "selection_index": frozen.selection_index,
        "group_order_index": frozen.group_order_index,
        "connectivity_identity_sha256": frozen.connectivity_identity_sha256,
        "member_id": frozen.member_id,
        "sdf_record_index": frozen.sdf_record_index,
        "storage_key": prepared.storage_key,
        "wire_bytes": len(payload),
        "atom_input_token_count": len(loaded.atom_record.input_ids),
        "motif_input_token_count": len(loaded.motif_record.input_ids),
        "atom_count": prepared.atom_count,
        "motif_count": prepared.motif_count,
        "edge_count": prepared.edge_count,
        "macro_identity_occurrences": sum(
            mode == "macro" for mode in loaded.surface_summary.motif_identity_modes
        ),
        "fallback_identity_occurrences": sum(
            mode == "fallback" for mode in loaded.surface_summary.motif_identity_modes
        ),
        "effective_geometry_content_sha256": (
            prepared.effective_geometry_content_sha256
        ),
    }


def _iter_output_membership(path: Path) -> Iterator[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise PF1PairedReleaseError(
                    "output membership line {} is invalid JSON".format(line_number)
                ) from exc
            if not isinstance(row, dict) or row.get("schema_version") != OUTPUT_MEMBERSHIP_SCHEMA:
                raise PF1PairedReleaseError("output membership schema differs")
            yield row


def _replay_release(
    *,
    lmdb_path: Path,
    membership_paths: Sequence[Path],
    expected_entries: int,
    sentinel_id_count: int,
    lmdb_module: Any,
    decoder: Callable[[bytes], paired_wire.LoadedPairedTrainingRecord] = (
        paired_wire.decode_paired_training_record
    ),
) -> dict[str, object]:
    environment = lmdb_module.open(
        str(lmdb_path),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=8,
    )
    atom_lengths: list[int] = []
    motif_lengths: list[int] = []
    wire_bytes: list[int] = []
    capacity: dict[str, dict[str, list[int]]] = {
        "atom": {"units": [], "identity_tokens": [], "targets": [], "sentinels": []},
        "motif": {"units": [], "identity_tokens": [], "targets": [], "sentinels": []},
    }
    observed = 0
    try:
        with environment.begin(write=False) as transaction:
            if int(transaction.stat().get("entries", -1)) != expected_entries:
                raise PF1PairedReleaseError("paired LMDB entry count differs")
            for membership_path in membership_paths:
                for row in _iter_output_membership(membership_path):
                    raw = transaction.get(str(row["storage_key"]).encode("ascii"))
                    if raw is None:
                        raise PF1PairedReleaseError("paired LMDB is missing a membership row")
                    payload = bytes(raw)
                    loaded = decoder(payload)
                    if not (
                        loaded.schedule_index == row["selection_index"]
                        and loaded.sdf_record_index == row["sdf_record_index"]
                        and loaded.atom_record.record_id == row["member_id"]
                    ):
                        raise PF1PairedReleaseError("paired LMDB replay differs from membership")
                    atom_lengths.append(len(loaded.atom_record.input_ids))
                    motif_lengths.append(len(loaded.motif_record.input_ids))
                    wire_bytes.append(len(payload))
                    for family, spans in (
                        ("atom", loaded.atom_record.atom_identity_spans),
                        ("motif", loaded.motif_record.identity_spans),
                    ):
                        units = len(spans)
                        identity_tokens = sum(span.stop - span.start for span in spans)
                        target = identity_tokens + units + 2
                        sentinels = units + 1
                        if (
                            units <= 0
                            or identity_tokens <= 0
                            or sentinels > sentinel_id_count
                            or target > MAX_SEQUENCE_LENGTH
                        ):
                            raise PF1PairedReleaseError(
                                "all-mask target or sentinel capacity is insufficient"
                            )
                        values = capacity[family]
                        values["units"].append(units)
                        values["identity_tokens"].append(identity_tokens)
                        values["targets"].append(target)
                        values["sentinels"].append(sentinels)
                    observed += 1
    finally:
        environment.close()
    if observed != expected_entries:
        raise PF1PairedReleaseError("membership replay count differs")
    return {
        "records_replayed": observed,
        "atom_input_tokens": _value_distribution(atom_lengths),
        "motif_input_tokens": _value_distribution(motif_lengths),
        "wire_bytes": _value_distribution(wire_bytes),
        "sentinel_id_count": sentinel_id_count,
        "all_masks_all_epochs_proven": True,
        "atom": {key: _value_distribution(values) for key, values in capacity["atom"].items()},
        "motif": {key: _value_distribution(values) for key, values in capacity["motif"].items()},
    }


def _write_failure(
    staging_root: Path,
    rejects: Sequence[Mapping[str, object]],
    *,
    expected_members: int,
) -> None:
    rejects_path = staging_root / REJECTS_NAME
    if not rejects_path.exists():
        with rejects_path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rejects:
                _write_jsonl_row(handle, row)
    manifest_path = staging_root / MANIFEST_NAME
    if not manifest_path.exists():
        _write_json(
            manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "scheduled_members": expected_members,
                "rejected_members": len(rejects),
                "rejects_by_stage": dict(
                    sorted(Counter(str(row["stage"]) for row in rejects).items())
                ),
                "no_replacement": True,
                "sequence_truncation": False,
            },
        )


def run(args: argparse.Namespace) -> dict[str, object]:
    frozen_path = Path(args.frozen_membership).expanduser().resolve()
    release_root = Path(args.release_root).expanduser().resolve()
    source_archive = Path(args.source_archive).expanduser().resolve()
    e3fp_source = Path(args.e3fp_source).expanduser().resolve()
    base_tokenizer = Path(args.base_tokenizer).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    staging_root = output_dir.with_name(output_dir.name + STAGING_SUFFIX)
    if not frozen_path.is_file():
        raise PF1PairedReleaseError("frozen membership is absent")
    if not release_root.is_dir() or not source_archive.is_file():
        raise PF1PairedReleaseError("production release or source archive is absent")
    if not e3fp_source.exists() or not base_tokenizer.is_dir():
        raise PF1PairedReleaseError("E3FP source or base tokenizer is absent")
    if output_dir.exists() or staging_root.exists():
        raise PF1PairedReleaseError("output and sibling staging paths must be new")
    if args.workers < 1 or args.max_pending < args.workers:
        raise PF1PairedReleaseError("workers must be positive and max-pending >= workers")
    if args.commit_every < 1 or args.lmdb_map_size_gib < 1:
        raise PF1PairedReleaseError("LMDB commit/map-size settings must be positive")

    try:
        import lmdb
        import numpy as np
        import selfies as sf
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise PF1PairedReleaseError(
            "NumPy, RDKit, SELFIES and python-lmdb are required"
        ) from exc

    members = load_frozen_membership(
        frozen_path, expected_members=args.expected_members
    )
    by_ordinal = {row.sdf_record_index: row for row in members}
    staging_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir()
    spool_path = staging_root / "prepared_spool.sqlite3"
    spool = _PreparedSpool(spool_path, create=True)
    rejects: list[dict[str, object]] = []
    train_selfies_symbols: set[str] = set()
    dev_selfies_symbols: set[str] = set()
    train_identity_counts: Counter[str] = Counter()
    structure: dict[str, list[int]] = {
        "atoms": [],
        "motifs": [],
        "cross_motif_edges": [],
        "e3fp_levels": [],
    }
    sdf_observation: dict[str, object] = {}
    binding_stream = _ProductionBindingStream(
        release_root=release_root,
        members=members,
        np=np,
        lmdb_module=lmdb,
    )
    configuration = binding_stream.manifest.get("configuration", {})
    source_record_count = configuration.get("source_record_count")
    locked_member = configuration.get("locked_sdf_member")
    archive_lock = configuration.get("staged_inputs", {}).get("train_3d_sdf_archive")
    if (
        not isinstance(source_record_count, int)
        or not isinstance(locked_member, dict)
        or not isinstance(archive_lock, dict)
        or source_archive.stat().st_size != archive_lock.get("bytes")
    ):
        binding_stream.close()
        spool.close()
        raise PF1PairedReleaseError("production SDF binding is incomplete")

    linearizer_sha = release_reader.sha256_file(Path(mol_linearizer.__file__).resolve())

    def tasks() -> Iterator[
        tuple[FrozenMember, bytes | None, str | None, Mapping[str, object]]
    ]:
        for ordinal, mol_binary, parse_error in _iter_selected_sdf(
            Chem,
            archive_path=source_archive,
            locked_member=locked_member,
            selected_ordinals=by_ordinal,
            expected_record_count=source_record_count,
            observation=sdf_observation,
            progress_every=args.progress_every,
        ):
            yield (
                by_ordinal[ordinal],
                mol_binary,
                parse_error,
                binding_stream.binding_for(ordinal),
            )

    try:
        results = production.ordered_bounded_map(
            _prepare_one,
            tasks(),
            args.workers,
            args.max_pending,
            initializer=_init_prepare_worker,
            initargs=(str(e3fp_source), linearizer_sha),
        )
        for result in results:
            if result.get("status") != "pass":
                reject = result.get("reject")
                if not isinstance(reject, dict):
                    raise PF1PairedReleaseError("worker reject row is absent")
                rejects.append(reject)
                continue
            prepared = result.get("prepared")
            if not isinstance(prepared, PreparedMember):
                raise PF1PairedReleaseError("worker prepared row has an unknown type")
            spool.put(prepared)
            if prepared.frozen.split == "train":
                train_selfies_symbols.update(
                    prepared.prepared_surfaces.atom_surface.selfies_symbols
                )
                train_identity_counts.update(
                    motif.identity_smiles
                    for motif in prepared.prepared_surfaces.graph_encoding.motifs
                )
            else:
                dev_selfies_symbols.update(
                    prepared.prepared_surfaces.atom_surface.selfies_symbols
                )
            structure["atoms"].append(prepared.atom_count)
            structure["motifs"].append(prepared.motif_count)
            structure["cross_motif_edges"].append(prepared.edge_count)
            structure["e3fp_levels"].append(len(prepared.inherited_e3fp[0]))
        spool.commit()
    finally:
        binding_stream.close()

    if rejects or spool.count() != len(members):
        spool.close()
        _write_failure(staging_root, rejects, expected_members=len(members))
        raise PF1PairedReleaseError("surface discovery rejected frozen PF-1 members")
    phase_a_spool_bytes = int(spool_path.stat().st_size)

    robust_selfies_symbols = set(sf.get_semantic_robust_alphabet())
    selfies_coverage = split_selfies_coverage(
        train_observed=train_selfies_symbols,
        dev_observed=dev_selfies_symbols,
        robust_alphabet=robust_selfies_symbols,
    )
    # PF-1 macro frequency and rank are fitted on train only.  A dev identity
    # absent from this mapping is represented by the existing lossless byte
    # fallback and therefore cannot influence vocabulary selection.
    macro_registry, macro_summary = canary.build_macro_registry(
        train_identity_counts
    )
    tokenizer_build = union_builder.build_canary_union_tokenizer(
        base_snapshot=base_tokenizer,
        output_dir=staging_root / TOKENIZER_DIRECTORY,
        selfies_distribution_version=atom_codec.SELFIES_DISTRIBUTION_VERSION,
        robust_selfies_symbols=robust_selfies_symbols,
        observed_selfies_symbols=selfies_coverage["cohort_observed"],
        motif_macro_registry=macro_registry,
    )
    macro_by_identity = {
        str(row["identity"]): str(row["token"]) for row in macro_registry
    }
    normalized_macro_registry = tokenizer_build.manifest["contract"][
        "motif_macro_registry"
    ]
    _write_json(
        staging_root / MACRO_REGISTRY_NAME,
        {
            "schema_version": MACRO_REGISTRY_SCHEMA,
            "policy": {
                "minimum_occurrences": canary.MACRO_MIN_OCCURRENCES,
                "frequency_and_rank_source_split": "train",
                "fallback_is_lossless_gports_byte_surface": True,
                "scope": "pf1_sample_bound_provisional",
                "final_pretraining_k": False,
            },
            "summary": macro_summary,
            "rows": normalized_macro_registry,
        },
    )

    binding_base = _binding_base(
        production_manifest=binding_stream.manifest,
        production_manifest_path=binding_stream.manifest_path,
        frozen_membership=frozen_path,
    )
    lmdb_path = staging_root / LMDB_DIRECTORY
    environment = lmdb.open(
        str(lmdb_path),
        subdir=True,
        map_size=int(args.lmdb_map_size_gib) * 1024**3,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        max_dbs=1,
    )
    train_path = staging_root / TRAIN_MEMBERSHIP_NAME
    dev_path = staging_root / DEV_MEMBERSHIP_NAME
    split_counts = {"train": 0, "dev": 0}
    materialized_modes: Counter[str] = Counter()
    materialized_modes_by_split: dict[str, Counter[str]] = {
        "train": Counter(),
        "dev": Counter(),
    }
    pair_rejects: list[dict[str, object]] = []
    try:
        with train_path.open("x", encoding="utf-8", newline="\n") as train_handle, dev_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as dev_handle:
            for start in range(0, len(members), args.commit_every):
                chunk = members[start : start + args.commit_every]
                with environment.begin(write=True) as transaction:
                    for frozen in chunk:
                        prepared = spool.get(frozen.selection_index)
                        if prepared.frozen != frozen:
                            raise PF1PairedReleaseError(
                                "prepared spool order/binding differs from frozen membership"
                            )
                        try:
                            bindings = P1ArtifactBindings(
                                **binding_base,
                                geometry_record_content_sha256=(
                                    prepared.effective_geometry_content_sha256
                                ),
                                tokenizer_contract_sha256=(
                                    tokenizer_build.runtime.tokenizer_contract_sha256
                                ),
                                tokenizer_snapshot_sha256=(
                                    tokenizer_build.runtime.tokenizer_snapshot_sha256
                                ),
                            )
                            pair = paired.build_production_paired_identity_records_from_prepared(
                                prepared=prepared.prepared_surfaces,
                                member=P1MemberRef(frozen.member_id, prepared.storage_key),
                                bindings=bindings,
                                base_geometry_record_content_sha256=(
                                    prepared.base_record_content_sha256
                                ),
                                effective_inherited_overlay_content_sha256=(
                                    prepared.effective_geometry_content_sha256
                                ),
                                source_atom_count=prepared.source_atom_count,
                                model_to_source_atom_index=(
                                    prepared.model_to_source_atom_index
                                ),
                                inherited_e3fp=prepared.inherited_e3fp,
                                union_tokenizer=tokenizer_build.tokenizer,
                                tokenizer_binding=tokenizer_build.runtime,
                                macro_by_identity=macro_by_identity,
                            )
                            payload = paired_wire.encode_paired_training_record(
                                pair,
                                schedule_index=frozen.selection_index,
                                sdf_record_index=frozen.sdf_record_index,
                            )
                            loaded = paired_wire.decode_paired_training_record(payload)
                            if (
                                len(loaded.atom_record.input_ids) > MAX_SEQUENCE_LENGTH
                                or len(loaded.motif_record.input_ids) > MAX_SEQUENCE_LENGTH
                            ):
                                raise PF1PairedReleaseError(
                                    "A or M input exceeds 512; truncation is forbidden"
                                )
                            if not transaction.put(
                                prepared.storage_key.encode("ascii"), payload, overwrite=False
                            ):
                                raise PF1PairedReleaseError("paired LMDB key collision")
                        except Exception as exc:
                            pair_rejects.append(
                                _reject_row(
                                    frozen,
                                    stage="PAIRED_BUILD",
                                    reason="{}: {}".format(type(exc).__name__, exc),
                                )
                            )
                            continue
                        split_index = split_counts[frozen.split]
                        row = _output_membership_row(
                            prepared, loaded, payload, split_index=split_index
                        )
                        _write_jsonl_row(
                            train_handle if frozen.split == "train" else dev_handle,
                            row,
                        )
                        split_counts[frozen.split] += 1
                        materialized_modes.update(
                            loaded.surface_summary.motif_identity_modes
                        )
                        materialized_modes_by_split[frozen.split].update(
                            loaded.surface_summary.motif_identity_modes
                        )
        environment.sync(True)
    finally:
        environment.close()
        spool.close()

    if pair_rejects or sum(split_counts.values()) != len(members):
        _write_failure(
            staging_root, pair_rejects, expected_members=len(members)
        )
        raise PF1PairedReleaseError("paired materialization rejected frozen PF-1 members")

    with (staging_root / REJECTS_NAME).open(
        "x", encoding="utf-8", newline="\n"
    ):
        pass
    replay = _replay_release(
        lmdb_path=lmdb_path,
        membership_paths=(train_path, dev_path),
        expected_entries=len(members),
        sentinel_id_count=len(tokenizer_build.runtime.sentinel_token_ids),
        lmdb_module=lmdb,
    )
    lmdb_data_bytes = int((lmdb_path / "data.mdb").stat().st_size)
    published_data_bytes_before_manifest = sum(
        (
            lmdb_data_bytes,
            int(train_path.stat().st_size),
            int(dev_path.stat().st_size),
            int((staging_root / REJECTS_NAME).stat().st_size),
            int((staging_root / MACRO_REGISTRY_NAME).stat().st_size),
            _tree_file_bytes(staging_root / TOKENIZER_DIRECTORY),
        )
    )
    peak_materialization_artifact_bytes = (
        phase_a_spool_bytes + published_data_bytes_before_manifest
    )
    # The spool is one explicit temporary file; the published release contains
    # only model-consumed records and scientific lineage artifacts.
    spool_path.unlink()

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "created_utc": _utc_now(),
        "scope": "pf1_one_percent_failure_screen",
        "counts": {
            "scheduled_members": len(members),
            "train_members": split_counts["train"],
            "dev_members": split_counts["dev"],
            "paired_records": sum(split_counts.values()),
            "rejects": 0,
            "observed_selfies_symbols": len(
                set(train_selfies_symbols).union(dev_selfies_symbols)
            ),
            "train_observed_selfies_symbols": len(train_selfies_symbols),
            "dev_observed_selfies_symbols": len(dev_selfies_symbols),
            "robust_selfies_symbols": len(robust_selfies_symbols),
            "dev_only_nonrobust_symbols": len(
                selfies_coverage["dev_only_nonrobust"]
            ),
            "materialized_macro_occurrences": materialized_modes["macro"],
            "materialized_fallback_occurrences": materialized_modes["fallback"],
        },
        "selection": {
            "source_membership": str(frozen_path),
            "membership_schema": selection.MEMBERSHIP_SCHEMA,
            "selection_order_preserved_within_each_split": True,
            "train_dev_connectivity_group_disjoint": True,
            "no_replacement": True,
        },
        "structure": {
            key: _value_distribution(values) for key, values in structure.items()
        },
        "selfies_vocabulary_coverage": {
            "registry_role": "lossless_atom_selfies_syntax",
            "registry_scope": "complete_frozen_unlabeled_pf1_cohort",
            "labels_or_performance_signals_used": False,
            "frequency_or_rank_used": False,
            "robust_alphabet_always_included": True,
            "cohort_observed": list(selfies_coverage["cohort_observed"]),
            "train_observed": list(selfies_coverage["train_observed"]),
            "dev_observed": list(selfies_coverage["dev_observed"]),
            "train_nonrobust": list(selfies_coverage["train_nonrobust"]),
            "dev_nonrobust": list(selfies_coverage["dev_nonrobust"]),
            "dev_only_nonrobust": list(
                selfies_coverage["dev_only_nonrobust"]
            ),
            "dev_only_symbols_registered_before_model_optimization": True,
            "dev_records_used_for_model_optimization": False,
        },
        "macro_policy": {
            "scope": "pf1_sample_bound_provisional",
            "frequency_and_rank_source_split": "train",
            "minimum_occurrences": canary.MACRO_MIN_OCCURRENCES,
            "final_pretraining_k": False,
            "train_summary": macro_summary,
            "materialized_modes": {
                split: dict(sorted(counts.items()))
                for split, counts in materialized_modes_by_split.items()
            },
            "dev_identity_absent_from_train_macro_uses_lossless_fallback": True,
        },
        "replay": replay,
        "disk_usage": {
            "phase_a_temporary_spool_bytes": phase_a_spool_bytes,
            "published_data_bytes_before_manifest": (
                published_data_bytes_before_manifest
            ),
            "observed_peak_materialization_artifact_bytes": (
                peak_materialization_artifact_bytes
            ),
            "lmdb_map_size_gib_virtual_limit": args.lmdb_map_size_gib,
            "temporary_spool_removed_before_publication": True,
        },
        "inputs": {
            "production_release_id": binding_stream.manifest["release_id"],
            "production_logical_release_root_sha256": binding_stream.manifest[
                "logical_release_root_sha256"
            ],
            "production_shards_opened_read_only": binding_stream.shard_receipts,
            "source_archive_bytes": archive_lock["bytes"],
            "source_sdf": sdf_observation,
        },
        "artifacts": {
            "paired_lmdb": {
                "relative_path": LMDB_DIRECTORY,
                "entry_count": len(members),
                "data_mdb_bytes": lmdb_data_bytes,
            },
            "train_membership": {
                "relative_path": TRAIN_MEMBERSHIP_NAME,
                "row_count": split_counts["train"],
            },
            "dev_membership": {
                "relative_path": DEV_MEMBERSHIP_NAME,
                "row_count": split_counts["dev"],
            },
            "rejects": {"relative_path": REJECTS_NAME, "row_count": 0},
            "macro_registry": {
                "relative_path": MACRO_REGISTRY_NAME,
                "row_count": len(normalized_macro_registry),
            },
            "union_tokenizer": {
                "relative_path": TOKENIZER_DIRECTORY,
                "vocab_size": tokenizer_build.runtime.vocab_size,
                "tokenizer_contract_sha256": (
                    tokenizer_build.runtime.tokenizer_contract_sha256
                ),
                "tokenizer_snapshot_sha256": (
                    tokenizer_build.runtime.tokenizer_snapshot_sha256
                ),
            },
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "selfies": importlib_metadata.version("selfies"),
            "lmdb": getattr(lmdb, "__version__", "unknown"),
            "workers": args.workers,
            "max_pending": args.max_pending,
        },
        "method_boundary": {
            "single_sdf_scan": True,
            "production_release_read_only": True,
            "explicit_inherited_e3fp": True,
            "macro_fit_uses_train_split_only": True,
            "selfies_syntax_registry_uses_full_unlabeled_cohort": True,
            "selfies_syntax_frequency_or_rank_used": False,
            "parent_process_single_lmdb_writer": True,
            "bounded_worker_tasks": True,
            "full_molecule_or_production_record_residency": False,
            "second_pass_chemistry_recomputed": False,
            "sequence_truncation": False,
            "complete_lmdb_decode_replay": True,
        },
    }
    _write_json(staging_root / MANIFEST_NAME, manifest)
    staging_root.rename(output_dir)
    return manifest


class PF1PairedReleaseReader:
    """Concrete ``PF1RecordReader`` over one published paired LMDB release."""

    def __init__(
        self,
        release_root: Path,
        *,
        lmdb_module: Any | None = None,
        decoder: Callable[[bytes], Any] = paired_wire.decode_paired_training_record,
    ) -> None:
        self.release_root = Path(release_root).expanduser().resolve()
        if lmdb_module is None:
            try:
                import lmdb as lmdb_module
            except ImportError as exc:
                raise PF1PairedReleaseError("python-lmdb is required") from exc
        self.lmdb_module = lmdb_module
        self.decoder = decoder
        self._uses_canonical_decoder = decoder is paired_wire.decode_paired_training_record
        self._decoded_cache: dict[str, Any] | None = None
        self._decoded_cache_lock = threading.Lock()
        self._decoded_cache_hits = 0
        self._decoded_cache_misses = 0
        self._decoded_cache_warmup: dict[str, object] | None = None
        manifest_path = self.release_root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise PF1PairedReleaseError("PF-1 paired manifest is absent")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("status") != "pass"
        ):
            raise PF1PairedReleaseError("PF-1 paired manifest is not a passed release")
        self.manifest = manifest
        self.lmdb_path = self.release_root / LMDB_DIRECTORY
        if not self.lmdb_path.is_dir():
            raise PF1PairedReleaseError("paired LMDB directory is absent")
        self._train_rows = self._load_split("train", TRAIN_MEMBERSHIP_NAME)
        self._dev_rows = self._load_split("dev", DEV_MEMBERSHIP_NAME)
        train_keys = {str(row["storage_key"]) for row in self._train_rows}
        dev_keys = {str(row["storage_key"]) for row in self._dev_rows}
        if train_keys.intersection(dev_keys):
            raise PF1PairedReleaseError("one paired LMDB key crosses train/dev")
        self.train_member_count = len(self._train_rows)
        self.dev_member_count = len(self._dev_rows)
        counts = manifest.get("counts", {})
        if not (
            counts.get("train_members") == self.train_member_count
            and counts.get("dev_members") == self.dev_member_count
            and counts.get("paired_records")
            == self.train_member_count + self.dev_member_count
        ):
            raise PF1PairedReleaseError("PF-1 reader counts differ from manifest")

    @staticmethod
    def _validate_record_against_membership(
        record: Any,
        row: Mapping[str, object],
    ) -> None:
        if not (
            record.schedule_index == row["selection_index"]
            and record.sdf_record_index == row["sdf_record_index"]
            and record.atom_record.record_id == row["member_id"]
        ):
            raise PF1PairedReleaseError(
                "decoded training row differs from membership"
            )

    def enable_decoded_record_cache(self) -> None:
        """Enable one process-local cache populated only by strict decoding."""

        with self._decoded_cache_lock:
            if self._decoded_cache is None:
                self._decoded_cache = {}

    def decoded_record_cache_stats(self) -> dict[str, object]:
        with self._decoded_cache_lock:
            entries = len(self._decoded_cache or {})
            return {
                "enabled": self._decoded_cache is not None,
                "entries": entries,
                "expected_entries": self.train_member_count + self.dev_member_count,
                "complete": entries
                == self.train_member_count + self.dev_member_count,
                "hits": self._decoded_cache_hits,
                "strict_decode_misses": self._decoded_cache_misses,
                "warmup": dict(self._decoded_cache_warmup)
                if self._decoded_cache_warmup is not None
                else None,
                "process_local_only": True,
                "persistent_artifact": False,
            }

    def warm_decoded_record_cache(
        self,
        *,
        workers: int = 4,
        max_pending: int = 16,
    ) -> dict[str, object]:
        """Strictly decode all uncached rows once with bounded ordered workers.

        This is a runtime optimization, not a second data release.  The cache
        is never serialized and every entry still passes the canonical wire
        decoder plus the frozen membership join before insertion.
        """

        if workers <= 0 or max_pending < workers:
            raise PF1PairedReleaseError(
                "cache workers must be positive and max_pending must cover them"
            )
        if not self._uses_canonical_decoder:
            raise PF1PairedReleaseError(
                "parallel cache warmup requires the canonical paired-wire decoder"
            )
        self.enable_decoded_record_cache()
        all_rows = self._train_rows + self._dev_rows
        row_by_key = {str(row["storage_key"]): row for row in all_rows}
        started = time.perf_counter()
        scheduled = 0
        inserted = 0
        environment = self.lmdb_module.open(
            str(self.lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=8,
        )

        def tasks() -> Iterator[tuple[str, bytes]]:
            nonlocal scheduled
            with environment.begin(write=False) as transaction:
                for row in all_rows:
                    storage_key = str(row["storage_key"])
                    with self._decoded_cache_lock:
                        already_cached = storage_key in (self._decoded_cache or {})
                    if already_cached:
                        continue
                    raw = transaction.get(storage_key.encode("ascii"))
                    if raw is None:
                        raise PF1PairedReleaseError(
                            "paired LMDB row is absent during cache warmup"
                        )
                    scheduled += 1
                    yield storage_key, bytes(raw)

        try:
            if workers == 1:
                results: Iterable[tuple[str, Any]] = (
                    _decode_paired_wire_cache_worker(item) for item in tasks()
                )
            else:
                # Spawn is intentional even on Linux: the training process may
                # already have queried CUDA, and cache workers must never
                # inherit that runtime through fork.
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )

                def bounded_results() -> Iterator[tuple[str, Any]]:
                    pending: deque[concurrent.futures.Future] = deque()
                    try:
                        for item in tasks():
                            pending.append(
                                executor.submit(
                                    _decode_paired_wire_cache_worker, item
                                )
                            )
                            if len(pending) >= max_pending:
                                yield pending.popleft().result()
                        while pending:
                            yield pending.popleft().result()
                    finally:
                        executor.shutdown(wait=True)

                results = bounded_results()
            for storage_key, record in results:
                row = row_by_key.get(storage_key)
                if row is None:
                    raise PF1PairedReleaseError(
                        "cache worker returned an unknown storage key"
                    )
                self._validate_record_against_membership(record, row)
                with self._decoded_cache_lock:
                    assert self._decoded_cache is not None
                    if storage_key in self._decoded_cache:
                        raise PF1PairedReleaseError(
                            "cache warmup produced a duplicate storage key"
                        )
                    self._decoded_cache[storage_key] = record
                    self._decoded_cache_misses += 1
                inserted += 1
        finally:
            environment.close()
        elapsed = time.perf_counter() - started
        with self._decoded_cache_lock:
            assert self._decoded_cache is not None
            complete = len(self._decoded_cache) == len(all_rows)
        if not complete or inserted != scheduled:
            raise PF1PairedReleaseError(
                "decoded record cache warmup did not close the release domain"
            )
        report = {
            "workers": workers,
            "max_pending": max_pending,
            "scheduled_strict_decodes": scheduled,
            "inserted_records": inserted,
            "seconds": elapsed,
            "records_per_second": inserted / elapsed if elapsed else None,
            "order_preserved": True,
            "membership_join_rechecked": True,
        }
        with self._decoded_cache_lock:
            self._decoded_cache_warmup = dict(report)
        return report

    def _load_split(self, split: str, filename: str) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        seen_keys: set[str] = set()
        for row in _iter_output_membership(self.release_root / filename):
            key = row.get("storage_key")
            if (
                row.get("split") != split
                or row.get("split_index") != len(rows)
                or not isinstance(key, str)
                or not key
                or key in seen_keys
            ):
                raise PF1PairedReleaseError(
                    "{} membership order or storage key is invalid".format(split)
                )
            seen_keys.add(key)
            rows.append(row)
        if not rows:
            raise PF1PairedReleaseError("{} membership is empty".format(split))
        return tuple(rows)

    def _iter_batches(
        self, rows: Sequence[Mapping[str, object]], batch_size: int
    ) -> Iterator[tuple[Any, ...]]:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise PF1PairedReleaseError("batch_size must be positive")
        environment = self.lmdb_module.open(
            str(self.lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=8,
        )
        try:
            with environment.begin(write=False) as transaction:
                for start in range(0, len(rows), batch_size):
                    decoded = []
                    for row in rows[start : start + batch_size]:
                        storage_key = str(row["storage_key"])
                        record = None
                        with self._decoded_cache_lock:
                            if self._decoded_cache is not None:
                                record = self._decoded_cache.get(storage_key)
                                if record is not None:
                                    self._decoded_cache_hits += 1
                        if record is None:
                            raw = transaction.get(storage_key.encode("ascii"))
                            if raw is None:
                                raise PF1PairedReleaseError(
                                    "paired LMDB row is absent during training replay"
                                )
                            record = self.decoder(bytes(raw))
                            self._validate_record_against_membership(record, row)
                            with self._decoded_cache_lock:
                                if self._decoded_cache is not None:
                                    self._decoded_cache[storage_key] = record
                                    self._decoded_cache_misses += 1
                        else:
                            self._validate_record_against_membership(record, row)
                        decoded.append(record)
                    yield tuple(decoded)
        finally:
            environment.close()

    def iter_train_epoch(self, *, epoch: int, batch_size: int) -> Iterator[tuple[Any, ...]]:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise PF1PairedReleaseError("epoch must be a non-negative integer")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise PF1PairedReleaseError("batch_size must be positive")
        yield from self._iter_batches(self._train_rows, batch_size)

    def iter_dev(self, *, batch_size: int) -> Iterator[tuple[Any, ...]]:
        yield from self._iter_batches(self._dev_rows, batch_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-membership", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--e3fp-source", required=True)
    parser.add_argument("--base-tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-members", type=int, default=selection.TARGET_MEMBERS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-pending", type=int, default=DEFAULT_MAX_PENDING)
    parser.add_argument("--lmdb-map-size-gib", type=int, default=DEFAULT_MAP_SIZE_GIB)
    parser.add_argument("--commit-every", type=int, default=DEFAULT_COMMIT_EVERY)
    parser.add_argument("--progress-every", type=int, default=250_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_members <= 0 or args.progress_every <= 0:
        parser.error("expected-members and progress-every must be positive")
    manifest = run(args)
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PF1PairedReleaseError",
    "PF1PairedReleaseReader",
    "PreparedMember",
    "FrozenMember",
    "load_frozen_membership",
    "run",
]
