#!/usr/bin/env python3
"""Materialize a frozen PF-1/PF-10 paired A/M training release.

The builder is deliberately two-phase.  Phase A scans the locked PCQM SDF
member once, keeps at most ``max_pending`` molecule/production-record tasks in
flight, and spools tokenizer-independent paired surfaces to one temporary
SQLite file.  Phase B freezes the shared union tokenizer and motif macro
registry, then the parent process alone writes ``paired_records.lmdb``.
An externally supplied, complete Phase-A spool can restart Phase B in a new
staging directory without rescanning the SDF or reusing any partial LMDB.

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
import io
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from most_t5_next.p1 import freeze_pf1_connectivity_sample_v1 as selection
from most_t5_next.p1.runtime_bridge import P1ArtifactBindings, P1MemberRef
from most_t5_next.r1.adapter import build_p1_inherited_e3fp_overlay_v1 as overlay
from most_t5_next.r1.adapter import build_p1_paired_canary_v1 as canary
from most_t5_next.r1.adapter import build_pcqm_p1_geometry_production_v1 as production
from most_t5_next.r1.adapter import graphports_donor_atom_map_sidecar_v1 as donor_atom_map
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
DONOR_ATOM_MAP_NAME = "donor_atom_maps.jsonl"
STAGING_SUFFIX = ".staging"
DEFAULT_WORKERS = 16
DEFAULT_MAX_PENDING = 24
DEFAULT_PHASE_B_WORKERS = 28
DEFAULT_PHASE_B_MAX_PENDING = 84
DEFAULT_MAP_SIZE_GIB = 4
DEFAULT_COMMIT_EVERY = 512
MAX_SEQUENCE_LENGTH = 512
PF1_RELEASE_PROFILE = "pf1-one-percent-failure-screen-v1"
PF10_RELEASE_PROFILE = "pf10-ten-percent-causal-gate-v1"
RELEASE_PROFILES = {
    PF1_RELEASE_PROFILE: {
        "expected_members": selection.TARGET_MEMBERS,
        "scope": "pf1_one_percent_failure_screen",
        "macro_scope": "pf1_sample_bound_provisional",
        "syntax_registry_scope": "complete_frozen_unlabeled_pf1_cohort",
        "binding_suffix": "pf1-inherited-e3fp-v1",
    },
    PF10_RELEASE_PROFILE: {
        "expected_members": selection.PF10_TARGET_MEMBERS,
        "scope": "pf10_ten_percent_causal_gate",
        "macro_scope": "pf10_sample_bound_provisional",
        "syntax_registry_scope": "complete_frozen_unlabeled_pf10_cohort",
        "binding_suffix": "pf10-inherited-e3fp-v1",
    },
}

_WORKER_STATE: dict[str, Any] = {}


class PF1PairedReleaseError(RuntimeError):
    """The frozen PF-1 paired release could not be materialized exactly."""


def resolve_release_profile(
    args: argparse.Namespace,
) -> tuple[str, Mapping[str, object], int]:
    """Freeze profile-sensitive counts and scientific scope before I/O."""

    profile_id = getattr(args, "release_profile", PF1_RELEASE_PROFILE)
    try:
        profile = RELEASE_PROFILES[profile_id]
    except KeyError as exc:
        raise PF1PairedReleaseError("unknown paired-release profile") from exc
    expected = int(profile["expected_members"])
    requested = getattr(args, "expected_members", None)
    if requested is not None and requested != expected:
        raise PF1PairedReleaseError(
            "named paired-release profile forbids a different expected member count"
        )
    return profile_id, profile, expected


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

    def __init__(
        self,
        path: Path,
        *,
        create: bool,
        immutable: bool = False,
    ) -> None:
        self.path = Path(path)
        self.read_only = not create
        if create:
            self.connection = sqlite3.connect(str(self.path))
        else:
            if not self.path.is_file():
                raise PF1PairedReleaseError("prepared spool is absent")
            query = "?mode=ro&immutable=1" if immutable else "?mode=ro"
            self.connection = sqlite3.connect(
                "{}{}".format(self.path.resolve().as_uri(), query),
                uri=True,
            )
        if create:
            self.connection.execute("PRAGMA journal_mode=OFF")
            self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute(
                "CREATE TABLE prepared (selection_index INTEGER PRIMARY KEY, payload BLOB NOT NULL)"
            )

    def put(self, member: PreparedMember) -> None:
        if self.read_only:
            raise PF1PairedReleaseError("prepared spool is read only")
        self.connection.execute(
            "INSERT INTO prepared(selection_index,payload) VALUES (?,?)",
            (
                member.frozen.selection_index,
                sqlite3.Binary(pickle.dumps(member, protocol=pickle.HIGHEST_PROTOCOL)),
            ),
        )

    def commit(self) -> None:
        if self.read_only:
            raise PF1PairedReleaseError("prepared spool is read only")
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
        value = _load_prepared_pickle(bytes(row[0]))
        if not isinstance(value, PreparedMember):
            raise PF1PairedReleaseError("prepared spool contains an unknown value")
        return value

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM prepared").fetchone()[0])

    def dense_selection_span(self) -> tuple[int, int, int]:
        row = self.connection.execute(
            "SELECT COUNT(*), MIN(selection_index), MAX(selection_index) FROM prepared"
        ).fetchone()
        if row is None or row[1] is None or row[2] is None:
            return (0, -1, -1)
        return (int(row[0]), int(row[1]), int(row[2]))

    def close(self) -> None:
        self.connection.close()


class _PreparedMemberUnpickler(pickle.Unpickler):
    """Map spools produced by ``python -m`` back to canonical dataclasses."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "FrozenMember":
            return FrozenMember
        if module == "__main__" and name == "PreparedMember":
            return PreparedMember
        return super().find_class(module, name)


def _load_prepared_pickle(payload: bytes) -> object:
    return _PreparedMemberUnpickler(io.BytesIO(payload)).load()


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
    binding_suffix: str,
) -> dict[str, str]:
    graph_source = Path(graph_codec.__file__).resolve()
    atom_source = Path(atom_codec.__file__).resolve()
    paired_source = Path(paired.__file__).resolve()
    return {
        "release_id": "{}:{}".format(
            production_manifest["release_id"], binding_suffix
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


def _output_donor_atom_map_row(prepared: PreparedMember) -> dict[str, object]:
    """Persist the GraphPorts atom isomorphism needed only by F3D planning."""

    frozen = prepared.frozen
    row = donor_atom_map.build_release_row(
        selection_index=frozen.selection_index,
        member_id=frozen.member_id,
        sdf_record_index=frozen.sdf_record_index,
        split=frozen.split,
        storage_key=prepared.storage_key,
        graph_encoding=prepared.prepared_surfaces.graph_encoding,
    )
    if row["motif_count"] != prepared.motif_count:
        raise PF1PairedReleaseError(
            "donor atom-map motif count differs from prepared surfaces"
        )
    return row


def _materialize_prepared_member(
    prepared: PreparedMember,
    *,
    union_tokenizer: Any,
    tokenizer_binding: Any,
    binding_base: Mapping[str, str],
    macro_by_identity: Mapping[str, str],
) -> dict[str, object]:
    """Build one canonical paired row without performing any release writes."""

    frozen = prepared.frozen
    bindings = P1ArtifactBindings(
        **binding_base,
        geometry_record_content_sha256=prepared.effective_geometry_content_sha256,
        tokenizer_contract_sha256=tokenizer_binding.tokenizer_contract_sha256,
        tokenizer_snapshot_sha256=tokenizer_binding.tokenizer_snapshot_sha256,
    )
    pair = paired.build_production_paired_identity_records_from_prepared(
        prepared=prepared.prepared_surfaces,
        member=P1MemberRef(frozen.member_id, prepared.storage_key),
        bindings=bindings,
        base_geometry_record_content_sha256=prepared.base_record_content_sha256,
        effective_inherited_overlay_content_sha256=(
            prepared.effective_geometry_content_sha256
        ),
        source_atom_count=prepared.source_atom_count,
        model_to_source_atom_index=prepared.model_to_source_atom_index,
        inherited_e3fp=prepared.inherited_e3fp,
        union_tokenizer=union_tokenizer,
        tokenizer_binding=tokenizer_binding,
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
    return {
        "status": "pass",
        "frozen": frozen,
        "storage_key": prepared.storage_key,
        "payload": payload,
        "membership": _output_membership_row(
            prepared, loaded, payload, split_index=0
        ),
        "donor_atom_map": _output_donor_atom_map_row(prepared),
        "motif_identity_modes": tuple(
            loaded.surface_summary.motif_identity_modes
        ),
    }


def _init_phase_b_worker(
    spool_path: str,
    base_tokenizer: str,
    tokenizer_directory: str,
    binding_base: Mapping[str, str],
    macro_by_identity: Mapping[str, str],
) -> None:
    tokenizer_build = union_builder.load_verified_canary_union_tokenizer(
        base_snapshot=Path(base_tokenizer),
        output_dir=Path(tokenizer_directory),
    )
    _WORKER_STATE["phase_b_spool"] = _PreparedSpool(
        Path(spool_path), create=False, immutable=True
    )
    _WORKER_STATE["phase_b_tokenizer"] = tokenizer_build.tokenizer
    _WORKER_STATE["phase_b_tokenizer_binding"] = tokenizer_build.runtime
    _WORKER_STATE["phase_b_binding_base"] = dict(binding_base)
    _WORKER_STATE["phase_b_macro_by_identity"] = dict(macro_by_identity)


def _materialize_spooled_member(frozen: FrozenMember) -> dict[str, object]:
    """Phase-B worker: read one immutable spool row and emit canonical bytes."""

    try:
        spool = _WORKER_STATE["phase_b_spool"]
        prepared = spool.get(frozen.selection_index)
        if prepared.frozen != frozen:
            raise PF1PairedReleaseError(
                "prepared spool order/binding differs from frozen membership"
            )
        return _materialize_prepared_member(
            prepared,
            union_tokenizer=_WORKER_STATE["phase_b_tokenizer"],
            tokenizer_binding=_WORKER_STATE["phase_b_tokenizer_binding"],
            binding_base=_WORKER_STATE["phase_b_binding_base"],
            macro_by_identity=_WORKER_STATE["phase_b_macro_by_identity"],
        )
    except Exception as exc:
        return {
            "status": "reject",
            "reject": _reject_row(
                frozen,
                stage="PAIRED_BUILD",
                reason="{}: {}".format(type(exc).__name__, exc),
            ),
        }


def _ordered_phase_b_map(
    members: Iterable[FrozenMember],
    *,
    workers: int,
    max_pending: int,
    initargs: tuple[object, ...],
) -> Iterator[dict[str, object]]:
    """Bound a spawn pool while yielding Phase-B results in selection order."""

    if workers < 1 or max_pending < workers:
        raise PF1PairedReleaseError(
            "phase-b-workers must be positive and phase-b-max-pending >= workers"
        )
    if workers == 1:
        _init_phase_b_worker(*initargs)  # type: ignore[arg-type]
        for frozen in members:
            yield _materialize_spooled_member(frozen)
        return
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_phase_b_worker,
        initargs=initargs,
    ) as executor:
        pending: deque[concurrent.futures.Future[dict[str, object]]] = deque()
        for frozen in members:
            pending.append(executor.submit(_materialize_spooled_member, frozen))
            if len(pending) >= max_pending:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _replay_donor_atom_map_sidecar(
    path: Path, members: Sequence[FrozenMember]
) -> dict[str, int | bool]:
    """Stream-replay the sidecar in frozen order without retaining its rows."""

    rows = donor_atom_map.iter_release_rows(path)
    motif_count = 0
    atom_mapping_count = 0
    for expected in members:
        try:
            row = next(rows)
        except StopIteration as exc:
            raise PF1PairedReleaseError(
                "donor atom-map sidecar ends before frozen membership"
            ) from exc
        if not all(
            (
                row["selection_index"] == expected.selection_index,
                row["member_id"] == expected.member_id,
                row["sdf_record_index"] == expected.sdf_record_index,
                row["split"] == expected.split,
            )
        ):
            raise PF1PairedReleaseError(
                "donor atom-map sidecar lineage differs from frozen membership"
            )
        motif_count += int(row["motif_count"])
        planning = row["overlay_planning_sidecar"]
        atom_mapping_count += sum(
            len(atom_map)
            for atom_map in planning["canonical_local_atom_to_model_atom"]  # type: ignore[index]
        )
    try:
        next(rows)
    except StopIteration:
        pass
    else:
        raise PF1PairedReleaseError(
            "donor atom-map sidecar contains rows outside frozen membership"
        )
    return {
        "rows_replayed": len(members),
        "motifs_replayed": motif_count,
        "atom_mappings_replayed": atom_mapping_count,
        "selection_order_preserved": True,
        "full_rows_retained_in_memory": False,
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
    release_profile_id, release_profile, expected_members = resolve_release_profile(
        args
    )
    frozen_path = Path(args.frozen_membership).expanduser().resolve()
    release_root = Path(args.release_root).expanduser().resolve()
    source_archive = Path(args.source_archive).expanduser().resolve()
    e3fp_source = Path(args.e3fp_source).expanduser().resolve()
    base_tokenizer = Path(args.base_tokenizer).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    staging_root = output_dir.with_name(output_dir.name + STAGING_SUFFIX)
    prepared_spool_arg = getattr(args, "prepared_spool", None)
    external_spool_path = (
        Path(prepared_spool_arg).expanduser().resolve()
        if prepared_spool_arg is not None
        else None
    )
    if not frozen_path.is_file():
        raise PF1PairedReleaseError("frozen membership is absent")
    if not release_root.is_dir() or not source_archive.is_file():
        raise PF1PairedReleaseError("production release or source archive is absent")
    if not e3fp_source.exists() or not base_tokenizer.is_dir():
        raise PF1PairedReleaseError("E3FP source or base tokenizer is absent")
    if output_dir.exists() or staging_root.exists():
        raise PF1PairedReleaseError("output and sibling staging paths must be new")
    if external_spool_path is not None and not external_spool_path.is_file():
        raise PF1PairedReleaseError("external prepared spool is absent")
    if args.workers < 1 or args.max_pending < args.workers:
        raise PF1PairedReleaseError("workers must be positive and max-pending >= workers")
    if (
        args.phase_b_workers < 1
        or args.phase_b_max_pending < args.phase_b_workers
    ):
        raise PF1PairedReleaseError(
            "phase-b-workers must be positive and phase-b-max-pending >= workers"
        )
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
        frozen_path, expected_members=expected_members
    )
    by_ordinal = {row.sdf_record_index: row for row in members}
    staging_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir()
    spool_owned_by_builder = external_spool_path is None
    spool_path = (
        staging_root / "prepared_spool.sqlite3"
        if external_spool_path is None
        else external_spool_path
    )
    spool = _PreparedSpool(
        spool_path,
        create=spool_owned_by_builder,
        immutable=not spool_owned_by_builder,
    )
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

    def observe_prepared(prepared: PreparedMember) -> None:
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
        if spool_owned_by_builder:
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
                    raise PF1PairedReleaseError(
                        "worker prepared row has an unknown type"
                    )
                spool.put(prepared)
                observe_prepared(prepared)
            spool.commit()
        else:
            dense_span = spool.dense_selection_span()
            if dense_span != (len(members), 0, len(members) - 1):
                raise PF1PairedReleaseError(
                    "external prepared spool is not a complete dense selection"
                )
            for frozen in members:
                prepared = spool.get(frozen.selection_index)
                if prepared.frozen != frozen:
                    raise PF1PairedReleaseError(
                        "external prepared spool differs from frozen membership"
                    )
                observe_prepared(prepared)
            sdf_observation.update(
                {
                    "scan_performed_this_invocation": False,
                    "reason": "reused_complete_prepared_spool",
                    "expected_source_record_count": source_record_count,
                }
            )
    except Exception:
        spool.close()
        raise
    finally:
        binding_stream.close()

    if rejects or spool.count() != len(members):
        spool.close()
        _write_failure(staging_root, rejects, expected_members=len(members))
        raise PF1PairedReleaseError("surface discovery rejected frozen paired members")
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
                "scope": str(release_profile["macro_scope"]),
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
        binding_suffix=str(release_profile["binding_suffix"]),
    )
    spool.close()
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
    donor_atom_map_path = staging_root / DONOR_ATOM_MAP_NAME
    split_counts = {"train": 0, "dev": 0}
    donor_atom_map_rows = 0
    donor_atom_map_payload_bytes = 0
    materialized_modes: Counter[str] = Counter()
    materialized_modes_by_split: dict[str, Counter[str]] = {
        "train": Counter(),
        "dev": Counter(),
    }
    pair_rejects: list[dict[str, object]] = []
    phase_b_results = _ordered_phase_b_map(
        members,
        workers=args.phase_b_workers,
        max_pending=args.phase_b_max_pending,
        initargs=(
            str(spool_path),
            str(base_tokenizer),
            str(staging_root / TOKENIZER_DIRECTORY),
            binding_base,
            macro_by_identity,
        ),
    )
    try:
        with train_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as train_handle, dev_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as dev_handle, donor_atom_map_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as donor_atom_map_handle:
            for start in range(0, len(members), args.commit_every):
                chunk = members[start : start + args.commit_every]
                with environment.begin(write=True) as transaction:
                    for frozen in chunk:
                        result = next(phase_b_results)
                        if result.get("status") != "pass":
                            reject = result.get("reject")
                            if not isinstance(reject, dict):
                                raise PF1PairedReleaseError(
                                    "phase-B worker reject row is absent"
                                )
                            pair_rejects.append(reject)
                            raise PF1PairedReleaseError(
                                "phase-B stopped at the first rejected member"
                            )
                        if result.get("frozen") != frozen:
                            raise PF1PairedReleaseError(
                                "phase-B result order differs from frozen membership"
                            )
                        storage_key = result.get("storage_key")
                        payload = result.get("payload")
                        row_value = result.get("membership")
                        donor_value = result.get("donor_atom_map")
                        modes = result.get("motif_identity_modes")
                        if not (
                            isinstance(storage_key, str)
                            and isinstance(payload, bytes)
                            and isinstance(row_value, dict)
                            and isinstance(donor_value, dict)
                            and isinstance(modes, tuple)
                        ):
                            raise PF1PairedReleaseError(
                                "phase-B worker returned an invalid materialized row"
                            )
                        if not transaction.put(
                            storage_key.encode("ascii"), payload, overwrite=False
                        ):
                            raise PF1PairedReleaseError("paired LMDB key collision")
                        split_index = split_counts[frozen.split]
                        row = dict(row_value)
                        row["split_index"] = split_index
                        _write_jsonl_row(
                            train_handle if frozen.split == "train" else dev_handle,
                            row,
                        )
                        donor_atom_map_payload_bytes += donor_atom_map.write_release_row(
                            donor_atom_map_handle, donor_value
                        )
                        donor_atom_map_rows += 1
                        split_counts[frozen.split] += 1
                        materialized_modes.update(modes)
                        materialized_modes_by_split[frozen.split].update(modes)
            try:
                next(phase_b_results)
            except StopIteration:
                pass
            else:
                raise PF1PairedReleaseError(
                    "phase-B produced rows outside frozen membership"
                )
        environment.sync(True)
    except Exception:
        if pair_rejects:
            _write_failure(
                staging_root, pair_rejects, expected_members=len(members)
            )
        raise
    finally:
        phase_b_results.close()
        environment.close()

    if pair_rejects or sum(split_counts.values()) != len(members):
        _write_failure(
            staging_root, pair_rejects, expected_members=len(members)
        )
        raise PF1PairedReleaseError("paired materialization rejected frozen paired members")
    if donor_atom_map_rows != len(members):
        raise PF1PairedReleaseError(
            "donor atom-map sidecar count differs from paired records"
        )
    donor_atom_map_replay = _replay_donor_atom_map_sidecar(
        donor_atom_map_path, members
    )

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
            int(donor_atom_map_path.stat().st_size),
            int((staging_root / REJECTS_NAME).stat().st_size),
            int((staging_root / MACRO_REGISTRY_NAME).stat().st_size),
            _tree_file_bytes(staging_root / TOKENIZER_DIRECTORY),
        )
    )
    peak_materialization_artifact_bytes = (
        phase_a_spool_bytes + published_data_bytes_before_manifest
    )
    # A builder-owned spool is temporary.  An explicitly supplied Phase-A
    # boundary is immutable input and is never removed or modified.
    if spool_owned_by_builder:
        spool_path.unlink()

    phase_a_resume = {
        "phase_a_mode": (
            "fresh_sdf_scan"
            if spool_owned_by_builder
            else "reused_complete_spool"
        ),
        "prepared_spool_path": str(spool_path),
        "prepared_spool_bytes": phase_a_spool_bytes,
        "prepared_spool_owned_by_builder": spool_owned_by_builder,
        "external_spool_preserved": not spool_owned_by_builder,
        "dense_selection_rows_validated": len(members),
        "frozen_member_rows_fully_revalidated": len(members),
        "aggregates_recomputed_from_spool": not spool_owned_by_builder,
        "sdf_rescanned": spool_owned_by_builder,
        "phase_b_started_from_selection_index": 0,
        "prior_partial_lmdb_reused": False,
        "new_output_staging": str(staging_root),
    }

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "created_utc": _utc_now(),
        "scope": str(release_profile["scope"]),
        "release_profile": release_profile_id,
        "counts": {
            "scheduled_members": len(members),
            "train_members": split_counts["train"],
            "dev_members": split_counts["dev"],
            "paired_records": sum(split_counts.values()),
            "donor_atom_map_rows": donor_atom_map_rows,
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
        "phase_a_resume": phase_a_resume,
        "structure": {
            key: _value_distribution(values) for key, values in structure.items()
        },
        "selfies_vocabulary_coverage": {
            "registry_role": "lossless_atom_selfies_syntax",
            "registry_scope": str(release_profile["syntax_registry_scope"]),
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
            "scope": str(release_profile["macro_scope"]),
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
        "donor_atom_map_replay": donor_atom_map_replay,
        "disk_usage": {
            "phase_a_spool_bytes": phase_a_spool_bytes,
            "phase_a_temporary_spool_bytes": (
                phase_a_spool_bytes if spool_owned_by_builder else 0
            ),
            "phase_a_spool_owned_by_builder": spool_owned_by_builder,
            "external_phase_a_spool_preserved": not spool_owned_by_builder,
            "published_data_bytes_before_manifest": (
                published_data_bytes_before_manifest
            ),
            "observed_peak_materialization_artifact_bytes": (
                peak_materialization_artifact_bytes
            ),
            "lmdb_map_size_gib_virtual_limit": args.lmdb_map_size_gib,
            "temporary_spool_removed_before_publication": (
                spool_owned_by_builder
            ),
        },
        "inputs": {
            "production_release_id": binding_stream.manifest["release_id"],
            "production_logical_release_root_sha256": binding_stream.manifest[
                "logical_release_root_sha256"
            ],
            "production_shards_opened_read_only": binding_stream.shard_receipts,
            "source_archive_bytes": archive_lock["bytes"],
            "source_sdf": sdf_observation,
            "prepared_spool": {
                "path": str(spool_path),
                "bytes": phase_a_spool_bytes,
                "role": (
                    "builder_temporary_phase_boundary"
                    if spool_owned_by_builder
                    else "external_complete_phase_a_boundary"
                ),
            },
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
            "donor_atom_maps": {
                "relative_path": DONOR_ATOM_MAP_NAME,
                "schema_version": donor_atom_map.ROW_SCHEMA,
                "row_count": donor_atom_map_rows,
                "payload_bytes": donor_atom_map_payload_bytes,
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
            "phase_b_workers": args.phase_b_workers,
            "phase_b_max_pending": args.phase_b_max_pending,
        },
        "method_boundary": {
            "single_sdf_scan": spool_owned_by_builder,
            "phase_a_sdf_scan_performed_this_invocation": (
                spool_owned_by_builder
            ),
            "external_complete_spool_replayed": not spool_owned_by_builder,
            "phase_b_worker_reads_immutable_sqlite": True,
            "phase_b_ordered_results": True,
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
            "donor_atom_maps_published_as_separate_planning_sidecar": True,
            "donor_atom_map_rows_streamed_without_release_residency": True,
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

    def iter_donor_atom_maps(
        self,
        *,
        split: str | None = None,
        max_rows: int | None = None,
    ) -> Iterator[dict[str, object]]:
        """Stream F3D planning maps; never cache the release-wide sidecar."""

        if split not in {None, "train", "dev"}:
            raise PF1PairedReleaseError("donor atom-map split must be train, dev or None")
        if max_rows is not None and (
            isinstance(max_rows, bool)
            or not isinstance(max_rows, int)
            or max_rows <= 0
        ):
            raise PF1PairedReleaseError("donor atom-map max_rows must be positive")
        path = self.release_root / DONOR_ATOM_MAP_NAME
        if not path.is_file():
            raise PF1PairedReleaseError("donor atom-map planning sidecar is absent")
        selected = 0
        total = 0
        rows = donor_atom_map.iter_release_rows(path)
        try:
            for row in rows:
                total += 1
                if split is not None and row["split"] != split:
                    continue
                yield row
                selected += 1
                if max_rows is not None and selected == max_rows:
                    return
        finally:
            rows.close()
        expected_total = self.train_member_count + self.dev_member_count
        expected_selected = (
            expected_total
            if split is None
            else self.train_member_count
            if split == "train"
            else self.dev_member_count
        )
        if total != expected_total or selected != expected_selected:
            raise PF1PairedReleaseError(
                "donor atom-map stream count differs from paired membership"
            )

    def iter_raw_motif_documents(
        self,
        *,
        split: str,
    ) -> Iterator[tuple[dict[str, object], dict[str, object]]]:
        """Stream persisted motif documents in frozen split order.

        This narrow planning interface is intentionally separate from the
        training decoder.  It exposes fields such as cross-motif bonds and
        slot atoms that are validated at release construction but omitted
        from the compact in-memory training record.
        """

        if split == "train":
            rows = self._train_rows
        elif split == "dev":
            rows = self._dev_rows
        else:
            raise PF1PairedReleaseError("raw motif-document split must be train or dev")
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
                for row in rows:
                    storage_key = str(row["storage_key"])
                    raw = transaction.get(storage_key.encode("ascii"))
                    if raw is None:
                        raise PF1PairedReleaseError(
                            "paired LMDB row is absent during planning replay"
                        )
                    try:
                        envelope = json.loads(bytes(raw))
                        motif_document = envelope["motif_training_document"]
                        member = motif_document["member"]
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise PF1PairedReleaseError(
                            "paired LMDB planning document is malformed"
                        ) from exc
                    if not (
                        isinstance(envelope, dict)
                        and isinstance(motif_document, dict)
                        and member["member_id"] == row["member_id"]
                        and member["storage_key"] == storage_key
                    ):
                        raise PF1PairedReleaseError(
                            "paired planning document differs from membership"
                        )
                    yield dict(row), motif_document
        finally:
            environment.close()

    def benchmark_donor_atom_map_prefix(
        self, *, max_rows: int = donor_atom_map.DEFAULT_BENCHMARK_ROWS
    ) -> dict[str, int | float | bool]:
        """Replay the bounded 1,024-row planning interface."""

        path = self.release_root / DONOR_ATOM_MAP_NAME
        if not path.is_file():
            raise PF1PairedReleaseError("donor atom-map planning sidecar is absent")
        return donor_atom_map.benchmark_release_prefix(path, max_rows=max_rows)

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

    def iter_strict_parallel_split(
        self,
        *,
        split: str,
        max_rows: int | None = None,
        workers: int = 4,
        max_pending: int = 16,
    ) -> Iterator[Any]:
        """Strictly decode one ordered split stream without retaining it.

        Unlike :meth:`warm_decoded_record_cache`, this interface is intended
        for one-time derived-artifact compilation.  Only the selected rows are
        submitted, workers return results in membership order, and no decoded
        Python record is inserted into the process-local training cache.
        """

        if split not in {"train", "dev"}:
            raise PF1PairedReleaseError("parallel decode split must be train or dev")
        if (
            max_rows is not None
            and (
                isinstance(max_rows, bool)
                or not isinstance(max_rows, int)
                or max_rows <= 0
            )
        ):
            raise PF1PairedReleaseError("parallel decode max_rows must be positive")
        if workers <= 0 or max_pending < workers:
            raise PF1PairedReleaseError(
                "parallel decode workers must be positive and bounded"
            )
        if not self._uses_canonical_decoder:
            raise PF1PairedReleaseError(
                "parallel split decode requires the canonical paired-wire decoder"
            )
        source = self._train_rows if split == "train" else self._dev_rows
        rows = source if max_rows is None else source[:max_rows]
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
            with environment.begin(write=False) as transaction:
                for row in rows:
                    storage_key = str(row["storage_key"])
                    raw = transaction.get(storage_key.encode("ascii"))
                    if raw is None:
                        raise PF1PairedReleaseError(
                            "paired LMDB row is absent during parallel split decode"
                        )
                    yield storage_key, bytes(raw)

        executor: concurrent.futures.ProcessPoolExecutor | None = None
        try:
            if workers == 1:
                results: Iterable[tuple[str, Any]] = (
                    _decode_paired_wire_cache_worker(item) for item in tasks()
                )
            else:
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )

                def bounded_results() -> Iterator[tuple[str, Any]]:
                    pending: deque[concurrent.futures.Future] = deque()
                    for item in tasks():
                        pending.append(
                            executor.submit(  # type: ignore[union-attr]
                                _decode_paired_wire_cache_worker, item
                            )
                        )
                        if len(pending) >= max_pending:
                            yield pending.popleft().result()
                    while pending:
                        yield pending.popleft().result()

                results = bounded_results()

            decoded_count = 0
            for expected_row, (storage_key, record) in zip(rows, results):
                if storage_key != str(expected_row["storage_key"]):
                    raise PF1PairedReleaseError(
                        "parallel split decode changed membership order"
                    )
                self._validate_record_against_membership(record, expected_row)
                decoded_count += 1
                yield record
            if decoded_count != len(rows):
                raise PF1PairedReleaseError(
                    "parallel split decode returned the wrong row count"
                )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
            environment.close()

    def iter_selected_split_indices(
        self,
        *,
        split: str,
        split_indices: Sequence[int],
        batch_size: int,
    ) -> Iterator[tuple[Any, ...]]:
        """Decode an explicit ordered subset without scanning intervening rows."""

        if split not in {"train", "dev"}:
            raise PF1PairedReleaseError("selected split must be train or dev")
        indices = tuple(split_indices)
        if not indices or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        ):
            raise PF1PairedReleaseError(
                "selected split indices must be nonnegative integers"
            )
        if tuple(sorted(set(indices))) != indices:
            raise PF1PairedReleaseError(
                "selected split indices must be unique and increasing"
            )
        source = self._train_rows if split == "train" else self._dev_rows
        if indices[-1] >= len(source):
            raise PF1PairedReleaseError("selected split index is out of range")
        rows = tuple(source[index] for index in indices)
        yield from self._iter_batches(rows, batch_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-membership", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--e3fp-source", required=True)
    parser.add_argument("--base-tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prepared-spool",
        help=(
            "complete Phase-A prepared_spool.sqlite3; skips the SDF scan and "
            "restarts Phase B in a new output staging directory"
        ),
    )
    parser.add_argument(
        "--release-profile",
        choices=tuple(RELEASE_PROFILES),
        default=PF1_RELEASE_PROFILE,
    )
    parser.add_argument(
        "--expected-members",
        type=int,
        help="optional assertion; must equal the named release profile",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-pending", type=int, default=DEFAULT_MAX_PENDING)
    parser.add_argument(
        "--phase-b-workers", type=int, default=DEFAULT_PHASE_B_WORKERS
    )
    parser.add_argument(
        "--phase-b-max-pending", type=int, default=DEFAULT_PHASE_B_MAX_PENDING
    )
    parser.add_argument("--lmdb-map-size-gib", type=int, default=DEFAULT_MAP_SIZE_GIB)
    parser.add_argument("--commit-every", type=int, default=DEFAULT_COMMIT_EVERY)
    parser.add_argument("--progress-every", type=int, default=250_000)
    return parser


def run_phase_b_resume(
    args: argparse.Namespace,
    *,
    prepared_spool: Path,
) -> dict[str, object]:
    """Public API for restarting Phase B from one complete Phase-A spool."""

    resumed = argparse.Namespace(**vars(args))
    resumed.prepared_spool = str(Path(prepared_spool).expanduser().resolve())
    return run(resumed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.expected_members is not None
        and args.expected_members <= 0
    ) or args.progress_every <= 0:
        parser.error("expected-members and progress-every must be positive")
    manifest = run(args)
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DONOR_ATOM_MAP_NAME",
    "PF1_RELEASE_PROFILE",
    "PF10_RELEASE_PROFILE",
    "PF1PairedReleaseError",
    "PF1PairedReleaseReader",
    "PreparedMember",
    "FrozenMember",
    "load_frozen_membership",
    "resolve_release_profile",
    "run",
    "run_phase_b_resume",
]
