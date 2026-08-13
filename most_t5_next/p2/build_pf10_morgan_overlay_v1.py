"""Build the coordinate-blind PF-10 Morgan atom-state control overlay.

The overlay is deliberately separate from the paired identity/geometry release.
Each worker strictly decodes one published paired record, derives radius-0..3
Morgan states from its persisted SELFIES surface, and returns a compact row to
the parent process.  The parent is the only LMDB/JSONL writer and preserves the
published train/dev order.

The same pass also freezes the fair S-stage intersection: a motif is retained
only when at least two of its atoms have populated level-1 and level-2 states
in both B2D (Morgan) and F3D (E3FP).  Thus either view can mask one atom while
leaving at least one same-motif state atom visible.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import concurrent.futures
from dataclasses import dataclass
from importlib import metadata
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

from most_t5_next.p2.morgan_atom_state_v1 import (
    MORGAN_STATE_ID,
    derive_morgan_atom_state,
)
from most_t5_next.r1.adapter import paired_record_wire_v1 as paired_wire


SCHEMA_VERSION = "most-t5-p2/pf10-morgan-overlay/v1"
MEMBER_SCHEMA_VERSION = "most-t5-p2/pf10-morgan-overlay-member/v1"
COMMON_MEMBER_SCHEMA_VERSION = "most-t5-p2/pf10-common-state-member/v1"
LMDB_DIRECTORY = "morgan_states.lmdb"
TRAIN_MEMBERSHIP = "train_membership.jsonl"
DEV_MEMBERSHIP = "dev_membership.jsonl"
COMMON_TRAIN_MEMBERSHIP = "common_train_state_eligible_membership.jsonl"
COMMON_DEV_MEMBERSHIP = "common_dev_state_eligible_membership.jsonl"
MANIFEST_NAME = "manifest.json"
REJECTS_NAME = "rejects.jsonl"


class PF10MorganOverlayError(ValueError):
    """The published paired release cannot form the frozen B2D control."""


@dataclass(frozen=True)
class _SourceMember:
    split: str
    split_index: int
    selection_index: int
    record_id: str
    storage_key: str
    atom_count: int


_WORKER_ENV = None
_WORKER_SELFIES = None
_WORKER_CHEM = None
_WORKER_FP = None


def _canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_source_members(path: Path, split: str) -> tuple[_SourceMember, ...]:
    rows: list[_SourceMember] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise PF10MorganOverlayError(
                    f"blank {split} membership row at line {line_number}"
                )
            row = json.loads(line)
            member = _SourceMember(
                split=split,
                split_index=int(row["split_index"]),
                selection_index=int(row["selection_index"]),
                record_id=str(row["member_id"]),
                storage_key=str(row["storage_key"]),
                atom_count=int(row["atom_count"]),
            )
            if member.split_index != len(rows):
                raise PF10MorganOverlayError(
                    f"{split} membership split_index is not dense and ordered"
                )
            rows.append(member)
    if not rows:
        raise PF10MorganOverlayError(f"{split} membership is empty")
    return tuple(rows)


def _init_worker(lmdb_path: str, map_size: int) -> None:
    global _WORKER_ENV, _WORKER_SELFIES, _WORKER_CHEM, _WORKER_FP
    import lmdb
    import selfies
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    _WORKER_ENV = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=2048,
        map_size=int(map_size),
        subdir=True,
    )
    _WORKER_SELFIES = selfies
    _WORKER_CHEM = Chem
    _WORKER_FP = rdFingerprintGenerator


def _common_eligible_motifs(
    motif_record: Any,
    b2d_rows: Sequence[Sequence[int]],
) -> tuple[dict[str, object], ...]:
    motif_count = len(motif_record.identity_spans)
    if not (
        len(b2d_rows)
        == len(motif_record.full_e3fp_ids)
        == len(motif_record.atom_valid_mask)
        == len(motif_record.atom_to_logical_motif)
    ):
        raise PF10MorganOverlayError("B2D/F3D motif atom domains disagree")
    by_motif: list[list[int]] = [[] for _ in range(motif_count)]
    for atom_index, (b2d, f3d, valid, motif_id) in enumerate(
        zip(
            b2d_rows,
            motif_record.full_e3fp_ids,
            motif_record.atom_valid_mask,
            motif_record.atom_to_logical_motif,
        )
    ):
        motif_id = int(motif_id)
        if not valid:
            continue
        if not 0 <= motif_id < motif_count or len(b2d) != 4 or len(f3d) != 4:
            raise PF10MorganOverlayError("invalid motif ownership or state width")
        if int(b2d[1]) >= 0 and int(b2d[2]) >= 0 and int(f3d[1]) >= 0 and int(f3d[2]) >= 0:
            by_motif[motif_id].append(atom_index)
    output = []
    for motif_id, atom_indices in enumerate(by_motif):
        if len(atom_indices) < 2:
            continue
        span = motif_record.identity_spans[motif_id]
        output.append(
            {
                "motif_id": motif_id,
                "identity_span_length": int(span.stop - span.start),
                "eligible_atom_indices": atom_indices,
            }
        )
    return tuple(output)


def _build_one(member: _SourceMember) -> dict[str, object]:
    if _WORKER_ENV is None:
        raise PF10MorganOverlayError("Morgan worker was not initialized")
    with _WORKER_ENV.begin(write=False) as transaction:
        payload = transaction.get(member.storage_key.encode("ascii"))
    if payload is None:
        raise PF10MorganOverlayError("paired LMDB record is absent")
    loaded = paired_wire.decode_paired_training_record(bytes(payload))
    atom = loaded.atom_record
    motif = loaded.motif_record
    if not (
        loaded.schedule_index == member.selection_index
        and atom.record_id == motif.record_id == member.record_id
        and atom.storage_key == motif.storage_key == member.storage_key
        and len(atom.atom_to_carrier) == member.atom_count
    ):
        raise PF10MorganOverlayError("paired record differs from source membership")
    state = derive_morgan_atom_state(
        Chem=_WORKER_CHEM,
        rdFingerprintGenerator=_WORKER_FP,
        selfies_decoder=_WORKER_SELFIES.decoder,
        selfies=atom.selfies,
        atom_to_carrier=atom.atom_to_carrier,
    )
    if len(state.state_ids) != member.atom_count:
        raise PF10MorganOverlayError("Morgan state atom count differs from membership")
    common = _common_eligible_motifs(motif, state.state_ids)
    state_document = {
        "schema_version": MEMBER_SCHEMA_VERSION,
        "state_kind": MORGAN_STATE_ID,
        "record_id": member.record_id,
        "storage_key": member.storage_key,
        "state_ids": [list(row) for row in state.state_ids],
    }
    membership = {
        "schema_version": MEMBER_SCHEMA_VERSION,
        "split": member.split,
        "split_index": member.split_index,
        "selection_index": member.selection_index,
        "record_id": member.record_id,
        "storage_key": member.storage_key,
        "atom_count": member.atom_count,
    }
    common_document = None
    if common:
        common_document = {
            "schema_version": COMMON_MEMBER_SCHEMA_VERSION,
            "split": member.split,
            "split_index": member.split_index,
            "selection_index": member.selection_index,
            "record_id": member.record_id,
            "storage_key": member.storage_key,
            "eligible_motifs": list(common),
        }
    return {
        "state_payload": _canonical_json_bytes(state_document),
        "membership": membership,
        "common_membership": common_document,
        "common_motif_count": len(common),
        "common_atom_count": sum(
            len(row["eligible_atom_indices"]) for row in common
        ),
    }


def _ordered_bounded_map(
    function,
    iterable: Iterable[_SourceMember],
    *,
    workers: int,
    max_pending: int,
    initializer,
    initargs: tuple[object, ...],
) -> Iterator[dict[str, object]]:
    if workers < 1 or max_pending < workers:
        raise PF10MorganOverlayError("workers must be positive and pending >= workers")
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=initializer,
        initargs=initargs,
    ) as executor:
        pending: deque[concurrent.futures.Future] = deque()
        for item in iterable:
            pending.append(executor.submit(function, item))
            if len(pending) >= max_pending:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


class MorganAtomStateProvider:
    """Read-only provider implementing the factorized collator protocol."""

    state_kind = MORGAN_STATE_ID

    def __init__(self, overlay_root: Path, *, lmdb_module=None) -> None:
        root = Path(overlay_root).expanduser().resolve()
        manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "pass":
            raise PF10MorganOverlayError("Morgan overlay is not a passed release")
        if lmdb_module is None:
            import lmdb as lmdb_module
        self._env = lmdb_module.open(
            str(root / LMDB_DIRECTORY),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=256,
            subdir=True,
        )

    def get(self, record_id: str) -> tuple[tuple[int, int, int, int], ...]:
        with self._env.begin(write=False) as transaction:
            payload = transaction.get(str(record_id).encode("utf-8"))
        if payload is None:
            raise PF10MorganOverlayError("Morgan state record is absent")
        document = json.loads(bytes(payload))
        if document.get("record_id") != record_id or document.get("state_kind") != MORGAN_STATE_ID:
            raise PF10MorganOverlayError("Morgan state record identity mismatch")
        rows = tuple(tuple(int(value) for value in row) for row in document["state_ids"])
        if not rows or any(len(row) != 4 for row in rows):
            raise PF10MorganOverlayError("Morgan state matrix has an invalid shape")
        return rows

    def close(self) -> None:
        self._env.close()


def build_pf10_morgan_overlay(
    *,
    paired_release: Path,
    output_dir: Path,
    workers: int = 28,
    max_pending: int = 84,
    lmdb_map_size_gib: int = 2,
    commit_every: int = 1024,
) -> dict[str, object]:
    start = time.perf_counter()
    paired_release = Path(paired_release).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    staging = output_dir.with_name(output_dir.name + ".staging")
    if output_dir.exists() or staging.exists():
        raise PF10MorganOverlayError("output or sibling staging path already exists")
    source_manifest = json.loads(
        (paired_release / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    if source_manifest.get("status") != "pass":
        raise PF10MorganOverlayError("paired source release is not passed")
    train = _load_source_members(paired_release / TRAIN_MEMBERSHIP, "train")
    dev = _load_source_members(paired_release / DEV_MEMBERSHIP, "dev")
    members = train + dev
    if len({row.record_id for row in members}) != len(members):
        raise PF10MorganOverlayError("paired membership repeats record_id")
    lmdb_path = paired_release / "paired_records.lmdb"
    if not lmdb_path.is_dir():
        raise PF10MorganOverlayError("paired source LMDB is absent")
    staging.mkdir(parents=True)
    rejects_path = staging / REJECTS_NAME
    rejects_path.write_text("", encoding="utf-8")

    import lmdb

    state_lmdb = staging / LMDB_DIRECTORY
    state_lmdb.mkdir()
    map_size = int(lmdb_map_size_gib) * 1024**3
    env = lmdb.open(str(state_lmdb), map_size=map_size, subdir=True, max_dbs=1)
    transaction = env.begin(write=True)
    split_handles = {
        "train": (staging / TRAIN_MEMBERSHIP).open("w", encoding="utf-8", newline="\n"),
        "dev": (staging / DEV_MEMBERSHIP).open("w", encoding="utf-8", newline="\n"),
    }
    common_handles = {
        "train": (staging / COMMON_TRAIN_MEMBERSHIP).open("w", encoding="utf-8", newline="\n"),
        "dev": (staging / COMMON_DEV_MEMBERSHIP).open("w", encoding="utf-8", newline="\n"),
    }
    counts = Counter()
    processed = 0
    try:
        results = _ordered_bounded_map(
            _build_one,
            members,
            workers=int(workers),
            max_pending=int(max_pending),
            initializer=_init_worker,
            initargs=(str(lmdb_path), int(source_manifest.get("disk_usage", {}).get("lmdb_map_size_bytes", 64 * 1024**3))),
        )
        for member, result in zip(members, results):
            if not transaction.put(
                member.record_id.encode("utf-8"),
                result["state_payload"],
                overwrite=False,
            ):
                raise PF10MorganOverlayError("Morgan LMDB key repeats")
            membership = result["membership"]
            split_handles[member.split].write(
                json.dumps(membership, sort_keys=True, separators=(",", ":")) + "\n"
            )
            common = result["common_membership"]
            if common is not None:
                common_handles[member.split].write(
                    json.dumps(common, sort_keys=True, separators=(",", ":")) + "\n"
                )
                counts[f"common_{member.split}_records"] += 1
            counts[f"common_{member.split}_motifs"] += int(result["common_motif_count"])
            counts[f"common_{member.split}_atoms"] += int(result["common_atom_count"])
            processed += 1
            if processed % int(commit_every) == 0:
                transaction.commit()
                transaction = env.begin(write=True)
        transaction.commit()
        transaction = None
        env.sync(True)
    except Exception as exc:
        if transaction is not None:
            transaction.abort()
        rejects_path.write_text(
            json.dumps(
                {"stage": "MORGAN_OVERLAY_BUILD", "processed": processed, "error": str(exc)},
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        for handle in (*split_handles.values(), *common_handles.values()):
            handle.close()
        env.close()

    if processed != len(members):
        raise PF10MorganOverlayError("Morgan overlay did not exhaust membership")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "state_kind": MORGAN_STATE_ID,
        "source": {
            "paired_release": str(paired_release),
            "paired_release_schema_version": source_manifest.get("schema_version"),
            "paired_records": len(members),
        },
        "counts": {
            "records": len(members),
            "train_records": len(train),
            "dev_records": len(dev),
            **dict(sorted(counts.items())),
        },
        "state_contract": {
            "radius": 3,
            "fp_size": 4096,
            "include_chirality": True,
            "use_bond_types": True,
            "include_redundant_environments": True,
            "coordinate_input_used": False,
        },
        "common_s_stage_contract": {
            "views": ["B2D_Morgan", "F3D_E3FP"],
            "required_levels": [1, 2],
            "minimum_eligible_atoms_per_motif": 2,
            "masked_atoms_per_selected_motif": 1,
            "same_motif_visible_state_atom_guaranteed": True,
        },
        "runtime": {
            "workers": int(workers),
            "max_pending": int(max_pending),
            "multiprocessing_start_method": "spawn",
            "rdkit": metadata.version("rdkit"),
            "selfies": metadata.version("selfies"),
            "wall_seconds": time.perf_counter() - start,
        },
        "method_boundary": {
            "paired_release_modified": False,
            "parent_process_single_lmdb_writer": True,
            "source_membership_order_preserved": True,
            "train_dev_split_preserved": True,
            "no_replacement": True,
            "sequence_truncation": False,
            "per_record_sha256_used": False,
        },
    }
    _write_json(staging / MANIFEST_NAME, manifest)
    staging.rename(output_dir)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--max-pending", type=int, default=84)
    parser.add_argument("--lmdb-map-size-gib", type=int, default=2)
    parser.add_argument("--commit-every", type=int, default=1024)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_pf10_morgan_overlay(
        paired_release=args.paired_release,
        output_dir=args.output_dir,
        workers=args.workers,
        max_pending=args.max_pending,
        lmdb_map_size_gib=args.lmdb_map_size_gib,
        commit_every=args.commit_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMMON_DEV_MEMBERSHIP",
    "COMMON_TRAIN_MEMBERSHIP",
    "MANIFEST_NAME",
    "MorganAtomStateProvider",
    "PF10MorganOverlayError",
    "SCHEMA_VERSION",
    "build_pf10_morgan_overlay",
]
