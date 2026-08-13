#!/usr/bin/env python3
"""Reproject the Phase-II PubChem train molecules into the current motif surface.

The legacy LMDB is treated only as a hash-locked source of ``(CID, SMILES)``.
Stored legacy ``motif_seq`` values are deliberately ignored because their
anchor and stereochemistry semantics differ from the current anchored surface.
The result is a compact, deterministic motif-ID/offset cache suitable for a
joint Phase-I/Phase-II vocabulary analysis; it is not a training release.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, deque
import concurrent.futures
import datetime as dt
import hashlib
import json
import multiprocessing
from numbers import Integral
from pathlib import Path
import pickle
import re
import sys
import time
from typing import Iterator, Sequence

from most_t5_next.p1.build_registered_downstream_pure_motif_census_v1 import (
    _project_smiles,
)


SCHEMA_VERSION = "most-t5-next/phase2-anchored-pure-motif-census/v1"
ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_TRUSTED_PICKLE_CAN_EXECUTE_CODE"
CONFIG_SCHEMA = "most-t5-r1/identity-collection-extraction-config/v1"
EXPECTED_FIELDS = {
    "atom_to_motif_map",
    "atoms",
    "cid",
    "coordinates",
    "description",
    "e3fp",
    "enriched_description",
    "motif_seq",
    "raw_smiles",
    "smiles",
}
PAYLOAD_KEY_RE = re.compile(rb"^[1-9][0-9]*$")


class Phase2AnchoredCensusError(RuntimeError):
    """The locked Phase-II train source cannot be projected exactly."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _load_source_config(path: Path, source: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise Phase2AnchoredCensusError("source config schema differs")
    collection = config.get("collection")
    source_row = config.get("source")
    mapping = config.get("mapping")
    if not isinstance(collection, dict) or collection.get("phase") != "p2":
        raise Phase2AnchoredCensusError("source is not registered as Phase II")
    if collection.get("split") != "train":
        raise Phase2AnchoredCensusError("only Phase-II train may influence vocabulary")
    if not isinstance(source_row, dict) or source_row.get("format") != "legacy_lmdb_pickle":
        raise Phase2AnchoredCensusError("source format differs")
    if not isinstance(mapping, dict) or mapping.get("smiles_field") != "smiles":
        raise Phase2AnchoredCensusError("current projection must consume the smiles field")
    expected_bytes = source_row.get("expected_bytes")
    expected_sha = source_row.get("expected_sha256")
    options = source_row.get("format_options")
    lmdb_options = options.get("lmdb") if isinstance(options, dict) else None
    trusted_sha = (
        lmdb_options.get("trusted_pickle_source_sha256")
        if isinstance(lmdb_options, dict)
        else None
    )
    permitted_metadata = (
        lmdb_options.get("metadata_keys_permitted")
        if isinstance(lmdb_options, dict)
        else None
    )
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or trusted_sha != expected_sha
        or not isinstance(permitted_metadata, list)
        or any(not isinstance(item, str) or not item for item in permitted_metadata)
    ):
        raise Phase2AnchoredCensusError("source byte/pickle trust lock is invalid")
    if not source.is_file() or source.is_symlink():
        raise Phase2AnchoredCensusError("source LMDB must be one regular file")
    if source.stat().st_size != expected_bytes:
        raise Phase2AnchoredCensusError("source LMDB byte count differs")
    if _sha256_file(source) != expected_sha:
        raise Phase2AnchoredCensusError("source LMDB SHA-256 differs")
    return config


def _payload_smiles(source_key: str, value: bytes) -> str:
    payload = pickle.loads(value)
    if not isinstance(payload, dict) or set(payload) != EXPECTED_FIELDS:
        raise Phase2AnchoredCensusError("legacy payload schema differs")
    cid = payload["cid"]
    if isinstance(cid, Integral) and not isinstance(cid, bool):
        normalized_cid = str(int(cid))
    elif isinstance(cid, str) and cid.isascii() and cid.isdigit():
        normalized_cid = str(int(cid))
    else:
        raise Phase2AnchoredCensusError("payload CID is invalid")
    if normalized_cid != source_key:
        raise Phase2AnchoredCensusError("LMDB key and payload CID differ")
    smiles = payload["smiles"]
    if not isinstance(smiles, str) or not smiles.strip():
        raise Phase2AnchoredCensusError("payload smiles is empty")
    return smiles.strip()


_WORKER_ENV = None


def _init_worker(source_path: str) -> None:
    global _WORKER_ENV
    import lmdb

    _WORKER_ENV = lmdb.open(
        source_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )


def _project_key(task: tuple[int, str]) -> dict[str, object]:
    selection_index, source_key = task
    try:
        if _WORKER_ENV is None:
            raise Phase2AnchoredCensusError("worker LMDB is not initialized")
        with _WORKER_ENV.begin(write=False, buffers=True) as transaction:
            value = transaction.get(source_key.encode("ascii"))
            if value is None:
                raise Phase2AnchoredCensusError("payload disappeared from locked LMDB")
            value_bytes = bytes(value)
        smiles = _payload_smiles(source_key, value_bytes)
        result = _project_smiles((selection_index, source_key, smiles))
        result["source_key"] = source_key
        return result
    except Exception as exc:
        return {
            "row_index": selection_index,
            "record_id": source_key,
            "source_key": source_key,
            "pure_motifs": (),
            "model_facing_anchor_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _ordered_bounded_map(
    tasks: Sequence[tuple[int, str]],
    workers: int,
    max_pending: int,
    source_path: str,
) -> Iterator[dict[str, object]]:
    if workers <= 0 or max_pending < workers:
        raise Phase2AnchoredCensusError("max_pending must be at least workers")
    if workers == 1:
        _init_worker(source_path)
        try:
            for task in tasks:
                yield _project_key(task)
        finally:
            global _WORKER_ENV
            if _WORKER_ENV is not None:
                _WORKER_ENV.close()
                _WORKER_ENV = None
        return
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(source_path,),
    ) as executor:
        pending: deque[concurrent.futures.Future] = deque()
        iterator = iter(tasks)
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < max_pending:
                try:
                    pending.append(executor.submit(_project_key, next(iterator)))
                except StopIteration:
                    exhausted = True
            if pending:
                yield pending.popleft().result()


def _source_keys(
    source: Path,
    expected_records: int,
    permitted_metadata: set[str],
) -> tuple[str, ...]:
    import lmdb

    environment = lmdb.open(
        str(source),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    try:
        with environment.begin(write=False, buffers=True) as transaction:
            keys = []
            observed_metadata: set[str] = set()
            for key, _value in transaction.cursor():
                key_bytes = bytes(key)
                if PAYLOAD_KEY_RE.fullmatch(key_bytes) is not None:
                    keys.append(key_bytes.decode("ascii"))
                else:
                    try:
                        observed_metadata.add(key_bytes.decode("utf-8"))
                    except UnicodeDecodeError as exc:
                        raise Phase2AnchoredCensusError(
                            "source contains an unknown non-UTF8 metadata key"
                        ) from exc
    finally:
        environment.close()
    if len(keys) != expected_records or len(set(keys)) != len(keys):
        raise Phase2AnchoredCensusError("Phase-II payload membership count differs")
    if not observed_metadata <= permitted_metadata:
        raise Phase2AnchoredCensusError("source contains an undeclared metadata key")
    return tuple(sorted(keys, key=int))


def build(args: argparse.Namespace) -> dict[str, object]:
    if args.legacy_pickle_acknowledgement != ACKNOWLEDGEMENT:
        raise Phase2AnchoredCensusError("exact trusted-pickle acknowledgement is required")
    if sys.byteorder != "little":
        raise Phase2AnchoredCensusError("compact cache requires a little-endian host")
    source = Path(args.source_lmdb).expanduser().resolve()
    config_path = Path(args.source_config).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists():
        raise Phase2AnchoredCensusError("output and sibling staging must be absent")
    if args.expected_records <= 0:
        raise Phase2AnchoredCensusError("expected_records must be positive")
    config = _load_source_config(config_path, source)
    permitted_metadata = set(
        config["source"]["format_options"]["lmdb"]["metadata_keys_permitted"]
    )
    keys = _source_keys(source, args.expected_records, permitted_metadata)
    staging.mkdir(parents=False)
    started = time.perf_counter()

    motif_ids = array("I")
    offsets = array("Q", [0])
    source_keys = array("Q")
    anchor_counts = array("H")
    pure_to_id: dict[str, int] = {}
    occurrences: Counter[str] = Counter()
    molecule_counts: Counter[str] = Counter()
    rejects_path = staging / "rejects.jsonl"
    rejected = 0
    with rejects_path.open("x", encoding="utf-8", newline="\n") as rejects:
        results = _ordered_bounded_map(
            tuple(enumerate(keys)),
            args.workers,
            args.max_pending,
            str(source),
        )
        for processed, result in enumerate(results, 1):
            error = result["error"]
            if error is not None:
                rejected += 1
                rejects.write(
                    _canonical_json(
                        {
                            "selection_index": result["row_index"],
                            "source_key": result["source_key"],
                            "reason": error,
                        }
                    )
                    + "\n"
                )
                continue
            motifs = result["pure_motifs"]
            if not isinstance(motifs, tuple) or not motifs:
                raise Phase2AnchoredCensusError("projector returned no motifs")
            for pure in motifs:
                pure_id = pure_to_id.setdefault(pure, len(pure_to_id))
                motif_ids.append(pure_id)
                occurrences[pure] += 1
            molecule_counts.update(set(motifs))
            offsets.append(len(motif_ids))
            source_keys.append(int(result["source_key"]))
            anchor_count = int(result["model_facing_anchor_count"])
            if anchor_count > 65535:
                raise Phase2AnchoredCensusError("anchor count exceeds uint16")
            anchor_counts.append(anchor_count)
            if args.progress_every and processed % args.progress_every == 0:
                print(f"phase2-current-motif-census {processed}/{len(keys)}", flush=True)

    array_paths = {
        "motif_ids": staging / "motif_ids.u32",
        "offsets": staging / "offsets.u64",
        "source_keys": staging / "source_keys.u64",
        "anchor_counts": staging / "anchor_counts.u16",
    }
    for name, values in (
        ("motif_ids", motif_ids),
        ("offsets", offsets),
        ("source_keys", source_keys),
        ("anchor_counts", anchor_counts),
    ):
        with array_paths[name].open("xb") as handle:
            values.tofile(handle)

    registry_path = staging / "pure_motif_registry.jsonl"
    ranking = sorted(
        pure_to_id,
        key=lambda pure: (-occurrences[pure], pure.encode("utf-8")),
    )
    with registry_path.open("x", encoding="utf-8", newline="\n") as handle:
        for rank, pure in enumerate(ranking):
            handle.write(
                _canonical_json(
                    {
                        "rank": rank,
                        "pure_motif_id": pure_to_id[pure],
                        "pure_motif": pure,
                        "pure_motif_sha256": hashlib.sha256(
                            pure.encode("utf-8")
                        ).hexdigest(),
                        "occurrences": occurrences[pure],
                        "molecules": molecule_counts[pure],
                    }
                )
                + "\n"
            )

    passed = rejected == 0 and len(source_keys) == len(keys)
    artifacts = {
        name: _artifact(path)
        for name, path in {
            **array_paths,
            "pure_motif_registry": registry_path,
            "rejects": rejects_path,
        }.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if passed else "failed",
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "scope": "phase2_train_current_anchored_surface_vocabulary_evidence",
        "source": {
            "lmdb": str(source),
            "sha256": config["source"]["expected_sha256"],
            "config": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "split": "train",
            "smiles_field": "smiles",
        },
        "counts": {
            "scheduled_records": len(keys),
            "admitted_records": len(source_keys),
            "rejected_records": rejected,
            "pure_motif_occurrences": len(motif_ids),
            "unique_pure_motifs": len(pure_to_id),
        },
        "runtime": {
            "workers": args.workers,
            "max_pending": args.max_pending,
            "wall_seconds": time.perf_counter() - started,
        },
        "contracts": {
            "legacy_motif_seq_used": False,
            "stored_e3fp_recomputed": False,
            "text_fields_used_for_vocabulary": False,
            "validation_or_test_used_for_vocabulary": False,
            "current_frozen_linearizer_used": True,
            "molecule_local_anchor_ids_removed_before_counting": True,
            "training_admission": False,
        },
        "artifacts": artifacts,
    }
    with (staging / "manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    if not passed:
        raise Phase2AnchoredCensusError(
            f"Phase-II projection rejected {rejected} record(s); staging retained"
        )
    staging.rename(output)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lmdb", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-records", type=int, default=301655)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-pending", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=4096)
    parser.add_argument("--legacy-pickle-acknowledgement", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        manifest = build(_parser().parse_args(argv))
    except Exception as exc:
        print(f"Phase-II current motif census failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
