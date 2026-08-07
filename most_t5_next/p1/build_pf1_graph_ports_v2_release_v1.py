"""Derive a GraphPorts-v2 PF-1 release from one validated GraphPorts-v1 release.

Only the model-facing connection-table surface changes.  The source paired
wire is decoded through the authoritative loader, the endpoint-pair stream is
rebuilt from the persisted cross-motif bond table, and the derived wire is
decoded again before publication.  Atom SELFIES, motif identities, masks,
geometry, membership order, tokenizer IDs, and macro policy are inherited
verbatim.  No SDF, motif partition, E3FP, or vocabulary fitting is repeated.

This is a codec-screen artifact, not a new chemistry or training admission.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import shutil
from typing import Any, Iterable, Iterator, Mapping, Sequence

from most_t5_next.p1.build_pf1_paired_release_v1 import (
    DEV_MEMBERSHIP_NAME,
    LMDB_DIRECTORY,
    MACRO_REGISTRY_NAME,
    MANIFEST_NAME,
    PF1PairedReleaseError,
    REJECTS_NAME,
    SCHEMA_VERSION,
    TOKENIZER_DIRECTORY,
    TRAIN_MEMBERSHIP_NAME,
)
from most_t5_next.p1.runtime_bridge import P1ArtifactBindings, P1MemberRef
from most_t5_next.r1.adapter import paired_record_wire_v1 as paired_wire
from most_t5_next.r1.adapter import production_paired_identity_records_v1 as paired_identity
from most_t5_next.r1.tokenizer import production_graph_ports_codec_v1 as graph_v1
from most_t5_next.r1.tokenizer import production_graph_ports_codec_v2 as graph_v2


DERIVED_SCHEMA = "most-t5-next/pf1-graphports-v2-derived-release/v1"
DERIVED_SCOPE = "pf1_graphports_v2_codec_screen_candidate"
DEFAULT_WORKERS = 4
DEFAULT_MAX_PENDING = 16
DEFAULT_MAP_SIZE_MIB = 512
DEFAULT_COMMIT_EVERY = 512


class PF1GraphPortsV2ReleaseError(RuntimeError):
    """The v1-to-v2 derived release cannot be proved lossless."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PF1GraphPortsV2ReleaseError(f"cannot load JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PF1GraphPortsV2ReleaseError(f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json_bytes(value) + b"\n")


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise PF1GraphPortsV2ReleaseError(
                    f"blank JSONL row at {path}:{line_number}"
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PF1GraphPortsV2ReleaseError(
                    f"invalid JSONL row at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise PF1GraphPortsV2ReleaseError(
                    f"non-object JSONL row at {path}:{line_number}"
                )
            yield row


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_json_bytes(dict(row)) + b"\n")


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        raise PF1GraphPortsV2ReleaseError("cannot summarize an empty value set")
    ordered = sorted(values)

    def quantile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": ordered[-1],
        "mean": float(sum(ordered) / len(ordered)),
    }


def _directory_fingerprint(root: Path) -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return _sha256_bytes(_canonical_json_bytes(rows))


def _connections_from_document(
    motif_count: int,
    cross_bonds: object,
) -> tuple[graph_v1.ConnectionRecord, ...]:
    if not isinstance(cross_bonds, list):
        raise PF1GraphPortsV2ReleaseError("cross_motif_bonds must be one list")
    ordered = sorted(cross_bonds, key=lambda row: int(row["edge_id"]))
    connections: list[graph_v1.ConnectionRecord] = []
    for expected_zero_id, row in enumerate(ordered):
        if not isinstance(row, dict) or int(row.get("edge_id", -1)) != expected_zero_id:
            raise PF1GraphPortsV2ReleaseError(
                "cross-motif edge ids are not contiguous from zero"
            )
        if str(row.get("bond_type", "")).lower() != "single":
            raise PF1GraphPortsV2ReleaseError(
                "cross-motif edge is outside the SINGLE GraphPorts domain"
            )
        left = row.get("left")
        right = row.get("right")
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise PF1GraphPortsV2ReleaseError("cross-motif endpoint is malformed")
        endpoint_a = graph_v1.PortRef(
            int(left["logical_motif_index"]), int(left["slot_ordinal"]) + 1
        )
        endpoint_b = graph_v1.PortRef(
            int(right["logical_motif_index"]), int(right["slot_ordinal"]) + 1
        )
        if endpoint_b < endpoint_a:
            endpoint_a, endpoint_b = endpoint_b, endpoint_a
        if not (
            0 <= endpoint_a.motif_id < motif_count
            and 0 <= endpoint_b.motif_id < motif_count
        ):
            raise PF1GraphPortsV2ReleaseError(
                "cross-motif endpoint names an out-of-domain motif"
            )
        connections.append(
            graph_v1.ConnectionRecord(
                connection_id=expected_zero_id + 1,
                endpoint_a=endpoint_a,
                endpoint_b=endpoint_b,
                bond_type="SINGLE",
                bond_stereo="STEREONONE",
                stereo_atoms=None,
            )
        )
    return tuple(connections)


def _identity_graph_offset(motif_document: Mapping[str, object]) -> int:
    logical = motif_document["logical_motif_domain"]
    if not isinstance(logical, dict):
        raise PF1GraphPortsV2ReleaseError("logical_motif_domain is malformed")
    spans = logical.get("identity_spans")
    if not isinstance(spans, list) or not spans:
        raise PF1GraphPortsV2ReleaseError("identity_spans must be non-empty")
    cursor = 1
    for motif_id, span in enumerate(spans):
        if (
            not isinstance(span, list)
            or len(span) != 2
            or int(span[0]) != cursor
            or int(span[1]) <= cursor
        ):
            raise PF1GraphPortsV2ReleaseError(
                f"identity span {motif_id} is not one contiguous prefix segment"
            )
        cursor = int(span[1])
    return cursor


def _derived_identity_codec_sha256(
    *,
    parent_identity_codec_sha256: str,
    parent_connection_codec_sha256: str,
    replacement_connection_codec_sha256: str,
) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema_version": "pf1-derived-identity-codec-binding/v1",
                "parent_identity_codec_sha256": parent_identity_codec_sha256,
                "parent_connection_codec_sha256": parent_connection_codec_sha256,
                "replacement_connection_codec_sha256": (
                    replacement_connection_codec_sha256
                ),
                "transformation": "graph-surface-only-endpoint-pair-v2",
            }
        )
    )


def _normalize_expected_surface_differences(document: Mapping[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(dict(document))
    motif = normalized["motif_training_document"]
    motif["bindings"]["identity_codec_sha256"] = "<codec-variant>"
    motif["bindings"]["connection_codec_sha256"] = "<codec-variant>"
    motif["dimensions"]["token_count"] = "<surface-variant>"
    motif["token_domain"] = "<surface-variant>"
    motif["logical_motif_domain"]["connection_token_indices"] = (
        "<surface-variant>"
    )
    normalized["surface_summary"]["motif_input_token_count"] = (
        "<surface-variant>"
    )
    normalized["surface_summary"]["graph_token_count"] = "<surface-variant>"
    return normalized


@dataclass(frozen=True)
class TransformedWire:
    payload: bytes
    motif_input_token_count: int
    graph_token_count: int
    source_motif_input_token_count: int
    source_graph_token_count: int
    edge_count: int


def transform_paired_wire_to_graphports_v2(
    source_payload: bytes,
    *,
    declared_token_ids: Mapping[str, int],
    replacement_connection_codec_sha256: str,
) -> TransformedWire:
    """Strictly decode, reserialize only the graph suffix, and decode again."""

    source_loaded = paired_wire.decode_paired_training_record(source_payload)
    source_document = json.loads(source_payload)
    target = copy.deepcopy(source_document)
    motif = target["motif_training_document"]
    dimensions = motif["dimensions"]
    token_domain = motif["token_domain"]
    logical = motif["logical_motif_domain"]
    summary = target["surface_summary"]
    if not all(
        isinstance(value, dict)
        for value in (motif, dimensions, token_domain, logical, summary)
    ):
        raise PF1GraphPortsV2ReleaseError("paired motif document is malformed")

    motif_count = int(dimensions["logical_motif_count"])
    connections = _connections_from_document(
        motif_count, logical["cross_motif_bonds"]
    )
    components = graph_v1._connected_component_motifs(motif_count, connections)
    identity_mapping = tuple(range(motif_count))
    stream = graph_v2._build_endpoint_pair_graph_token_stream(
        components, connections, identity_mapping
    )
    decoded = graph_v2._decode_endpoint_pair_graph_token_stream(
        stream, components, identity_mapping
    )
    if decoded != connections:
        raise PF1GraphPortsV2ReleaseError(
            "derived v2 graph stream does not decode to the source connection table"
        )

    try:
        graph_ids = [int(declared_token_ids[token]) for token in stream.tokens]
    except (KeyError, TypeError, ValueError) as exc:
        raise PF1GraphPortsV2ReleaseError(
            "frozen union tokenizer does not cover the v2 graph stream"
        ) from exc
    if len(graph_ids) != len(stream.tokens) or any(value < 0 for value in graph_ids):
        raise PF1GraphPortsV2ReleaseError("v2 graph token ids are invalid")

    graph_offset = _identity_graph_offset(motif)
    source_ids = token_domain["input_ids"]
    source_roles = token_domain["token_role"]
    source_mapping = token_domain["token_to_logical_motif"]
    source_attention = token_domain["attention_mask"]
    if not all(
        isinstance(value, list)
        for value in (source_ids, source_roles, source_mapping, source_attention)
    ):
        raise PF1GraphPortsV2ReleaseError("source token arrays are malformed")
    if not (
        len(source_ids)
        == len(source_roles)
        == len(source_mapping)
        == len(source_attention)
        == int(dimensions["token_count"])
        == int(summary["motif_input_token_count"])
    ):
        raise PF1GraphPortsV2ReleaseError("source token-array lengths disagree")
    source_graph_count = len(source_ids) - graph_offset - 1
    if source_graph_count != int(summary["graph_token_count"]):
        raise PF1GraphPortsV2ReleaseError(
            "source graph suffix differs from its surface summary"
        )
    if any(value is not True for value in source_attention):
        raise PF1GraphPortsV2ReleaseError(
            "source uncorrupted record has a non-true attention entry"
        )

    token_domain["input_ids"] = [*source_ids[:graph_offset], *graph_ids, source_ids[-1]]
    token_domain["token_role"] = [
        *source_roles[:graph_offset],
        *stream.token_roles,
        source_roles[-1],
    ]
    token_domain["token_to_logical_motif"] = [
        *source_mapping[:graph_offset],
        *stream.token_to_logical_motif,
        source_mapping[-1],
    ]
    token_domain["attention_mask"] = [True] * len(token_domain["input_ids"])
    logical["connection_token_indices"] = [
        [graph_offset + index for index in row]
        for row in stream.connection_token_indices
    ]
    dimensions["token_count"] = len(token_domain["input_ids"])
    summary["graph_token_count"] = len(stream.tokens)
    summary["motif_input_token_count"] = len(token_domain["input_ids"])

    bindings_document = motif["bindings"]
    if not isinstance(bindings_document, dict):
        raise PF1GraphPortsV2ReleaseError("motif bindings are malformed")
    parent_identity = str(bindings_document["identity_codec_sha256"])
    parent_connection = str(bindings_document["connection_codec_sha256"])
    bindings_document["connection_codec_sha256"] = (
        replacement_connection_codec_sha256
    )
    bindings_document["identity_codec_sha256"] = _derived_identity_codec_sha256(
        parent_identity_codec_sha256=parent_identity,
        parent_connection_codec_sha256=parent_connection,
        replacement_connection_codec_sha256=replacement_connection_codec_sha256,
    )

    bindings = P1ArtifactBindings(**bindings_document)
    atom = source_loaded.atom_record
    target["atom_document"]["record_artifact_sha256"] = (
        paired_identity._atom_record_artifact_sha256(
            member=P1MemberRef(atom.record_id, atom.storage_key),
            bindings=bindings,
            alignment=atom,
            source_atom_count=atom.source_atom_count,
            model_to_source_atom_index=atom.model_to_source_atom_index,
            inherited_e3fp=atom.full_e3fp_ids,
        )
    )

    target_payload = _canonical_json_bytes(target)
    target_loaded = paired_wire.decode_paired_training_record(target_payload)
    if target_loaded.atom_record != source_loaded.atom_record:
        raise PF1GraphPortsV2ReleaseError(
            "GraphPorts transformation changed the atom/SELFIES record"
        )
    if (
        target_loaded.receipt != source_loaded.receipt
        or target_loaded.motif_record.identity_spans
        != source_loaded.motif_record.identity_spans
        or target_loaded.motif_record.logical_to_carrier
        != source_loaded.motif_record.logical_to_carrier
        or target_loaded.motif_record.exact_identity_sha256
        != source_loaded.motif_record.exact_identity_sha256
        or target_loaded.motif_record.full_e3fp_ids
        != source_loaded.motif_record.full_e3fp_ids
        or target_loaded.motif_record.model_to_source_atom_index
        != source_loaded.motif_record.model_to_source_atom_index
    ):
        raise PF1GraphPortsV2ReleaseError(
            "GraphPorts transformation changed identity, geometry, or receipt semantics"
        )
    if _normalize_expected_surface_differences(source_document) != (
        _normalize_expected_surface_differences(target)
    ):
        raise PF1GraphPortsV2ReleaseError(
            "derived paired wire changed a field outside the declared graph surface"
        )
    return TransformedWire(
        payload=target_payload,
        motif_input_token_count=len(token_domain["input_ids"]),
        graph_token_count=len(stream.tokens),
        source_motif_input_token_count=int(
            source_document["surface_summary"]["motif_input_token_count"]
        ),
        source_graph_token_count=int(
            source_document["surface_summary"]["graph_token_count"]
        ),
        edge_count=len(connections),
    )


_WORKER_DECLARED_TOKEN_IDS: dict[str, int] | None = None
_WORKER_CONNECTION_CODEC_SHA256: str | None = None


def _initialize_worker(
    declared_token_ids: Mapping[str, int],
    connection_codec_sha256: str,
) -> None:
    global _WORKER_DECLARED_TOKEN_IDS, _WORKER_CONNECTION_CODEC_SHA256
    _WORKER_DECLARED_TOKEN_IDS = dict(declared_token_ids)
    _WORKER_CONNECTION_CODEC_SHA256 = connection_codec_sha256


def _transform_worker(item: tuple[str, bytes]) -> tuple[str, TransformedWire]:
    if _WORKER_DECLARED_TOKEN_IDS is None or _WORKER_CONNECTION_CODEC_SHA256 is None:
        raise PF1GraphPortsV2ReleaseError("transform worker was not initialized")
    storage_key, payload = item
    return storage_key, transform_paired_wire_to_graphports_v2(
        payload,
        declared_token_ids=_WORKER_DECLARED_TOKEN_IDS,
        replacement_connection_codec_sha256=_WORKER_CONNECTION_CODEC_SHA256,
    )


def _bounded_ordered_map(
    executor: concurrent.futures.ProcessPoolExecutor,
    items: Iterable[tuple[str, bytes]],
    *,
    max_pending: int,
) -> Iterator[tuple[str, TransformedWire]]:
    pending: list[concurrent.futures.Future] = []
    for item in items:
        pending.append(executor.submit(_transform_worker, item))
        if len(pending) >= max_pending:
            yield pending.pop(0).result()
    while pending:
        yield pending.pop(0).result()


def _membership_rows(source_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train = list(_iter_jsonl(source_root / TRAIN_MEMBERSHIP_NAME))
    dev = list(_iter_jsonl(source_root / DEV_MEMBERSHIP_NAME))
    for split, rows in (("train", train), ("dev", dev)):
        for split_index, row in enumerate(rows):
            if row.get("split") != split or row.get("split_index") != split_index:
                raise PF1GraphPortsV2ReleaseError(
                    f"source {split} membership order is invalid"
                )
    return train, dev


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        import lmdb
    except ImportError as exc:
        raise PF1GraphPortsV2ReleaseError("python-lmdb is required") from exc

    source_root = Path(args.source_release).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    staging_root = output_root.with_name(output_root.name + ".staging")
    if output_root.exists() or staging_root.exists():
        raise PF1GraphPortsV2ReleaseError(
            "output and sibling staging paths must both be absent"
        )
    source_manifest_path = source_root / MANIFEST_NAME
    source_manifest = _load_json(source_manifest_path)
    if (
        source_manifest.get("schema_version") != SCHEMA_VERSION
        or source_manifest.get("status") != "pass"
    ):
        raise PF1GraphPortsV2ReleaseError(
            "source is not one passed PF-1 paired release"
        )
    train_rows, dev_rows = _membership_rows(source_root)
    source_counts = source_manifest.get("counts")
    if not isinstance(source_counts, dict) or not (
        source_counts.get("train_members") == len(train_rows)
        and source_counts.get("dev_members") == len(dev_rows)
        and source_counts.get("paired_records") == len(train_rows) + len(dev_rows)
    ):
        raise PF1GraphPortsV2ReleaseError(
            "source membership differs from its manifest counts"
        )

    tokenizer_manifest = _load_json(
        source_root / TOKENIZER_DIRECTORY / MANIFEST_NAME
    )
    contract = tokenizer_manifest.get("contract")
    if not isinstance(contract, dict) or not isinstance(
        contract.get("declared_token_ids"), dict
    ):
        raise PF1GraphPortsV2ReleaseError(
            "source tokenizer omits declared token ids"
        )
    declared_token_ids = {
        str(token): int(token_id)
        for token, token_id in contract["declared_token_ids"].items()
    }
    missing = [
        token for token in graph_v2.GPORTS_V2_UNION_TOKENS
        if token not in declared_token_ids
    ]
    if missing:
        raise PF1GraphPortsV2ReleaseError(
            f"source tokenizer omits v2 graph tokens: {missing!r}"
        )
    replacement_connection_sha = _sha256_file(Path(graph_v2.__file__).resolve())

    staging_root.mkdir(parents=True)
    shutil.copytree(
        source_root / TOKENIZER_DIRECTORY,
        staging_root / TOKENIZER_DIRECTORY,
    )
    shutil.copy2(
        source_root / MACRO_REGISTRY_NAME,
        staging_root / MACRO_REGISTRY_NAME,
    )
    (staging_root / REJECTS_NAME).write_bytes(b"")
    if _directory_fingerprint(source_root / TOKENIZER_DIRECTORY) != (
        _directory_fingerprint(staging_root / TOKENIZER_DIRECTORY)
    ):
        raise PF1GraphPortsV2ReleaseError("copied tokenizer snapshot changed bytes")
    if _sha256_file(source_root / MACRO_REGISTRY_NAME) != _sha256_file(
        staging_root / MACRO_REGISTRY_NAME
    ):
        raise PF1GraphPortsV2ReleaseError("copied macro registry changed bytes")

    source_environment = lmdb.open(
        str(source_root / LMDB_DIRECTORY),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=8,
    )
    target_environment = lmdb.open(
        str(staging_root / LMDB_DIRECTORY),
        subdir=True,
        map_size=int(args.lmdb_map_size_mib) * (1 << 20),
        lock=True,
        writemap=False,
        meminit=False,
        max_readers=8,
    )
    all_source_rows = train_rows + dev_rows
    transformed_by_key: dict[str, TransformedWire] = {}

    def source_items() -> Iterator[tuple[str, bytes]]:
        with source_environment.begin(write=False) as transaction:
            for row in all_source_rows:
                storage_key = str(row["storage_key"])
                raw = transaction.get(storage_key.encode("ascii"))
                if raw is None:
                    raise PF1GraphPortsV2ReleaseError(
                        "source LMDB omits a membership row"
                    )
                yield storage_key, bytes(raw)

    try:
        if args.workers == 1:
            _initialize_worker(declared_token_ids, replacement_connection_sha)
            results: Iterable[tuple[str, TransformedWire]] = (
                _transform_worker(item) for item in source_items()
            )
            executor = None
        else:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=int(args.workers),
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(declared_token_ids, replacement_connection_sha),
            )
            results = _bounded_ordered_map(
                executor,
                source_items(),
                max_pending=int(args.max_pending),
            )
        transaction = target_environment.begin(write=True)
        try:
            for index, (storage_key, transformed) in enumerate(results, start=1):
                expected_key = str(all_source_rows[index - 1]["storage_key"])
                if storage_key != expected_key or storage_key in transformed_by_key:
                    raise PF1GraphPortsV2ReleaseError(
                        "transform workers changed or duplicated membership order"
                    )
                if not transaction.put(
                    storage_key.encode("ascii"), transformed.payload, overwrite=False
                ):
                    raise PF1GraphPortsV2ReleaseError(
                        "derived LMDB contains a duplicate storage key"
                    )
                transformed_by_key[storage_key] = transformed
                if index % int(args.commit_every) == 0:
                    transaction.commit()
                    transaction = target_environment.begin(write=True)
            transaction.commit()
            transaction = None
        finally:
            if transaction is not None:
                transaction.abort()
            if executor is not None:
                executor.shutdown(wait=True)
        target_environment.sync(True)
    finally:
        source_environment.close()
        target_environment.close()

    if len(transformed_by_key) != len(all_source_rows):
        raise PF1GraphPortsV2ReleaseError(
            "derived LMDB did not close the source membership domain"
        )

    def transformed_membership(
        rows: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        output = []
        for row in rows:
            storage_key = str(row["storage_key"])
            transformed = transformed_by_key[storage_key]
            new_row = dict(row)
            new_row["motif_input_token_count"] = (
                transformed.motif_input_token_count
            )
            new_row["wire_bytes"] = len(transformed.payload)
            unchanged = dict(new_row)
            unchanged["motif_input_token_count"] = row["motif_input_token_count"]
            unchanged["wire_bytes"] = row["wire_bytes"]
            if unchanged != dict(row):
                raise PF1GraphPortsV2ReleaseError(
                    "derived membership changed a non-surface field"
                )
            output.append(new_row)
        return output

    derived_train = transformed_membership(train_rows)
    derived_dev = transformed_membership(dev_rows)
    _write_jsonl(staging_root / TRAIN_MEMBERSHIP_NAME, derived_train)
    _write_jsonl(staging_root / DEV_MEMBERSHIP_NAME, derived_dev)

    replay_environment = lmdb.open(
        str(staging_root / LMDB_DIRECTORY),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=4,
    )
    replayed = 0
    try:
        with replay_environment.begin(write=False) as transaction:
            for row in (*derived_train, *derived_dev):
                storage_key = str(row["storage_key"])
                raw = transaction.get(storage_key.encode("ascii"))
                if raw is None or len(raw) != int(row["wire_bytes"]):
                    raise PF1GraphPortsV2ReleaseError(
                        "derived replay differs from membership bytes"
                    )
                loaded = paired_wire.decode_paired_training_record(bytes(raw))
                if not (
                    loaded.atom_record.record_id == row["member_id"]
                    and loaded.schedule_index == row["selection_index"]
                    and loaded.sdf_record_index == row["sdf_record_index"]
                    and len(loaded.motif_record.input_ids)
                    == row["motif_input_token_count"]
                ):
                    raise PF1GraphPortsV2ReleaseError(
                        "derived replay differs from membership semantics"
                    )
                replayed += 1
    finally:
        replay_environment.close()
    if replayed != len(all_source_rows):
        raise PF1GraphPortsV2ReleaseError("derived full replay is incomplete")

    transformed_values = list(transformed_by_key.values())
    source_motif_lengths = [row.source_motif_input_token_count for row in transformed_values]
    target_motif_lengths = [row.motif_input_token_count for row in transformed_values]
    source_graph_lengths = [row.source_graph_token_count for row in transformed_values]
    target_graph_lengths = [row.graph_token_count for row in transformed_values]
    target_lmdb_file_bytes = (
        staging_root / LMDB_DIRECTORY / "data.mdb"
    ).stat().st_size
    stats_environment = lmdb.open(
        str(staging_root / LMDB_DIRECTORY),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    try:
        lmdb_info = stats_environment.info()
        lmdb_stats = stats_environment.stat()
        target_lmdb_logical_bytes = (
            int(lmdb_info["last_pgno"]) + 1
        ) * int(lmdb_stats["psize"])
    finally:
        stats_environment.close()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "derived_schema_version": DERIVED_SCHEMA,
        "status": "pass",
        "scope": DERIVED_SCOPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_admission": False,
        "codec_screen_candidate": True,
        "counts": {
            "scheduled_members": len(all_source_rows),
            "paired_records": len(all_source_rows),
            "train_members": len(train_rows),
            "dev_members": len(dev_rows),
            "rejected_members": 0,
        },
        "source_release": {
            "manifest_sha256": _sha256_file(source_manifest_path),
            "schema_version": source_manifest["schema_version"],
            "scope": source_manifest.get("scope"),
            "status": source_manifest["status"],
        },
        "codec": {
            "source_format_version": graph_v1.FORMAT_VERSION,
            "target_format_version": graph_v2.FORMAT_VERSION,
            "target_source_sha256": replacement_connection_sha,
            "target_required_union_tokens": len(graph_v2.GPORTS_V2_UNION_TOKENS),
            "new_token_ids": 0,
        },
        "paired_invariants": {
            "same_member_and_split_order": True,
            "same_atom_selfies_record": True,
            "same_motif_identity_spans_and_carriers": True,
            "same_cross_motif_bond_table": True,
            "same_mask_decision": True,
            "same_geometry_and_e3fp": True,
            "same_receipt": True,
            "same_macro_registry": True,
            "same_union_tokenizer_snapshot": True,
            "identity_targets_equal_for_every_corruption_epoch_by_construction": True,
        },
        "lengths": {
            "motif_surface_v1": _distribution(source_motif_lengths),
            "motif_surface_v2": _distribution(target_motif_lengths),
            "graph_surface_v1": _distribution(source_graph_lengths),
            "graph_surface_v2": _distribution(target_graph_lengths),
            "mean_motif_fraction_reduction": (
                1.0 - sum(target_motif_lengths) / sum(source_motif_lengths)
            ),
            "mean_graph_fraction_reduction": (
                1.0 - sum(target_graph_lengths) / sum(source_graph_lengths)
            ),
        },
        "artifacts": {
            "paired_lmdb": {
                "relative_path": LMDB_DIRECTORY,
                "entry_count": len(all_source_rows),
                "data_mdb_file_bytes": target_lmdb_file_bytes,
                "logical_used_bytes": target_lmdb_logical_bytes,
                "configured_map_size_mib": int(args.lmdb_map_size_mib),
            },
            "train_membership": {
                "relative_path": TRAIN_MEMBERSHIP_NAME,
                "row_count": len(derived_train),
            },
            "dev_membership": {
                "relative_path": DEV_MEMBERSHIP_NAME,
                "row_count": len(derived_dev),
            },
            "rejects": {"relative_path": REJECTS_NAME, "row_count": 0},
            "macro_registry": {
                "relative_path": MACRO_REGISTRY_NAME,
                "sha256": _sha256_file(staging_root / MACRO_REGISTRY_NAME),
            },
            "union_tokenizer": {
                "relative_path": TOKENIZER_DIRECTORY,
                "directory_fingerprint_sha256": _directory_fingerprint(
                    staging_root / TOKENIZER_DIRECTORY
                ),
                "tokenizer_contract_sha256": tokenizer_manifest[
                    "tokenizer_contract_sha256"
                ],
                "tokenizer_snapshot_sha256": tokenizer_manifest[
                    "tokenizer_snapshot_sha256"
                ],
            },
        },
        "replay": {
            "strict_source_decode_records": len(all_source_rows),
            "strict_derived_decode_records": replayed,
            "zero_rejects": True,
        },
        "method_boundary": {
            "derived_from_published_v1_wire": True,
            "source_sdf_read": False,
            "motif_partition_recomputed": False,
            "e3fp_recomputed": False,
            "selfies_recomputed": False,
            "macro_or_vocabulary_refit": False,
            "graph_surface_only": True,
            "sequence_truncation": False,
            "no_replacement": True,
        },
        "runtime": {
            "workers": int(args.workers),
            "max_pending": int(args.max_pending),
        },
    }
    _write_json(staging_root / MANIFEST_NAME, manifest)
    staging_root.rename(output_root)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-pending", type=int, default=DEFAULT_MAX_PENDING)
    parser.add_argument("--lmdb-map-size-mib", type=int, default=DEFAULT_MAP_SIZE_MIB)
    parser.add_argument("--commit-every", type=int, default=DEFAULT_COMMIT_EVERY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.workers <= 0
        or args.max_pending < args.workers
        or args.lmdb_map_size_mib <= 0
        or args.commit_every <= 0
    ):
        parser.error(
            "workers/map-size/commit-every must be positive and max-pending must cover workers"
        )
    manifest = run(args)
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PF1GraphPortsV2ReleaseError",
    "TransformedWire",
    "transform_paired_wire_to_graphports_v2",
    "run",
]
