"""Compiled mmap training ABI for PF-10 motif/E3FP records.

The authoritative paired LMDB remains the scientific source.  This module
compiles its already validated, deterministic fields into flat binary arrays
plus offsets.  Epoch/view corruption, sentinel construction, padding, and
dropout remain online operations owned by the existing V3 collator/model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from most_t5_next.p1.build_pf1_paired_release_v1 import (
    DONOR_ATOM_MAP_NAME,
    MANIFEST_NAME as PAIRED_MANIFEST_NAME,
    PF1PairedReleaseReader,
)
from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
)
from most_t5_next.p1.bound_record import Span

from .build_pf10_morgan_overlay_v1 import (
    MANIFEST_NAME as MORGAN_MANIFEST_NAME,
    MorganAtomStateProvider,
)
from .three_d_motif_training_views_v3 import collate_3d_motif_training_view_v3


SCHEMA_VERSION = "most-t5-p2/pf10-training-tensor-cache/v1"
MANIFEST_NAME = "manifest.json"
ROLE_TO_ID = {"boundary": 0, "identity": 1, "connection": 2}
ID_TO_ROLE = tuple(ROLE_TO_ID)
BUFFER_BYTES = 8 * 1024 * 1024


ARRAY_DTYPES = {
    "input_ids": "<i4",
    "token_to_motif": "<i4",
    "token_role": "u1",
    "endpoint_token_to_atom": "<i4",
    "identity_spans": "<i4",
    "logical_to_carrier": "<i4",
    "exact_identity_sha256": "u1",
    "e3fp": "<i4",
    "morgan": "<i4",
    "atom_valid": "u1",
    "atom_to_motif": "<i4",
    "atom_is_attachment": "u1",
    "model_to_source_atom": "<i4",
    "atom_local_position": "<i4",
    "record_id_bytes": "u1",
    "storage_key_bytes": "u1",
    "record_artifact_sha256": "u1",
    "geometry_sha256": "u1",
    "source_atom_count": "<i4",
    "token_offsets": "<i8",
    "motif_offsets": "<i8",
    "atom_offsets": "<i8",
    "record_id_offsets": "<i8",
    "storage_key_offsets": "<i8",
    "train_indices": "<i8",
    "dev_indices": "<i8",
}


class PF10TrainingTensorCacheError(RuntimeError):
    """The derived training ABI is incomplete or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _BufferedArrayWriter:
    def __init__(self, path: Path, dtype: str) -> None:
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.handle = self.path.open("wb", buffering=BUFFER_BYTES)
        self.elements = 0

    def write(self, values: object) -> None:
        array = np.asarray(values, dtype=self.dtype)
        self.handle.write(array.tobytes(order="C"))
        self.elements += int(array.size)

    def write_bytes(self, value: bytes) -> None:
        if self.dtype != np.dtype("u1"):
            raise PF10TrainingTensorCacheError("byte writes require uint8 arrays")
        self.handle.write(value)
        self.elements += len(value)

    def close(self) -> None:
        self.handle.close()


def _hex_bytes(value: object, name: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise PF10TrainingTensorCacheError(f"{name} is not one SHA-256")
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise PF10TrainingTensorCacheError(f"{name} is not hexadecimal") from exc
    if len(result) != 32:
        raise PF10TrainingTensorCacheError(f"{name} has the wrong width")
    return result


def _canonical_local_positions(
    donor_row: Mapping[str, object], record: ProductionMotifRecord
) -> tuple[int, ...]:
    if (
        donor_row.get("member_id") != record.record_id
        or donor_row.get("storage_key") != record.storage_key
    ):
        raise PF10TrainingTensorCacheError(
            "donor atom-map order differs from paired records"
        )
    try:
        sidecar = donor_row["overlay_planning_sidecar"]
        maps = sidecar["canonical_local_atom_to_model_atom"]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise PF10TrainingTensorCacheError("donor atom-map row is malformed") from exc
    positions = [-1] * len(record.atom_valid_mask)
    owners = [-1] * len(record.atom_valid_mask)
    for motif_id, atom_map in enumerate(maps):
        for local_id, model_atom in atom_map:
            index = int(model_atom)
            if not 0 <= index < len(positions) or positions[index] != -1:
                raise PF10TrainingTensorCacheError(
                    "donor atom-map does not form one model-atom partition"
                )
            positions[index] = int(local_id) - 1
            owners[index] = motif_id
    if (
        -1 in positions
        or tuple(owners) != tuple(record.atom_to_logical_motif)
        or any(value < 0 for value in positions)
    ):
        raise PF10TrainingTensorCacheError(
            "donor atom-map ownership differs from the paired record"
        )
    return tuple(positions)


def _selected_rows(
    reader: PF1PairedReleaseReader,
    *,
    split: str,
    limit: int | None,
    workers: int = 0,
    max_pending: int = 128,
    batch_size: int = 256,
) -> Iterator[Any]:
    if workers:
        parallel = getattr(reader, "iter_strict_parallel_split", None)
        if not callable(parallel):
            raise PF10TrainingTensorCacheError(
                "reader lacks bounded selected-row parallel decode"
            )
        yield from parallel(
            split=split,
            max_rows=limit,
            workers=workers,
            max_pending=max_pending,
        )
        return
    source = (
        reader.iter_train_epoch(epoch=0, batch_size=batch_size)
        if split == "train"
        else reader.iter_dev(batch_size=batch_size)
    )
    yielded = 0
    for batch in source:
        for row in batch:
            if limit is not None and yielded >= limit:
                return
            yield row
            yielded += 1


def build_pf10_training_tensor_cache(
    *,
    paired_release: Path,
    morgan_overlay: Path,
    output_dir: Path,
    max_train_records: int | None = None,
    max_dev_records: int | None = None,
    decode_workers: int = 0,
    decode_max_pending: int = 128,
    reader_factory: Any = PF1PairedReleaseReader,
    morgan_provider_factory: Any = MorganAtomStateProvider,
    source_extensions: Mapping[str, object] | None = None,
    donor_atom_maps_path: Path | None = None,
) -> dict[str, object]:
    """Compile one immutable training cache without freezing runtime masks."""

    for name, value in (
        ("max_train_records", max_train_records),
        ("max_dev_records", max_dev_records),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise PF10TrainingTensorCacheError(f"{name} must be positive")
    if (
        isinstance(decode_workers, bool)
        or not isinstance(decode_workers, int)
        or decode_workers < 0
        or isinstance(decode_max_pending, bool)
        or not isinstance(decode_max_pending, int)
        or decode_max_pending <= 0
        or (decode_workers > 0 and decode_max_pending < decode_workers)
    ):
        raise PF10TrainingTensorCacheError("parallel decode settings are invalid")
    if source_extensions is not None:
        if not isinstance(source_extensions, Mapping):
            raise PF10TrainingTensorCacheError("source_extensions must be a mapping")
        try:
            source_extensions = json.loads(
                json.dumps(
                    dict(source_extensions),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PF10TrainingTensorCacheError(
                "source_extensions must be canonical-JSON serializable"
            ) from exc
    paired_release = Path(paired_release).expanduser().resolve()
    morgan_overlay = Path(morgan_overlay).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    staging = output_dir.with_name(output_dir.name + ".staging")
    if output_dir.exists() or staging.exists():
        raise PF10TrainingTensorCacheError("output or staging path already exists")
    staging.mkdir(parents=True)

    paired_manifest = paired_release / PAIRED_MANIFEST_NAME
    donor_path = (
        Path(donor_atom_maps_path).expanduser().resolve()
        if donor_atom_maps_path is not None
        else paired_release / DONOR_ATOM_MAP_NAME
    )
    morgan_manifest = morgan_overlay / MORGAN_MANIFEST_NAME
    if not paired_manifest.is_file() or not donor_path.is_file() or not morgan_manifest.is_file():
        raise PF10TrainingTensorCacheError("one authoritative source artifact is absent")

    reader = reader_factory(paired_release)
    morgan = morgan_provider_factory(morgan_overlay)
    stream_names = tuple(
        name
        for name in ARRAY_DTYPES
        if name
        not in {
            "token_offsets",
            "motif_offsets",
            "atom_offsets",
            "record_id_offsets",
            "storage_key_offsets",
            "train_indices",
            "dev_indices",
            "source_atom_count",
        }
    )
    writers = {
        name: _BufferedArrayWriter(staging / f"{name}.bin", ARRAY_DTYPES[name])
        for name in stream_names
    }
    token_offsets = [0]
    motif_offsets = [0]
    atom_offsets = [0]
    record_id_offsets = [0]
    storage_key_offsets = [0]
    source_atom_counts: list[int] = []
    split_indices: dict[str, list[int]] = {"train": [], "dev": []}
    releases: set[str] = set()
    tokenizer_contracts: set[str] = set()
    tokenizer_snapshots: set[str] = set()
    record_count = 0
    started = __import__("time").perf_counter()

    try:
        for split, limit in (
            ("train", max_train_records),
            ("dev", max_dev_records),
        ):
            donor_rows = iter(reader.iter_donor_atom_maps(split=split, max_rows=limit))
            split_seen = 0
            for loaded in _selected_rows(
                reader,
                split=split,
                limit=limit,
                workers=decode_workers,
                max_pending=decode_max_pending,
            ):
                record = getattr(loaded, "motif_record", None)
                if not isinstance(record, ProductionMotifRecord):
                    raise PF10TrainingTensorCacheError(
                        "paired reader row lacks a production motif record"
                    )
                try:
                    donor_row = next(donor_rows)
                except StopIteration as exc:
                    raise PF10TrainingTensorCacheError(
                        "donor atom-map ends before paired records"
                    ) from exc
                positions = _canonical_local_positions(donor_row, record)
                morgan_rows = np.asarray(morgan.get(record.record_id), dtype=np.int32)
                e3fp_rows = np.asarray(record.full_e3fp_ids, dtype=np.int32)
                if morgan_rows.shape != e3fp_rows.shape or e3fp_rows.ndim != 2 or e3fp_rows.shape[1] != 4:
                    raise PF10TrainingTensorCacheError(
                        "Morgan and E3FP atom-state matrices differ"
                    )
                roles = []
                for role in record.token_role:
                    if role not in ROLE_TO_ID:
                        raise PF10TrainingTensorCacheError(
                            "production token role is outside the cache vocabulary"
                        )
                    roles.append(ROLE_TO_ID[role])
                if len(record.connection_token_to_atom) != len(record.input_ids):
                    raise PF10TrainingTensorCacheError(
                        "V3 endpoint address is absent from a production record"
                    )

                writers["input_ids"].write(record.input_ids)
                writers["token_to_motif"].write(record.token_to_logical_motif)
                writers["token_role"].write(roles)
                writers["endpoint_token_to_atom"].write(
                    record.connection_token_to_atom
                )
                writers["identity_spans"].write(
                    [(span.start, span.stop) for span in record.identity_spans]
                )
                writers["logical_to_carrier"].write(record.logical_to_carrier)
                for value in record.exact_identity_sha256:
                    writers["exact_identity_sha256"].write_bytes(
                        _hex_bytes(value, "exact_identity_sha256")
                    )
                writers["e3fp"].write(e3fp_rows)
                writers["morgan"].write(morgan_rows)
                writers["atom_valid"].write(record.atom_valid_mask)
                writers["atom_to_motif"].write(record.atom_to_logical_motif)
                writers["atom_is_attachment"].write(record.atom_is_attachment)
                writers["model_to_source_atom"].write(
                    record.model_to_source_atom_index
                )
                writers["atom_local_position"].write(positions)
                record_id_bytes = record.record_id.encode("utf-8")
                storage_key_bytes = record.storage_key.encode("utf-8")
                writers["record_id_bytes"].write_bytes(record_id_bytes)
                writers["storage_key_bytes"].write_bytes(storage_key_bytes)
                writers["record_artifact_sha256"].write_bytes(
                    _hex_bytes(record.record_artifact_sha256, "record_artifact_sha256")
                )
                writers["geometry_sha256"].write_bytes(
                    _hex_bytes(
                        record.geometry_record_content_sha256,
                        "geometry_record_content_sha256",
                    )
                )

                token_offsets.append(token_offsets[-1] + len(record.input_ids))
                motif_offsets.append(motif_offsets[-1] + len(record.identity_spans))
                atom_offsets.append(atom_offsets[-1] + len(record.atom_valid_mask))
                record_id_offsets.append(record_id_offsets[-1] + len(record_id_bytes))
                storage_key_offsets.append(
                    storage_key_offsets[-1] + len(storage_key_bytes)
                )
                source_atom_counts.append(int(record.source_atom_count))
                split_indices[split].append(record_count)
                releases.add(record.release_id)
                tokenizer_contracts.add(record.tokenizer_contract_sha256)
                tokenizer_snapshots.add(record.tokenizer_snapshot_sha256)
                record_count += 1
                split_seen += 1
            try:
                next(donor_rows)
            except StopIteration:
                pass
            else:
                raise PF10TrainingTensorCacheError(
                    "donor atom-map contains rows beyond selected paired records"
                )
            expected = limit
            if expected is None:
                expected = (
                    reader.train_member_count if split == "train" else reader.dev_member_count
                )
            if split_seen != expected:
                raise PF10TrainingTensorCacheError(
                    f"{split} cache row count differs from selection"
                )
    finally:
        for writer in writers.values():
            writer.close()
        close = getattr(morgan, "close", None)
        if callable(close):
            close()

    if (
        record_count <= 0
        or len(releases) != 1
        or len(tokenizer_contracts) != 1
        or len(tokenizer_snapshots) != 1
    ):
        raise PF10TrainingTensorCacheError(
            "cache-wide release/tokenizer identity is not singular"
        )
    fixed_arrays = {
        "token_offsets": token_offsets,
        "motif_offsets": motif_offsets,
        "atom_offsets": atom_offsets,
        "record_id_offsets": record_id_offsets,
        "storage_key_offsets": storage_key_offsets,
        "source_atom_count": source_atom_counts,
        "train_indices": split_indices["train"],
        "dev_indices": split_indices["dev"],
    }
    for name, values in fixed_arrays.items():
        np.asarray(values, dtype=ARRAY_DTYPES[name]).tofile(staging / f"{name}.bin")

    total_tokens = token_offsets[-1]
    total_motifs = motif_offsets[-1]
    total_atoms = atom_offsets[-1]
    shapes = {
        "input_ids": [total_tokens],
        "token_to_motif": [total_tokens],
        "token_role": [total_tokens],
        "endpoint_token_to_atom": [total_tokens],
        "identity_spans": [total_motifs, 2],
        "logical_to_carrier": [total_motifs],
        "exact_identity_sha256": [total_motifs, 32],
        "e3fp": [total_atoms, 4],
        "morgan": [total_atoms, 4],
        "atom_valid": [total_atoms],
        "atom_to_motif": [total_atoms],
        "atom_is_attachment": [total_atoms],
        "model_to_source_atom": [total_atoms],
        "atom_local_position": [total_atoms],
        "record_id_bytes": [record_id_offsets[-1]],
        "storage_key_bytes": [storage_key_offsets[-1]],
        "record_artifact_sha256": [record_count, 32],
        "geometry_sha256": [record_count, 32],
        "source_atom_count": [record_count],
        "token_offsets": [record_count + 1],
        "motif_offsets": [record_count + 1],
        "atom_offsets": [record_count + 1],
        "record_id_offsets": [record_count + 1],
        "storage_key_offsets": [record_count + 1],
        "train_indices": [len(split_indices["train"])],
        "dev_indices": [len(split_indices["dev"])],
    }
    arrays = {}
    for name, shape in shapes.items():
        path = staging / f"{name}.bin"
        expected_bytes = int(np.prod(shape)) * np.dtype(ARRAY_DTYPES[name]).itemsize
        if path.stat().st_size != expected_bytes:
            raise PF10TrainingTensorCacheError(
                f"compiled array {name} has the wrong byte length"
            )
        arrays[name] = {
            "file": path.name,
            "dtype": ARRAY_DTYPES[name],
            "shape": shape,
            "bytes": expected_bytes,
            "sha256": _sha256_file(path),
        }

    elapsed = __import__("time").perf_counter() - started
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "role_vocabulary": ROLE_TO_ID,
        "counts": {
            "records": record_count,
            "train_records": len(split_indices["train"]),
            "dev_records": len(split_indices["dev"]),
            "tokens": total_tokens,
            "motifs": total_motifs,
            "atoms": total_atoms,
        },
        "source": {
            "paired_manifest_sha256": _sha256_file(paired_manifest),
            "donor_atom_maps_sha256": _sha256_file(donor_path),
            "morgan_manifest_sha256": _sha256_file(morgan_manifest),
            "release_id": next(iter(releases)),
            "tokenizer_contract_sha256": next(iter(tokenizer_contracts)),
            "tokenizer_snapshot_sha256": next(iter(tokenizer_snapshots)),
            **(
                {"derived_representation": source_extensions}
                if source_extensions is not None
                else {}
            ),
        },
        "arrays": arrays,
        "runtime_boundary": {
            "epoch_corruption_cached": False,
            "sentinel_outputs_cached": False,
            "padded_batches_cached": False,
            "training_views_cached": False,
            "dropout_cached": False,
            "dynamic_corruption_required": True,
            "authoritative_release_replaced": False,
        },
        "build": {
            "seconds": elapsed,
            "records_per_second": record_count / elapsed,
            "atomic_staging_rename": True,
            "decode_workers": decode_workers,
            "decode_max_pending": decode_max_pending,
            "strict_ordered_selected_decode": True,
            "decoded_records_retained_in_python_cache": False,
        },
    }
    (staging / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.rename(output_dir)
    return manifest


@dataclass(frozen=True)
class CachedProductionMotifRecord(ProductionMotifRecord):
    cache_index: int = -1


class PF10TrainingTensorCache(torch.utils.data.Dataset):
    """Read-only mmap Dataset which recreates lightweight production rows."""

    def __init__(self, root: Path, *, verify_hashes: bool = True) -> None:
        self.root = Path(root).expanduser().resolve()
        try:
            manifest = json.loads((self.root / MANIFEST_NAME).read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PF10TrainingTensorCacheError("cache manifest is unreadable") from exc
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "pass":
            raise PF10TrainingTensorCacheError("cache is not a passed V1 artifact")
        if manifest.get("role_vocabulary") != ROLE_TO_ID:
            raise PF10TrainingTensorCacheError("cache token-role vocabulary differs")
        self.manifest = manifest
        self.arrays: dict[str, np.memmap] = {}
        for name, spec in manifest["arrays"].items():
            path = self.root / spec["file"]
            dtype = np.dtype(spec["dtype"])
            shape = tuple(int(value) for value in spec["shape"])
            expected = int(np.prod(shape)) * dtype.itemsize
            if not path.is_file() or path.stat().st_size != expected:
                raise PF10TrainingTensorCacheError(
                    f"cache array {name} is absent or truncated"
                )
            if verify_hashes and _sha256_file(path) != spec.get("sha256"):
                raise PF10TrainingTensorCacheError(
                    f"cache array {name} differs from its manifest digest"
                )
            self.arrays[name] = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        self.release_id = str(manifest["source"]["release_id"])
        self.tokenizer_contract_sha256 = str(
            manifest["source"]["tokenizer_contract_sha256"]
        )
        self.tokenizer_snapshot_sha256 = str(
            manifest["source"]["tokenizer_snapshot_sha256"]
        )

    def __getstate__(self) -> dict[str, object]:
        return {"root": self.root}

    def __setstate__(self, state: Mapping[str, object]) -> None:
        # The parent process verified the immutable files once. Spawned loader
        # workers only reopen their mmap handles and do not repeat whole-file I/O.
        self.__init__(Path(state["root"]), verify_hashes=False)

    def __len__(self) -> int:
        return int(self.manifest["counts"]["records"])

    def close(self) -> None:
        """Release mmap handles explicitly (needed before Windows cleanup)."""

        for array in self.arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.arrays.clear()

    def _bytes(self, name: str, offsets: str, index: int) -> bytes:
        bounds = self.arrays[offsets]
        start, stop = int(bounds[index]), int(bounds[index + 1])
        return bytes(self.arrays[name][start:stop])

    def record_id_at(self, index: int) -> str:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        return self._bytes("record_id_bytes", "record_id_offsets", index).decode(
            "utf-8"
        )

    def _slice(self, name: str, offsets: str, index: int) -> np.ndarray:
        bounds = self.arrays[offsets]
        start, stop = int(bounds[index]), int(bounds[index + 1])
        return self.arrays[name][start:stop]

    def __getitem__(self, index: int) -> CachedProductionMotifRecord:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        token_start = int(self.arrays["token_offsets"][index])
        token_stop = int(self.arrays["token_offsets"][index + 1])
        motif_start = int(self.arrays["motif_offsets"][index])
        motif_stop = int(self.arrays["motif_offsets"][index + 1])
        atom_start = int(self.arrays["atom_offsets"][index])
        atom_stop = int(self.arrays["atom_offsets"][index + 1])
        token_roles = tuple(
            ID_TO_ROLE[int(value)]
            for value in self.arrays["token_role"][token_start:token_stop]
        )
        token_owners = tuple(
            int(value)
            for value in self.arrays["token_to_motif"][token_start:token_stop]
        )
        motif_count = motif_stop - motif_start
        connection_indices = tuple(
            tuple(
                offset
                for offset, (role, owner) in enumerate(zip(token_roles, token_owners))
                if role == "connection" and owner == motif_id
            )
            for motif_id in range(motif_count)
        )
        spans = tuple(
            Span(int(row[0]), int(row[1]))
            for row in self.arrays["identity_spans"][motif_start:motif_stop]
        )
        exact_hashes = tuple(
            bytes(row).hex()
            for row in self.arrays["exact_identity_sha256"][motif_start:motif_stop]
        )
        return CachedProductionMotifRecord(
            record_artifact_sha256=bytes(
                self.arrays["record_artifact_sha256"][index]
            ).hex(),
            record_id=self.record_id_at(index),
            storage_key=self._bytes(
                "storage_key_bytes", "storage_key_offsets", index
            ).decode("utf-8"),
            release_id=self.release_id,
            geometry_record_content_sha256=bytes(
                self.arrays["geometry_sha256"][index]
            ).hex(),
            tokenizer_contract_sha256=self.tokenizer_contract_sha256,
            tokenizer_snapshot_sha256=self.tokenizer_snapshot_sha256,
            input_ids=tuple(
                int(value) for value in self.arrays["input_ids"][token_start:token_stop]
            ),
            token_to_logical_motif=token_owners,
            token_role=token_roles,
            identity_spans=spans,
            connection_token_indices=connection_indices,
            logical_to_carrier=tuple(
                int(value)
                for value in self.arrays["logical_to_carrier"][motif_start:motif_stop]
            ),
            exact_identity_sha256=exact_hashes,
            source_atom_count=int(self.arrays["source_atom_count"][index]),
            full_e3fp_ids=tuple(
                tuple(int(value) for value in row)
                for row in self.arrays["e3fp"][atom_start:atom_stop]
            ),
            atom_valid_mask=tuple(
                bool(value) for value in self.arrays["atom_valid"][atom_start:atom_stop]
            ),
            model_to_source_atom_index=tuple(
                int(value)
                for value in self.arrays["model_to_source_atom"][atom_start:atom_stop]
            ),
            atom_to_logical_motif=tuple(
                int(value)
                for value in self.arrays["atom_to_motif"][atom_start:atom_stop]
            ),
            atom_is_attachment=tuple(
                bool(value)
                for value in self.arrays["atom_is_attachment"][atom_start:atom_stop]
            ),
            connection_token_to_atom=tuple(
                int(value)
                for value in self.arrays["endpoint_token_to_atom"][token_start:token_stop]
            ),
            cache_index=index,
        )

    def split_indices(self, split: str) -> tuple[int, ...]:
        if split not in {"train", "dev"}:
            raise PF10TrainingTensorCacheError("split must be train or dev")
        return tuple(int(value) for value in self.arrays[f"{split}_indices"])

    def atom_local_positions(self, record: ProductionMotifRecord) -> tuple[int, ...]:
        index = getattr(record, "cache_index", -1)
        if not 0 <= index < len(self):
            raise PF10TrainingTensorCacheError("record is not owned by this cache")
        row = self._slice("atom_local_position", "atom_offsets", index)
        return tuple(int(value) for value in row)

    def morgan_state(
        self, record_id: str, *, cache_index: int
    ) -> tuple[tuple[int, int, int, int], ...]:
        if not 0 <= cache_index < len(self):
            raise PF10TrainingTensorCacheError("Morgan cache index is invalid")
        if self.record_id_at(cache_index) != record_id:
            raise PF10TrainingTensorCacheError("Morgan record identity differs")
        return tuple(
            tuple(int(value) for value in row)
            for row in self._slice("morgan", "atom_offsets", cache_index)
        )


class CachedCanonicalAtomAddressProvider:
    def __init__(self, cache: PF10TrainingTensorCache) -> None:
        self.cache = cache

    def get(self, record: ProductionMotifRecord) -> tuple[int, ...]:
        return self.cache.atom_local_positions(record)


class CachedMorganAtomStateProvider:
    state_kind = "most-t5-p2/coordinate-blind-morgan-atom-state/r3-fp4096-v1"

    def __init__(
        self,
        cache: PF10TrainingTensorCache,
        records: Sequence[CachedProductionMotifRecord],
    ) -> None:
        self.cache = cache
        self.indices = {record.record_id: record.cache_index for record in records}
        if len(self.indices) != len(tuple(records)):
            raise PF10TrainingTensorCacheError("one cache batch repeats a record")

    def get(self, record_id: str) -> tuple[tuple[int, int, int, int], ...]:
        try:
            index = self.indices[record_id]
        except KeyError as exc:
            raise PF10TrainingTensorCacheError("Morgan record is absent") from exc
        return self.cache.morgan_state(record_id, cache_index=index)


@dataclass(frozen=True)
class CacheTrainingIndex:
    record_index: int
    epoch: int
    view_id: str


@dataclass(frozen=True)
class CacheTrainingSample:
    record: CachedProductionMotifRecord
    epoch: int
    view_id: str


class IndexedPF10TrainingTensorCache(PF10TrainingTensorCache):
    def __getitem__(self, index: object) -> object:
        if isinstance(index, CacheTrainingIndex):
            return CacheTrainingSample(
                record=super().__getitem__(index.record_index),
                epoch=index.epoch,
                view_id=index.view_id,
            )
        return super().__getitem__(index)  # type: ignore[arg-type]


class V3EpochViewBatchSampler:
    """Finite microbatch schedule with short tails and epoch-aware views."""

    def __init__(
        self,
        indices: Sequence[int],
        *,
        cell: str,
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        total_updates: int,
        shuffle_seed: int | None = None,
        fixed_view_id: str | None = None,
    ) -> None:
        from .run_pf10_3d_motif_v3_short_v1 import view_for_update

        self.indices = tuple(int(value) for value in indices)
        self.cell = cell
        self.micro_batch_size = int(micro_batch_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.total_updates = int(total_updates)
        self.shuffle_seed = shuffle_seed
        self.fixed_view_id = fixed_view_id
        self._view_for_update = view_for_update
        if (
            not self.indices
            or len(set(self.indices)) != len(self.indices)
            or self.micro_batch_size <= 0
            or self.gradient_accumulation_steps <= 0
            or self.total_updates <= 0
        ):
            raise PF10TrainingTensorCacheError("training sampler contract is invalid")
        if fixed_view_id is not None and fixed_view_id not in {
            "m_only",
            "m_plus_g",
            "g_only",
        }:
            raise PF10TrainingTensorCacheError("fixed training view is invalid")
        if shuffle_seed is not None and (
            isinstance(shuffle_seed, bool)
            or not isinstance(shuffle_seed, int)
            or not 0 <= shuffle_seed < 2**64
        ):
            raise PF10TrainingTensorCacheError("shuffle_seed must fit uint64")

    def _epoch_indices(self, epoch: int) -> tuple[int, ...]:
        if self.shuffle_seed is None:
            return self.indices
        generator = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([self.shuffle_seed, epoch]))
        )
        order = generator.permutation(len(self.indices))
        return tuple(self.indices[int(position)] for position in order)

    def __len__(self) -> int:
        return self.total_updates * self.gradient_accumulation_steps

    def __iter__(self) -> Iterator[list[CacheTrainingIndex]]:
        cursor = 0
        epoch = 0
        epoch_indices = self._epoch_indices(epoch)
        for update in range(1, self.total_updates + 1):
            view = (
                self.fixed_view_id
                if self.fixed_view_id is not None
                else self._view_for_update(self.cell, update)
            )
            for _ in range(self.gradient_accumulation_steps):
                if cursor == len(self.indices):
                    cursor = 0
                    epoch += 1
                    epoch_indices = self._epoch_indices(epoch)
                stop = min(cursor + self.micro_batch_size, len(self.indices))
                batch = [
                    CacheTrainingIndex(index, epoch, view)
                    for index in epoch_indices[cursor:stop]
                ]
                cursor = stop
                if not batch:
                    raise PF10TrainingTensorCacheError("sampler emitted an empty batch")
                yield batch


@dataclass(frozen=True)
class CachedV3Batch:
    view_id: str
    epoch: int
    record_ids: tuple[str, ...]
    exact_identity_sha256: tuple[tuple[str, ...], ...]
    inputs: Mapping[str, object]
    labels: torch.Tensor

    def pin_memory(self) -> "CachedV3Batch":
        values = {
            key: value.pin_memory() if isinstance(value, torch.Tensor) else value
            for key, value in self.inputs.items()
        }
        return CachedV3Batch(
            self.view_id,
            self.epoch,
            self.record_ids,
            self.exact_identity_sha256,
            values,
            values["labels"],  # type: ignore[arg-type]
        )

    def to(self, device: object, *, non_blocking: bool = True) -> "CachedV3Batch":
        values = {
            key: value.to(device=device, non_blocking=non_blocking)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in self.inputs.items()
        }
        return CachedV3Batch(
            self.view_id,
            self.epoch,
            self.record_ids,
            self.exact_identity_sha256,
            values,
            values["labels"],  # type: ignore[arg-type]
        )


class CachedV3Collator:
    """Worker-side dynamic corruption/padding over mmap-backed records."""

    def __init__(
        self,
        *,
        cache: PF10TrainingTensorCache,
        tokenizer: ProductionTokenizerRuntime,
        cell: str,
        seed: int,
        num_e3fp_embeddings: int = 4096,
    ) -> None:
        self.cache = cache
        self.tokenizer = tokenizer
        self.cell = cell
        self.seed = int(seed)
        self.num_e3fp_embeddings = int(num_e3fp_embeddings)
        self.addresses = CachedCanonicalAtomAddressProvider(cache)

    def __call__(self, samples: Sequence[CacheTrainingSample]) -> CachedV3Batch:
        rows = tuple(samples)
        if not rows or len({row.epoch for row in rows}) != 1 or len({row.view_id for row in rows}) != 1:
            raise PF10TrainingTensorCacheError(
                "one loader microbatch must share epoch and training view"
            )
        view_id = rows[0].view_id
        records = tuple(row.record for row in rows)
        states = (
            CachedMorganAtomStateProvider(self.cache, records)
            if self.cell == "B2D"
            else None
        )
        batch = collate_3d_motif_training_view_v3(
            records,
            view_id=view_id,
            tokenizer=self.tokenizer,
            seed=self.seed,
            epoch=rows[0].epoch,
            atom_address_provider=self.addresses,
            atom_state_provider=states,
            num_e3fp_embeddings=self.num_e3fp_embeddings,
            device=None,
        )
        inputs = batch.model_inputs()
        labels = inputs.get("labels")
        if not isinstance(labels, torch.Tensor):
            raise PF10TrainingTensorCacheError("V3 training view lacks CE labels")
        return CachedV3Batch(
            view_id=view_id,
            epoch=rows[0].epoch,
            record_ids=tuple(row.record.record_id for row in rows),
            exact_identity_sha256=tuple(
                tuple(row.record.exact_identity_sha256) for row in rows
            ),
            inputs=inputs,
            labels=labels,
        )


def build_v3_cache_dataloader(
    *,
    cache_root: Path,
    tokenizer: ProductionTokenizerRuntime,
    cell: str,
    seed: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    total_updates: int,
    num_workers: int,
    prefetch_factor: int = 4,
    shuffle_seed: int | None = None,
    fixed_view_id: str | None = None,
) -> torch.utils.data.DataLoader:
    """Construct the worker/pinning boundary used by future V3 runners."""

    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
        or isinstance(prefetch_factor, bool)
        or not isinstance(prefetch_factor, int)
        or prefetch_factor <= 0
    ):
        raise PF10TrainingTensorCacheError(
            "cache DataLoader workers must be nonnegative and prefetch positive"
        )
    dataset = IndexedPF10TrainingTensorCache(cache_root)
    sampler = V3EpochViewBatchSampler(
        dataset.split_indices("train"),
        cell=cell,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        total_updates=total_updates,
        shuffle_seed=shuffle_seed,
        fixed_view_id=fixed_view_id,
    )
    collator = CachedV3Collator(
        cache=dataset,
        tokenizer=tokenizer,
        cell=cell,
        seed=seed,
    )
    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "collate_fn": collator,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            multiprocessing_context="spawn",
        )
    return torch.utils.data.DataLoader(
        **loader_kwargs,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--morgan-overlay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-dev-records", type=int)
    parser.add_argument("--decode-workers", type=int, default=0)
    parser.add_argument("--decode-max-pending", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_pf10_training_tensor_cache(
        paired_release=args.paired_release,
        morgan_overlay=args.morgan_overlay,
        output_dir=args.output_dir,
        max_train_records=args.max_train_records,
        max_dev_records=args.max_dev_records,
        decode_workers=args.decode_workers,
        decode_max_pending=args.decode_max_pending,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
