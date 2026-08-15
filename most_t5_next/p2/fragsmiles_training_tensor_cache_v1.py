"""Compact mmap ABI for fragSMILES plus atom-level E3FP pretraining.

The immutable PCQM geometry release remains the scientific source.  This
module stores only deterministic fields consumed by the model/collator as
flat native arrays with record offsets.  Random span corruption, task-view
selection, padding and dropout deliberately remain online operations.

The layout follows the established processed-dataset pattern used by nearby
molecular models: expensive chemistry is materialized once, while workers
open read-only memory maps and construct dynamic batches without decoding
canonical JSON or rebuilding Python chemistry objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .fragsmiles_geometry_sidecar_v1 import FragSmilesGeometrySidecar


SCHEMA_VERSION = "most-t5-next/fragsmiles-training-tensor-cache/v1"
MANIFEST_NAME = "manifest.json"
MAX_SEQUENCE_LENGTH = 512
REFERENCE_DATALOADER_WORKERS = 8
REFERENCE_PREFETCH_FACTOR = 5

ROLE_TO_ID = {
    "control": 0,
    "fragment_phrase": 1,
    "connector_endpoint": 2,
    "branch": 3,
    "component": 4,
    "stereo_record": 5,
    "molecule_boundary": 6,
    "syntax_glyph": 7,
    "atom_glyph": 8,
}
ID_TO_ROLE = tuple(ROLE_TO_ID)
MODE_TO_ID = {"compact": 0, "whole_molecule_fallback": 1}
ID_TO_MODE = tuple(MODE_TO_ID)
REPRESENTATION_TO_ID = {"macro": 0, "fragment_lexer": 1}
ID_TO_REPRESENTATION = tuple(REPRESENTATION_TO_ID)

# Closed-right bins: <=64, <=128, <=256, <=384, <=512, >512.
LENGTH_BIN_UPPER_BOUNDS = (64, 128, 256, 384, 512)

ARRAY_DTYPES = {
    "input_ids": "<i4",
    "token_role": "u1",
    "token_to_fragment": "<i4",
    "fragment_span": "<i4",
    "fragment_carrier": "<i4",
    "fragment_component": "<u2",
    "fragment_representation": "u1",
    "atom_to_fragment": "<i4",
    "atom_local_index": "<i4",
    "atom_component": "<u2",
    "atom_carrier": "<i4",
    "atom_is_attachment": "u1",
    "e3fp": "<i4",
    # connector, side(0/1), fragment, e3fp row, carrier token, explicit flag
    "endpoint": "<i4",
    "token_offsets": "<i8",
    "fragment_offsets": "<i8",
    "atom_offsets": "<i8",
    "endpoint_offsets": "<i8",
    "record_ordinal": "<i4",
    "source_segment": "u1",
    "mode": "u1",
    "component_count": "<u2",
    "molecule_carrier": "<i4",
    "sequence_length": "<u2",
    "length_class": "u1",
}


class FragSmilesTrainingCacheError(RuntimeError):
    """The compiled training artifact is incomplete or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def length_class(length: int) -> int:
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise FragSmilesTrainingCacheError("sequence length must be positive")
    for index, upper in enumerate(LENGTH_BIN_UPPER_BOUNDS):
        if length <= upper:
            return index
    return len(LENGTH_BIN_UPPER_BOUNDS)


@dataclass(frozen=True)
class CompiledFragSmilesRecord:
    """One unpadded record after deterministic chemistry/tokenization."""

    ordinal: int
    source_segment: int
    input_ids: tuple[int, ...]
    token_roles: tuple[int, ...]
    token_to_fragment: tuple[int, ...]
    fragment_spans: tuple[tuple[int, int], ...]
    fragment_carriers: tuple[int, ...]
    fragment_components: tuple[int, ...]
    fragment_representations: tuple[int, ...]
    atom_to_fragment: tuple[int, ...]
    atom_local_index: tuple[int, ...]
    atom_components: tuple[int, ...]
    atom_carriers: tuple[int, ...]
    atom_is_attachment: tuple[bool, ...]
    e3fp: tuple[tuple[int, int, int, int], ...]
    endpoints: tuple[tuple[int, int, int, int, int, int], ...]
    mode: int
    component_count: int
    molecule_carrier: int

    def __post_init__(self) -> None:
        token_count = len(self.input_ids)
        fragment_count = len(self.fragment_spans)
        atom_count = len(self.e3fp)
        if (
            self.ordinal < 0
            or self.source_segment < 0
            or token_count <= 0
            or not (
                token_count == len(self.token_roles) == len(self.token_to_fragment)
            )
            or not (
                fragment_count
                == len(self.fragment_carriers)
                == len(self.fragment_components)
                == len(self.fragment_representations)
            )
            or not (
                atom_count
                == len(self.atom_to_fragment)
                == len(self.atom_local_index)
                == len(self.atom_components)
                == len(self.atom_carriers)
                == len(self.atom_is_attachment)
            )
            or self.mode not in range(len(MODE_TO_ID))
            or self.component_count <= 0
        ):
            raise FragSmilesTrainingCacheError("compiled record dimensions disagree")
        if any(not 0 <= value < len(ROLE_TO_ID) for value in self.token_roles):
            raise FragSmilesTrainingCacheError("unknown token role ID")
        if any(not 0 <= value < len(REPRESENTATION_TO_ID) for value in self.fragment_representations):
            raise FragSmilesTrainingCacheError("unknown fragment representation ID")
        if any(len(row) != 4 for row in self.e3fp):
            raise FragSmilesTrainingCacheError("E3FP rows must have four shell IDs")
        if any(len(row) != 6 for row in self.endpoints):
            raise FragSmilesTrainingCacheError("endpoint rows must have fixed width six")
        if any(not (0 <= start < stop <= token_count) for start, stop in self.fragment_spans):
            raise FragSmilesTrainingCacheError("fragment span is outside the token row")
        if any(not 0 <= value < token_count for value in self.fragment_carriers):
            raise FragSmilesTrainingCacheError("fragment carrier is outside the token row")
        if any(not 0 <= value < token_count for value in self.atom_carriers):
            raise FragSmilesTrainingCacheError("atom carrier is outside the token row")
        if any(value < -1 or value >= fragment_count for value in self.atom_to_fragment):
            raise FragSmilesTrainingCacheError("atom owner is outside the fragment row")
        if any(value < -1 or value >= fragment_count for value in self.token_to_fragment):
            raise FragSmilesTrainingCacheError("token owner is outside the fragment row")
        if self.mode == MODE_TO_ID["compact"]:
            if fragment_count <= 0 or self.molecule_carrier != -1:
                raise FragSmilesTrainingCacheError("compact carrier contract failed")
            owned_atom_counts = [0] * fragment_count
            for owner in self.atom_to_fragment:
                if owner < 0:
                    raise FragSmilesTrainingCacheError(
                        "compact atom has no fragment owner"
                    )
                owned_atom_counts[owner] += 1
            if any(count == 0 for count in owned_atom_counts):
                raise FragSmilesTrainingCacheError(
                    "compact fragment owns no retained heavy atom"
                )
            endpoint_sides = {}
            for connector, side, owner, atom, token, explicit in self.endpoints:
                if (
                    connector < 0
                    or side not in (0, 1)
                    or not 0 <= owner < fragment_count
                    or not 0 <= atom < atom_count
                    or not 0 <= token < token_count
                    or explicit not in (0, 1)
                ):
                    raise FragSmilesTrainingCacheError(
                        "endpoint address is outside the compact record"
                    )
                sides = endpoint_sides.setdefault(connector, set())
                if side in sides:
                    raise FragSmilesTrainingCacheError(
                        "connector repeats one endpoint side"
                    )
                sides.add(side)
                if (
                    self.atom_to_fragment[atom] != owner
                    or not self.atom_is_attachment[atom]
                ):
                    raise FragSmilesTrainingCacheError(
                        "endpoint atom and fragment ownership disagree"
                    )
                if explicit:
                    if (
                        self.token_roles[token] != ROLE_TO_ID["connector_endpoint"]
                        or self.token_to_fragment[token] != owner
                    ):
                        raise FragSmilesTrainingCacheError(
                            "explicit endpoint token and owner disagree"
                        )
                elif token != self.fragment_carriers[owner]:
                    raise FragSmilesTrainingCacheError(
                        "implicit endpoint does not use its fragment carrier"
                    )
            if endpoint_sides and (
                sorted(endpoint_sides) != list(range(len(endpoint_sides)))
                or any(sides != {0, 1} for sides in endpoint_sides.values())
            ):
                raise FragSmilesTrainingCacheError(
                    "connector endpoints do not form dense left-right pairs"
                )
        else:
            valid_molecule_carriers = {token_count - 1}
            # The formal T5 input appends ordinary </s> after the molecular
            # envelope.  In that case <eom> remains the molecule-summary
            # carrier at the penultimate position; EOS is an unowned control
            # token and never receives geometry.
            if (
                token_count >= 2
                and self.token_roles[-1] == ROLE_TO_ID["control"]
                and self.token_to_fragment[-1] == -1
            ):
                valid_molecule_carriers.add(token_count - 2)
            if (
                fragment_count
                or self.endpoints
                or self.molecule_carrier not in valid_molecule_carriers
            ):
                raise FragSmilesTrainingCacheError("fallback carrier contract failed")


def compile_sidecar_record(
    *,
    ordinal: int,
    source_segment: int,
    input_ids: Sequence[int],
    sidecar: FragSmilesGeometrySidecar,
    e3fp: Sequence[Sequence[int]],
) -> CompiledFragSmilesRecord:
    """Project the rich sidecar into the minimal model-facing cache schema."""

    if len(input_ids) not in (len(sidecar.model_tokens), len(sidecar.model_tokens) + 1):
        raise FragSmilesTrainingCacheError(
            "token IDs must cover the model surface with an optional T5 EOS"
        )
    if len(e3fp) != sum(row.has_e3fp_row for row in sidecar.atoms):
        raise FragSmilesTrainingCacheError("E3FP rows do not cover sidecar geometry")
    atoms_by_e3fp = {
        int(row.e3fp_row): row for row in sidecar.atoms if row.e3fp_row is not None
    }
    if sorted(atoms_by_e3fp) != list(range(len(e3fp))):
        raise FragSmilesTrainingCacheError("E3FP row domain is not dense")

    atom_owner: list[int] = []
    atom_local: list[int] = []
    atom_component: list[int] = []
    atom_carrier: list[int] = []
    atom_attachment: list[bool] = []
    for e3fp_row in range(len(e3fp)):
        atom = atoms_by_e3fp[e3fp_row]
        owner = -1 if atom.fragment_index is None else int(atom.fragment_index)
        local = -1 if atom.fragment_local_atom_index is None else int(atom.fragment_local_atom_index)
        if atom.fragment_carrier_token_index is not None:
            carrier = int(atom.fragment_carrier_token_index)
        elif atom.token_stop is not None:
            # Whole-molecule fallback keeps atom-level geometry aligned to the
            # final glyph of the atom lexeme without inventing another token.
            carrier = int(atom.token_stop) - 1
        else:
            raise FragSmilesTrainingCacheError("E3FP atom has no model carrier")
        atom_owner.append(owner)
        atom_local.append(local)
        atom_component.append(int(atom.component_index))
        atom_carrier.append(carrier)
        atom_attachment.append(bool(atom.is_attachment))

    endpoints = []
    for connector in sidecar.connectors:
        for side_id, endpoint in enumerate((connector.left, connector.right)):
            endpoints.append(
                (
                    int(connector.connector_index),
                    side_id,
                    int(endpoint.fragment_index),
                    int(endpoint.e3fp_row),
                    int(endpoint.carrier_token_index),
                    int(bool(endpoint.explicit_in_surface)),
                )
            )

    try:
        token_roles = tuple(ROLE_TO_ID[value] for value in sidecar.token_roles)
        mode = MODE_TO_ID[sidecar.mode]
        fragment_representations = tuple(
            REPRESENTATION_TO_ID[row.representation] for row in sidecar.fragments
        )
    except KeyError as exc:
        raise FragSmilesTrainingCacheError("sidecar contains an unknown enum") from exc
    return CompiledFragSmilesRecord(
        ordinal=int(ordinal),
        source_segment=int(source_segment),
        input_ids=tuple(int(value) for value in input_ids),
        token_roles=(
            token_roles
            if len(input_ids) == len(sidecar.model_tokens)
            else token_roles + (ROLE_TO_ID["control"],)
        ),
        token_to_fragment=(
            tuple(int(value) for value in sidecar.token_to_fragment)
            if len(input_ids) == len(sidecar.model_tokens)
            else tuple(int(value) for value in sidecar.token_to_fragment) + (-1,)
        ),
        fragment_spans=tuple((int(row.token_start), int(row.token_stop)) for row in sidecar.fragments),
        fragment_carriers=tuple(int(row.carrier_token_index) for row in sidecar.fragments),
        fragment_components=tuple(int(row.component_index) for row in sidecar.fragments),
        fragment_representations=fragment_representations,
        atom_to_fragment=tuple(atom_owner),
        atom_local_index=tuple(atom_local),
        atom_components=tuple(atom_component),
        atom_carriers=tuple(atom_carrier),
        atom_is_attachment=tuple(atom_attachment),
        e3fp=tuple(tuple(int(value) for value in row) for row in e3fp),
        endpoints=tuple(endpoints),
        mode=mode,
        component_count=int(sidecar.component_count),
        molecule_carrier=(
            -1
            if sidecar.molecule_carrier_token_index is None
            else int(sidecar.molecule_carrier_token_index)
        ),
    )


class _Writer:
    def __init__(self, path: Path, dtype: str) -> None:
        self.path = path
        self.dtype = np.dtype(dtype)
        self.handle = path.open("wb", buffering=8 * 1024 * 1024)
        self.elements = 0

    def write(self, value: object) -> None:
        array = np.asarray(value, dtype=self.dtype)
        self.handle.write(array.tobytes(order="C"))
        self.elements += int(array.size)

    def close(self) -> None:
        self.handle.close()


def write_training_cache(
    records: Iterable[CompiledFragSmilesRecord],
    *,
    output_dir: Path,
    source: Mapping[str, object],
    tokenizer: Mapping[str, object],
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    exclude_oversize: bool = True,
) -> dict[str, object]:
    """Write one immutable cache and a transparent oversize exclusion ledger."""

    output_dir = Path(output_dir).expanduser().resolve()
    staging = output_dir.with_name(output_dir.name + ".staging")
    if output_dir.exists() or staging.exists():
        raise FragSmilesTrainingCacheError("output or staging path already exists")
    if max_sequence_length <= 0 or max_sequence_length > np.iinfo(np.uint16).max:
        raise FragSmilesTrainingCacheError("max sequence length is invalid")
    staging.mkdir(parents=True)
    variable = set(ARRAY_DTYPES) - {
        "token_offsets", "fragment_offsets", "atom_offsets", "endpoint_offsets",
        "record_ordinal", "source_segment", "mode", "component_count",
        "molecule_carrier", "sequence_length", "length_class",
    }
    writers = {name: _Writer(staging / f"{name}.bin", ARRAY_DTYPES[name]) for name in variable}
    offsets = {"token": [0], "fragment": [0], "atom": [0], "endpoint": [0]}
    fixed = {name: [] for name in (
        "record_ordinal", "source_segment", "mode", "component_count",
        "molecule_carrier", "sequence_length", "length_class",
    )}
    mode_counts = {name: 0 for name in MODE_TO_ID}
    length_counts = [0] * (len(LENGTH_BIN_UPPER_BOUNDS) + 1)
    excluded = 0
    seen_ordinals: set[int] = set()
    exclusion_path = staging / "length_exclusions.jsonl"
    try:
        with exclusion_path.open("w", encoding="utf-8", newline="\n") as exclusions:
            for record in records:
                if record.ordinal in seen_ordinals:
                    raise FragSmilesTrainingCacheError("record ordinal repeats")
                seen_ordinals.add(record.ordinal)
                sequence_length = len(record.input_ids)
                category = length_class(sequence_length)
                if sequence_length > max_sequence_length:
                    exclusions.write(json.dumps({
                        "ordinal": record.ordinal,
                        "source_segment": record.source_segment,
                        "mode": ID_TO_MODE[record.mode],
                        "sequence_length": sequence_length,
                        "reason": "MODEL_INPUT_EXCEEDS_MAX_SEQUENCE_LENGTH",
                    }, sort_keys=True, separators=(",", ":")) + "\n")
                    excluded += 1
                    if not exclude_oversize:
                        raise FragSmilesTrainingCacheError("oversize record encountered")
                    continue
                writers["input_ids"].write(record.input_ids)
                writers["token_role"].write(record.token_roles)
                writers["token_to_fragment"].write(record.token_to_fragment)
                writers["fragment_span"].write(record.fragment_spans)
                writers["fragment_carrier"].write(record.fragment_carriers)
                writers["fragment_component"].write(record.fragment_components)
                writers["fragment_representation"].write(record.fragment_representations)
                writers["atom_to_fragment"].write(record.atom_to_fragment)
                writers["atom_local_index"].write(record.atom_local_index)
                writers["atom_component"].write(record.atom_components)
                writers["atom_carrier"].write(record.atom_carriers)
                writers["atom_is_attachment"].write(record.atom_is_attachment)
                writers["e3fp"].write(record.e3fp)
                writers["endpoint"].write(record.endpoints)
                offsets["token"].append(offsets["token"][-1] + sequence_length)
                offsets["fragment"].append(offsets["fragment"][-1] + len(record.fragment_spans))
                offsets["atom"].append(offsets["atom"][-1] + len(record.e3fp))
                offsets["endpoint"].append(offsets["endpoint"][-1] + len(record.endpoints))
                fixed["record_ordinal"].append(record.ordinal)
                fixed["source_segment"].append(record.source_segment)
                fixed["mode"].append(record.mode)
                fixed["component_count"].append(record.component_count)
                fixed["molecule_carrier"].append(record.molecule_carrier)
                fixed["sequence_length"].append(sequence_length)
                fixed["length_class"].append(category)
                mode_counts[ID_TO_MODE[record.mode]] += 1
                length_counts[category] += 1
    finally:
        for writer in writers.values():
            writer.close()

    record_count = len(fixed["record_ordinal"])
    if record_count <= 0:
        raise FragSmilesTrainingCacheError("cache contains no trainable records")
    for axis, values in offsets.items():
        np.asarray(values, dtype=ARRAY_DTYPES[f"{axis}_offsets"]).tofile(
            staging / f"{axis}_offsets.bin"
        )
    for name, values in fixed.items():
        np.asarray(values, dtype=ARRAY_DTYPES[name]).tofile(staging / f"{name}.bin")

    shapes = {
        "input_ids": [offsets["token"][-1]],
        "token_role": [offsets["token"][-1]],
        "token_to_fragment": [offsets["token"][-1]],
        "fragment_span": [offsets["fragment"][-1], 2],
        "fragment_carrier": [offsets["fragment"][-1]],
        "fragment_component": [offsets["fragment"][-1]],
        "fragment_representation": [offsets["fragment"][-1]],
        "atom_to_fragment": [offsets["atom"][-1]],
        "atom_local_index": [offsets["atom"][-1]],
        "atom_component": [offsets["atom"][-1]],
        "atom_carrier": [offsets["atom"][-1]],
        "atom_is_attachment": [offsets["atom"][-1]],
        "e3fp": [offsets["atom"][-1], 4],
        "endpoint": [offsets["endpoint"][-1], 6],
        "token_offsets": [record_count + 1],
        "fragment_offsets": [record_count + 1],
        "atom_offsets": [record_count + 1],
        "endpoint_offsets": [record_count + 1],
        **{name: [record_count] for name in fixed},
    }
    arrays = {}
    total_bytes = 0
    for name, shape in shapes.items():
        path = staging / f"{name}.bin"
        expected = int(np.prod(shape)) * np.dtype(ARRAY_DTYPES[name]).itemsize
        if path.stat().st_size != expected:
            raise FragSmilesTrainingCacheError(f"array {name} byte length differs")
        arrays[name] = {
            "file": path.name,
            "dtype": ARRAY_DTYPES[name],
            "shape": shape,
            "bytes": expected,
            "sha256": _sha256_file(path),
        }
        total_bytes += expected

    lengths = np.asarray(fixed["sequence_length"], dtype=np.int64)
    percentiles = {
        name: int(np.percentile(lengths, value, method="nearest"))
        for name, value in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))
    }
    labels = [f"le_{value}" for value in LENGTH_BIN_UPPER_BOUNDS] + ["gt_512"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "training_admission": True,
        "source": json.loads(json.dumps(dict(source), sort_keys=True)),
        "tokenizer": json.loads(json.dumps(dict(tokenizer), sort_keys=True)),
        "counts": {
            "records": record_count,
            "excluded_oversize_records": excluded,
            "tokens": offsets["token"][-1],
            "fragments": offsets["fragment"][-1],
            "atoms": offsets["atom"][-1],
            "endpoint_rows": offsets["endpoint"][-1],
            "modes": mode_counts,
        },
        "lengths": {
            "maximum_model_length": max_sequence_length,
            "no_truncation": True,
            "minimum": int(lengths.min()),
            "maximum": int(lengths.max()),
            "mean": float(lengths.mean()),
            **percentiles,
            "classes": dict(zip(labels, length_counts)),
        },
        "arrays": arrays,
        "storage": {
            "layout": "flat_native_arrays_plus_int64_offsets",
            "read_mode": "read_only_mmap",
            "total_array_bytes": total_bytes,
            "coordinates_cached": False,
            "audit_hashes_cached_per_record": False,
            "record_strings_cached": False,
            "morgan_cached": False,
        },
        "online_training_boundary": {
            "corruption_cached": False,
            "padding_cached": False,
            "task_views_cached": False,
            "dropout_cached": False,
            "dynamic_corruption_and_padding_required": True,
        },
        "artifacts": {
            "length_exclusions": {
                "path": exclusion_path.name,
                "bytes": exclusion_path.stat().st_size,
                "sha256": _sha256_file(exclusion_path),
                "rows": excluded,
            }
        },
    }
    (staging / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.rename(output_dir)
    return manifest


@dataclass(frozen=True)
class CachedFragSmilesRecord:
    cache_index: int
    ordinal: int
    source_segment: int
    mode: str
    component_count: int
    molecule_carrier: int
    input_ids: np.ndarray
    token_roles: np.ndarray
    token_to_fragment: np.ndarray
    fragment_spans: np.ndarray
    fragment_carriers: np.ndarray
    fragment_components: np.ndarray
    fragment_representations: np.ndarray
    atom_to_fragment: np.ndarray
    atom_local_index: np.ndarray
    atom_components: np.ndarray
    atom_carriers: np.ndarray
    atom_is_attachment: np.ndarray
    e3fp: np.ndarray
    endpoints: np.ndarray


class FragSmilesTrainingTensorCache:
    """Spawn-safe, read-only mmap dataset for the formal pretraining path."""

    def __init__(self, root: Path, *, verify_hashes: bool = True) -> None:
        self.root = Path(root).expanduser().resolve()
        try:
            manifest = json.loads((self.root / MANIFEST_NAME).read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FragSmilesTrainingCacheError("cache manifest is unreadable") from exc
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("status") != "pass"
            or manifest.get("training_admission") is not True
        ):
            raise FragSmilesTrainingCacheError("cache is not training-admitted")
        self.manifest = manifest
        self.arrays: dict[str, np.ndarray] = {}
        for name, spec in manifest["arrays"].items():
            path = self.root / spec["file"]
            shape = tuple(int(value) for value in spec["shape"])
            dtype = np.dtype(spec["dtype"])
            if not path.is_file() or path.stat().st_size != int(np.prod(shape)) * dtype.itemsize:
                raise FragSmilesTrainingCacheError(f"cache array {name} is absent or truncated")
            if verify_hashes and _sha256_file(path) != spec["sha256"]:
                raise FragSmilesTrainingCacheError(f"cache array {name} hash differs")
            if int(np.prod(shape)) == 0:
                empty = np.empty(shape, dtype=dtype)
                empty.flags.writeable = False
                self.arrays[name] = empty
            else:
                self.arrays[name] = np.memmap(
                    path, dtype=dtype, mode="r", shape=shape
                )

    def __getstate__(self) -> dict[str, object]:
        return {"root": self.root}

    def __setstate__(self, state: Mapping[str, object]) -> None:
        self.__init__(Path(state["root"]), verify_hashes=False)

    def __len__(self) -> int:
        return int(self.manifest["counts"]["records"])

    def _slice(self, name: str, axis: str, index: int) -> np.ndarray:
        offsets = self.arrays[f"{axis}_offsets"]
        return self.arrays[name][int(offsets[index]) : int(offsets[index + 1])]

    def __getitem__(self, index: int) -> CachedFragSmilesRecord:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        return CachedFragSmilesRecord(
            cache_index=index,
            ordinal=int(self.arrays["record_ordinal"][index]),
            source_segment=int(self.arrays["source_segment"][index]),
            mode=ID_TO_MODE[int(self.arrays["mode"][index])],
            component_count=int(self.arrays["component_count"][index]),
            molecule_carrier=int(self.arrays["molecule_carrier"][index]),
            input_ids=self._slice("input_ids", "token", index),
            token_roles=self._slice("token_role", "token", index),
            token_to_fragment=self._slice("token_to_fragment", "token", index),
            fragment_spans=self._slice("fragment_span", "fragment", index),
            fragment_carriers=self._slice("fragment_carrier", "fragment", index),
            fragment_components=self._slice("fragment_component", "fragment", index),
            fragment_representations=self._slice("fragment_representation", "fragment", index),
            atom_to_fragment=self._slice("atom_to_fragment", "atom", index),
            atom_local_index=self._slice("atom_local_index", "atom", index),
            atom_components=self._slice("atom_component", "atom", index),
            atom_carriers=self._slice("atom_carrier", "atom", index),
            atom_is_attachment=self._slice("atom_is_attachment", "atom", index),
            e3fp=self._slice("e3fp", "atom", index),
            endpoints=self._slice("endpoint", "endpoint", index),
        )

    def close(self) -> None:
        for array in self.arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.arrays.clear()


@dataclass(frozen=True)
class CachedFragSmilesSample:
    """One cache row plus the epoch used by online random corruption."""

    record: CachedFragSmilesRecord
    epoch: int


@dataclass(frozen=True)
class FragSmilesTrainingIndex:
    cache_index: int
    epoch: int


class IndexedFragSmilesTrainingTensorCache(FragSmilesTrainingTensorCache):
    def __getitem__(self, index: object) -> object:
        if isinstance(index, FragSmilesTrainingIndex):
            return CachedFragSmilesSample(
                record=super().__getitem__(index.cache_index),
                epoch=index.epoch,
            )
        return super().__getitem__(index)  # type: ignore[arg-type]


class FragSmilesEpochBatchSampler:
    """Deterministic epoch shuffle with an explicit, retained short tail."""

    def __init__(
        self,
        record_count: int,
        *,
        micro_batch_size: int,
        shuffle_seed: int | None,
        epoch: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.record_count = int(record_count)
        self.micro_batch_size = int(micro_batch_size)
        self.shuffle_seed = shuffle_seed
        self.epoch = int(epoch)
        self.drop_last = bool(drop_last)
        if (
            self.record_count <= 0
            or self.micro_batch_size <= 0
            or self.epoch < 0
            or (
                shuffle_seed is not None
                and (
                    isinstance(shuffle_seed, bool)
                    or not isinstance(shuffle_seed, int)
                    or not 0 <= shuffle_seed < 2**64
                )
            )
        ):
            raise FragSmilesTrainingCacheError("epoch sampler settings are invalid")

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise FragSmilesTrainingCacheError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        quotient, remainder = divmod(self.record_count, self.micro_batch_size)
        return quotient if self.drop_last or remainder == 0 else quotient + 1

    def __iter__(self) -> Iterator[list[FragSmilesTrainingIndex]]:
        order = np.arange(self.record_count, dtype=np.int64)
        if self.shuffle_seed is not None:
            generator = np.random.Generator(
                np.random.PCG64(
                    np.random.SeedSequence([self.shuffle_seed, self.epoch])
                )
            )
            order = generator.permutation(order)
        usable = self.record_count
        if self.drop_last:
            usable -= self.record_count % self.micro_batch_size
        for start in range(0, usable, self.micro_batch_size):
            stop = min(start + self.micro_batch_size, usable)
            yield [
                FragSmilesTrainingIndex(int(index), self.epoch)
                for index in order[start:stop]
            ]


def collate_cached_fragsmiles(
    records: Sequence[CachedFragSmilesRecord], *, pad_token_id: int
) -> dict[str, object]:
    """Dynamically pad one batch without collapsing fragSMILES endpoints.

    Endpoints deliberately retain their own ``E`` axis.  A token-axis
    ``token -> atom`` map cannot represent the official fragSMILES case where
    several implicit connector endpoints share one fragment carrier.  Online
    corruption may transform token indices later, but it must preserve these
    endpoint rows and their owning-fragment addresses.
    """

    if not records:
        raise FragSmilesTrainingCacheError("cannot collate an empty batch")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime boundary
        raise RuntimeError("PyTorch is required for batch collation") from exc
    batch = len(records)
    max_tokens = max(len(row.input_ids) for row in records)
    max_atoms = max(len(row.e3fp) for row in records)
    max_fragments = max(len(row.fragment_spans) for row in records)
    max_endpoints = max(len(row.endpoints) for row in records)
    input_ids = torch.full((batch, max_tokens), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch, max_tokens), dtype=torch.bool)
    token_roles = torch.zeros((batch, max_tokens), dtype=torch.long)
    token_to_fragment = torch.full((batch, max_tokens), -1, dtype=torch.long)
    e3fp = torch.full((batch, max_atoms, 4), -1, dtype=torch.long)
    atom_mask = torch.zeros((batch, max_atoms), dtype=torch.bool)
    atom_to_fragment = torch.full((batch, max_atoms), -1, dtype=torch.long)
    atom_carrier = torch.full((batch, max_atoms), -1, dtype=torch.long)
    atom_is_attachment = torch.zeros((batch, max_atoms), dtype=torch.bool)
    fragment_mask = torch.zeros((batch, max_fragments), dtype=torch.bool)
    fragment_to_carrier = torch.full(
        (batch, max_fragments), -1, dtype=torch.long
    )
    identity_span_bounds = torch.full(
        (batch, max_fragments, 2), -1, dtype=torch.long
    )
    fragment_representation = torch.full(
        (batch, max_fragments), -1, dtype=torch.long
    )
    endpoint_mask = torch.zeros((batch, max_endpoints), dtype=torch.bool)
    endpoint_to_atom = torch.full((batch, max_endpoints), -1, dtype=torch.long)
    endpoint_to_token = torch.full((batch, max_endpoints), -1, dtype=torch.long)
    endpoint_to_fragment = torch.full(
        (batch, max_endpoints), -1, dtype=torch.long
    )
    endpoint_connector = torch.full((batch, max_endpoints), -1, dtype=torch.long)
    endpoint_side = torch.full((batch, max_endpoints), -1, dtype=torch.long)
    endpoint_is_explicit = torch.zeros((batch, max_endpoints), dtype=torch.bool)
    molecule_carrier = torch.full((batch,), -1, dtype=torch.long)
    for batch_index, row in enumerate(records):
        token_count = len(row.input_ids)
        atom_count = len(row.e3fp)
        fragment_count = len(row.fragment_spans)
        endpoint_count = len(row.endpoints)
        input_ids[batch_index, :token_count] = torch.tensor(
            np.asarray(row.input_ids), dtype=torch.long
        )
        attention_mask[batch_index, :token_count] = True
        token_roles[batch_index, :token_count] = torch.tensor(
            np.asarray(row.token_roles), dtype=torch.long
        )
        token_to_fragment[batch_index, :token_count] = torch.tensor(
            np.asarray(row.token_to_fragment), dtype=torch.long
        )
        e3fp[batch_index, :atom_count] = torch.tensor(
            np.asarray(row.e3fp), dtype=torch.long
        )
        atom_mask[batch_index, :atom_count] = True
        atom_to_fragment[batch_index, :atom_count] = torch.tensor(
            np.asarray(row.atom_to_fragment), dtype=torch.long
        )
        atom_carrier[batch_index, :atom_count] = torch.tensor(
            np.asarray(row.atom_carriers), dtype=torch.long
        )
        atom_is_attachment[batch_index, :atom_count] = torch.tensor(
            np.asarray(row.atom_is_attachment), dtype=torch.bool
        )
        if fragment_count:
            fragment_mask[batch_index, :fragment_count] = True
            fragment_to_carrier[batch_index, :fragment_count] = torch.tensor(
                np.asarray(row.fragment_carriers), dtype=torch.long
            )
            identity_span_bounds[batch_index, :fragment_count] = torch.tensor(
                np.asarray(row.fragment_spans), dtype=torch.long
            )
            fragment_representation[batch_index, :fragment_count] = torch.tensor(
                np.asarray(row.fragment_representations), dtype=torch.long
            )
        if endpoint_count:
            endpoint_rows = np.asarray(row.endpoints)
            if endpoint_rows.shape != (endpoint_count, 6):
                raise FragSmilesTrainingCacheError(
                    "cached endpoint rows must have shape [E,6]"
                )
            endpoint_mask[batch_index, :endpoint_count] = True
            endpoint_connector[batch_index, :endpoint_count] = torch.tensor(
                endpoint_rows[:, 0], dtype=torch.long
            )
            endpoint_side[batch_index, :endpoint_count] = torch.tensor(
                endpoint_rows[:, 1], dtype=torch.long
            )
            endpoint_to_fragment[batch_index, :endpoint_count] = torch.tensor(
                endpoint_rows[:, 2], dtype=torch.long
            )
            endpoint_to_atom[batch_index, :endpoint_count] = torch.tensor(
                endpoint_rows[:, 3], dtype=torch.long
            )
            endpoint_to_token[batch_index, :endpoint_count] = torch.tensor(
                endpoint_rows[:, 4], dtype=torch.long
            )
            endpoint_is_explicit[batch_index, :endpoint_count] = torch.tensor(
                endpoint_rows[:, 5], dtype=torch.bool
            )
        molecule_carrier[batch_index] = int(row.molecule_carrier)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_roles": token_roles,
        "connector_endpoint_mask": (
            token_roles == ROLE_TO_ID["connector_endpoint"]
        ) & attention_mask,
        "token_to_fragment": token_to_fragment,
        "e3fp_ids": e3fp,
        "e3fp_atom_mask": atom_mask,
        "atom_to_fragment": atom_to_fragment,
        "atom_to_token": atom_carrier,
        "atom_is_attachment": atom_is_attachment,
        "fragment_mask": fragment_mask,
        "fragment_to_carrier": fragment_to_carrier,
        "identity_span_bounds": identity_span_bounds,
        "fragment_representation": fragment_representation,
        "endpoint_mask": endpoint_mask,
        "endpoint_to_atom": endpoint_to_atom,
        "endpoint_to_token": endpoint_to_token,
        "endpoint_to_fragment": endpoint_to_fragment,
        "endpoint_connector": endpoint_connector,
        "endpoint_side": endpoint_side,
        "endpoint_is_explicit": endpoint_is_explicit,
        "molecule_carrier": molecule_carrier,
        "records": tuple(records),
    }


def collate_cached_fragsmiles_samples(
    samples: Sequence[CachedFragSmilesSample], *, pad_token_id: int
) -> dict[str, object]:
    if not samples or len({sample.epoch for sample in samples}) != 1:
        raise FragSmilesTrainingCacheError(
            "one online batch must contain one nonempty epoch"
        )
    batch = collate_cached_fragsmiles(
        tuple(sample.record for sample in samples), pad_token_id=pad_token_id
    )
    batch["epoch"] = samples[0].epoch
    return batch


def build_fragsmiles_cache_dataloader(
    *,
    cache_root: Path,
    micro_batch_size: int,
    pad_token_id: int,
    num_workers: int = REFERENCE_DATALOADER_WORKERS,
    prefetch_factor: int = REFERENCE_PREFETCH_FACTOR,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = False,
    shuffle_seed: int | None = None,
    epoch: int = 0,
    collate_fn: Callable[[Sequence[CachedFragSmilesSample]], object] | None = None,
    multiprocessing_context: str | None = None,
    require_admission_receipt: bool = True,
):
    """Build the reference-style hot-path DataLoader over the mmap cache.

    The default collator performs dynamic padding only.  Formal Phase-I
    training supplies its epoch-aware corruption collator through ``collate_fn``;
    corruption is intentionally absent from the immutable cache.  Record order
    remains the cache order and the short final batch is retained.
    """

    if (
        isinstance(micro_batch_size, bool)
        or not isinstance(micro_batch_size, int)
        or micro_batch_size <= 0
        or isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
        or isinstance(prefetch_factor, bool)
        or not isinstance(prefetch_factor, int)
        or prefetch_factor <= 0
    ):
        raise FragSmilesTrainingCacheError("DataLoader settings are invalid")
    if num_workers == 0 and persistent_workers:
        raise FragSmilesTrainingCacheError(
            "persistent workers require at least one DataLoader worker"
        )
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - runtime boundary
        raise RuntimeError("PyTorch is required for the training DataLoader") from exc

    cache_root = Path(cache_root).expanduser().resolve()
    if require_admission_receipt:
        receipt_path = cache_root / "training_admission_receipt.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FragSmilesTrainingCacheError(
                "formal DataLoader requires a training admission receipt"
            ) from exc
        if (
            receipt.get("schema_version")
            != "most-t5-next/fragsmiles-training-cache-admission/v1"
            or receipt.get("status") != "pass"
            or receipt.get("training_admission") is not True
            or receipt.get("semantic_schemas", {}).get("training_cache")
            != SCHEMA_VERSION
        ):
            raise FragSmilesTrainingCacheError(
                "formal DataLoader admission receipt is incompatible"
            )
    dataset = IndexedFragSmilesTrainingTensorCache(cache_root)
    batch_sampler = FragSmilesEpochBatchSampler(
        len(dataset),
        micro_batch_size=micro_batch_size,
        shuffle_seed=shuffle_seed,
        epoch=epoch,
        drop_last=drop_last,
    )
    resolved_collator = collate_fn or partial(
        collate_cached_fragsmiles_samples, pad_token_id=int(pad_token_id)
    )
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
        "collate_fn": resolved_collator,
    }
    if num_workers:
        kwargs.update(
            {
                "persistent_workers": bool(persistent_workers),
                "prefetch_factor": prefetch_factor,
            }
        )
        if multiprocessing_context is not None:
            kwargs["multiprocessing_context"] = multiprocessing_context
    elif multiprocessing_context is not None:
        dataset.close()
        raise FragSmilesTrainingCacheError(
            "multiprocessing context requires DataLoader workers"
        )
    try:
        return DataLoader(**kwargs)
    except Exception:
        dataset.close()
        raise


__all__ = [
    "ARRAY_DTYPES",
    "CachedFragSmilesRecord",
    "CachedFragSmilesSample",
    "CompiledFragSmilesRecord",
    "FragSmilesTrainingCacheError",
    "FragSmilesTrainingTensorCache",
    "FragSmilesEpochBatchSampler",
    "FragSmilesTrainingIndex",
    "IndexedFragSmilesTrainingTensorCache",
    "LENGTH_BIN_UPPER_BOUNDS",
    "MAX_SEQUENCE_LENGTH",
    "REFERENCE_DATALOADER_WORKERS",
    "REFERENCE_PREFETCH_FACTOR",
    "ROLE_TO_ID",
    "SCHEMA_VERSION",
    "collate_cached_fragsmiles",
    "collate_cached_fragsmiles_samples",
    "build_fragsmiles_cache_dataloader",
    "compile_sidecar_record",
    "length_class",
    "write_training_cache",
]
