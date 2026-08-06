#!/usr/bin/env python3
"""Build the clean, molecule-group-disjoint QM9 instruction split.

This is an offline derivation over the author-released 3D-MolT5 Hugging Face
artifact.  It deliberately consumes only ``train`` and ``validation`` from the
frozen revision below: the released ``test`` file is a byte-identical copy of
``validation`` and is therefore forbidden as an input to the clean view.

The derivation has two related identity levels:

* canonical non-isomeric connectivity defines the molecule group that must
  stay in one output split; every stereoisomer/state of that connectivity is
  assigned together; and
* a model-visible duplicate requires equal semantic signature, byte-for-byte
  SELFIES, and equal canonical serialization of the *complete* ``molecule_fp``
  value.  A later record satisfying all three conditions is removed in stable
  source order.  Equal semantics with a different fingerprint is retained as a
  distinct state in the same molecule group.

No source file is modified or copied.  Parquet input requires ``pyarrow``;
small JSONL fixtures are supported so the scientific contract can be tested
without that optional dependency or a network download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only in broken environments.
    raise RuntimeError(
        "build_qm9_identity_split.py requires NumPy for the frozen PCG64 split"
    ) from exc

try:
    from rdkit import Chem, rdBase
except ImportError as exc:  # pragma: no cover - exercised only in broken environments.
    raise RuntimeError(
        "build_qm9_identity_split.py requires RDKit for canonical isomeric SMILES"
    ) from exc

from most_t5_next.r1.overlap import shared_identity_normalization_v1 as identity_normalization


SOURCE_REVISION = "QizhiPei/e3fp-mol-instructions-qm9@bfe55090be9ebf1c9cbbe6687a5796711ac0edd8"
ALLOWED_SOURCE_SPLITS = ("train", "validation")
FORBIDDEN_SOURCE_SPLITS = frozenset(("test",))
SOURCE_SPLIT_ORDER = {name: index for index, name in enumerate(ALLOWED_SOURCE_SPLITS)}

RNG_SEED = 42
DEFAULT_TRAIN_GROUP_COUNT = 110_000
DEFAULT_VALIDATION_GROUP_COUNT = 10_000
PRODUCTION_RDKIT_VERSION = "2024.03.5"
PRODUCTION_EXPECTED_COUNTS = {
    "input_rows": 349_702,
    "retained_rows": 349_660,
    "removed_model_visible_duplicates": 42,
    "molecule_groups": 128_783,
    "output_groups": {"train": 110_000, "validation": 10_000, "test": 8_783},
}

SPLIT_PROTOCOL_ID = "qm9-3dmolt5-connectivity-group-110k10k-rest-s42-v2"

RELEASED_NUMERIC_TARGET_PATTERN = re.compile(
    r"(?P<numeric_literal>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<terminal_sentence_period>\.)?"
)

REQUIRED_COLUMNS = ("smiles", "selfies", "instruction", "output", "molecule_fp")
SOURCE_MANIFEST_FILENAME = "source_manifest.json"
SPLIT_MANIFEST_FILENAME = "split_manifest.jsonl"
DUPLICATE_REPORT_FILENAME = "duplicate_report.json"
SPLIT_SUMMARY_FILENAME = "split_summary.json"

SOURCE_MANIFEST_SCHEMA = "most-t5-r1/qm9-clean-source-manifest/v2"
SPLIT_ROW_SCHEMA = "most-t5-r1/qm9-clean-split-member/v2"
DUPLICATE_REPORT_SCHEMA = "most-t5-r1/qm9-model-visible-duplicate-report/v1"
SPLIT_SUMMARY_SCHEMA = "most-t5-r1/qm9-clean-split-summary/v2"
IDENTITY_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "contracts"
    / "pcqm4mv2_identity_normalization_contract.json"
)


class Qm9SplitProtocolError(ValueError):
    """Raised when an input would violate the frozen scientific protocol."""


@dataclass(frozen=True)
class SourceFile:
    split: str
    file_ordinal: int
    path: Path
    bytes: int
    sha256: str


@dataclass(frozen=True)
class RetainedRecord:
    source_split: str
    source_file_ordinal: int
    source_file_sha256: str
    source_row_index: int
    source_ordinal: int
    group_id: str
    strict_canonical_isomeric_smiles: str
    strict_canonical_isomeric_smiles_sha256: str
    canonical_connectivity_smiles: str
    canonical_connectivity_smiles_sha256: str
    instruction_stripped: str
    normalized_numeric_target: str
    semantic_signature_sha256: str
    selfies_sha256: str
    molecule_fp_serialized_sha256: str
    molecule_fp_serialized_bytes: int

    @property
    def record_id(self) -> str:
        return (
            "qm9-clean-source:"
            + self.source_split
            + ":"
            + str(self.source_file_ordinal)
            + ":"
            + str(self.source_row_index)
        )

    def source_reference(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "source_split": self.source_split,
            "source_file_ordinal": self.source_file_ordinal,
            "source_file_sha256": self.source_file_sha256,
            "source_row_index": self.source_row_index,
            "stable_source_ordinal": self.source_ordinal,
        }


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Qm9SplitProtocolError("value is not canonical-JSON serializable") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            byte_count += len(block)
            digest.update(block)
    return byte_count, digest.hexdigest()


def _json_compatible(value: Any) -> Any:
    """Convert Arrow/NumPy containers while preserving the complete value."""
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise Qm9SplitProtocolError("molecule_fp mapping keys must be strings")
            converted[key] = _json_compatible(item)
        return converted
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Qm9SplitProtocolError("molecule_fp contains a non-finite float")
        return value
    raise Qm9SplitProtocolError(
        "molecule_fp contains unsupported value type " + type(value).__name__
    )


def serialize_complete_molecule_fp(value: Any) -> bytes:
    if value is None:
        raise Qm9SplitProtocolError("molecule_fp must not be null")
    return canonical_json_bytes(_json_compatible(value))


def normalize_numeric_target(value: Any) -> str:
    """Normalize one finite numeric literal in the released output grammar.

    A string must full-match exactly one numeric literal followed by at most
    one terminal sentence period.  No substring search or unit stripping is
    permitted.
    """
    if isinstance(value, bool) or value is None:
        raise Qm9SplitProtocolError("numeric target must be a finite decimal literal")
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, (int, np.integer)):
        decimal_value = Decimal(int(value))
    elif isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise Qm9SplitProtocolError("numeric target must be finite")
        decimal_value = Decimal(str(numeric))
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise Qm9SplitProtocolError("numeric target must not be empty")
        match = RELEASED_NUMERIC_TARGET_PATTERN.fullmatch(stripped)
        if match is None:
            raise Qm9SplitProtocolError(
                "numeric target must be exactly one numeric literal optionally "
                "followed by one terminal sentence period"
            )
        try:
            decimal_value = Decimal(match.group("numeric_literal"))
        except InvalidOperation as exc:
            raise Qm9SplitProtocolError(
                "numeric target literal is not a valid decimal"
            ) from exc
    else:
        raise Qm9SplitProtocolError(
            "numeric target has unsupported type " + type(value).__name__
        )
    if not decimal_value.is_finite():
        raise Qm9SplitProtocolError("numeric target must be finite")
    if decimal_value.is_zero():
        return "0"
    return format(decimal_value.normalize(), "f")


def canonicalize_identity_forms(value: Any) -> tuple[str, str]:
    try:
        forms = identity_normalization.canonical_forms_from_smiles(value)
    except identity_normalization.IdentityNormalizationError as exc:
        raise Qm9SplitProtocolError(str(exc)) from exc
    return forms.strict_isomeric_smiles, forms.connectivity_smiles


def canonicalize_isomeric_smiles(value: Any) -> str:
    return canonicalize_identity_forms(value)[0]


def canonicalize_connectivity_smiles(value: Any) -> str:
    """Canonical non-isomeric identity used for grouping and protection."""
    return canonicalize_identity_forms(value)[1]


def group_id_for_connectivity_smiles(canonical_smiles: str) -> str:
    digest = sha256_bytes(canonical_smiles.encode("utf-8"))
    return "qm9-canonical-connectivity-smiles-sha256:" + digest


def _require_source_mapping(source_paths: Mapping[str, Sequence[Path | str]]) -> None:
    provided = set(source_paths)
    forbidden = sorted(provided & FORBIDDEN_SOURCE_SPLITS)
    if forbidden:
        raise Qm9SplitProtocolError(
            "the clean QM9 derivation forbids released test input because it duplicates validation"
        )
    unknown = sorted(provided - set(ALLOWED_SOURCE_SPLITS))
    if unknown:
        raise Qm9SplitProtocolError("unknown source split(s): " + ", ".join(unknown))
    missing = [split for split in ALLOWED_SOURCE_SPLITS if not source_paths.get(split)]
    if missing:
        raise Qm9SplitProtocolError(
            "both frozen source splits are required; missing " + ", ".join(missing)
        )


def bind_source_files(
    source_paths: Mapping[str, Sequence[Path | str]],
) -> list[SourceFile]:
    _require_source_mapping(source_paths)
    bound: list[SourceFile] = []
    for split in ALLOWED_SOURCE_SPLITS:
        paths = sorted(
            (Path(value) for value in source_paths[split]),
            key=lambda path: (path.name, str(path)),
        )
        for file_ordinal, path in enumerate(paths):
            if not path.is_file():
                raise Qm9SplitProtocolError("source is not a regular file: " + str(path))
            byte_count, digest = sha256_file(path)
            bound.append(
                SourceFile(
                    split=split,
                    file_ordinal=file_ordinal,
                    path=path,
                    bytes=byte_count,
                    sha256=digest,
                )
            )
    return bound


def require_production_protocol(
    source_files: Sequence[SourceFile],
    *,
    train_group_count: int,
    validation_group_count: int,
) -> None:
    """Require the scientific input layout and split scale for production.

    File names, byte sizes, and SHA-256 observations are deliberately absent
    from this admission decision.  The official Hugging Face revision, full
    row parsing, required schema, pinned canonicalization version, and final
    semantic census define the source protocol instead.
    """

    if rdBase.rdkitVersion != PRODUCTION_RDKIT_VERSION:
        raise Qm9SplitProtocolError(
            "production QM9 canonicalization requires RDKit "
            + PRODUCTION_RDKIT_VERSION
            + "; observed "
            + rdBase.rdkitVersion
        )

    by_split: dict[str, list[SourceFile]] = defaultdict(list)
    for source in source_files:
        by_split[source.split].append(source)
    for split in ALLOWED_SOURCE_SPLITS:
        files = by_split[split]
        if len(files) != 1:
            raise Qm9SplitProtocolError(
                f"production {split} source must contain exactly one Parquet file"
            )
        source = files[0]
        if source.path.suffix.lower() != ".parquet":
            raise Qm9SplitProtocolError(
                f"production {split} source must be one Parquet file"
            )
    if train_group_count != DEFAULT_TRAIN_GROUP_COUNT:
        raise Qm9SplitProtocolError(
            "production train_group_count must equal "
            + str(DEFAULT_TRAIN_GROUP_COUNT)
        )
    if validation_group_count != DEFAULT_VALIDATION_GROUP_COUNT:
        raise Qm9SplitProtocolError(
            "production validation_group_count must equal "
            + str(DEFAULT_VALIDATION_GROUP_COUNT)
        )


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise Qm9SplitProtocolError(
                    f"blank JSONL row at {path.name}:{line_number}"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Qm9SplitProtocolError(
                    f"invalid JSON at {path.name}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise Qm9SplitProtocolError(
                    f"JSONL row must be an object at {path.name}:{line_number}"
                )
            yield value


def _iter_parquet(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "Parquet input requires pyarrow; install it in the CPU processing environment"
        ) from exc
    parquet_file = parquet.ParquetFile(path)
    # ``schema.names`` exposes Parquet leaf names (for a list it can report
    # ``element``); ``schema_arrow.names`` is the required top-level contract.
    missing = sorted(set(REQUIRED_COLUMNS) - set(parquet_file.schema_arrow.names))
    if missing:
        raise Qm9SplitProtocolError(
            "Parquet source is missing required columns: " + ", ".join(missing)
        )
    for batch in parquet_file.iter_batches(columns=list(REQUIRED_COLUMNS)):
        for row in batch.to_pylist():
            yield row


def iter_source_rows(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        yield from _iter_jsonl(path)
        return
    if suffix == ".parquet":
        yield from _iter_parquet(path)
        return
    raise Qm9SplitProtocolError(
        "unsupported source format; expected .parquet or .jsonl: " + path.name
    )


def _require_record_columns(row: Mapping[str, Any], source_label: str) -> None:
    missing = [key for key in REQUIRED_COLUMNS if key not in row]
    if missing:
        raise Qm9SplitProtocolError(
            source_label + " is missing required columns: " + ", ".join(missing)
        )


def _semantic_material(
    canonical_smiles: str, instruction_stripped: str, normalized_target: str
) -> bytes:
    return canonical_json_bytes(
        {
            "canonical_isomeric_smiles": canonical_smiles,
            "instruction_stripped": instruction_stripped,
            "normalized_numeric_target": normalized_target,
        }
    )


def _process_sources(
    source_files: Sequence[SourceFile],
) -> tuple[
    list[RetainedRecord],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[tuple[str, int], int],
]:
    retained: list[RetainedRecord] = []
    removed_duplicates: list[dict[str, object]] = []
    source_row_counts: dict[tuple[str, int], int] = {}
    first_by_model_visible_key: dict[tuple[bytes, bytes, bytes], RetainedRecord] = {}
    retained_by_semantic: dict[str, list[RetainedRecord]] = defaultdict(list)
    stable_source_ordinal = 0

    for source in source_files:
        row_count = 0
        for row_index, row in enumerate(iter_source_rows(source.path)):
            source_label = f"{source.split}[{source.file_ordinal}] row {row_index}"
            _require_record_columns(row, source_label)
            canonical_smiles, connectivity_smiles = canonicalize_identity_forms(
                row["smiles"]
            )
            instruction = row["instruction"]
            if not isinstance(instruction, str):
                raise Qm9SplitProtocolError(source_label + " instruction must be a string")
            instruction_stripped = instruction.strip()
            if not instruction_stripped:
                raise Qm9SplitProtocolError(source_label + " instruction must not be empty")
            normalized_target = normalize_numeric_target(row["output"])
            semantic_material = _semantic_material(
                canonical_smiles, instruction_stripped, normalized_target
            )
            semantic_sha256 = sha256_bytes(semantic_material)
            selfies = row["selfies"]
            if not isinstance(selfies, str) or not selfies:
                raise Qm9SplitProtocolError(source_label + " selfies must be a non-empty string")
            selfies_bytes = selfies.encode("utf-8")
            fp_bytes = serialize_complete_molecule_fp(row["molecule_fp"])
            model_visible_key = (semantic_material, selfies_bytes, fp_bytes)
            canonical_smiles_sha256 = sha256_bytes(canonical_smiles.encode("utf-8"))
            candidate = RetainedRecord(
                source_split=source.split,
                source_file_ordinal=source.file_ordinal,
                source_file_sha256=source.sha256,
                source_row_index=row_index,
                source_ordinal=stable_source_ordinal,
                group_id=group_id_for_connectivity_smiles(connectivity_smiles),
                strict_canonical_isomeric_smiles=canonical_smiles,
                strict_canonical_isomeric_smiles_sha256=canonical_smiles_sha256,
                canonical_connectivity_smiles=connectivity_smiles,
                canonical_connectivity_smiles_sha256=sha256_bytes(
                    connectivity_smiles.encode("utf-8")
                ),
                instruction_stripped=instruction_stripped,
                normalized_numeric_target=normalized_target,
                semantic_signature_sha256=semantic_sha256,
                selfies_sha256=sha256_bytes(selfies_bytes),
                molecule_fp_serialized_sha256=sha256_bytes(fp_bytes),
                molecule_fp_serialized_bytes=len(fp_bytes),
            )
            first = first_by_model_visible_key.get(model_visible_key)
            if first is None:
                first_by_model_visible_key[model_visible_key] = candidate
                retained.append(candidate)
                retained_by_semantic[semantic_sha256].append(candidate)
            else:
                removed_duplicates.append(
                    {
                        "reason": "later_model_visible_duplicate",
                        "semantic_signature_sha256": semantic_sha256,
                        "selfies_sha256": candidate.selfies_sha256,
                        "molecule_fp_serialized_sha256": candidate.molecule_fp_serialized_sha256,
                        "molecule_fp_serialized_bytes": candidate.molecule_fp_serialized_bytes,
                        "kept": first.source_reference(),
                        "removed": candidate.source_reference(),
                    }
                )
            row_count += 1
            stable_source_ordinal += 1
        source_row_counts[(source.split, source.file_ordinal)] = row_count

    retained_state_variants: list[dict[str, object]] = []
    for semantic_sha256 in sorted(retained_by_semantic):
        records = retained_by_semantic[semantic_sha256]
        fp_hashes = sorted({record.molecule_fp_serialized_sha256 for record in records})
        if len(fp_hashes) <= 1:
            continue
        retained_state_variants.append(
            {
                "semantic_signature_sha256": semantic_sha256,
                "group_id": records[0].group_id,
                "retained_record_count": len(records),
                "distinct_molecule_fp_serialization_count": len(fp_hashes),
                "molecule_fp_serialized_sha256": fp_hashes,
                "records": [record.source_reference() for record in records],
                "interpretation": "retained_equal_semantics_with_distinct_complete_e3fp_state",
            }
        )
    return retained, removed_duplicates, retained_state_variants, source_row_counts


def assign_group_splits(
    records: Sequence[RetainedRecord],
    train_group_count: int,
    validation_group_count: int,
) -> tuple[dict[str, str], dict[str, object]]:
    if train_group_count < 0 or validation_group_count < 0:
        raise Qm9SplitProtocolError("group counts must be non-negative")
    group_ids = sorted({record.group_id for record in records})
    required = train_group_count + validation_group_count
    if len(group_ids) <= required:
        raise Qm9SplitProtocolError(
            "group count must exceed train_group_count + validation_group_count so test is non-empty"
        )
    generator = np.random.default_rng(RNG_SEED)
    if type(generator.bit_generator).__name__ != "PCG64":
        raise RuntimeError("NumPy default_rng(42) is not backed by PCG64 in this environment")
    permutation = generator.permutation(len(group_ids))
    permuted_group_ids = [group_ids[int(index)] for index in permutation]
    assignment: dict[str, str] = {}
    boundaries = (
        ("train", 0, train_group_count),
        ("validation", train_group_count, required),
        ("test", required, len(group_ids)),
    )
    for split, start, end in boundaries:
        for group_id in permuted_group_ids[start:end]:
            assignment[group_id] = split
    if set(assignment) != set(group_ids):
        raise RuntimeError("internal split assignment did not cover each molecule group exactly once")
    return assignment, {
        "generator": "numpy.random.default_rng",
        "numpy_version": np.__version__,
        "bit_generator": "PCG64",
        "seed": RNG_SEED,
        "input_group_order": "group_id_unicode_lexicographic_ascending",
        "assignment_order": "permutation_then_train_validation_test_contiguous_slices",
        "train_group_count": train_group_count,
        "validation_group_count": validation_group_count,
        "test_group_count": len(group_ids) - required,
    }


def _write_new(path: Path, payload: bytes) -> dict[str, object]:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _write_json_new(path: Path, value: object) -> dict[str, object]:
    return _write_new(path, canonical_json_bytes(value) + b"\n")


def require_production_semantic_census(observed: Mapping[str, object]) -> None:
    """Reject production materialization whose semantic census has drifted."""

    admission_view = {
        key: observed.get(key) for key in PRODUCTION_EXPECTED_COUNTS
    }
    if admission_view != PRODUCTION_EXPECTED_COUNTS:
        raise Qm9SplitProtocolError(
            "QM9 semantic census differs from the preregistered production counts: "
            + canonical_json_bytes(admission_view).decode("utf-8")
        )


def build_qm9_identity_split(
    source_paths: Mapping[str, Sequence[Path | str]],
    output_dir: Path | str,
    *,
    train_group_count: int = DEFAULT_TRAIN_GROUP_COUNT,
    validation_group_count: int = DEFAULT_VALIDATION_GROUP_COUNT,
    enforce_production_protocol: bool = True,
) -> dict[str, object]:
    """Materialize the molecule-group-disjoint split and return its summary.

    ``train_group_count`` and ``validation_group_count`` exist only so unit
    fixtures can exercise the exact algorithm with a few groups.  Production
    CLI runs fix the official revision, one-Parquet-per-split layout,
    110000/10000 boundaries, full semantic census, and all row-level protocol
    checks.  File names, byte sizes, and SHA-256 values remain observations and
    do not decide admission.  Tests using synthetic JSONL must explicitly set
    ``enforce_production_protocol=False``.
    """
    _require_source_mapping(source_paths)
    output_path = Path(output_dir)
    if output_path.exists():
        raise Qm9SplitProtocolError("output directory already exists: " + str(output_path))
    source_files = bind_source_files(source_paths)
    if not isinstance(enforce_production_protocol, bool):
        raise Qm9SplitProtocolError("enforce_production_protocol must be Boolean")
    if enforce_production_protocol:
        require_production_protocol(
            source_files,
            train_group_count=train_group_count,
            validation_group_count=validation_group_count,
        )
    retained, removed, state_variants, source_row_counts = _process_sources(source_files)
    if not retained:
        raise Qm9SplitProtocolError("no records were retained from the frozen sources")
    assignments, rng_contract = assign_group_splits(
        retained,
        train_group_count=train_group_count,
        validation_group_count=validation_group_count,
    )

    row_counts = Counter(assignments[record.group_id] for record in retained)
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for record in retained:
        groups_by_split[assignments[record.group_id]].add(record.group_id)
    input_row_count = sum(source_row_counts.values())
    if input_row_count != len(retained) + len(removed):
        raise RuntimeError("internal row accounting mismatch")
    if sum(row_counts.values()) != len(retained):
        raise RuntimeError("internal split row accounting mismatch")
    group_union = set().union(
        *(groups_by_split[name] for name in ("train", "validation", "test"))
    )
    if len(group_union) != sum(
        len(groups_by_split[name]) for name in ("train", "validation", "test")
    ):
        raise RuntimeError("a molecule group crossed output splits")
    observed_counts = {
        "input_rows": input_row_count,
        "retained_rows": len(retained),
        "removed_model_visible_duplicates": len(removed),
        "molecule_groups": len(group_union),
        "output_rows": {
            split: row_counts[split] for split in ("train", "validation", "test")
        },
        "output_groups": {
            split: len(groups_by_split[split])
            for split in ("train", "validation", "test")
        },
    }
    if enforce_production_protocol:
        require_production_semantic_census(observed_counts)

    identity_spec_id = sha256_bytes(IDENTITY_CONTRACT_PATH.read_bytes())

    output_path.mkdir(parents=True, exist_ok=False)

    canonicalization_contract = {
        "library": "RDKit",
        "rdkit_version": rdBase.rdkitVersion,
        "production_required_version": (
            PRODUCTION_RDKIT_VERSION if enforce_production_protocol else None
        ),
        "input_parser": "Chem.MolFromSmiles(smiles.strip())",
        "serializer": "Chem.MolToSmiles",
        "split_molecule_identity": {
            "name": "canonical_non_isomeric_connectivity_smiles",
            "parameters": {
                "canonical": True,
                "isomericSmiles": False,
                "kekuleSmiles": False,
            },
            "group_identity": "sha256_utf8_canonical_non_isomeric_connectivity_smiles",
            "stereoisomer_policy": "retain_all_states_and_assign_them_with_the_connectivity_group",
        },
        "protected_union_identity": {
            "name": "canonical_non_isomeric_connectivity_smiles",
            "parameters": {
                "canonical": True,
                "isomericSmiles": False,
                "kekuleSmiles": False,
            },
            "role": "same_identity_used_for_split_grouping_and_p1_p2_decontamination",
        },
        "explicit_hydrogen_projection": {
            "operation": "Chem.RemoveHs(Chem.Mol(mol), RemoveHsParameters(), sanitize=True)",
            "removeDefiningBondStereo": True,
            "post_steps": ["Chem.SanitizeMol", "Chem.AssignStereochemistry(cleanIt=True,force=True)"],
        },
    }
    source_entries = []
    for source in source_files:
        source_entries.append(
            {
                "source_split": source.split,
                "source_file_ordinal": source.file_ordinal,
                "file_name": source.path.name,
                "bytes": source.bytes,
                "sha256": source.sha256,
                "observation_role": "provenance_metadata_not_admission_criterion",
                "row_count": source_row_counts[(source.split, source.file_ordinal)],
            }
        )
    source_manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "dataset_id": "3dmolt5-e3fp-mol-instructions-qm9-clean-view",
        "split_protocol_id": SPLIT_PROTOCOL_ID,
        "frozen_source_revision": SOURCE_REVISION,
        "production_protocol_enforced": enforce_production_protocol,
        "source_file_observation_policy": (
            "file_name_bytes_and_sha256_are_recorded_for_provenance_only_"
            "and_do_not_decide_scientific_admission"
        ),
        "expected_production_counts": (
            PRODUCTION_EXPECTED_COUNTS if enforce_production_protocol else None
        ),
        "accepted_source_splits": list(ALLOWED_SOURCE_SPLITS),
        "forbidden_source_split": {
            "name": "test",
            "reason": "released test artifact is byte-identical to validation",
        },
        "stable_source_order": "train_then_validation; within split file_name_then_supplied_path_unicode_lexicographic; within file row_index_ascending",
        "required_columns": list(REQUIRED_COLUMNS),
        "source_files": source_entries,
        "canonicalization": canonicalization_contract,
        "identity_normalization_contract_sha256": identity_spec_id,
        "semantic_signature": {
            "fields": [
                "strict_canonical_isomeric_smiles",
                "instruction.strip()",
                "Decimal_normalized_numeric_target",
            ],
            "serialization": "canonical_json_utf8_v1",
            "digest": "sha256",
            "released_numeric_target_grammar": {
                "scope": "full stripped output string; substring extraction is forbidden",
                "numeric_literal_regex": RELEASED_NUMERIC_TARGET_PATTERN.pattern,
                "terminal_sentence_period": "optional; at most one",
                "units_extra_tokens_and_malformed_punctuation": "rejected",
            },
            "numeric_normalization": "parse the matched literal as finite Decimal; map signed zero to 0; Decimal.normalize(); format with f and no exponent",
        },
        "model_visible_duplicate_identity": {
            "fields": [
                "semantic_signature_sha256",
                "exact_utf8_selfies",
                "complete_molecule_fp_canonical_json_serialization",
            ],
            "keep_rule": "retain_first_in_stable_source_order",
            "different_e3fp_rule": "retain_as_state_in_same_molecule_group",
        },
    }
    source_artifact = _write_json_new(output_path / SOURCE_MANIFEST_FILENAME, source_manifest)

    split_rows = []
    for record in retained:  # already in stable source order
        split_rows.append(
            {
                "schema_version": SPLIT_ROW_SCHEMA,
                "record_id": record.record_id,
                "assigned_split": assignments[record.group_id],
                "source": record.source_reference(),
                "group_id": record.group_id,
                "strict_canonical_isomeric_smiles": record.strict_canonical_isomeric_smiles,
                "strict_canonical_isomeric_smiles_sha256": record.strict_canonical_isomeric_smiles_sha256,
                "canonical_connectivity_smiles": record.canonical_connectivity_smiles,
                "canonical_connectivity_smiles_sha256": record.canonical_connectivity_smiles_sha256,
                "instruction_stripped": record.instruction_stripped,
                "normalized_numeric_target": record.normalized_numeric_target,
                "semantic_signature_sha256": record.semantic_signature_sha256,
                "selfies_sha256": record.selfies_sha256,
                "molecule_fp_serialized_sha256": record.molecule_fp_serialized_sha256,
                "molecule_fp_serialized_bytes": record.molecule_fp_serialized_bytes,
            }
        )
    split_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in split_rows)
    split_artifact = _write_new(output_path / SPLIT_MANIFEST_FILENAME, split_payload)
    split_artifact["row_count"] = len(split_rows)

    duplicate_report = {
        "schema_version": DUPLICATE_REPORT_SCHEMA,
        "frozen_source_revision": SOURCE_REVISION,
        "input_row_count": sum(source_row_counts.values()),
        "retained_record_count": len(retained),
        "removed_model_visible_duplicate_count": len(removed),
        "removed_model_visible_duplicates": removed,
        "retained_equal_semantics_distinct_e3fp_signature_count": len(state_variants),
        "retained_equal_semantics_distinct_e3fp_states": state_variants,
        "decision_boundary": "remove only when semantic signature, exact SELFIES, and complete serialized molecule_fp are all equal",
    }
    duplicate_artifact = _write_json_new(
        output_path / DUPLICATE_REPORT_FILENAME, duplicate_report
    )

    summary = {
        "schema_version": SPLIT_SUMMARY_SCHEMA,
        "dataset_id": "3dmolt5-e3fp-mol-instructions-qm9-clean-view",
        "split_protocol_id": SPLIT_PROTOCOL_ID,
        "frozen_source_revision": SOURCE_REVISION,
        "canonicalization": canonicalization_contract,
        "identity_normalization_contract_sha256": identity_spec_id,
        "rng_contract": rng_contract,
        "counts": {
            "input_rows": input_row_count,
            "retained_rows": len(retained),
            "removed_model_visible_duplicates": len(removed),
            "molecule_groups": len(group_union),
            "source_rows": {
                split: sum(
                    count
                    for (source_split, _), count in source_row_counts.items()
                    if source_split == split
                )
                for split in ALLOWED_SOURCE_SPLITS
            },
            "output_rows": {
                split: row_counts[split] for split in ("train", "validation", "test")
            },
            "output_groups": {
                split: len(groups_by_split[split])
                for split in ("train", "validation", "test")
            },
        },
        "invariants": {
            "source_test_not_consumed": True,
            "production_semantic_census_enforced": enforce_production_protocol,
            "all_input_rows_accounted_for": input_row_count == len(retained) + len(removed),
            "each_retained_row_assigned_once": sum(row_counts.values()) == len(retained),
            "connectivity_groups_are_split_disjoint": True,
            "all_stereoisomer_states_follow_their_connectivity_group": True,
            "different_e3fp_states_are_retained": True,
        },
        "artifacts": {
            "source_manifest": source_artifact,
            "split_manifest": split_artifact,
            "duplicate_report": duplicate_artifact,
        },
    }
    _write_json_new(output_path / SPLIT_SUMMARY_FILENAME, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        action="append",
        required=True,
        type=Path,
        help="Frozen source train Parquet path; repeat for multiple shards.",
    )
    parser.add_argument(
        "--validation",
        action="append",
        required=True,
        type=Path,
        help="Frozen source validation Parquet path; repeat for multiple shards.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    summary = build_qm9_identity_split(
        {"train": arguments.train, "validation": arguments.validation},
        arguments.output_dir,
        train_group_count=DEFAULT_TRAIN_GROUP_COUNT,
        validation_group_count=DEFAULT_VALIDATION_GROUP_COUNT,
        enforce_production_protocol=True,
    )
    print(canonical_json_bytes(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Qm9SplitProtocolError, RuntimeError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
