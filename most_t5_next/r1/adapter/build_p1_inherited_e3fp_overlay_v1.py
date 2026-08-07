#!/usr/bin/env python3
"""Build the bounded 128-record inherited-E3FP overlay.

The completed PCQM production-v2 release remains immutable and keeps its raw
shell-identifier semantic.  This adapter reuses its exact sparse release
reader and SDF transport, replays the frozen hydrogen projection, and derives
raw plus explicit duplicate-pointer inheritance from one E3FP run per selected
molecule.  The replayed raw matrix, atom universe and coordinates must match
the production payload before any output path is created.

The result is a sample-scope-only LMDB overlay.  It is not a training-data
admission.  Values use the existing sidecar-v2 wire codec, and the manifest is
published only after all staged values have been decoded and revalidated.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from most_t5_next.r1.adapter import run_p1_topology_canary_v1 as release_reader
from most_t5_next.r1.adapter import sidecar_v2_codec
from most_t5_next.r1.gates import pcqm_e3fp_preflight as projection
from most_t5_next.r1.semantic import e3fp_duplicate_inheritance_v1 as inheritance


SOURCE_RECORD_COUNT = 3_378_606
SAMPLE_COUNT = 128
SCHEDULE_SCHEMA = "most-t5-r1/p1-inherited-e3fp-overlay-schedule/v1"
RECORD_SCHEMA = "most-t5-r1/p1-inherited-e3fp-overlay-record/v1"
MEMBERSHIP_SCHEMA = "most-t5-r1/p1-inherited-e3fp-overlay-membership/v1"
MANIFEST_SCHEMA = "most-t5-r1/p1-inherited-e3fp-overlay-manifest/v1"
EXPECTED_BASE_RECORD_SCHEMA = release_reader.PRODUCTION_RECORD_SCHEMA
SCHEDULE_RULE = "floor(source_record_count*schedule_index/128)"
STAGING_LMDB_NAME = "inherited_e3fp_overlay.lmdb.staging"
FINAL_LMDB_NAME = "inherited_e3fp_overlay.lmdb"
STAGING_MEMBERSHIP_NAME = "membership.jsonl.staging"
FINAL_MEMBERSHIP_NAME = "membership.jsonl"
STAGING_SCHEDULE_NAME = "schedule.json.staging"
FINAL_SCHEDULE_NAME = "schedule.json"
MANIFEST_NAME = "manifest.json"


class InheritedE3FPOverlayError(RuntimeError):
    """The bounded overlay cannot be bound to production-v2 exactly."""


def frozen_schedule() -> tuple[int, ...]:
    """Return the immutable 128-record schedule without admission fallback."""

    ordinals = tuple((SOURCE_RECORD_COUNT * index) // SAMPLE_COUNT for index in range(SAMPLE_COUNT))
    if (
        len(ordinals) != SAMPLE_COUNT
        or len(set(ordinals)) != SAMPLE_COUNT
        or ordinals[0] != 0
        or ordinals[-1] != 3_352_210
        or any(left >= right for left, right in zip(ordinals, ordinals[1:]))
    ):
        raise InheritedE3FPOverlayError("internal frozen schedule invariant failed")
    return ordinals


def schedule_sha256(ordinals: Sequence[int] | None = None) -> str:
    values = tuple(frozen_schedule() if ordinals is None else ordinals)
    return release_reader.sha256_json(list(values))


def build_schedule_document() -> dict:
    ordinals = frozen_schedule()
    return {
        "schema_version": SCHEDULE_SCHEMA,
        "sample_scope_only": True,
        "training_admission": False,
        "source_record_count": SOURCE_RECORD_COUNT,
        "sample_count": SAMPLE_COUNT,
        "selection_rule": SCHEDULE_RULE,
        "ordered_ordinals_sha256": schedule_sha256(ordinals),
        "ordinals": list(ordinals),
    }


def _require_sha256(value, label: str) -> None:
    if not release_reader._is_sha256(value):
        raise InheritedE3FPOverlayError(f"{label} must be a lower-case SHA-256")


def _array_sha256(array) -> str:
    return sidecar_v2_codec.sha256_bytes(array.tobytes(order="C"))


def _require_exact_keys(value, expected, label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(expected):
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise InheritedE3FPOverlayError(f"{label} fields differ: {observed}")


def validate_base_binding(np, binding: Mapping[str, object], ordinal: int) -> tuple[dict, dict]:
    """Validate the base key/member/logical-content binding used by the overlay."""

    if not isinstance(binding, dict):
        raise InheritedE3FPOverlayError("base binding must be an object")
    record = binding.get("record")
    membership = binding.get("membership")
    if not isinstance(record, dict) or not isinstance(membership, dict):
        raise InheritedE3FPOverlayError("base binding lacks record or membership")
    if record.get("record_schema_version") != EXPECTED_BASE_RECORD_SCHEMA:
        raise InheritedE3FPOverlayError("base payload is not production-v2")
    member = record.get("member")
    atom_universe = record.get("atom_universe")
    geometry = record.get("geometry")
    if not all(isinstance(value, dict) for value in (member, atom_universe, geometry)):
        raise InheritedE3FPOverlayError("base payload lacks member, atom universe or geometry")

    expected_key = f"{ordinal:09d}"
    if not (
        membership.get("disposition") == "admit"
        and membership.get("sdf_record_index") == ordinal
        and membership.get("record_storage_key") == expected_key
        and member.get("sdf_record_index") == ordinal
        and member.get("member_id") == membership.get("member_id")
        and member.get("storage_key") == expected_key
    ):
        raise InheritedE3FPOverlayError("base member and storage key are not exactly bound")
    expected_content = membership.get("record_content_sha256")
    _require_sha256(expected_content, "base record content hash")
    observed_content = sidecar_v2_codec.logical_record_sha256(np, record)
    if observed_content != expected_content:
        raise InheritedE3FPOverlayError("base payload logical content differs from membership")

    model_atom_count = atom_universe.get("model_atom_count")
    source_atom_count = atom_universe.get("source_atom_count")
    mapping = atom_universe.get("model_to_source_atom_index")
    coordinates = geometry.get("coordinates")
    base_e3fp = geometry.get("e3fp")
    if (
        not isinstance(model_atom_count, int)
        or isinstance(model_atom_count, bool)
        or model_atom_count <= 0
        or not isinstance(source_atom_count, int)
        or isinstance(source_atom_count, bool)
        or source_atom_count <= 0
    ):
        raise InheritedE3FPOverlayError("base atom counts are invalid")
    if (
        not isinstance(mapping, np.ndarray)
        or mapping.dtype != np.int32
        or mapping.shape != (model_atom_count,)
        or not mapping.flags.c_contiguous
    ):
        raise InheritedE3FPOverlayError("base model-to-source mapping is invalid")
    if (
        not isinstance(coordinates, np.ndarray)
        or coordinates.dtype != np.float32
        or coordinates.shape != (model_atom_count, 3)
        or not coordinates.flags.c_contiguous
    ):
        raise InheritedE3FPOverlayError("base coordinates are invalid")
    if (
        not isinstance(base_e3fp, np.ndarray)
        or base_e3fp.dtype != np.int32
        or base_e3fp.shape != (model_atom_count, inheritance.FP_LEVEL + 1)
        or not base_e3fp.flags.c_contiguous
    ):
        raise InheritedE3FPOverlayError("base E3FP matrix is invalid")
    if geometry.get("coordinates_sha256") != _array_sha256(coordinates):
        raise InheritedE3FPOverlayError("base coordinate hash does not describe its array")
    if geometry.get("e3fp_sha256") != _array_sha256(base_e3fp):
        raise InheritedE3FPOverlayError("base E3FP hash does not describe its array")
    return record, membership


def validate_overlay_release_record(
    record: Mapping[str, object], membership: Mapping[str, object], ordinal: int
) -> None:
    """Validate the release join without importing topology-only locks."""

    if not isinstance(record, dict) or not isinstance(membership, dict):
        raise InheritedE3FPOverlayError("selected release row is not an object")
    member = record.get("member")
    if record.get("record_schema_version") != EXPECTED_BASE_RECORD_SCHEMA or not isinstance(
        member, dict
    ):
        raise InheritedE3FPOverlayError("selected release payload is not production-v2")
    expected_key = f"{ordinal:09d}"
    if not (
        membership.get("disposition") == "admit"
        and membership.get("sdf_record_index") == ordinal
        and membership.get("record_storage_key") == expected_key
        and member.get("sdf_record_index") == ordinal
        and member.get("member_id") == membership.get("member_id")
        and member.get("storage_key") == expected_key
    ):
        raise InheritedE3FPOverlayError("selected release member/key join is invalid")


def _resolved_config_sha256(resolved: Mapping[str, object]) -> str:
    if not isinstance(resolved, dict) or not resolved:
        raise InheritedE3FPOverlayError("resolved E3FP configuration is empty")
    return release_reader.sha256_json(resolved)


def validate_overlay_record(np, record: Mapping[str, object], expected_schedule_sha256: str | None = None) -> None:
    """Validate one decoded overlay record without consulting its producer."""

    _require_exact_keys(
        record,
        ("record_schema_version", "overlay", "selection", "member", "e3fp"),
        "overlay record",
    )
    if record["record_schema_version"] != RECORD_SCHEMA:
        raise InheritedE3FPOverlayError("overlay record schema mismatch")
    overlay = record["overlay"]
    _require_exact_keys(
        overlay,
        (
            "semantics_id",
            "sample_scope_only",
            "training_admission",
            "source_full_release_manifest_sha256",
            "source_logical_release_root_sha256",
            "resolved_e3fp_config_sha256",
            "inheritance_implementation_sha256",
        ),
        "overlay metadata",
    )
    if not (
        overlay["semantics_id"] == inheritance.SEMANTICS_ID
        and overlay["sample_scope_only"] is True
        and overlay["training_admission"] is False
    ):
        raise InheritedE3FPOverlayError("overlay semantic/admission boundary is invalid")
    for key in (
        "source_full_release_manifest_sha256",
        "source_logical_release_root_sha256",
        "resolved_e3fp_config_sha256",
        "inheritance_implementation_sha256",
    ):
        _require_sha256(overlay[key], f"overlay.{key}")

    selection = record["selection"]
    _require_exact_keys(selection, ("schedule_index", "schedule_sha256"), "selection")
    if (
        not isinstance(selection["schedule_index"], int)
        or isinstance(selection["schedule_index"], bool)
        or not 0 <= selection["schedule_index"] < SAMPLE_COUNT
    ):
        raise InheritedE3FPOverlayError("schedule index is invalid")
    _require_sha256(selection["schedule_sha256"], "selection.schedule_sha256")
    if expected_schedule_sha256 is not None and selection["schedule_sha256"] != expected_schedule_sha256:
        raise InheritedE3FPOverlayError("overlay record uses a different frozen schedule")

    member = record["member"]
    _require_exact_keys(
        member,
        (
            "member_id",
            "sdf_record_index",
            "record_storage_key",
            "shard_index",
            "base_record_content_sha256",
            "base_e3fp_sha256",
            "source_address_sha256",
            "geometry_mol_identity_sha256",
            "model_atom_count",
        ),
        "member",
    )
    schedule_index = selection["schedule_index"]
    expected_ordinal = frozen_schedule()[schedule_index]
    if not (
        isinstance(member["member_id"], str)
        and member["member_id"]
        and member["sdf_record_index"] == expected_ordinal
        and member["record_storage_key"] == f"{expected_ordinal:09d}"
        and isinstance(member["shard_index"], int)
        and not isinstance(member["shard_index"], bool)
        and member["shard_index"] >= 0
        and isinstance(member["model_atom_count"], int)
        and not isinstance(member["model_atom_count"], bool)
        and member["model_atom_count"] > 0
    ):
        raise InheritedE3FPOverlayError("overlay member does not match its schedule slot")
    for key in (
        "base_record_content_sha256",
        "base_e3fp_sha256",
        "source_address_sha256",
        "geometry_mol_identity_sha256",
    ):
        _require_sha256(member[key], f"member.{key}")

    e3fp = record["e3fp"]
    _require_exact_keys(
        e3fp,
        (
            "shape",
            "raw_replay_sha256",
            "raw_matches_frozen",
            "inherited",
            "inherited_sha256",
            "duplicate_mask",
            "duplicate_mask_sha256",
            "summary",
        ),
        "e3fp",
    )
    atom_count = member["model_atom_count"]
    inherited = e3fp["inherited"]
    duplicate_mask = e3fp["duplicate_mask"]
    if (
        e3fp["shape"] != [atom_count, inheritance.FP_LEVEL + 1]
        or e3fp["raw_matches_frozen"] is not True
        or not isinstance(inherited, np.ndarray)
        or inherited.dtype != np.int32
        or inherited.shape != (atom_count, inheritance.FP_LEVEL + 1)
        or not inherited.flags.c_contiguous
        or not isinstance(duplicate_mask, np.ndarray)
        or duplicate_mask.dtype != np.bool_
        or duplicate_mask.shape != inherited.shape
        or not duplicate_mask.flags.c_contiguous
    ):
        raise InheritedE3FPOverlayError("overlay E3FP arrays violate shape/dtype/admission parity")
    if (
        bool(np.any(inherited < -1))
        or bool(np.any(inherited >= inheritance.FP_BITS))
        or bool(np.any(inherited[:, 0] == -1))
        or bool(np.any(duplicate_mask[:, 0]))
        or bool(np.any(duplicate_mask & (inherited == -1)))
    ):
        raise InheritedE3FPOverlayError("overlay E3FP values or duplicate mask are invalid")
    for key in ("raw_replay_sha256", "inherited_sha256", "duplicate_mask_sha256"):
        _require_sha256(e3fp[key], f"e3fp.{key}")
    if e3fp["raw_replay_sha256"] != member["base_e3fp_sha256"]:
        raise InheritedE3FPOverlayError("raw replay is not bound to the base E3FP array")
    if e3fp["inherited_sha256"] != _array_sha256(inherited):
        raise InheritedE3FPOverlayError("inherited E3FP hash mismatch")
    if e3fp["duplicate_mask_sha256"] != _array_sha256(duplicate_mask):
        raise InheritedE3FPOverlayError("duplicate mask hash mismatch")

    summary = e3fp["summary"]
    expected_summary_keys = {
        "semantics_id",
        "shells_seen",
        "slots_populated",
        "duplicate_slots",
        "duplicate_atoms",
        "changed_identifier_slots",
        "changed_token_slots",
    }
    _require_exact_keys(summary, expected_summary_keys, "e3fp.summary")
    if summary["semantics_id"] != inheritance.SEMANTICS_ID:
        raise InheritedE3FPOverlayError("E3FP summary semantic ID mismatch")
    for key in expected_summary_keys - {"semantics_id"}:
        if not isinstance(summary[key], int) or isinstance(summary[key], bool) or summary[key] < 0:
            raise InheritedE3FPOverlayError(f"E3FP summary {key} is invalid")
    if summary["duplicate_slots"] != int(duplicate_mask.sum()):
        raise InheritedE3FPOverlayError("duplicate-mask count differs from E3FP summary")


def build_overlay_record(
    Chem,
    np,
    *,
    schedule_index: int,
    binding: Mapping[str, object],
    source_mol,
    e3fp_api: Mapping[str, object],
    source_full_release_manifest_sha256: str,
    source_logical_release_root_sha256: str,
    inheritance_implementation_sha256: str,
) -> tuple[dict, dict]:
    """Replay one source molecule and construct an unwritten overlay row."""

    if not isinstance(schedule_index, int) or isinstance(schedule_index, bool) or not 0 <= schedule_index < SAMPLE_COUNT:
        raise InheritedE3FPOverlayError("schedule index is outside the frozen sample")
    ordinal = frozen_schedule()[schedule_index]
    record, membership = validate_base_binding(np, binding, ordinal)
    for value, label in (
        (source_full_release_manifest_sha256, "source full release manifest"),
        (source_logical_release_root_sha256, "source logical release root"),
        (inheritance_implementation_sha256, "inheritance implementation"),
    ):
        _require_sha256(value, label)

    tagged, source_atom_count, _ = projection.tag_source_atoms(Chem, source_mol)
    geometry_mol, model_to_source = projection.project_hydrogens(
        Chem, tagged, source_atom_count
    )
    atom_universe = record["atom_universe"]
    geometry = record["geometry"]
    expected_mapping = atom_universe["model_to_source_atom_index"]
    observed_mapping = np.ascontiguousarray(np.asarray(model_to_source, dtype=np.int32))
    if not (
        source_atom_count == atom_universe["source_atom_count"]
        and int(geometry_mol.GetNumAtoms()) == atom_universe["model_atom_count"]
        and observed_mapping.shape == expected_mapping.shape
        and bool(np.array_equal(observed_mapping, expected_mapping))
    ):
        raise InheritedE3FPOverlayError(
            "replayed source/model atom mapping differs from production-v2"
        )
    try:
        replayed_coordinates = np.ascontiguousarray(
            np.asarray(geometry_mol.GetConformer(0).GetPositions(), dtype=np.float32)
        )
    except (OSError, json.JSONDecodeError, release_reader.TopologyCanaryError) as exc:
        raise InheritedE3FPOverlayError("replayed geometry coordinates are unavailable") from exc
    if not bool(np.array_equal(replayed_coordinates, geometry["coordinates"])):
        raise InheritedE3FPOverlayError("replayed coordinates differ from production-v2")

    raw, inherited_ids, duplicate_mask, summary, resolved = (
        inheritance.generate_e3fp_projection_pair(
            np, e3fp_api, geometry_mol, ordinal
        )
    )
    base_e3fp = geometry["e3fp"]
    if not bool(np.array_equal(raw, base_e3fp)):
        raise InheritedE3FPOverlayError(
            "replayed raw E3FP differs from frozen production-v2"
        )
    raw_sha = _array_sha256(raw)
    if raw_sha != geometry["e3fp_sha256"]:
        raise InheritedE3FPOverlayError("raw replay hash differs from frozen base E3FP hash")

    inherited_ids = np.ascontiguousarray(inherited_ids, dtype=np.int32)
    duplicate_mask = np.ascontiguousarray(duplicate_mask, dtype=np.bool_)
    overlay_record = {
        "record_schema_version": RECORD_SCHEMA,
        "overlay": {
            "semantics_id": inheritance.SEMANTICS_ID,
            "sample_scope_only": True,
            "training_admission": False,
            "source_full_release_manifest_sha256": source_full_release_manifest_sha256,
            "source_logical_release_root_sha256": source_logical_release_root_sha256,
            "resolved_e3fp_config_sha256": _resolved_config_sha256(resolved),
            "inheritance_implementation_sha256": inheritance_implementation_sha256,
        },
        "selection": {
            "schedule_index": int(schedule_index),
            "schedule_sha256": schedule_sha256(),
        },
        "member": {
            "member_id": membership["member_id"],
            "sdf_record_index": int(ordinal),
            "record_storage_key": membership["record_storage_key"],
            "shard_index": int(binding["shard_index"]),
            "base_record_content_sha256": membership["record_content_sha256"],
            "base_e3fp_sha256": geometry["e3fp_sha256"],
            "source_address_sha256": record["member"]["source_address_sha256"],
            "geometry_mol_identity_sha256": atom_universe["geometry_mol_identity_sha256"],
            "model_atom_count": int(atom_universe["model_atom_count"]),
        },
        "e3fp": {
            "shape": [int(inherited_ids.shape[0]), int(inherited_ids.shape[1])],
            "raw_replay_sha256": raw_sha,
            "raw_matches_frozen": True,
            "inherited": inherited_ids,
            "inherited_sha256": _array_sha256(inherited_ids),
            "duplicate_mask": duplicate_mask,
            "duplicate_mask_sha256": _array_sha256(duplicate_mask),
            "summary": dict(summary),
        },
    }
    validate_overlay_record(np, overlay_record, schedule_sha256())
    return overlay_record, resolved


def _membership_row(np, record: dict, payload: bytes) -> dict:
    member = record["member"]
    logical_hash = sidecar_v2_codec.logical_record_sha256(np, record)
    return {
        "schema_version": MEMBERSHIP_SCHEMA,
        "schedule_index": record["selection"]["schedule_index"],
        "schedule_sha256": record["selection"]["schedule_sha256"],
        "member_id": member["member_id"],
        "sdf_record_index": member["sdf_record_index"],
        "record_storage_key": member["record_storage_key"],
        "base_record_content_sha256": member["base_record_content_sha256"],
        "base_e3fp_sha256": member["base_e3fp_sha256"],
        "overlay_record_content_sha256": logical_hash,
        "overlay_wire_bytes": int(len(payload)),
        "overlay_wire_sha256": sidecar_v2_codec.sha256_bytes(payload),
    }


def _validate_membership(row: Mapping[str, object], record: Mapping[str, object]) -> None:
    _require_exact_keys(
        row,
        (
            "schema_version",
            "schedule_index",
            "schedule_sha256",
            "member_id",
            "sdf_record_index",
            "record_storage_key",
            "base_record_content_sha256",
            "base_e3fp_sha256",
            "overlay_record_content_sha256",
            "overlay_wire_bytes",
            "overlay_wire_sha256",
        ),
        "overlay membership",
    )
    member = record["member"]
    if not (
        row["schema_version"] == MEMBERSHIP_SCHEMA
        and row["schedule_index"] == record["selection"]["schedule_index"]
        and row["schedule_sha256"] == record["selection"]["schedule_sha256"]
        and row["member_id"] == member["member_id"]
        and row["sdf_record_index"] == member["sdf_record_index"]
        and row["record_storage_key"] == member["record_storage_key"]
        and row["base_record_content_sha256"] == member["base_record_content_sha256"]
        and row["base_e3fp_sha256"] == member["base_e3fp_sha256"]
    ):
        raise InheritedE3FPOverlayError("overlay membership is not bound to its record/base")
    for key in (
        "schedule_sha256",
        "base_record_content_sha256",
        "base_e3fp_sha256",
        "overlay_record_content_sha256",
        "overlay_wire_sha256",
    ):
        _require_sha256(row[key], f"membership.{key}")
    if not isinstance(row["overlay_wire_bytes"], int) or row["overlay_wire_bytes"] <= 0:
        raise InheritedE3FPOverlayError("overlay membership wire byte count is invalid")


def _write_json(path: Path, value: object) -> None:
    with path.open("xb") as handle:
        handle.write(release_reader.canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(release_reader.canonical_json_bytes(row) + b"\n")


def _read_jsonl(path: Path) -> list[dict]:
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise InheritedE3FPOverlayError("staged membership row is not an object")
            result.append(value)
    return result


def load_overlay_readonly(
    overlay_root: Path,
    *,
    np,
    lmdb_module,
) -> tuple[dict[int, dict], dict]:
    """Load the published 128-record overlay without opening a write handle.

    The returned mapping is keyed by the frozen source-record ordinal.  Each
    value contains the decoded ``record`` and its exact ``membership`` row;
    the second return value is the pass manifest that bound the artifacts.
    """

    overlay_root = Path(overlay_root).expanduser().resolve()
    if not overlay_root.is_dir():
        raise InheritedE3FPOverlayError("overlay root is not a directory")
    try:
        manifest = release_reader.load_json(
            overlay_root / MANIFEST_NAME, "inherited-E3FP overlay manifest"
        )
    except Exception as exc:
        raise InheritedE3FPOverlayError("overlay manifest could not be loaded") from exc
    if not (
        manifest.get("schema_version") == MANIFEST_SCHEMA
        and manifest.get("status") == "pass"
        and manifest.get("sample_scope_only") is True
        and manifest.get("training_admission") is False
    ):
        raise InheritedE3FPOverlayError("a pass sample-scope overlay manifest is required")

    expected_schedule_sha = schedule_sha256()
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or not (
        selection.get("schedule_schema_version") == SCHEDULE_SCHEMA
        and selection.get("schedule_sha256") == expected_schedule_sha
        and selection.get("sample_count") == SAMPLE_COUNT
        and selection.get("source_record_count") == SOURCE_RECORD_COUNT
        and selection.get("selection_rule") == SCHEDULE_RULE
        and selection.get("no_next_admitted_replacement") is True
    ):
        raise InheritedE3FPOverlayError("overlay manifest does not bind the frozen schedule")
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or not (
        counts.get("scheduled_records") == SAMPLE_COUNT
        and counts.get("overlay_records") == SAMPLE_COUNT
        and counts.get("raw_parity_count") == SAMPLE_COUNT
        and counts.get("failed_records") == 0
    ):
        raise InheritedE3FPOverlayError("overlay manifest record counts are not a complete pass")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise InheritedE3FPOverlayError("overlay manifest lacks artifact bindings")
    membership_artifact = artifacts.get("membership")
    schedule_artifact = artifacts.get("schedule")
    lmdb_artifact = artifacts.get("overlay_lmdb")
    if not (
        isinstance(membership_artifact, dict)
        and membership_artifact.get("relative_path") == FINAL_MEMBERSHIP_NAME
        and membership_artifact.get("row_count") == SAMPLE_COUNT
        and isinstance(schedule_artifact, dict)
        and schedule_artifact.get("relative_path") == FINAL_SCHEDULE_NAME
        and isinstance(lmdb_artifact, dict)
        and lmdb_artifact.get("relative_path") == FINAL_LMDB_NAME
        and lmdb_artifact.get("subdir") is True
    ):
        raise InheritedE3FPOverlayError("overlay manifest artifact paths or counts differ")

    membership_path = overlay_root / FINAL_MEMBERSHIP_NAME
    schedule_path = overlay_root / FINAL_SCHEDULE_NAME
    lmdb_path = overlay_root / FINAL_LMDB_NAME
    data_path = lmdb_path / "data.mdb"
    for path, artifact, byte_key, hash_key, label in (
        (membership_path, membership_artifact, "bytes", "sha256", "membership"),
        (schedule_path, schedule_artifact, "bytes", "sha256", "schedule"),
        (data_path, lmdb_artifact, "data_mdb_bytes", "data_mdb_sha256", "LMDB data"),
    ):
        if not path.is_file():
            raise InheritedE3FPOverlayError(f"overlay {label} artifact is missing")
        if (
            artifact.get(byte_key) != int(path.stat().st_size)
            or artifact.get(hash_key) != release_reader.sha256_file(path)
        ):
            raise InheritedE3FPOverlayError(f"overlay {label} artifact differs from manifest")

    try:
        with schedule_path.open("r", encoding="utf-8") as handle:
            schedule_document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise InheritedE3FPOverlayError("overlay schedule could not be read") from exc
    if schedule_document != build_schedule_document():
        raise InheritedE3FPOverlayError("published schedule is not the frozen 128-record schedule")
    try:
        memberships = _read_jsonl(membership_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise InheritedE3FPOverlayError("overlay membership could not be read") from exc
    if len(memberships) != SAMPLE_COUNT:
        raise InheritedE3FPOverlayError("overlay membership must contain exactly 128 rows")

    ordinals = frozen_schedule()
    bound: dict[int, dict] = {}
    environment = lmdb_module.open(
        str(lmdb_path),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=8,
    )
    try:
        with environment.begin(write=False) as transaction:
            if int(transaction.stat().get("entries", -1)) != SAMPLE_COUNT:
                raise InheritedE3FPOverlayError("overlay LMDB must contain exactly 128 entries")
            for schedule_index, (ordinal, membership) in enumerate(zip(ordinals, memberships)):
                expected_key = f"{ordinal:09d}"
                if not (
                    membership.get("schedule_index") == schedule_index
                    and membership.get("schedule_sha256") == expected_schedule_sha
                    and membership.get("sdf_record_index") == ordinal
                    and membership.get("record_storage_key") == expected_key
                ):
                    raise InheritedE3FPOverlayError(
                        "overlay membership order differs from the frozen schedule"
                    )
                raw = transaction.get(expected_key.encode("ascii"))
                if raw is None:
                    raise InheritedE3FPOverlayError("overlay LMDB payload is missing")
                if (
                    len(raw) != membership.get("overlay_wire_bytes")
                    or sidecar_v2_codec.sha256_bytes(raw)
                    != membership.get("overlay_wire_sha256")
                ):
                    raise InheritedE3FPOverlayError(
                        "overlay LMDB wire bytes differ from membership"
                    )
                record, logical_hash = sidecar_v2_codec.decode_record(np, raw)
                validate_overlay_record(np, record, expected_schedule_sha)
                _validate_membership(membership, record)
                if logical_hash != membership.get("overlay_record_content_sha256"):
                    raise InheritedE3FPOverlayError(
                        "overlay decoded logical content differs from membership"
                    )
                bound[ordinal] = {"record": record, "membership": membership}
    finally:
        environment.close()
    if len(bound) != SAMPLE_COUNT:
        raise InheritedE3FPOverlayError("overlay did not load all frozen ordinals")
    return bound, manifest


def _lmdb_artifact(path: Path) -> dict:
    data_path = path / "data.mdb"
    if not data_path.is_file():
        raise InheritedE3FPOverlayError("final overlay LMDB lacks data.mdb")
    return {
        "relative_path": path.name,
        "subdir": True,
        "data_mdb_bytes": int(data_path.stat().st_size),
        "data_mdb_sha256": release_reader.sha256_file(data_path),
    }


def _aggregate_summary(records: Sequence[Mapping[str, object]]) -> dict:
    totals = Counter()
    config_hashes = set()
    for record in records:
        totals.update(
            {
                key: int(record["e3fp"]["summary"][key])
                for key in (
                    "shells_seen",
                    "slots_populated",
                    "duplicate_slots",
                    "changed_identifier_slots",
                    "changed_token_slots",
                )
            }
        )
        config_hashes.add(record["overlay"]["resolved_e3fp_config_sha256"])
    if len(config_hashes) != 1:
        raise InheritedE3FPOverlayError("selected records resolved multiple E3FP configurations")
    return {
        "raw_parity_count": len(records),
        "shells_seen": totals["shells_seen"],
        "slots_populated": totals["slots_populated"],
        "duplicate_slots": totals["duplicate_slots"],
        "changed_identifier_slots": totals["changed_identifier_slots"],
        "changed_token_slots": totals["changed_token_slots"],
        "resolved_e3fp_config_sha256": next(iter(config_hashes)),
    }


def write_overlay_outputs(
    output_dir: Path,
    records: Sequence[dict],
    *,
    np,
    lmdb_module,
    manifest_metadata: Mapping[str, object],
) -> dict:
    """Stage, fully replay, then publish the overlay and its manifest."""

    if output_dir.exists():
        raise InheritedE3FPOverlayError("--output-dir must be a new path")
    if len(records) != SAMPLE_COUNT:
        raise InheritedE3FPOverlayError("overlay publication requires exactly 128 records")
    expected_schedule_sha = schedule_sha256()
    payloads = []
    memberships = []
    seen_keys = set()
    for schedule_index, record in enumerate(records):
        validate_overlay_record(np, record, expected_schedule_sha)
        if record["selection"]["schedule_index"] != schedule_index:
            raise InheritedE3FPOverlayError("overlay records are not in frozen schedule order")
        key = record["member"]["record_storage_key"]
        if key in seen_keys:
            raise InheritedE3FPOverlayError("overlay repeats a base storage key")
        seen_keys.add(key)
        payload = sidecar_v2_codec.encode_record(np, record)
        membership = _membership_row(np, record, payload)
        _validate_membership(membership, record)
        payloads.append((key, payload))
        memberships.append(membership)

    schedule_document = build_schedule_document()
    output_dir.mkdir(parents=True)
    staging_lmdb = output_dir / STAGING_LMDB_NAME
    staging_membership = output_dir / STAGING_MEMBERSHIP_NAME
    staging_schedule = output_dir / STAGING_SCHEDULE_NAME
    map_size = max(64 * 1024 * 1024, sum(len(payload) for _, payload in payloads) * 4)
    environment = lmdb_module.open(
        str(staging_lmdb), subdir=True, map_size=map_size, readonly=False, lock=True,
        readahead=False, meminit=False, max_dbs=1,
    )
    try:
        with environment.begin(write=True) as transaction:
            for key, payload in payloads:
                if not transaction.put(key.encode("ascii"), payload, overwrite=False):
                    raise InheritedE3FPOverlayError("staged LMDB key collision")
        # python-lmdb 1.7 exposes ``force`` as a positional-only argument.
        environment.sync(True)
    finally:
        environment.close()
    _write_jsonl(staging_membership, memberships)
    _write_json(staging_schedule, schedule_document)

    # Complete readback occurs against staging.  No manifest exists yet.
    replayed_memberships = _read_jsonl(staging_membership)
    with staging_schedule.open("r", encoding="utf-8") as handle:
        replayed_schedule = json.load(handle)
    if replayed_memberships != memberships or replayed_schedule != schedule_document:
        raise InheritedE3FPOverlayError("staged JSON artifacts failed complete readback")
    replay = lmdb_module.open(
        str(staging_lmdb), subdir=True, readonly=True, lock=False, readahead=False,
        meminit=False, max_readers=8,
    )
    try:
        with replay.begin(write=False) as transaction:
            if int(transaction.stat().get("entries", -1)) != SAMPLE_COUNT:
                raise InheritedE3FPOverlayError("staged overlay LMDB entry count mismatch")
            for expected_record, membership in zip(records, memberships):
                raw = transaction.get(membership["record_storage_key"].encode("ascii"))
                if raw is None:
                    raise InheritedE3FPOverlayError("staged overlay payload is missing")
                if (
                    len(raw) != membership["overlay_wire_bytes"]
                    or sidecar_v2_codec.sha256_bytes(raw) != membership["overlay_wire_sha256"]
                ):
                    raise InheritedE3FPOverlayError("staged overlay wire bytes differ from membership")
                decoded, logical_hash = sidecar_v2_codec.decode_record(np, raw)
                validate_overlay_record(np, decoded, expected_schedule_sha)
                if logical_hash != membership["overlay_record_content_sha256"]:
                    raise InheritedE3FPOverlayError("staged overlay logical hash mismatch")
                if decoded["member"] != expected_record["member"]:
                    raise InheritedE3FPOverlayError("staged overlay member binding changed on replay")
    finally:
        replay.close()

    final_lmdb = output_dir / FINAL_LMDB_NAME
    final_membership = output_dir / FINAL_MEMBERSHIP_NAME
    final_schedule = output_dir / FINAL_SCHEDULE_NAME
    staging_lmdb.rename(final_lmdb)
    staging_membership.rename(final_membership)
    staging_schedule.rename(final_schedule)

    manifest = dict(manifest_metadata)
    manifest.update(
        {
            "schema_version": MANIFEST_SCHEMA,
            "status": "pass",
            "created_utc": release_reader.utc_now(),
            "sample_scope_only": True,
            "training_admission": False,
            "selection": {
                "schedule_schema_version": SCHEDULE_SCHEMA,
                "schedule_sha256": expected_schedule_sha,
                "sample_count": SAMPLE_COUNT,
                "source_record_count": SOURCE_RECORD_COUNT,
                "selection_rule": SCHEDULE_RULE,
                "no_next_admitted_replacement": True,
            },
            "counts": {
                "scheduled_records": SAMPLE_COUNT,
                "overlay_records": SAMPLE_COUNT,
                "failed_records": 0,
                **_aggregate_summary(records),
            },
            "artifacts": {
                "overlay_lmdb": _lmdb_artifact(final_lmdb),
                "membership": release_reader._artifact(final_membership, SAMPLE_COUNT),
                "schedule": release_reader._artifact(final_schedule),
            },
            "publication": {
                "staging_fully_replayed": True,
                "manifest_written_after_complete_readback": True,
                "frozen_release_modified": False,
            },
        }
    )
    manifest_path = output_dir / MANIFEST_NAME
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def run(args: argparse.Namespace) -> dict:
    release_root = Path(args.release_root).expanduser().resolve()
    archive_path = Path(args.source_archive).expanduser().resolve()
    e3fp_source = Path(args.e3fp_source).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not release_root.is_dir() or not archive_path.is_file():
        raise InheritedE3FPOverlayError("release root and source archive must exist")
    if output_dir.exists():
        raise InheritedE3FPOverlayError("--output-dir must be a new path")

    candidate_path = release_root / "full_release_manifest.json"
    candidate = release_reader.load_json(candidate_path, "production-v2 full release manifest")
    release_selection = {
        "release": {
            "release_id": candidate.get("release_id"),
            "full_release_manifest_sha256": release_reader.sha256_file(candidate_path),
            "logical_release_root_sha256": candidate.get("logical_release_root_sha256"),
        }
    }
    manifest_path, release_manifest = release_reader.load_release_manifest(
        release_root, release_selection
    )
    configuration = release_manifest.get("configuration", {})
    if configuration.get("source_record_count") != SOURCE_RECORD_COUNT:
        raise InheritedE3FPOverlayError("production release has a different source record count")
    locked_member = configuration.get("locked_sdf_member")
    archive_lock = configuration.get("staged_inputs", {}).get("train_3d_sdf_archive")
    if not isinstance(locked_member, dict) or not isinstance(archive_lock, dict):
        raise InheritedE3FPOverlayError("production release lacks its SDF source lock")
    if archive_path.stat().st_size != archive_lock.get("bytes"):
        raise InheritedE3FPOverlayError("source archive byte size differs from production-v2")

    try:
        import lmdb
        import numpy as np
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise InheritedE3FPOverlayError("NumPy, RDKit and python-lmdb are required") from exc

    ordinals = frozen_schedule()
    items = tuple(
        release_reader.SelectionItem("inherited_e3fp_overlay", index, ordinal, ())
        for index, ordinal in enumerate(ordinals)
    )
    bound, shard_receipts = release_reader.load_bound_records(
        release_root,
        release_manifest,
        items,
        np,
        lmdb,
        record_validator=validate_overlay_release_record,
    )
    if len({receipt["shard_index"] for receipt in shard_receipts}) != SAMPLE_COUNT:
        raise InheritedE3FPOverlayError("frozen schedule no longer covers 128 distinct shards")

    def report_progress(observed: int, expected: int) -> None:
        print(
            f"[inherited-e3fp-overlay] scanned {observed:,}/{expected:,} SDF records",
            file=sys.stderr,
            flush=True,
        )

    molecules, member_observation = release_reader.stream_selected_sdf(
        Chem,
        archive_path,
        locked_member,
        ordinals,
        SOURCE_RECORD_COUNT,
        progress_every=args.progress_every,
        progress=report_progress,
    )
    import_root, package_root, e3fp_files = projection.resolve_e3fp_source(e3fp_source)
    e3fp_api = projection.import_locked_e3fp(import_root, package_root)
    full_release_sha = release_reader.sha256_file(manifest_path)
    logical_release_sha = release_manifest["logical_release_root_sha256"]
    implementation_sha = release_reader.sha256_file(Path(inheritance.__file__).resolve())

    # All records are built and parity-checked before write_overlay_outputs
    # creates the output directory or opens a writable LMDB.
    records = []
    for schedule_index, ordinal in enumerate(ordinals):
        overlay_record, _ = build_overlay_record(
            Chem,
            np,
            schedule_index=schedule_index,
            binding=bound[ordinal],
            source_mol=molecules[ordinal],
            e3fp_api=e3fp_api,
            source_full_release_manifest_sha256=full_release_sha,
            source_logical_release_root_sha256=logical_release_sha,
            inheritance_implementation_sha256=implementation_sha,
        )
        records.append(overlay_record)

    manifest_metadata = {
        "production_release": {
            "release_id": release_manifest["release_id"],
            "full_release_manifest_sha256": full_release_sha,
            "logical_release_root_sha256": logical_release_sha,
            "opened_read_only": True,
            "shards_read": shard_receipts,
        },
        "source_sdf": {
            "archive_observation": {
                "bytes": archive_lock["bytes"],
                "sha256": archive_lock["sha256"],
                "rehashed_by_this_builder": False,
            },
            "member": member_observation,
        },
        "code_sha256": {
            "builder": release_reader.sha256_file(Path(__file__).resolve()),
            "inheritance": implementation_sha,
            "hydrogen_projection": release_reader.sha256_file(Path(projection.__file__).resolve()),
            "payload_codec": release_reader.sha256_file(Path(sidecar_v2_codec.__file__).resolve()),
            "release_sdf_reader": release_reader.sha256_file(Path(release_reader.__file__).resolve()),
            "e3fp_source_files": {
                name: release_reader.sha256_file(path) for name, path in sorted(e3fp_files.items())
            },
        },
        "runtime": {
            "python": sys.version.split()[0],
            "rdkit": rdBase.rdkitVersion,
            "e3fp": e3fp_api["module_version"],
        },
        "method_boundary": {
            "production_worker_ipc_and_hydrogen_projection_replayed": True,
            "coordinates_required_equal_to_base": True,
            "raw_e3fp_required_equal_to_base": True,
            "inherited_e3fp_recomputed": True,
            "production_release_modified": False,
        },
    }
    return write_overlay_outputs(
        output_dir,
        records,
        np=np,
        lmdb_module=lmdb,
        manifest_metadata=manifest_metadata,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--e3fp-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=250_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be >= 0")
    manifest = run(args)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
