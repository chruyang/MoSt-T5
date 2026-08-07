#!/usr/bin/env python3
"""Build the exact 128-member paired A/M canary used by the P1 four-grid.

The builder joins the immutable production-v2 geometry release with the
bounded inherited-E3FP overlay, opens both LMDBs read-only, and streams the
official SDF archive exactly once.  Each selected molecule is projected and
linearized once.  Tokenizer-independent SELFIES and graph/ports surfaces are
kept in memory, so the second pass only freezes token IDs and materializes the
paired production records.

This is a sample-bound scientific canary, not a full-data training release.
Publication requires all 128 frozen ordinals, no replacement, no truncation,
complete LMDB decode/replay, and a CPU A0/A1/M0/M1 dry-collate whose raw and
corrupted inputs and labels are all at most 512 tokens.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import datetime as dt
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from most_t5_next.p1.atom_production_bridge import (
    collate_production_atom_batch,
    collate_production_atom_record,
)
from most_t5_next.p1.experiment_grid import validate_a1_m1_geometry_atom_parity
from most_t5_next.p1.production_bridge import (
    collate_production_batch,
    collate_production_motif_record,
)
from most_t5_next.p1.runtime_bridge import P1ArtifactBindings, P1MemberRef
from most_t5_next.p1.training_adapter import to_four_grid_batch_encoding
from most_t5_next.r1.adapter import build_p1_inherited_e3fp_overlay_v1 as overlay
from most_t5_next.r1.adapter import mol_linearizer
from most_t5_next.r1.adapter import paired_record_wire_v1 as paired_wire
from most_t5_next.r1.adapter import p1_topology_augmentation_v1 as topology
from most_t5_next.r1.adapter import production_paired_identity_records_v1 as paired
from most_t5_next.r1.adapter import run_p1_topology_canary_v1 as release_reader
from most_t5_next.r1.gates import pcqm_e3fp_preflight as projection
from most_t5_next.r1.tokenizer import build_p1_canary_union_tokenizer_v1 as union_builder
from most_t5_next.r1.tokenizer import production_atom_selfies_codec_v1 as atom_codec
from most_t5_next.r1.tokenizer import production_graph_ports_codec_v1 as graph_codec


SCHEMA_VERSION = "most-t5-r1/p1-paired-canary-manifest/v1"
MEMBERSHIP_SCHEMA = "most-t5-r1/p1-paired-canary-membership/v1"
REJECT_SCHEMA = "most-t5-r1/p1-paired-canary-reject/v1"
SAMPLE_COUNT = overlay.SAMPLE_COUNT
MAX_SEQUENCE_LENGTH = 512
TOKENIZER_DIRECTORY = "union_tokenizer"
LMDB_DIRECTORY = "paired_records.lmdb"
MEMBERSHIP_NAME = "membership.jsonl"
REJECTS_NAME = "rejects.jsonl"
MANIFEST_NAME = "manifest.json"
STAGING_SUFFIX = ".staging"
MACRO_MIN_OCCURRENCES = 2


class PairedCanaryBuildError(RuntimeError):
    """The exact paired canary cannot be published."""


class RecordRejected(ValueError):
    """One frozen member failed a declared chemistry or parity boundary."""

    def __init__(self, stage: str, reason: str) -> None:
        self.stage = stage
        self.reason = reason
        super().__init__("{}: {}".format(stage, reason))


@dataclass(frozen=True)
class PreparedMember:
    schedule_index: int
    sdf_record_index: int
    member_id: str
    storage_key: str
    source_atom_count: int
    model_to_source_atom_index: tuple[int, ...]
    inherited_e3fp: tuple[tuple[int, ...], ...]
    base_record_content_sha256: str
    effective_overlay_content_sha256: str
    prepared_surfaces: paired.PreparedPairedIdentitySurfaces
    atom_count: int
    motif_count: int
    edge_count: int


@dataclass(frozen=True)
class InputBundle:
    release_manifest_path: Path
    release_manifest: dict[str, object]
    base_bound: dict[int, dict]
    base_shard_receipts: tuple[dict, ...]
    overlay_bound: dict[int, dict]
    overlay_manifest: dict[str, object]
    locked_sdf_member: dict[str, object]
    source_record_count: int
    archive_lock: dict[str, object]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


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


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _write_json(path: Path, value: object) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise PairedCanaryBuildError(
                    "{} line {} is invalid JSON".format(path.name, line_number)
                ) from exc
            if not isinstance(row, dict):
                raise PairedCanaryBuildError(
                    "{} line {} is not an object".format(path.name, line_number)
                )
            rows.append(row)
    return rows


def _plain_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PairedCanaryBuildError("{} must be an integer".format(field))
    return int(value)


def _value_distribution(values: Iterable[int]) -> dict[str, int]:
    ordered = sorted(_plain_int(value, "length") for value in values)
    if not ordered or ordered[0] < 0:
        raise PairedCanaryBuildError("distribution must be non-empty and nonnegative")

    def percentile(fraction: float) -> int:
        index = max(0, min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1))
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _length_distribution(values: Iterable[int]) -> dict[str, int]:
    ordered = tuple(values)
    summary = _value_distribution(ordered)
    summary["over_512"] = sum(value > MAX_SEQUENCE_LENGTH for value in ordered)
    return summary


def build_macro_registry(
    identity_occurrences: Mapping[str, int],
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Select repeated port-aware identities and assign opaque ranked tokens."""

    rows: list[tuple[str, int]] = []
    total_occurrences = 0
    for identity, raw_count in identity_occurrences.items():
        if not isinstance(identity, str) or not identity:
            raise PairedCanaryBuildError("motif identity must be non-empty text")
        count = _plain_int(raw_count, "motif identity occurrence count")
        if count <= 0:
            raise PairedCanaryBuildError("motif identity occurrence count must be positive")
        total_occurrences += count
        if count >= MACRO_MIN_OCCURRENCES:
            rows.append((identity, count))
    rows.sort(key=lambda row: (-row[1], row[0].encode("utf-8")))
    registry = tuple(
        {
            "identity": identity,
            "token": "<MOST:M:{:06d}>".format(rank),
            "occurrence_count": count,
        }
        for rank, (identity, count) in enumerate(rows)
    )
    macro_occurrences = sum(int(row["occurrence_count"]) for row in registry)
    fallback_occurrences = total_occurrences - macro_occurrences
    summary = {
        "minimum_macro_occurrences": MACRO_MIN_OCCURRENCES,
        "unique_identity_count": len(identity_occurrences),
        "total_identity_occurrences": total_occurrences,
        "macro_identity_count": len(registry),
        "macro_occurrences": macro_occurrences,
        "fallback_identity_count": sum(
            count < MACRO_MIN_OCCURRENCES for count in identity_occurrences.values()
        ),
        "fallback_occurrences": fallback_occurrences,
    }
    if macro_occurrences <= 0 or fallback_occurrences <= 0:
        raise PairedCanaryBuildError(
            "the frozen sample must exercise both macro and fallback identity surfaces"
        )
    return registry, summary


def cross_edges_from_augmentation(document: Mapping[str, object]) -> tuple[graph_codec.CrossEdgeInput, ...]:
    """Adapt the validated topology document without inferring another schema."""

    try:
        bonds = document["logical_motif_domain"]["cross_motif_bonds"]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise RecordRejected("TOPOLOGY_PARITY", "augmentation lacks cross-motif bonds") from exc
    if not isinstance(bonds, list):
        raise RecordRejected("TOPOLOGY_PARITY", "cross-motif bonds must be a list")
    result = []
    for edge_index, bond in enumerate(bonds):
        try:
            if not isinstance(bond, dict) or bond["edge_id"] != edge_index:
                raise ValueError("edge IDs are not dense")
            left = bond["left"]
            right = bond["right"]
            atom_a = left["model_atom_index"]
            atom_b = right["model_atom_index"]
            bond_type = str(bond["source_bond_type"]).strip().upper()
            if (
                isinstance(atom_a, bool)
                or not isinstance(atom_a, int)
                or isinstance(atom_b, bool)
                or not isinstance(atom_b, int)
                or not bond_type
            ):
                raise ValueError("edge endpoint or bond type is invalid")
        except (KeyError, TypeError, ValueError) as exc:
            raise RecordRejected(
                "TOPOLOGY_PARITY",
                "augmentation edge {} cannot become CrossEdgeInput".format(edge_index),
            ) from exc
        result.append(graph_codec.CrossEdgeInput(atom_a, atom_b, bond_type))
    return tuple(result)


def _join_overlay_to_base(
    np: Any,
    *,
    ordinal: int,
    base_binding: Mapping[str, object],
    overlay_binding: Mapping[str, object],
) -> tuple[dict, dict, dict, dict]:
    base_record, base_membership = overlay.validate_base_binding(np, base_binding, ordinal)
    overlay_record = overlay_binding.get("record")
    overlay_membership = overlay_binding.get("membership")
    if not isinstance(overlay_record, dict) or not isinstance(overlay_membership, dict):
        raise RecordRejected("OVERLAY_JOIN", "overlay record or membership is absent")
    overlay.validate_overlay_record(np, overlay_record, overlay.schedule_sha256())
    base_member = base_record["member"]
    base_atoms = base_record["atom_universe"]
    base_geometry = base_record["geometry"]
    overlay_member = overlay_record["member"]
    if not (
        overlay_membership.get("sdf_record_index") == ordinal
        and overlay_membership.get("record_storage_key")
        == base_membership.get("record_storage_key")
        and overlay_membership.get("member_id") == base_membership.get("member_id")
        and overlay_membership.get("base_record_content_sha256")
        == base_membership.get("record_content_sha256")
        and overlay_member.get("base_record_content_sha256")
        == base_membership.get("record_content_sha256")
        and overlay_member.get("base_e3fp_sha256") == base_geometry.get("e3fp_sha256")
        and overlay_member.get("source_address_sha256")
        == base_member.get("source_address_sha256")
        and overlay_member.get("geometry_mol_identity_sha256")
        == base_atoms.get("geometry_mol_identity_sha256")
        and overlay_member.get("model_atom_count") == base_atoms.get("model_atom_count")
    ):
        raise RecordRejected("OVERLAY_JOIN", "overlay and base logical rows differ")
    if not release_reader._is_sha256(
        overlay_membership.get("overlay_record_content_sha256")
    ):
        raise RecordRejected("OVERLAY_JOIN", "overlay logical content digest is absent")
    return base_record, base_membership, overlay_record, overlay_membership


def prepare_member(
    Chem: Any,
    sf: Any,
    np: Any,
    *,
    schedule_index: int,
    ordinal: int,
    source_mol: Any,
    base_binding: Mapping[str, object],
    overlay_binding: Mapping[str, object],
    linearizer_sha256: str,
) -> PreparedMember:
    """Replay one molecule once and retain only tokenizer-independent evidence."""

    base_record, base_membership, overlay_record, overlay_membership = _join_overlay_to_base(
        np,
        ordinal=ordinal,
        base_binding=base_binding,
        overlay_binding=overlay_binding,
    )
    try:
        tagged, source_atom_count, _ = projection.tag_source_atoms(Chem, source_mol)
        projected_mol, model_to_source = projection.project_hydrogens(
            Chem, tagged, source_atom_count
        )
    except projection.RecordRejected as exc:
        raise RecordRejected(
            "PROJECTION_{}".format(str(exc.stage).upper()),
            str(exc.reason_code),
        ) from exc
    except (ValueError, RuntimeError) as exc:
        raise RecordRejected("PROJECTION", "hydrogen projection failed") from exc

    atom_universe = base_record["atom_universe"]
    expected_mapping = atom_universe["model_to_source_atom_index"]
    observed_mapping = np.ascontiguousarray(
        np.asarray(model_to_source, dtype=np.int32)
    )
    if not (
        source_atom_count == atom_universe["source_atom_count"]
        and int(projected_mol.GetNumAtoms()) == atom_universe["model_atom_count"]
        and observed_mapping.shape == expected_mapping.shape
        and bool(np.array_equal(observed_mapping, expected_mapping))
    ):
        raise RecordRejected(
            "PROJECTION_PARITY", "source/model atom mapping differs from production-v2"
        )
    try:
        coordinates = np.ascontiguousarray(
            np.asarray(
                projected_mol.GetConformer(0).GetPositions(), dtype=np.float32
            )
        )
    except Exception as exc:
        raise RecordRejected("PROJECTION_PARITY", "projected coordinates are absent") from exc
    base_coordinates = base_record["geometry"]["coordinates"]
    if (
        coordinates.dtype != np.float32
        or coordinates.shape != base_coordinates.shape
        or not bool(np.array_equal(coordinates, base_coordinates))
    ):
        raise RecordRejected(
            "PROJECTION_PARITY", "float32 coordinates differ from production-v2"
        )

    try:
        linearization = mol_linearizer.linearize_mol(projected_mol)
    except (ValueError, RuntimeError) as exc:
        raise RecordRejected("LINEARIZER", "frozen motif linearizer failed") from exc
    try:
        augmentation = topology.build_topology_augmentation(
            linearization_result=linearization,
            member_id=base_membership["member_id"],
            base_record_content_sha256=base_membership["record_content_sha256"],
            linearizer_spec_sha256=linearizer_sha256,
            expected_motif_atom_indices=base_record["topology"]["motif_atom_indices"],
            expected_motif_lexeme_sha256=base_record["topology"]["motif_lexeme_sha256"],
            source_atom_count=source_atom_count,
            model_to_source_atom_index=model_to_source,
        )
        groups = tuple(
            tuple(row)
            for row in augmentation["logical_motif_domain"]["motif_atom_indices"]
        )
        cross_edges = cross_edges_from_augmentation(augmentation)
    except topology.TopologyAugmentationError as exc:
        raise RecordRejected("TOPOLOGY_PARITY", str(exc)) from exc

    try:
        prepared_surfaces = paired.discover_production_paired_identity_surfaces(
            Chem, sf, projected_mol, groups, cross_edges
        )
    except atom_codec.AtomSelfiesAlignmentError as exc:
        raise RecordRejected(
            "SELFIES_{}".format(exc.stage.upper()), exc.reason_code
        ) from exc
    except graph_codec.GraphPortsContractError as exc:
        raise RecordRejected("GRAPH_PORTS", str(exc)) from exc
    except paired.ProductionPairedIdentityError as exc:
        raise RecordRejected("PAIRED_DISCOVERY", str(exc)) from exc

    inherited = overlay_record["e3fp"]["inherited"]
    inherited_rows = tuple(tuple(int(value) for value in row) for row in inherited)
    mapping = tuple(int(value) for value in model_to_source)
    return PreparedMember(
        schedule_index=schedule_index,
        sdf_record_index=ordinal,
        member_id=base_membership["member_id"],
        storage_key=base_membership["record_storage_key"],
        source_atom_count=source_atom_count,
        model_to_source_atom_index=mapping,
        inherited_e3fp=inherited_rows,
        base_record_content_sha256=base_membership["record_content_sha256"],
        effective_overlay_content_sha256=overlay_membership[
            "overlay_record_content_sha256"
        ],
        prepared_surfaces=prepared_surfaces,
        atom_count=int(projected_mol.GetNumAtoms()),
        motif_count=len(groups),
        edge_count=len(cross_edges),
    )


def _reject_row(
    *,
    schedule_index: int | None,
    ordinal: int | None,
    stage: str,
    reason: str,
) -> dict[str, object]:
    return {
        "schema_version": REJECT_SCHEMA,
        "schedule_index": schedule_index,
        "sdf_record_index": ordinal,
        "stage": stage,
        "reason": reason,
    }


def _load_inputs(
    *,
    release_root: Path,
    overlay_root: Path,
    source_archive: Path,
    np: Any,
    lmdb_module: Any,
) -> InputBundle:
    manifest_path = release_root / "full_release_manifest.json"
    candidate = release_reader.load_json(
        manifest_path, "production-v2 full release manifest"
    )
    selection = {
        "release": {
            "release_id": candidate.get("release_id"),
            "full_release_manifest_sha256": release_reader.sha256_file(manifest_path),
            "logical_release_root_sha256": candidate.get(
                "logical_release_root_sha256"
            ),
        }
    }
    verified_manifest_path, release_manifest = release_reader.load_release_manifest(
        release_root, selection
    )
    ordinals = overlay.frozen_schedule()
    items = tuple(
        release_reader.SelectionItem("paired_canary", index, ordinal, ())
        for index, ordinal in enumerate(ordinals)
    )
    base_bound, shard_receipts = release_reader.load_bound_records(
        release_root,
        release_manifest,
        items,
        np,
        lmdb_module,
        record_validator=release_reader.validate_bound_record,
    )
    overlay_bound, overlay_manifest = overlay.load_overlay_readonly(
        overlay_root, np=np, lmdb_module=lmdb_module
    )
    if set(base_bound) != set(ordinals) or set(overlay_bound) != set(ordinals):
        raise PairedCanaryBuildError("base/overlay rows do not equal the frozen schedule")

    configuration = release_manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise PairedCanaryBuildError("production release lacks configuration")
    source_record_count = configuration.get("source_record_count")
    locked_member = configuration.get("locked_sdf_member")
    staged_inputs = configuration.get("staged_inputs")
    archive_lock = (
        staged_inputs.get("train_3d_sdf_archive")
        if isinstance(staged_inputs, dict)
        else None
    )
    if (
        source_record_count != overlay.SOURCE_RECORD_COUNT
        or not isinstance(locked_member, dict)
        or not isinstance(archive_lock, dict)
    ):
        raise PairedCanaryBuildError("production release SDF lock is incomplete")
    if source_archive.stat().st_size != archive_lock.get("bytes"):
        raise PairedCanaryBuildError("source archive byte size differs from production-v2")

    production_binding = overlay_manifest.get("production_release")
    if not isinstance(production_binding, dict) or not (
        production_binding.get("release_id") == release_manifest.get("release_id")
        and production_binding.get("full_release_manifest_sha256")
        == release_reader.sha256_file(verified_manifest_path)
        and production_binding.get("logical_release_root_sha256")
        == release_manifest.get("logical_release_root_sha256")
    ):
        raise PairedCanaryBuildError("overlay was built from a different production release")
    return InputBundle(
        release_manifest_path=verified_manifest_path,
        release_manifest=release_manifest,
        base_bound=base_bound,
        base_shard_receipts=tuple(shard_receipts),
        overlay_bound=overlay_bound,
        overlay_manifest=overlay_manifest,
        locked_sdf_member=locked_member,
        source_record_count=int(source_record_count),
        archive_lock=archive_lock,
    )


def _artifact_bindings_base(
    *,
    input_bundle: InputBundle,
    overlay_root: Path,
) -> dict[str, str]:
    graph_source = Path(graph_codec.__file__).resolve()
    atom_source = Path(atom_codec.__file__).resolve()
    paired_source = Path(paired.__file__).resolve()
    descriptor = {
        "overlay_record_schema": overlay.RECORD_SCHEMA,
        "e3fp_semantics_id": next(
            iter(input_bundle.overlay_bound.values())
        )["record"]["overlay"]["semantics_id"],
    }
    return {
        "release_id": "{}:paired-inherited-e3fp-128-v1".format(
            input_bundle.release_manifest["release_id"]
        ),
        "data_release_manifest_sha256": release_reader.sha256_file(
            overlay_root / overlay.MANIFEST_NAME
        ),
        "geometry_record_schema_sha256": sha256_json(descriptor),
        "membership_manifest_sha256": release_reader.sha256_file(
            overlay_root / overlay.FINAL_MEMBERSHIP_NAME
        ),
        "identity_codec_sha256": sha256_json(
            {
                "atom_selfies_codec_sha256": release_reader.sha256_file(atom_source),
                "motif_graph_ports_codec_sha256": release_reader.sha256_file(
                    graph_source
                ),
                "paired_producer_sha256": release_reader.sha256_file(paired_source),
            }
        ),
        "connection_codec_sha256": release_reader.sha256_file(graph_source),
    }


def _membership_row(
    prepared_member: PreparedMember,
    loaded: paired_wire.LoadedPairedTrainingRecord,
    payload: bytes,
) -> dict[str, object]:
    return {
        "schema_version": MEMBERSHIP_SCHEMA,
        "schedule_index": prepared_member.schedule_index,
        "sdf_record_index": prepared_member.sdf_record_index,
        "member_id": prepared_member.member_id,
        "storage_key": prepared_member.storage_key,
        "wire_bytes": len(payload),
        "atom_record_artifact_sha256": loaded.atom_record.record_artifact_sha256,
        "motif_record_artifact_sha256": loaded.motif_record.record_artifact_sha256,
        "effective_overlay_content_sha256": (
            prepared_member.effective_overlay_content_sha256
        ),
        "atom_input_token_count": len(loaded.atom_record.input_ids),
        "motif_input_token_count": len(loaded.motif_record.input_ids),
        "motif_count": prepared_member.motif_count,
        "atom_count": prepared_member.atom_count,
        "edge_count": prepared_member.edge_count,
        "macro_identity_occurrences": sum(
            mode == "macro" for mode in loaded.surface_summary.motif_identity_modes
        ),
        "fallback_identity_occurrences": sum(
            mode == "fallback" for mode in loaded.surface_summary.motif_identity_modes
        ),
    }


def _write_lmdb_and_replay(
    *,
    staging_root: Path,
    payloads: Sequence[tuple[str, bytes]],
    expected_loaded: Sequence[paired_wire.LoadedPairedTrainingRecord],
    lmdb_module: Any,
) -> tuple[paired_wire.LoadedPairedTrainingRecord, ...]:
    if len(payloads) != SAMPLE_COUNT or len(expected_loaded) != SAMPLE_COUNT:
        raise PairedCanaryBuildError("LMDB publication requires exactly 128 pairs")
    lmdb_path = staging_root / LMDB_DIRECTORY
    map_size = max(64 * 1024 * 1024, sum(len(payload) for _, payload in payloads) * 4)
    environment = lmdb_module.open(
        str(lmdb_path),
        subdir=True,
        map_size=map_size,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        max_dbs=1,
    )
    try:
        with environment.begin(write=True) as transaction:
            for key, payload in payloads:
                if not transaction.put(key.encode("ascii"), payload, overwrite=False):
                    raise PairedCanaryBuildError("paired LMDB key collision")
        # python-lmdb 1.7 exposes force as positional-only.
        environment.sync(True)
    finally:
        environment.close()

    replayed: list[paired_wire.LoadedPairedTrainingRecord] = []
    replay = lmdb_module.open(
        str(lmdb_path),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=8,
    )
    try:
        with replay.begin(write=False) as transaction:
            if int(transaction.stat().get("entries", -1)) != SAMPLE_COUNT:
                raise PairedCanaryBuildError("paired LMDB entry count differs from 128")
            cursor_keys = tuple(key.decode("ascii") for key, _ in transaction.cursor())
            expected_keys = tuple(key for key, _ in payloads)
            if cursor_keys != tuple(sorted(expected_keys)):
                raise PairedCanaryBuildError("paired LMDB key set/order differs")
            for (key, expected_payload), expected_record in zip(
                payloads, expected_loaded
            ):
                raw = transaction.get(key.encode("ascii"))
                if raw is None or raw != expected_payload:
                    raise PairedCanaryBuildError("paired LMDB payload changed on readback")
                decoded = paired_wire.decode_paired_training_record(bytes(raw))
                if decoded != expected_record:
                    raise PairedCanaryBuildError("paired LMDB semantic replay differs")
                replayed.append(decoded)
    finally:
        replay.close()
    return tuple(replayed)


def static_all_mask_capacity(
    records: Sequence[paired_wire.LoadedPairedTrainingRecord],
    *,
    sentinel_id_count: int,
) -> dict[str, object]:
    """Prove target/sentinel capacity for every possible mask and epoch.

    Masking cannot make a corrupted input longer than its uncorrupted input.
    If all ``n`` identity units are selected, the target contains every
    identity token, one sentinel per unit, and the terminal sentinel/EOS pair:
    ``sum(span lengths) + n + 2``.  This is therefore a static upper bound for
    every seed, epoch and mask realization.
    """

    sentinel_id_count = _plain_int(sentinel_id_count, "sentinel_id_count")
    if sentinel_id_count <= 0:
        raise PairedCanaryBuildError("sentinel_id_count must be positive")
    if len(records) != SAMPLE_COUNT:
        raise PairedCanaryBuildError("static mask capacity requires exactly 128 pairs")

    family_values: dict[str, dict[str, list[int]]] = {
        "atom": {"units": [], "identity_tokens": [], "worst_target": [], "sentinels": []},
        "motif": {"units": [], "identity_tokens": [], "worst_target": [], "sentinels": []},
    }
    for row in records:
        for family, spans in (
            ("atom", row.atom_record.atom_identity_spans),
            ("motif", row.motif_record.identity_spans),
        ):
            unit_count = len(spans)
            identity_token_count = sum(span.stop - span.start for span in spans)
            worst_target = identity_token_count + unit_count + 2
            required_sentinels = unit_count + 1
            if unit_count <= 0 or identity_token_count <= 0:
                raise PairedCanaryBuildError(
                    "{} record {} has no maskable identity domain".format(
                        family, row.atom_record.record_id
                    )
                )
            if required_sentinels > sentinel_id_count:
                raise PairedCanaryBuildError(
                    "{} record {} needs {} sentinels but tokenizer has {}".format(
                        family,
                        row.atom_record.record_id,
                        required_sentinels,
                        sentinel_id_count,
                    )
                )
            if worst_target > MAX_SEQUENCE_LENGTH:
                raise PairedCanaryBuildError(
                    "{} record {} all-mask target upper bound {} exceeds 512".format(
                        family, row.atom_record.record_id, worst_target
                    )
                )
            values = family_values[family]
            values["units"].append(unit_count)
            values["identity_tokens"].append(identity_token_count)
            values["worst_target"].append(worst_target)
            values["sentinels"].append(required_sentinels)

    return {
        "sentinel_id_count": sentinel_id_count,
        "all_masks_all_epochs_proven": True,
        "corrupted_input_not_longer_than_uncorrupted": True,
        "target_upper_bound_formula": "sum(identity_span_lengths)+unit_count+2",
        "required_sentinels_formula": "unit_count+1",
        "atom": {
            "identity_units": _value_distribution(family_values["atom"]["units"]),
            "identity_tokens": _value_distribution(
                family_values["atom"]["identity_tokens"]
            ),
            "all_mask_target_upper_bound": _length_distribution(
                family_values["atom"]["worst_target"]
            ),
            "required_sentinels": _value_distribution(
                family_values["atom"]["sentinels"]
            ),
        },
        "motif": {
            "identity_units": _value_distribution(family_values["motif"]["units"]),
            "identity_tokens": _value_distribution(
                family_values["motif"]["identity_tokens"]
            ),
            "all_mask_target_upper_bound": _length_distribution(
                family_values["motif"]["worst_target"]
            ),
            "required_sentinels": _value_distribution(
                family_values["motif"]["sentinels"]
            ),
        },
    }


def _selected_identity_statistics(
    records: Sequence[Any],
    examples: Sequence[Any],
    *,
    span_field: str,
    selected_field: str,
    atom_owner_field: str | None = None,
) -> dict[str, object]:
    total_units = 0
    selected_units = 0
    total_identity_tokens = 0
    selected_identity_tokens = 0
    total_atoms = 0
    selected_atoms = 0
    for record, example in zip(records, examples):
        spans = getattr(record, span_field)
        selected = tuple(getattr(example, selected_field))
        total_units += len(spans)
        selected_units += len(selected)
        total_identity_tokens += sum(span.stop - span.start for span in spans)
        selected_identity_tokens += sum(
            spans[unit_id].stop - spans[unit_id].start for unit_id in selected
        )
        if atom_owner_field is None:
            total_atoms += len(spans)
            selected_atoms += len(selected)
        else:
            atom_owners = tuple(getattr(record, atom_owner_field))
            selected_set = set(selected)
            total_atoms += len(atom_owners)
            selected_atoms += sum(owner in selected_set for owner in atom_owners)
    if total_units <= 0 or total_identity_tokens <= 0 or total_atoms <= 0:
        raise PairedCanaryBuildError("dry-collate identity domain is empty")
    return {
        "selected_unit_count": selected_units,
        "total_unit_count": total_units,
        "selected_unit_ratio": selected_units / total_units,
        "selected_identity_token_count": selected_identity_tokens,
        "total_identity_token_count": total_identity_tokens,
        "selected_identity_token_ratio": (
            selected_identity_tokens / total_identity_tokens
        ),
        "selected_atom_count": selected_atoms,
        "total_atom_count": total_atoms,
        "selected_atom_ratio": selected_atoms / total_atoms,
    }


def run_cpu_four_grid_dry_collate(
    records: Sequence[paired_wire.LoadedPairedTrainingRecord],
    *,
    tokenizer_runtime: Any,
) -> dict[str, object]:
    """Exercise all four cells from reloaded rows without a model forward."""

    atom_records = tuple(row.atom_record for row in records)
    motif_records = tuple(row.motif_record for row in records)
    a0 = collate_production_atom_batch(
        atom_records,
        condition_id="A0",
        tokenizer=tokenizer_runtime,
        seed=0,
        epoch=0,
        mask_probability=0.15,
    )
    a1 = collate_production_atom_batch(
        atom_records,
        condition_id="A1",
        tokenizer=tokenizer_runtime,
        seed=0,
        epoch=0,
        mask_probability=0.15,
    )
    m0 = collate_production_batch(
        motif_records,
        condition_id="M0",
        tokenizer=tokenizer_runtime,
        seed=0,
        epoch=0,
        mask_probability=0.15,
    )
    m1 = collate_production_batch(
        motif_records,
        condition_id="M1",
        tokenizer=tokenizer_runtime,
        seed=0,
        epoch=0,
        mask_probability=0.15,
    )
    if a0.ce_batch != a1.ce_batch or m0.ce_batch != m1.ce_batch:
        raise PairedCanaryBuildError("3D toggle changed CE within a representation family")
    validate_a1_m1_geometry_atom_parity(a1, m1)
    tensor_interfaces: dict[str, object] = {}
    for batch in (a0, a1, m0, m1):
        # This is a CPU tensor-interface smoke, not a T5 model forward.
        encoded = to_four_grid_batch_encoding(batch, device="cpu")
        tensor_interfaces[batch.condition_id] = {
            "keys": sorted(encoded.keys()),
            "shapes": {
                key: list(value.shape) for key, value in encoded.items()
            },
            "dtypes": {
                key: str(value.dtype) for key, value in encoded.items()
            },
        }

    atom_examples = tuple(
        collate_production_atom_record(
            record,
            tokenizer=tokenizer_runtime,
            seed=0,
            epoch=0,
            mask_probability=0.15,
        )
        for record in atom_records
    )
    motif_examples = tuple(
        collate_production_motif_record(
            record,
            tokenizer=tokenizer_runtime,
            seed=0,
            epoch=0,
            mask_probability=0.15,
        )
        for record in motif_records
    )
    if (
        tuple(len(example.input_ids) for example in atom_examples)
        != a0.ce_batch.input_lengths
        or tuple(len(example.labels) for example in atom_examples)
        != a0.ce_batch.target_lengths
        or tuple(len(example.input_ids) for example in motif_examples)
        != m0.ce_batch.input_lengths
        or tuple(len(example.labels) for example in motif_examples)
        != m0.ce_batch.target_lengths
    ):
        raise PairedCanaryBuildError(
            "record-level and batch-level epoch-0 corruption disagree"
        )

    lengths = {
        "A_collated_input": _length_distribution(a0.ce_batch.input_lengths),
        "A_collated_label": _length_distribution(a0.ce_batch.target_lengths),
        "M_collated_input": _length_distribution(m0.ce_batch.input_lengths),
        "M_collated_label": _length_distribution(m0.ce_batch.target_lengths),
    }
    if any(summary["over_512"] for summary in lengths.values()):
        raise PairedCanaryBuildError(
            "an epoch-0 collated input or label exceeds 512 tokens; truncation is forbidden"
        )
    return {
        "seed": 0,
        "epoch": 0,
        "mask_probability": 0.15,
        "a0_a1_ce_equal": True,
        "m0_m1_ce_equal": True,
        "a1_m1_geometry_atom_parity": True,
        "four_grid_tensor_adapter_passed": True,
        "tensor_interfaces": tensor_interfaces,
        "atom_selection": _selected_identity_statistics(
            atom_records,
            atom_examples,
            span_field="atom_identity_spans",
            selected_field="selected_atom_ids_in_input_order",
        ),
        "motif_selection": _selected_identity_statistics(
            motif_records,
            motif_examples,
            span_field="identity_spans",
            selected_field="selected_logical_motif_ids_in_input_order",
            atom_owner_field="atom_to_logical_motif",
        ),
        "lengths": lengths,
    }


def _failure_artifacts(
    staging_root: Path,
    rejects: Sequence[Mapping[str, object]],
    *,
    reason: str,
) -> None:
    rejects_path = staging_root / REJECTS_NAME
    if not rejects_path.exists():
        _write_jsonl(rejects_path, rejects)
    failure_path = staging_root / "manifest.failed.json"
    if not failure_path.exists():
        _write_json(
            failure_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "created_utc": utc_now(),
                "reason": reason,
                "scheduled_records": SAMPLE_COUNT,
                "reject_count": len(rejects),
                "rejects_by_stage": dict(
                    sorted(Counter(row["stage"] for row in rejects).items())
                ),
                "publication": False,
            },
        )


def run(args: argparse.Namespace) -> dict[str, object]:
    release_root = Path(args.release_root).expanduser().resolve()
    source_archive = Path(args.source_archive).expanduser().resolve()
    overlay_root = Path(args.overlay_root).expanduser().resolve()
    base_tokenizer = Path(args.base_tokenizer).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    staging_root = output_dir.with_name(output_dir.name + STAGING_SUFFIX)
    if not release_root.is_dir() or not overlay_root.is_dir():
        raise PairedCanaryBuildError("release-root and overlay-root must be directories")
    if not source_archive.is_file() or not base_tokenizer.is_dir():
        raise PairedCanaryBuildError("source-archive and base-tokenizer must exist")
    if output_dir.exists() or staging_root.exists():
        raise PairedCanaryBuildError("output and sibling staging paths must both be new")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir()

    try:
        import lmdb
        import numpy as np
        import selfies as sf
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise PairedCanaryBuildError(
            "NumPy, RDKit, SELFIES and python-lmdb are required"
        ) from exc

    input_bundle = _load_inputs(
        release_root=release_root,
        overlay_root=overlay_root,
        source_archive=source_archive,
        np=np,
        lmdb_module=lmdb,
    )
    ordinals = overlay.frozen_schedule()
    if len(ordinals) != SAMPLE_COUNT:
        raise PairedCanaryBuildError("frozen schedule no longer contains 128 ordinals")

    progress_calls = 0

    def report_progress(observed: int, expected: int) -> None:
        nonlocal progress_calls
        progress_calls += 1
        print(
            "[paired-canary] scanned {:,}/{:,} SDF records".format(observed, expected),
            file=sys.stderr,
            flush=True,
        )

    # This is the only SDF stream invocation in the complete builder.
    molecules, sdf_observation = release_reader.stream_selected_sdf(
        Chem,
        source_archive,
        input_bundle.locked_sdf_member,
        ordinals,
        input_bundle.source_record_count,
        progress_every=args.progress_every,
        progress=report_progress,
    )
    if set(molecules) != set(ordinals):
        raise PairedCanaryBuildError("single SDF scan did not resolve all frozen ordinals")

    linearizer_sha = release_reader.sha256_file(Path(mol_linearizer.__file__).resolve())
    prepared_members: list[PreparedMember] = []
    rejects: list[dict[str, object]] = []
    observed_selfies_symbols: set[str] = set()
    identity_counts: Counter[str] = Counter()
    for schedule_index, ordinal in enumerate(ordinals):
        try:
            prepared_member = prepare_member(
                Chem,
                sf,
                np,
                schedule_index=schedule_index,
                ordinal=ordinal,
                source_mol=molecules[ordinal],
                base_binding=input_bundle.base_bound[ordinal],
                overlay_binding=input_bundle.overlay_bound[ordinal],
                linearizer_sha256=linearizer_sha,
            )
        except RecordRejected as exc:
            rejects.append(
                _reject_row(
                    schedule_index=schedule_index,
                    ordinal=ordinal,
                    stage=exc.stage,
                    reason=exc.reason,
                )
            )
            continue
        prepared_members.append(prepared_member)
        observed_selfies_symbols.update(
            prepared_member.prepared_surfaces.atom_surface.selfies_symbols
        )
        identity_counts.update(
            motif.identity_smiles
            for motif in prepared_member.prepared_surfaces.graph_encoding.motifs
        )
    if rejects or len(prepared_members) != SAMPLE_COUNT:
        _failure_artifacts(
            staging_root,
            rejects,
            reason="surface discovery did not accept all 128 frozen members",
        )
        raise PairedCanaryBuildError("surface discovery rejected frozen members")

    try:
        macro_registry, macro_summary = build_macro_registry(identity_counts)
    except PairedCanaryBuildError as exc:
        rejects.append(
            _reject_row(
                schedule_index=None,
                ordinal=None,
                stage="MACRO_POLICY",
                reason=str(exc),
            )
        )
        _failure_artifacts(staging_root, rejects, reason=str(exc))
        raise
    robust_selfies_symbols = set(sf.get_semantic_robust_alphabet())
    tokenizer_build = union_builder.build_canary_union_tokenizer(
        base_snapshot=base_tokenizer,
        output_dir=staging_root / TOKENIZER_DIRECTORY,
        selfies_distribution_version=atom_codec.SELFIES_DISTRIBUTION_VERSION,
        robust_selfies_symbols=robust_selfies_symbols,
        observed_selfies_symbols=observed_selfies_symbols,
        motif_macro_registry=macro_registry,
    )
    macro_by_identity = {
        str(row["identity"]): str(row["token"]) for row in macro_registry
    }
    binding_base = _artifact_bindings_base(
        input_bundle=input_bundle,
        overlay_root=overlay_root,
    )

    payloads: list[tuple[str, bytes]] = []
    loaded_records: list[paired_wire.LoadedPairedTrainingRecord] = []
    memberships: list[dict[str, object]] = []
    for prepared_member in prepared_members:
        try:
            bindings = P1ArtifactBindings(
                **binding_base,
                geometry_record_content_sha256=(
                    prepared_member.effective_overlay_content_sha256
                ),
                tokenizer_contract_sha256=(
                    tokenizer_build.runtime.tokenizer_contract_sha256
                ),
                tokenizer_snapshot_sha256=(
                    tokenizer_build.runtime.tokenizer_snapshot_sha256
                ),
            )
            pair = paired.build_production_paired_identity_records_from_prepared(
                prepared=prepared_member.prepared_surfaces,
                member=P1MemberRef(
                    prepared_member.member_id, prepared_member.storage_key
                ),
                bindings=bindings,
                base_geometry_record_content_sha256=(
                    prepared_member.base_record_content_sha256
                ),
                effective_inherited_overlay_content_sha256=(
                    prepared_member.effective_overlay_content_sha256
                ),
                source_atom_count=prepared_member.source_atom_count,
                model_to_source_atom_index=(
                    prepared_member.model_to_source_atom_index
                ),
                inherited_e3fp=prepared_member.inherited_e3fp,
                union_tokenizer=tokenizer_build.tokenizer,
                tokenizer_binding=tokenizer_build.runtime,
                macro_by_identity=macro_by_identity,
            )
            payload = paired_wire.encode_paired_training_record(
                pair,
                schedule_index=prepared_member.schedule_index,
                sdf_record_index=prepared_member.sdf_record_index,
            )
            loaded = paired_wire.decode_paired_training_record(payload)
        except (
            paired.ProductionPairedIdentityError,
            paired_wire.PairedRecordWireError,
            atom_codec.AtomSelfiesAlignmentError,
            graph_codec.GraphPortsContractError,
        ) as exc:
            rejects.append(
                _reject_row(
                    schedule_index=prepared_member.schedule_index,
                    ordinal=prepared_member.sdf_record_index,
                    stage="PAIRED_BUILD",
                    reason=str(exc),
                )
            )
            continue
        if (
            len(loaded.atom_record.input_ids) > MAX_SEQUENCE_LENGTH
            or len(loaded.motif_record.input_ids) > MAX_SEQUENCE_LENGTH
        ):
            rejects.append(
                _reject_row(
                    schedule_index=prepared_member.schedule_index,
                    ordinal=prepared_member.sdf_record_index,
                    stage="UNCORRUPTED_LENGTH",
                    reason="A or M input exceeds 512 tokens; truncation is forbidden",
                )
            )
            continue
        payloads.append((prepared_member.storage_key, payload))
        loaded_records.append(loaded)
        memberships.append(_membership_row(prepared_member, loaded, payload))
    if rejects or len(payloads) != SAMPLE_COUNT:
        _failure_artifacts(
            staging_root,
            rejects,
            reason="paired materialization did not accept all 128 frozen members",
        )
        raise PairedCanaryBuildError("paired materialization rejected frozen members")
    observed_modes = Counter(
        mode
        for row in loaded_records
        for mode in row.surface_summary.motif_identity_modes
    )
    identity_tokens_by_mode: Counter[str] = Counter()
    for row in loaded_records:
        for mode, count in zip(
            row.surface_summary.motif_identity_modes,
            row.surface_summary.motif_identity_token_counts,
        ):
            identity_tokens_by_mode[mode] += count
    if observed_modes["macro"] <= 0 or observed_modes["fallback"] <= 0:
        rejects.append(
            _reject_row(
                schedule_index=None,
                ordinal=None,
                stage="MACRO_POLICY",
                reason="materialized rows do not exercise both macro and fallback",
            )
        )
        _failure_artifacts(staging_root, rejects, reason=rejects[-1]["reason"])
        raise PairedCanaryBuildError(str(rejects[-1]["reason"]))

    try:
        sentinel_capacity = static_all_mask_capacity(
            loaded_records,
            sentinel_id_count=len(tokenizer_build.runtime.sentinel_token_ids),
        )
    except PairedCanaryBuildError as exc:
        rejects.append(
            _reject_row(
                schedule_index=None,
                ordinal=None,
                stage="STATIC_MASK_CAPACITY",
                reason=str(exc),
            )
        )
        _failure_artifacts(staging_root, rejects, reason=str(exc))
        raise

    replayed = _write_lmdb_and_replay(
        staging_root=staging_root,
        payloads=payloads,
        expected_loaded=loaded_records,
        lmdb_module=lmdb,
    )
    try:
        dry_collate = run_cpu_four_grid_dry_collate(
            replayed, tokenizer_runtime=tokenizer_build.runtime
        )
    except Exception as exc:
        rejects.append(
            _reject_row(
                schedule_index=None,
                ordinal=None,
                stage="FOUR_GRID_DRY_COLLATE",
                reason=str(exc),
            )
        )
        _failure_artifacts(staging_root, rejects, reason=str(exc))
        raise

    _write_jsonl(staging_root / MEMBERSHIP_NAME, memberships)
    _write_jsonl(staging_root / REJECTS_NAME, ())
    if _read_jsonl(staging_root / MEMBERSHIP_NAME) != memberships:
        raise PairedCanaryBuildError("paired membership failed complete readback")
    if _read_jsonl(staging_root / REJECTS_NAME) != []:
        raise PairedCanaryBuildError("successful paired reject ledger is not empty")

    atom_lengths = [len(row.atom_record.input_ids) for row in replayed]
    motif_lengths = [len(row.motif_record.input_ids) for row in replayed]
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "created_utc": utc_now(),
        "sample_scope_only": True,
        "training_admission": False,
        "canary_execution_ready": True,
        "selection": {
            "schedule_schema_version": overlay.SCHEDULE_SCHEMA,
            "schedule_sha256": overlay.schedule_sha256(),
            "sample_count": SAMPLE_COUNT,
            "source_record_count": input_bundle.source_record_count,
            "selection_rule": overlay.SCHEDULE_RULE,
            "no_next_admitted_replacement": True,
        },
        "counts": {
            "scheduled_records": SAMPLE_COUNT,
            "prepared_records": len(prepared_members),
            "paired_records": len(replayed),
            "replayed_records": len(replayed),
            "failed_records": 0,
            "rejects_by_stage": {},
            "observed_selfies_symbols": len(observed_selfies_symbols),
            "robust_selfies_symbols": len(robust_selfies_symbols),
            "observed_symbols_outside_robust": len(
                observed_selfies_symbols - robust_selfies_symbols
            ),
            **macro_summary,
            "materialized_macro_occurrences": observed_modes["macro"],
            "materialized_fallback_occurrences": observed_modes["fallback"],
            "materialized_macro_identity_tokens": identity_tokens_by_mode["macro"],
            "materialized_fallback_identity_tokens": identity_tokens_by_mode[
                "fallback"
            ],
            "materialized_graph_tokens": sum(
                row.surface_summary.graph_token_count for row in replayed
            ),
        },
        "lengths": {
            "uncorrupted_atom_input": _length_distribution(atom_lengths),
            "uncorrupted_motif_input": _length_distribution(motif_lengths),
            "all_mask_atom_target_upper_bound": sentinel_capacity["atom"][
                "all_mask_target_upper_bound"
            ],
            "all_mask_motif_target_upper_bound": sentinel_capacity["motif"][
                "all_mask_target_upper_bound"
            ],
            **dry_collate["lengths"],
        },
        "sentinel_capacity": sentinel_capacity,
        "structure": {
            "atoms": _value_distribution(row.atom_count for row in prepared_members),
            "motifs": _value_distribution(row.motif_count for row in prepared_members),
            "cross_motif_edges": _value_distribution(
                row.edge_count for row in prepared_members
            ),
            "e3fp_level_count": _value_distribution(
                len(row.inherited_e3fp[0]) for row in prepared_members
            ),
        },
        "selfies": {
            "distribution_version": atom_codec.SELFIES_DISTRIBUTION_VERSION,
            "observed_symbols_outside_robust": sorted(
                observed_selfies_symbols - robust_selfies_symbols,
                key=lambda value: value.encode("utf-8"),
            ),
        },
        "macro_registry": list(macro_registry),
        "wire_format": {
            "schema_version": paired_wire.PAIRED_RECORD_WIRE_SCHEMA,
            "encoding": "canonical_utf8_json",
            "lmdb_key": "base_record_storage_key_ascii",
            "one_atomic_value_contains": [
                "atom_document",
                "motif_training_document",
                "receipt",
                "surface_summary",
            ],
            "wire_bytes": _value_distribution(
                len(payload) for _key, payload in payloads
            ),
        },
        "four_grid_dry_collate": dry_collate,
        "interpretation_boundary": {
            "purpose": "runtime_and_dataflow_smoke_only",
            "not_an_a_m_effect_ranking": True,
            "reason": (
                "A masks atom identities while M masks motif identities; equal raw CE "
                "mask probability does not define equal semantic corruption difficulty"
            ),
        },
        "inputs": {
            "production_release_id": input_bundle.release_manifest["release_id"],
            "production_logical_release_root_sha256": input_bundle.release_manifest[
                "logical_release_root_sha256"
            ],
            "base_shards_opened_read_only": list(input_bundle.base_shard_receipts),
            "overlay_manifest_schema": input_bundle.overlay_manifest["schema_version"],
            "source_archive_bytes": input_bundle.archive_lock["bytes"],
            "source_sdf": sdf_observation,
        },
        "artifacts": {
            "paired_lmdb": {
                "relative_path": LMDB_DIRECTORY,
                "subdir": True,
                "entry_count": SAMPLE_COUNT,
                "data_mdb_bytes": int(
                    (staging_root / LMDB_DIRECTORY / "data.mdb").stat().st_size
                ),
            },
            "membership": {
                "relative_path": MEMBERSHIP_NAME,
                "row_count": SAMPLE_COUNT,
                "bytes": int((staging_root / MEMBERSHIP_NAME).stat().st_size),
            },
            "rejects": {
                "relative_path": REJECTS_NAME,
                "row_count": 0,
                "bytes": int((staging_root / REJECTS_NAME).stat().st_size),
            },
            "tokenizer": {
                "relative_path": TOKENIZER_DIRECTORY,
                "schema_version": tokenizer_build.manifest["schema_version"],
                "tokenizer_contract_sha256": tokenizer_build.runtime.tokenizer_contract_sha256,
                "tokenizer_snapshot_sha256": tokenizer_build.runtime.tokenizer_snapshot_sha256,
                "vocab_size": tokenizer_build.runtime.vocab_size,
            },
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "selfies_distribution": importlib_metadata.version("selfies"),
            "lmdb": getattr(lmdb, "__version__", "unknown"),
        },
        "method_boundary": {
            "base_release_opened_read_only": True,
            "inherited_overlay_opened_read_only": True,
            "sdf_stream_calls": 1,
            "sdf_progress_callbacks": progress_calls,
            "tag_project_calls": SAMPLE_COUNT,
            "linearizer_calls": SAMPLE_COUNT,
            "prepared_surface_discovery_calls": SAMPLE_COUNT,
            "model_to_source_mapping_required_equal_to_base": True,
            "float32_coordinates_required_equal_to_base": True,
            "frozen_motif_groups_and_lexemes_required_equal_to_base": True,
            "effective_geometry_binding_is_overlay_logical_content": True,
            "base_raw_geometry_digest_retained_only_in_receipt": True,
            "e3fp_recomputed": False,
            "second_pass_chemistry_recomputed": False,
            "sequence_truncation": False,
            "all_masks_all_epochs_target_and_sentinel_capacity_proven": True,
            "complete_lmdb_decode_replay": True,
            "publication_is_atomic_sibling_rename": True,
        },
    }
    if any(
        summary["over_512"]
        for summary in manifest["lengths"].values()  # type: ignore[union-attr]
    ):
        raise PairedCanaryBuildError("length manifest contains an over-512 row")
    _write_json(staging_root / MANIFEST_NAME, manifest)
    staging_root.rename(output_dir)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--overlay-root", required=True)
    parser.add_argument("--base-tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=250_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    manifest = run(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "InputBundle",
    "PairedCanaryBuildError",
    "PreparedMember",
    "RecordRejected",
    "build_macro_registry",
    "build_parser",
    "cross_edges_from_augmentation",
    "main",
    "prepare_member",
    "run",
    "run_cpu_four_grid_dry_collate",
]
