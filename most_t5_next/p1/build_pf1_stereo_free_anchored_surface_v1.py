#!/usr/bin/env python3
"""Derive the PF-1 stereo-free anchored motif surface without SDF/E3FP work.

The builder joins three already published authorities:

* PF-1 paired membership and its persisted slot/edge/atom sidecar;
* production-v2 motif lexeme digests for the same selected records;
* the production-v2 global digest-to-exact-lexeme census.

GraphPorts remains inside the source paired wire and is never copied to the
new model-facing surface.  The output is a sequential JSONL research release
for tokenizer and phrase-boundary experiments, not a training admission.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

from most_t5_next.r1.adapter import build_p1_inherited_e3fp_overlay_v1 as overlay
from most_t5_next.r1.adapter import paired_record_wire_v1 as paired_wire
from most_t5_next.r1.adapter import run_p1_topology_canary_v1 as release_reader
from most_t5_next.r1.adapter import sidecar_v2_codec
from most_t5_next.r1.tokenizer.stereo_free_anchored_motif_surface_v1 import (
    AnchoredMotifSurfaceError,
    build_stereo_free_anchored_surface_from_persisted_pair,
    surface_document,
)


SCHEMA_VERSION = "most-t5-next/pf1-stereo-free-anchored-surface-release/v1"
MEMBER_SCHEMA_VERSION = "most-t5-next/pf1-stereo-free-anchored-surface-member/v1"
REJECT_SCHEMA_VERSION = "most-t5-next/pf1-stereo-free-anchored-surface-reject/v1"
SURFACE_RECORD_SCHEMA_VERSION = "most-t5-next/pf1-stereo-free-anchored-surface-record/v1"
MANIFEST_NAME = "manifest.json"
SURFACE_RECORDS_NAME = "surface_records.jsonl"
MEMBERSHIP_NAME = "membership.jsonl"
PURE_MOTIF_CENSUS_NAME = "pure_motif_census.jsonl"
REJECTS_NAME = "rejects.jsonl"
STAGING_SUFFIX = ".staging"


class PF1AnchoredSurfaceReleaseError(RuntimeError):
    """The selected PF-1 surface release cannot be derived exactly."""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise PF1AnchoredSurfaceReleaseError(
                    f"{path.name} line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise PF1AnchoredSurfaceReleaseError(
                    f"{path.name} line {line_number} is not one JSON object"
                )
            yield line_number, row


def load_paired_membership(
    paired_release: Path, *, max_members: int | None = None
) -> tuple[dict[str, object], ...]:
    rows = []
    seen_members = set()
    seen_keys = set()
    for split, filename in (
        ("train", "train_membership.jsonl"),
        ("dev", "dev_membership.jsonl"),
    ):
        for _, row in _iter_jsonl(Path(paired_release) / filename):
            if row.get("split") != split:
                raise PF1AnchoredSurfaceReleaseError("paired membership split differs")
            selection_index = row.get("selection_index")
            ordinal = row.get("sdf_record_index")
            member_id = row.get("member_id")
            storage_key = row.get("storage_key")
            if (
                isinstance(selection_index, bool)
                or not isinstance(selection_index, int)
                or selection_index < 0
                or isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal < 0
                or not isinstance(member_id, str)
                or not member_id
                or not isinstance(storage_key, str)
                or not storage_key.isascii()
            ):
                raise PF1AnchoredSurfaceReleaseError("paired membership identity is invalid")
            if member_id in seen_members or storage_key in seen_keys:
                raise PF1AnchoredSurfaceReleaseError("paired membership repeats a member/key")
            seen_members.add(member_id)
            seen_keys.add(storage_key)
            rows.append(dict(row))
    rows.sort(key=lambda row: int(row["selection_index"]))
    if tuple(int(row["selection_index"]) for row in rows) != tuple(range(len(rows))):
        raise PF1AnchoredSurfaceReleaseError("paired selection indices are not dense")
    if max_members is not None:
        if isinstance(max_members, bool) or not isinstance(max_members, int) or max_members <= 0:
            raise PF1AnchoredSurfaceReleaseError("max_members must be positive")
        rows = rows[:max_members]
    return tuple(rows)


def load_motif_lexeme_census(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for _, row in _iter_jsonl(path):
        digest = row.get("motif_lexeme_sha256")
        fragment = row.get("motif_fragment")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(fragment, str)
            or not fragment
            or hashlib.sha256(fragment.encode("utf-8")).hexdigest() != digest
        ):
            raise PF1AnchoredSurfaceReleaseError("global motif census row is invalid")
        previous = result.setdefault(digest, fragment)
        if previous != fragment:
            raise PF1AnchoredSurfaceReleaseError("motif digest maps to distinct lexemes")
    if not result:
        raise PF1AnchoredSurfaceReleaseError("global motif census is empty")
    return result


class _ProductionRecordReader:
    """Validate selected production rows and read each required shard once."""

    def __init__(
        self,
        *,
        release_root: Path,
        members: Sequence[Mapping[str, object]],
        np: Any,
        lmdb_module: Any,
    ) -> None:
        self.release_root = Path(release_root)
        self.np = np
        self.lmdb_module = lmdb_module
        manifest_path = self.release_root / "full_release_manifest.json"
        candidate = release_reader.load_json(manifest_path, "production full manifest")
        selection = {
            "release": {
                "release_id": candidate.get("release_id"),
                "full_release_manifest_sha256": release_reader.sha256_file(manifest_path),
                "logical_release_root_sha256": candidate.get("logical_release_root_sha256"),
            }
        }
        self.manifest_path, self.manifest = release_reader.load_release_manifest(
            self.release_root, selection
        )
        by_shard: dict[int, list[int]] = defaultdict(list)
        top_entries: dict[int, dict[str, object]] = {}
        for member in members:
            ordinal = int(member["sdf_record_index"])
            shard = release_reader._shard_for_ordinal(self.manifest, ordinal)
            shard_index = int(shard["shard_index"])
            by_shard[shard_index].append(ordinal)
            top_entries[shard_index] = shard
        self.memberships: dict[int, dict[str, object]] = {}
        self.shard_by_ordinal: dict[int, tuple[int, Path]] = {}
        for shard_index in sorted(by_shard):
            top = top_entries[shard_index]
            shard_dir = self.release_root / f"shard-{shard_index:06d}"
            shard_manifest_path = shard_dir / "shard_manifest.json"
            shard_manifest = release_reader.load_json(
                shard_manifest_path, "production shard manifest"
            )
            if release_reader.sha256_file(shard_manifest_path) != top.get(
                "shard_manifest_sha256"
            ):
                raise PF1AnchoredSurfaceReleaseError("production shard binding differs")
            selected = release_reader._read_selected_membership(
                shard_dir / "membership.jsonl",
                int(shard_manifest["range_start"]),
                sorted(by_shard[shard_index]),
            )
            for ordinal in sorted(by_shard[shard_index]):
                membership = selected[ordinal]
                if membership.get("disposition") != "admit":
                    raise PF1AnchoredSurfaceReleaseError("selected member is rejected by production")
                self.memberships[ordinal] = membership
                self.shard_by_ordinal[ordinal] = (shard_index, shard_dir)
        self._current_shard: int | None = None
        self._environment: Any = None

    def get(self, member: Mapping[str, object]) -> dict[str, object]:
        ordinal = int(member["sdf_record_index"])
        membership = self.memberships[ordinal]
        if membership.get("member_id") != member.get("member_id"):
            raise PF1AnchoredSurfaceReleaseError("production membership member differs")
        shard_index, shard_dir = self.shard_by_ordinal[ordinal]
        if self._current_shard != shard_index:
            self.close()
            self._environment = self.lmdb_module.open(
                str(shard_dir / "geometry_records.lmdb"),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=8,
            )
            self._current_shard = shard_index
        with self._environment.begin(write=False) as transaction:
            raw = transaction.get(str(membership["record_storage_key"]).encode("ascii"))
        if raw is None:
            raise PF1AnchoredSurfaceReleaseError("production record is absent")
        record, logical_hash = sidecar_v2_codec.decode_record(self.np, bytes(raw))
        if logical_hash != membership.get("record_content_sha256"):
            raise PF1AnchoredSurfaceReleaseError("production logical hash differs")
        overlay.validate_overlay_release_record(record, membership, ordinal)
        return record

    def load_motif_lexeme_digests(
        self, members: Sequence[Mapping[str, object]]
    ) -> dict[int, tuple[str, ...]]:
        """Return selection-index keyed motif digests with one pass per shard.

        PF-1 membership is randomized by connectivity group, so following its
        selection order would repeatedly reopen the same 136 production LMDBs.
        Reading in ``(shard, ordinal)`` order preserves every record-level
        validation while reducing the I/O pattern to one open per used shard.
        The caller still emits the model-facing artifact in frozen selection
        order.
        """

        ordered = sorted(
            members,
            key=lambda member: (
                self.shard_by_ordinal[int(member["sdf_record_index"])][0],
                int(member["sdf_record_index"]),
            ),
        )
        result: dict[int, tuple[str, ...]] = {}
        for member in ordered:
            selection_index = int(member["selection_index"])
            if selection_index in result:
                raise PF1AnchoredSurfaceReleaseError(
                    "production digest preload repeats a selection index"
                )
            record = self.get(member)
            raw_digests = record.get("topology", {}).get("motif_lexeme_sha256")
            if not isinstance(raw_digests, (list, tuple)) or not raw_digests:
                raise PF1AnchoredSurfaceReleaseError(
                    "production motif digest sequence is absent"
                )
            digests = tuple(str(value) for value in raw_digests)
            if any(len(value) != 64 for value in digests):
                raise PF1AnchoredSurfaceReleaseError(
                    "production motif digest sequence is invalid"
                )
            result[selection_index] = digests
        if len(result) != len(members):
            raise PF1AnchoredSurfaceReleaseError(
                "production digest preload count differs from membership"
            )
        self.close()
        return result

    def close(self) -> None:
        if self._environment is not None:
            self._environment.close()
            self._environment = None
            self._current_shard = None


def _distribution(values: Iterable[int]) -> dict[str, int]:
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


def build_release(args: argparse.Namespace) -> dict[str, object]:
    try:
        import lmdb
        import numpy as np
    except ImportError as exc:
        raise PF1AnchoredSurfaceReleaseError("numpy and python-lmdb are required") from exc

    paired_root = Path(args.paired_release).expanduser().resolve()
    production_root = Path(args.production_release).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    staging = output.with_name(output.name + STAGING_SUFFIX)
    if output.exists() or staging.exists():
        raise PF1AnchoredSurfaceReleaseError("output and sibling staging must be absent")
    members = load_paired_membership(paired_root, max_members=args.max_members)
    if not members:
        raise PF1AnchoredSurfaceReleaseError("selected membership is empty")
    paired_manifest_path = paired_root / "manifest.json"
    with paired_manifest_path.open("r", encoding="utf-8") as handle:
        paired_manifest = json.load(handle)
    if not isinstance(paired_manifest, dict) or paired_manifest.get("status") != "pass":
        raise PF1AnchoredSurfaceReleaseError("paired input is not a passed release")
    census_path = production_root / "motif_census.jsonl"
    lexemes = load_motif_lexeme_census(census_path)
    staging.mkdir(parents=False)
    started = time.perf_counter()
    rejects_path = staging / REJECTS_NAME
    records_path = staging / SURFACE_RECORDS_NAME
    membership_path = staging / MEMBERSHIP_NAME
    pure_counts: dict[str, Counter[str]] = defaultdict(Counter)
    explicit_lengths = []
    implicit_lengths = []
    motif_counts = []
    anchor_counts = []
    record_bytes = []
    max_anchor_id = -1
    processed = 0
    rejected = 0
    production_reader = _ProductionRecordReader(
        release_root=production_root,
        members=members,
        np=np,
        lmdb_module=lmdb,
    )
    motif_digests_by_selection = production_reader.load_motif_lexeme_digests(members)
    paired_environment = lmdb.open(
        str(paired_root / "paired_records.lmdb"),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=8,
    )
    try:
        with records_path.open("xb") as records_handle, \
                membership_path.open(
                    "x", encoding="utf-8", newline="\n"
                ) as membership_handle, \
                rejects_path.open(
                    "x", encoding="utf-8", newline="\n"
                ) as rejects_handle, \
                paired_environment.begin(write=False) as paired_transaction:
            for member in members:
                try:
                    raw = paired_transaction.get(str(member["storage_key"]).encode("ascii"))
                    if raw is None:
                        raise PF1AnchoredSurfaceReleaseError("paired record is absent")
                    payload = bytes(raw)
                    loaded = paired_wire.decode_paired_training_record(payload)
                    envelope = json.loads(payload)
                    motif_document = envelope["motif_training_document"]
                    if not (
                        loaded.schedule_index == member["selection_index"]
                        and loaded.sdf_record_index == member["sdf_record_index"]
                        and loaded.atom_record.record_id == member["member_id"]
                    ):
                        raise PF1AnchoredSurfaceReleaseError("paired record differs from membership")
                    digests = motif_digests_by_selection[int(member["selection_index"])]
                    try:
                        exact_lexemes = tuple(lexemes[digest] for digest in digests)
                    except KeyError as exc:
                        raise PF1AnchoredSurfaceReleaseError(
                            "production motif digest is absent from the global census"
                        ) from exc
                    logical = motif_document["logical_motif_domain"]
                    atom = motif_document["atom_domain"]
                    dimensions = motif_document["dimensions"]
                    if len(exact_lexemes) != int(dimensions["logical_motif_count"]):
                        raise PF1AnchoredSurfaceReleaseError(
                            "production lexeme count differs from paired motif count"
                        )
                    surface = build_stereo_free_anchored_surface_from_persisted_pair(
                        member_id=str(member["member_id"]),
                        source_atom_count=int(dimensions["source_atom_count"]),
                        model_to_source_atom_index=atom["model_to_source_atom_index"],
                        atom_is_attachment=atom["atom_is_attachment"],
                        motif_atom_indices=logical["motif_atom_indices"],
                        exact_motif_lexemes=exact_lexemes,
                        motif_slot_atom_indices=logical["motif_slot_atom_indices"],
                        cross_motif_bonds=logical["cross_motif_bonds"],
                    )
                    explicit = surface.render("explicit")
                    implicit = surface.render("implicit")
                    document = {
                        "schema_version": SURFACE_RECORD_SCHEMA_VERSION,
                        "selection_index": member["selection_index"],
                        "sdf_record_index": member["sdf_record_index"],
                        "split": member["split"],
                        "storage_key": member["storage_key"],
                        "surface": surface_document(surface),
                        "renderings": {
                            "explicit": {
                                "tokens": list(explicit.tokens),
                                "phrase_spans": [list(row) for row in explicit.phrase_spans],
                                "motif_to_carrier": list(explicit.motif_to_carrier),
                                "anchor_token_positions": [list(row) for row in explicit.anchor_token_positions],
                                "component_token_ranges": [list(row) for row in explicit.component_token_ranges],
                            },
                            "implicit": {
                                "tokens": list(implicit.tokens),
                                "phrase_spans": [list(row) for row in implicit.phrase_spans],
                                "motif_to_carrier": list(implicit.motif_to_carrier),
                                "anchor_token_positions": [list(row) for row in implicit.anchor_token_positions],
                                "component_token_ranges": [list(row) for row in implicit.component_token_ranges],
                            },
                        },
                    }
                    encoded = _canonical_json_bytes(document) + b"\n"
                    offset = records_handle.tell()
                    records_handle.write(encoded)
                    membership_row = {
                        "schema_version": MEMBER_SCHEMA_VERSION,
                        "selection_index": member["selection_index"],
                        "sdf_record_index": member["sdf_record_index"],
                        "split": member["split"],
                        "member_id": member["member_id"],
                        "source_storage_key": member["storage_key"],
                        "surface_artifact_sha256": surface.artifact_sha256,
                        "record_offset": offset,
                        "record_length": len(encoded),
                    }
                    membership_handle.write(_canonical_json_bytes(membership_row).decode("utf-8") + "\n")
                    for phrase in surface.phrases:
                        pure_counts[phrase.pure_motif][str(member["split"])] += 1
                        for occurrence in phrase.anchors:
                            max_anchor_id = max(max_anchor_id, occurrence.anchor_id)
                    explicit_lengths.append(len(explicit.tokens))
                    implicit_lengths.append(len(implicit.tokens))
                    motif_counts.append(len(surface.phrases))
                    anchor_counts.append(sum(len(phrase.anchors) for phrase in surface.phrases))
                    record_bytes.append(len(encoded))
                    processed += 1
                except Exception as exc:
                    rejected += 1
                    reject = {
                        "schema_version": REJECT_SCHEMA_VERSION,
                        "selection_index": member.get("selection_index"),
                        "sdf_record_index": member.get("sdf_record_index"),
                        "member_id": member.get("member_id"),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                    rejects_handle.write(_canonical_json_bytes(reject).decode("utf-8") + "\n")
                    break
                if args.progress_every and processed % args.progress_every == 0:
                    print(f"anchored-surface {processed}/{len(members)}", flush=True)
    finally:
        paired_environment.close()
        production_reader.close()

    pure_path = staging / PURE_MOTIF_CENSUS_NAME
    with pure_path.open("x", encoding="utf-8", newline="\n") as handle:
        for pure in sorted(pure_counts, key=lambda value: value.encode("utf-8")):
            counts = pure_counts[pure]
            row = {
                "pure_motif": pure,
                "pure_motif_sha256": hashlib.sha256(pure.encode("utf-8")).hexdigest(),
                "train_occurrences": int(counts["train"]),
                "dev_occurrences": int(counts["dev"]),
                "total_occurrences": int(counts["train"] + counts["dev"]),
            }
            handle.write(_canonical_json_bytes(row).decode("utf-8") + "\n")

    passed = processed == len(members) and rejected == 0
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if passed else "failed",
        "created_at": _utc_now(),
        "scope": "pf1_full" if args.max_members is None else "bounded_prefix_smoke",
        "inputs": {
            "paired_release_manifest": {
                "path": str(paired_manifest_path),
                "sha256": _sha256_file(paired_manifest_path),
            },
            "production_release_manifest": {
                "path": str(production_reader.manifest_path),
                "sha256": _sha256_file(production_reader.manifest_path),
            },
            "production_motif_census": {
                "path": str(census_path),
                "sha256": _sha256_file(census_path),
                "unique_lexemes": len(lexemes),
            },
        },
        "counts": {
            "scheduled_members": len(members),
            "surface_records": processed,
            "rejected_members": rejected,
            "unique_pure_motifs": len(pure_counts),
        },
        "distributions": {
            "explicit_logical_tokens": _distribution(explicit_lengths),
            "implicit_logical_tokens": _distribution(implicit_lengths),
            "motifs_per_member": _distribution(motif_counts),
            "anchor_occurrences_per_member": _distribution(anchor_counts),
            "surface_record_bytes": _distribution(record_bytes),
        },
        "contracts": {
            "geometry_or_e3fp_recomputed": False,
            "source_sdf_read": False,
            "graphports_exposed_to_model": False,
            "stereo_markers_in_pure_motif": 0,
            "cross_motif_bond_domain": "SINGLE_only",
            "model_facing_anchor_ids": "canonical_edge_id_dense_per_molecule",
            "source_anchor_pairing_revalidated": True,
            "explicit_boundary_tokens_per_motif": 1,
            "explicit_and_implicit_renderings_same_logical_surface": True,
            "training_admission": False,
        },
        "observations": {
            "max_model_facing_anchor_id": max_anchor_id,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    manifest["artifacts"] = {
        SURFACE_RECORDS_NAME: _artifact(records_path),
        MEMBERSHIP_NAME: _artifact(membership_path),
        PURE_MOTIF_CENSUS_NAME: _artifact(pure_path),
        REJECTS_NAME: _artifact(rejects_path),
    }
    _write_json_new(staging / MANIFEST_NAME, manifest)
    if not passed:
        raise PF1AnchoredSurfaceReleaseError(
            f"anchored surface derivation rejected {rejected} member(s); staging retained"
        )
    staging.rename(output)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", required=True)
    parser.add_argument("--production-release", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-members", type=int)
    parser.add_argument("--progress-every", type=int, default=1024)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_release(args)
    except Exception as exc:
        print(f"PF1 anchored surface failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
