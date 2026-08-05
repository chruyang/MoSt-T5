#!/usr/bin/env python3
"""Build the frozen 32 + 256 topology-canary selection from production shard 0.

Only existing production-v2 membership and payload fields are read.  The
builder never opens the source SDF and never runs the linearizer, topology
augmentation, E3FP, or a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from most_t5_next.r1.adapter import run_p1_topology_canary_v1 as canary_runner
from most_t5_next.r1.adapter import sidecar_v2_codec


ALGORITHM_ID = "p1-topology-canary-shard0-feature-quota-sha256/v1"
SOURCE_SHARD_INDEX = 0
SMOKE_COUNT = canary_runner.SMOKE_COUNT
CANARY_COUNT = canary_runner.CANARY_COUNT

BOUNDARY_RULES = (
    {
        "tag": "motif_count_eq_1",
        "quota": 40,
        "predicate": {"feature": "motif_count", "operator": "eq", "value": 1},
    },
    {
        "tag": "motif_count_ge_8",
        "quota": 48,
        "predicate": {"feature": "motif_count", "operator": "ge", "value": 8},
    },
    {
        "tag": "model_atom_count_le_12",
        "quota": 40,
        "predicate": {"feature": "model_atom_count", "operator": "le", "value": 12},
    },
    {
        "tag": "model_atom_count_ge_18",
        "quota": 40,
        "predicate": {"feature": "model_atom_count", "operator": "ge", "value": 18},
    },
    {
        "tag": "singleton_fraction_ge_0p5",
        "quota": 48,
        "predicate": {
            "feature": "singleton_motif_fraction",
            "operator": "ge",
            "value": 0.5,
            "minimum_motif_count": 4,
        },
    },
    {
        "tag": "max_motif_size_ge_8",
        "quota": 40,
        "predicate": {"feature": "max_motif_size", "operator": "ge", "value": 8},
    },
)


class SelectionBuildError(RuntimeError):
    """The frozen shard cannot support the requested selection."""


@dataclass(frozen=True)
class MemberFeatures:
    member_id: str
    sdf_record_index: int
    model_atom_count: int
    motif_count: int
    motif_group_sizes: tuple[int, ...]

    @property
    def max_motif_size(self) -> int:
        return max(self.motif_group_sizes)

    @property
    def singleton_motif_fraction(self) -> float:
        return sum(size == 1 for size in self.motif_group_sizes) / self.motif_count


def _rank(seed: str, role: str, member_id: str) -> tuple[str, str]:
    payload = f"{seed}|{role}|{member_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), member_id


def _matches(member: MemberFeatures, rule: Mapping[str, object]) -> bool:
    tag = rule["tag"]
    if tag == "motif_count_eq_1":
        return member.motif_count == 1
    if tag == "motif_count_ge_8":
        return member.motif_count >= 8
    if tag == "model_atom_count_le_12":
        return member.model_atom_count <= 12
    if tag == "model_atom_count_ge_18":
        return member.model_atom_count >= 18
    if tag == "singleton_fraction_ge_0p5":
        return member.motif_count >= 4 and member.singleton_motif_fraction >= 0.5
    if tag == "max_motif_size_ge_8":
        return member.max_motif_size >= 8
    raise SelectionBuildError(f"unknown boundary rule: {tag}")


def _tags(member: MemberFeatures) -> list[str]:
    return sorted(rule["tag"] for rule in BOUNDARY_RULES if _matches(member, rule))


def _selection_row(member: MemberFeatures, tags: Sequence[str]) -> dict:
    return {
        "sdf_record_index": member.sdf_record_index,
        "selection_tags": sorted(set(tags)),
    }


def build_selection_document(
    *,
    features: Sequence[MemberFeatures],
    release_binding: Mapping[str, object],
    source_scope: Mapping[str, object],
    selection_id: str,
    seed: str,
) -> dict:
    """Apply the predeclared hash and feature-quota schedule."""

    if not selection_id or not seed:
        raise SelectionBuildError("selection_id and seed must be non-empty")
    unique = {member.member_id: member for member in features}
    unique_ordinals = {member.sdf_record_index for member in features}
    if len(unique) != len(features) or len(unique_ordinals) != len(features):
        raise SelectionBuildError("shard0 admitted feature rows contain duplicate identities")
    if len(features) < SMOKE_COUNT + CANARY_COUNT:
        raise SelectionBuildError("shard0 has fewer than 288 admitted members")

    ordered = sorted(features, key=lambda member: _rank(seed, "smoke", member.member_id))
    smoke = ordered[:SMOKE_COUNT]
    smoke_ids = {member.member_id for member in smoke}
    pool = [member for member in features if member.member_id not in smoke_ids]

    selected: list[MemberFeatures] = []
    selected_ids: set[str] = set()
    quota_stats: list[dict] = []
    for rule in BOUNDARY_RULES:
        eligible = [member for member in pool if _matches(member, rule)]
        ranked = sorted(
            eligible,
            key=lambda member, tag=rule["tag"]: _rank(seed, f"canary|{tag}", member.member_id),
        )
        chosen = [member for member in ranked if member.member_id not in selected_ids][
            : rule["quota"]
        ]
        selected.extend(chosen)
        selected_ids.update(member.member_id for member in chosen)
        quota_stats.append(
            {
                "tag": rule["tag"],
                "requested_quota": rule["quota"],
                "eligible_count": len(eligible),
                "selected_via_quota_count": len(chosen),
            }
        )

    if len(selected) > CANARY_COUNT:
        raise SelectionBuildError("fixed feature quotas exceed the canary budget")
    fill = sorted(
        (member for member in pool if member.member_id not in selected_ids),
        key=lambda member: _rank(seed, "canary|fill", member.member_id),
    )[: CANARY_COUNT - len(selected)]
    selected.extend(fill)
    if len(selected) != CANARY_COUNT:
        raise SelectionBuildError("shard0 cannot fill the 256-record canary")

    canary_rows = []
    fill_ids = {member.member_id for member in fill}
    for member in selected:
        tags = _tags(member)
        if member.member_id in fill_ids and not tags:
            tags = ["hash_fill"]
        canary_rows.append(_selection_row(member, tags))
    observed_boundary_tags = {tag for row in canary_rows for tag in row["selection_tags"]}
    required_boundary_tags = {rule["tag"] for rule in BOUNDARY_RULES}
    missing_tags = sorted(required_boundary_tags - observed_boundary_tags)
    if missing_tags:
        raise SelectionBuildError(
            "shard0 selection cannot cover boundary tags: {}".format(",".join(missing_tags))
        )

    return {
        "schema_version": canary_runner.SELECTION_SCHEMA,
        "selection_id": selection_id,
        "release": {
            "release_id": release_binding["release_id"],
            "full_release_manifest_sha256": release_binding["full_release_manifest_sha256"],
            "logical_release_root_sha256": release_binding["logical_release_root_sha256"],
        },
        "selection_algorithm": {
            "algorithm_id": ALGORITHM_ID,
            "seed": seed,
            "source_scope": dict(source_scope),
            "source_features": ["model_atom_count", "motif_count", "motif_group_sizes"],
            "smoke": {
                "target_count": SMOKE_COUNT,
                "ranking": "SHA256(seed|smoke|member_id), then member_id",
            },
            "canary": {
                "target_count": CANARY_COUNT,
                "rules_in_order": [dict(rule) for rule in BOUNDARY_RULES],
                "deduplication": "first selected rule retains the member",
                "fill_ranking": "SHA256(seed|canary|fill|member_id), then member_id",
                "quota_statistics": quota_stats,
                "hash_fill_count": len(fill),
            },
        },
        "groups": {
            "smoke": [_selection_row(member, ["smoke_hash_prefix"]) for member in smoke],
            "canary": canary_rows,
        },
    }


def _feature_from_record(record: Mapping[str, object], membership: Mapping[str, object]) -> MemberFeatures:
    ordinal = membership["sdf_record_index"]
    member = record.get("member", {})
    atom_universe = record.get("atom_universe", {})
    topology = record.get("topology", {})
    if not (
        record.get("record_schema_version") == canary_runner.PRODUCTION_RECORD_SCHEMA
        and member.get("member_id") == membership.get("member_id")
        and member.get("sdf_record_index") == ordinal
        and member.get("storage_key") == membership.get("record_storage_key")
    ):
        raise SelectionBuildError("admitted payload and membership do not bind")
    atom_count = atom_universe.get("model_atom_count")
    motif_count = topology.get("motif_count")
    groups = topology.get("motif_atom_indices")
    if not (
        isinstance(atom_count, int)
        and atom_count > 0
        and isinstance(motif_count, int)
        and motif_count > 0
        and isinstance(groups, list)
        and len(groups) == motif_count
    ):
        raise SelectionBuildError("admitted payload lacks usable topology counts")
    sizes = tuple(int(len(group)) for group in groups)
    if any(size <= 0 for size in sizes) or sum(sizes) != atom_count:
        raise SelectionBuildError("motif group sizes do not partition the model atoms")
    return MemberFeatures(membership["member_id"], ordinal, atom_count, motif_count, sizes)


def collect_admitted_features(
    membership_rows: Iterable[Mapping[str, object]],
    payload_loader: Callable[[Mapping[str, object]], tuple[Mapping[str, object], str]],
) -> list[MemberFeatures]:
    """Decode payloads only for rows whose frozen disposition is ``admit``."""

    features: list[MemberFeatures] = []
    for membership in membership_rows:
        if membership.get("disposition") != "admit":
            continue
        record, logical_hash = payload_loader(membership)
        if logical_hash != membership.get("record_content_sha256"):
            raise SelectionBuildError("admitted payload logical hash differs from membership")
        features.append(_feature_from_record(record, membership))
    return features


def read_shard0_features(release_root: Path, np, lmdb_module) -> tuple[list[MemberFeatures], dict, dict]:
    manifest_path = release_root / "full_release_manifest.json"
    manifest = canary_runner.load_json(manifest_path, "production-v2 full release manifest")
    if not (
        manifest.get("schema_version") == canary_runner.FULL_RELEASE_SCHEMA
        and manifest.get("release_status") == "complete"
    ):
        raise SelectionBuildError("completed production-v2 release is required")
    top_entry = next(
        (row for row in manifest.get("shards", []) if row.get("shard_index") == SOURCE_SHARD_INDEX),
        None,
    )
    if top_entry is None:
        raise SelectionBuildError("production release has no shard-000000")
    shard_dir = release_root / "shard-000000"
    shard_manifest_path = shard_dir / "shard_manifest.json"
    shard_manifest = canary_runner.load_json(shard_manifest_path, "shard0 manifest")
    if canary_runner.sha256_file(shard_manifest_path) != top_entry.get("shard_manifest_sha256"):
        raise SelectionBuildError("shard0 manifest differs from the full release")

    with (shard_dir / "membership.jsonl").open("r", encoding="utf-8") as handle:
        membership_rows = [json.loads(line) for line in handle]
    environment = lmdb_module.open(
        str(shard_dir / "geometry_records.lmdb"),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=4,
    )
    try:
        with environment.begin(write=False) as transaction:
            def load_payload(membership):
                raw = transaction.get(membership["record_storage_key"].encode("ascii"))
                if raw is None:
                    raise SelectionBuildError("admitted shard0 payload is missing")
                return sidecar_v2_codec.decode_record(np, raw)

            features = collect_admitted_features(membership_rows, load_payload)
    finally:
        environment.close()

    admitted_expected = shard_manifest.get("counts", {}).get("admitted_record_count")
    if len(features) != admitted_expected:
        raise SelectionBuildError("decoded admitted count differs from shard0 manifest")
    release_binding = {
        "release_id": manifest["release_id"],
        "full_release_manifest_sha256": canary_runner.sha256_file(manifest_path),
        "logical_release_root_sha256": manifest["logical_release_root_sha256"],
    }
    source_scope = {
        "shard_index": SOURCE_SHARD_INDEX,
        "range_start": shard_manifest["range_start"],
        "range_end": shard_manifest["range_end"],
        "admitted_record_count": len(features),
        "shard_manifest_sha256": top_entry["shard_manifest_sha256"],
    }
    return features, release_binding, source_scope


def write_selection(path: Path, document: Mapping[str, object]) -> None:
    if path.exists():
        raise SelectionBuildError("--output must be a new file")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict:
    release_root = Path(args.release_root).expanduser().resolve()
    if not release_root.is_dir():
        raise SelectionBuildError("--release-root must be a directory")
    try:
        import lmdb
        import numpy as np
    except ImportError as exc:
        raise SelectionBuildError("NumPy and python-lmdb are required") from exc
    features, release_binding, source_scope = read_shard0_features(release_root, np, lmdb)
    document = build_selection_document(
        features=features,
        release_binding=release_binding,
        source_scope=source_scope,
        selection_id=args.selection_id,
        seed=args.seed,
    )
    output = Path(args.output).expanduser().resolve()
    write_selection(output, document)
    return {
        "output": str(output),
        "sha256": canary_runner.sha256_file(output),
        "smoke_count": len(document["groups"]["smoke"]),
        "canary_count": len(document["groups"]["canary"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--selection-id", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
