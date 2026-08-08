#!/usr/bin/env python3
"""Freeze a named PF-1/PF-10 connectivity-group membership.

The input clean membership contains member IDs only.  This module therefore
streams the admitted PCQM identity rows, joins them to the permitted IDs and
uses the already-defined ``connectivity_identity_sha256`` as the scientific
group key.  Selection itself is performed by NumPy PCG64, never by member or
digest rank.

No molecule payload is copied.  Both named profiles are complete at the
connectivity-group boundary.  PF-10 replays the PF-1 group order and preserves
every supplied PF-1 baseline train/dev role; only newly admitted groups are
assigned to fill the larger dev quota.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "most-t5-p1/pf1-connectivity-sample-manifest/v1"
MEMBERSHIP_SCHEMA = "most-t5-p1/pf1-connectivity-sample-member/v1"
BENCHMARK_SCHEMA = "most-t5-p1/pf1-materialization-benchmark-member/v1"
INELIGIBLE_GROUP_SCHEMA = "most-t5-p1/pf1-ineligible-connectivity-group/v1"
PERMITTED_SCHEMA = "most-t5-r1/permitted-pretrain-member/v1"
IDENTITY_SCHEMA = "most-t5-r1/molecule-identity-row/v1"
MEMBER_PREFIX = "ogb_pcqm4mv2_train_row_index:"
EXPECTED_PERMITTED_MEMBERS = 3_360_067
TARGET_MEMBERS = math.floor(0.01 * EXPECTED_PERMITTED_MEMBERS)
PF10_TARGET_MEMBERS = math.floor(0.10 * EXPECTED_PERMITTED_MEMBERS)
DEV_FRACTION = 0.10
BENCHMARK_TARGET_MEMBERS = 1024
DEFAULT_SEED = 20_260_807
PF1_PROFILE_ID = "pf1-final-v4-1pct-v1"
PF10_PROFILE_ID = "pf10-final-v4-10pct-split-preserving-v1"
SELECTION_PROFILES = {
    PF1_PROFILE_ID: {
        "target_fraction": 0.01,
        "target_members": TARGET_MEMBERS,
        "requires_baseline": False,
        "preserve_baseline_splits": False,
    },
    PF10_PROFILE_ID: {
        "target_fraction": 0.10,
        "target_members": PF10_TARGET_MEMBERS,
        "requires_baseline": True,
        "preserve_baseline_splits": True,
    },
}


class PF1SelectionError(RuntimeError):
    """The requested group-complete selection cannot be frozen."""


def resolve_selection_profile(
    args: argparse.Namespace,
) -> tuple[str | None, int, bool, float | None]:
    """Resolve the named fidelity contract without changing fixture callers.

    Older programmatic callers pass an explicit ``target_members`` and no
    profile.  The formal CLI always supplies one of ``SELECTION_PROFILES``;
    PF-10 consequently cannot silently drift from 336,006 members, the frozen
    seed, or baseline split preservation.
    """

    profile_id = getattr(args, "selection_profile", None)
    requested_target = getattr(args, "target_members", None)
    explicit_preservation = bool(
        getattr(args, "preserve_baseline_splits", False)
    )
    if profile_id is None:
        if (
            isinstance(requested_target, bool)
            or not isinstance(requested_target, int)
            or requested_target <= 0
        ):
            raise PF1SelectionError("target_members must be a positive integer")
        return None, requested_target, explicit_preservation, None
    try:
        profile = SELECTION_PROFILES[profile_id]
    except KeyError as exc:
        raise PF1SelectionError("unknown connectivity selection profile") from exc
    target = int(profile["target_members"])
    if requested_target is not None and requested_target != target:
        raise PF1SelectionError(
            "named connectivity profile forbids a different target_members"
        )
    if getattr(args, "expected_permitted_members", None) != EXPECTED_PERMITTED_MEMBERS:
        raise PF1SelectionError(
            "named connectivity profile requires final-v4 permitted 3,360,067"
        )
    if getattr(args, "seed", None) != DEFAULT_SEED:
        raise PF1SelectionError(
            "named connectivity profile requires PCG64 seed 20260807"
        )
    preserve = bool(profile["preserve_baseline_splits"])
    if bool(profile["requires_baseline"]) and not getattr(
        args, "baseline_membership", None
    ):
        raise PF1SelectionError("PF-10 requires the frozen PF-1 baseline membership")
    return profile_id, target, preserve, float(profile["target_fraction"])


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise PF1SelectionError(
                    f"{path.name} line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise PF1SelectionError(
                    f"{path.name} line {line_number} is not an object"
                )
            yield line_number, value


def member_ordinal(member_id: str) -> int:
    if not isinstance(member_id, str) or not member_id.startswith(MEMBER_PREFIX):
        raise PF1SelectionError(f"invalid PCQM member_id: {member_id!r}")
    suffix = member_id[len(MEMBER_PREFIX) :]
    if not suffix.isdigit() or (len(suffix) > 1 and suffix.startswith("0")):
        raise PF1SelectionError(f"invalid PCQM member ordinal: {member_id!r}")
    return int(suffix)


def load_permitted_member_ids(path: Path) -> set[str]:
    permitted: set[str] = set()
    for line_number, row in _read_jsonl(path):
        if row.get("schema_version") != PERMITTED_SCHEMA or set(row) != {
            "schema_version",
            "member_id",
        }:
            raise PF1SelectionError(
                f"permitted membership line {line_number} has an unexpected schema"
            )
        member_id = row.get("member_id")
        member_ordinal(member_id)
        if member_id in permitted:
            raise PF1SelectionError(f"duplicate permitted member_id: {member_id}")
        permitted.add(member_id)
    if not permitted:
        raise PF1SelectionError("permitted membership is empty")
    return permitted


def count_permitted_connectivity_groups(
    identity_rows: Path,
    permitted_ids: set[str],
    *,
    expected_evidence_groups: Mapping[str, str] | None = None,
) -> tuple[dict[str, int], int]:
    """Stream the identity collection once and count only permitted groups."""

    group_sizes: dict[str, int] = {}
    matched = 0
    expected_evidence_groups = expected_evidence_groups or {}
    matched_evidence: set[str] = set()
    for line_number, row in _read_jsonl(identity_rows):
        if row.get("schema_version") != IDENTITY_SCHEMA:
            raise PF1SelectionError(
                f"identity line {line_number} has an unexpected schema"
            )
        member_id = row.get("member_id")
        if member_id not in permitted_ids:
            continue
        group_id = row.get("connectivity_identity_sha256")
        if not isinstance(group_id, str) or not group_id:
            raise PF1SelectionError(
                f"permitted identity line {line_number} lacks connectivity identity"
            )
        group_sizes[group_id] = group_sizes.get(group_id, 0) + 1
        matched += 1
        expected_group = expected_evidence_groups.get(member_id)
        if expected_group is not None:
            if group_id != expected_group:
                raise PF1SelectionError(
                    "ineligible evidence member {!r} belongs to group {!r}, not {!r}".format(
                        member_id, group_id, expected_group
                    )
                )
            matched_evidence.add(member_id)
    if matched != len(permitted_ids):
        raise PF1SelectionError(
            "identity rows matched {:,} of {:,} permitted members".format(
                matched, len(permitted_ids)
            )
        )
    if sum(group_sizes.values()) != matched:
        raise PF1SelectionError("connectivity group counts do not balance")
    missing_evidence = set(expected_evidence_groups) - matched_evidence
    if missing_evidence:
        raise PF1SelectionError(
            "ineligible evidence members are absent from permitted identities: "
            + ", ".join(sorted(missing_evidence))
        )
    return group_sizes, matched


def load_ineligible_connectivity_groups(path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Load group-level support exclusions with record-level evidence."""

    groups: dict[str, dict] = {}
    evidence_groups: dict[str, str] = {}
    for line_number, row in _read_jsonl(path):
        if row.get("schema_version") != INELIGIBLE_GROUP_SCHEMA or set(row) != {
            "schema_version",
            "connectivity_identity_sha256",
            "reason",
            "member_evidence",
        }:
            raise PF1SelectionError(
                f"ineligible group line {line_number} has an unexpected schema"
            )
        group_id = row.get("connectivity_identity_sha256")
        reason = row.get("reason")
        evidence = row.get("member_evidence")
        if not isinstance(group_id, str) or not group_id:
            raise PF1SelectionError(
                f"ineligible group line {line_number} lacks connectivity identity"
            )
        if group_id in groups:
            raise PF1SelectionError(f"duplicate ineligible group: {group_id}")
        if not isinstance(reason, str) or not reason:
            raise PF1SelectionError(
                f"ineligible group line {line_number} lacks a reason"
            )
        if not isinstance(evidence, list) or not evidence:
            raise PF1SelectionError(
                f"ineligible group line {line_number} lacks member evidence"
            )
        normalized_evidence: list[dict[str, object]] = []
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {
                "member_id",
                "sdf_record_index",
                "reason",
            }:
                raise PF1SelectionError(
                    f"ineligible group line {line_number} has malformed member evidence"
                )
            member_id = item.get("member_id")
            ordinal = item.get("sdf_record_index")
            member_reason = item.get("reason")
            parsed_ordinal = member_ordinal(member_id)
            if (
                not isinstance(ordinal, int)
                or isinstance(ordinal, bool)
                or ordinal != parsed_ordinal
                or not isinstance(member_reason, str)
                or not member_reason
            ):
                raise PF1SelectionError(
                    f"ineligible group line {line_number} has invalid member evidence"
                )
            previous_group = evidence_groups.get(member_id)
            if previous_group is not None:
                raise PF1SelectionError(
                    f"duplicate ineligible evidence member: {member_id}"
                )
            evidence_groups[member_id] = group_id
            normalized_evidence.append(
                {
                    "member_id": member_id,
                    "sdf_record_index": ordinal,
                    "reason": member_reason,
                }
            )
        groups[group_id] = {
            "connectivity_identity_sha256": group_id,
            "reason": reason,
            "member_evidence": normalized_evidence,
        }
    if not groups:
        raise PF1SelectionError("ineligible connectivity group file is empty")
    return groups, evidence_groups


def closest_complete_prefix(
    ordered_group_ids: Sequence[str],
    group_sizes: Mapping[str, int],
    target_members: int,
) -> tuple[str, ...]:
    """Return the non-empty whole-group prefix nearest a member target."""

    if target_members <= 0 or not ordered_group_ids:
        raise PF1SelectionError("group prefix target and group order must be non-empty")
    selected: list[str] = []
    member_count = 0
    for group_id in ordered_group_ids:
        size = group_sizes.get(group_id)
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise PF1SelectionError(f"invalid group size for {group_id!r}")
        candidate = member_count + size
        if candidate < target_members:
            selected.append(group_id)
            member_count = candidate
            continue
        if not selected or abs(candidate - target_members) <= abs(
            member_count - target_members
        ):
            selected.append(group_id)
        break
    if not selected:
        raise PF1SelectionError("group prefix selection is empty")
    return tuple(selected)


def freeze_group_plan(
    group_sizes: Mapping[str, int],
    *,
    seed: int,
    target_members: int,
    dev_fraction: float,
    benchmark_target_members: int,
    ineligible_group_ids: Iterable[str] = (),
    baseline_group_splits: Mapping[str, str] | None = None,
    preserve_baseline_splits: bool = False,
    np,
) -> dict[str, object]:
    """Use PCG64 to freeze PF-1 group order, split and benchmark prefix."""

    if not 0.0 < dev_fraction < 1.0:
        raise PF1SelectionError("dev_fraction must be strictly between zero and one")
    group_ids = tuple(group_sizes)
    if not group_ids:
        raise PF1SelectionError("no permitted connectivity groups were observed")
    rng = np.random.Generator(np.random.PCG64(seed))
    complete_group_order = tuple(
        group_ids[int(index)] for index in rng.permutation(len(group_ids))
    )
    ineligible_set = set(ineligible_group_ids)
    unknown_ineligible = ineligible_set - set(group_ids)
    if unknown_ineligible:
        raise PF1SelectionError(
            "ineligible connectivity groups are absent from permitted identities: "
            + ", ".join(sorted(unknown_ineligible))
        )
    selection_order = tuple(
        group_id for group_id in complete_group_order if group_id not in ineligible_set
    )
    selected_groups = closest_complete_prefix(
        selection_order, group_sizes, target_members
    )
    selected_member_count = sum(group_sizes[group_id] for group_id in selected_groups)

    split_order = tuple(
        selected_groups[int(index)]
        for index in rng.permutation(len(selected_groups))
    )
    dev_target = max(1, math.floor(dev_fraction * selected_member_count))
    baseline_group_splits = dict(baseline_group_splits or {})
    if preserve_baseline_splits and not baseline_group_splits:
        raise PF1SelectionError(
            "preserve_baseline_splits requires a nonempty baseline membership"
        )
    invalid_baseline_roles = {
        group_id: split
        for group_id, split in baseline_group_splits.items()
        if split not in {"train", "dev"}
    }
    if invalid_baseline_roles:
        raise PF1SelectionError("baseline group splits contain an invalid role")
    missing_baseline_groups = set(baseline_group_splits) - set(selected_groups)
    if preserve_baseline_splits and missing_baseline_groups:
        raise PF1SelectionError(
            "expanded selection does not retain every baseline connectivity group"
        )

    if preserve_baseline_splits:
        baseline_dev = {
            group_id
            for group_id, split in baseline_group_splits.items()
            if split == "dev"
        }
        existing_dev_members = sum(group_sizes[group_id] for group_id in baseline_dev)
        remaining_dev_target = max(0, dev_target - existing_dev_members)
        new_split_order = tuple(
            group_id
            for group_id in split_order
            if group_id not in baseline_group_splits
        )
        if remaining_dev_target > 0 and new_split_order:
            additional_dev = set(
                closest_complete_prefix(
                    new_split_order,
                    group_sizes,
                    remaining_dev_target,
                )
            )
        else:
            additional_dev = set()
        dev_set = baseline_dev | additional_dev
        dev_groups = tuple(
            group_id for group_id in selected_groups if group_id in dev_set
        )
    else:
        dev_groups = closest_complete_prefix(split_order, group_sizes, dev_target)
        dev_set = set(dev_groups)
    train_groups = tuple(
        group_id for group_id in selected_groups if group_id not in dev_set
    )
    if not train_groups or set(train_groups) & dev_set:
        raise PF1SelectionError("train/dev connectivity groups are not a partition")

    benchmark_groups = closest_complete_prefix(
        selected_groups, group_sizes, benchmark_target_members
    )
    return {
        "complete_group_order": complete_group_order,
        "selection_order": selected_groups,
        "train_groups": train_groups,
        "dev_groups": dev_groups,
        "benchmark_groups": benchmark_groups,
        "selected_member_count": selected_member_count,
        "train_member_count": sum(group_sizes[group_id] for group_id in train_groups),
        "dev_member_count": sum(group_sizes[group_id] for group_id in dev_groups),
        "benchmark_member_count": sum(
            group_sizes[group_id] for group_id in benchmark_groups
        ),
        "eligible_group_count": len(selection_order),
        "ineligible_group_count": len(ineligible_set),
        "baseline_split_preservation": {
            "enabled": bool(preserve_baseline_splits),
            "baseline_group_count": len(baseline_group_splits),
            "baseline_train_group_count": sum(
                split == "train" for split in baseline_group_splits.values()
            ),
            "baseline_dev_group_count": sum(
                split == "dev" for split in baseline_group_splits.values()
            ),
            "changed_baseline_group_count": sum(
                (
                    group_id in dev_set
                    and split != "dev"
                )
                or (
                    group_id not in dev_set
                    and split != "train"
                )
                for group_id, split in baseline_group_splits.items()
                if group_id in selected_groups
            ),
        },
    }


def load_baseline_membership(
    path: Path, group_sizes: Mapping[str, int]
) -> list[dict[str, object]]:
    """Load a previous frozen cohort and assert that it is group-complete."""

    rows: list[dict[str, object]] = []
    seen_members: set[str] = set()
    group_counts: Counter[str] = Counter()
    group_splits: dict[str, str] = {}
    for line_number, row in _read_jsonl(path):
        if row.get("schema_version") != MEMBERSHIP_SCHEMA:
            raise PF1SelectionError(
                f"baseline membership line {line_number} has an unexpected schema"
            )
        member_id = row.get("member_id")
        group_id = row.get("connectivity_identity_sha256")
        split = row.get("split")
        member_ordinal(member_id)
        if member_id in seen_members:
            raise PF1SelectionError(f"duplicate baseline member_id: {member_id}")
        if not isinstance(group_id, str) or group_id not in group_sizes:
            raise PF1SelectionError(
                f"baseline membership line {line_number} has an unknown group"
            )
        if split not in {"train", "dev"}:
            raise PF1SelectionError(
                f"baseline membership line {line_number} has an invalid split"
            )
        previous_split = group_splits.setdefault(group_id, split)
        if previous_split != split:
            raise PF1SelectionError(
                f"baseline connectivity group {group_id} crosses train/dev"
            )
        seen_members.add(member_id)
        group_counts[group_id] += 1
        rows.append(row)
    if not rows:
        raise PF1SelectionError("baseline membership is empty")
    partial_groups = {
        group_id: (count, group_sizes[group_id])
        for group_id, count in group_counts.items()
        if count != group_sizes[group_id]
    }
    if partial_groups:
        raise PF1SelectionError(
            "baseline membership contains partial connectivity groups: "
            + ", ".join(sorted(partial_groups))
        )
    return rows


def baseline_group_split_map(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    """Return the already-validated connectivity-group split assignment."""

    mapping: dict[str, str] = {}
    for row in rows:
        group_id = str(row["connectivity_identity_sha256"])
        split = str(row["split"])
        previous = mapping.setdefault(group_id, split)
        if previous != split:
            raise PF1SelectionError("baseline connectivity group crosses train/dev")
    return mapping


def compare_memberships(
    baseline: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    baseline_members = {str(row["member_id"]) for row in baseline}
    candidate_members = {str(row["member_id"]) for row in candidate}
    baseline_groups = {
        str(row["connectivity_identity_sha256"]) for row in baseline
    }
    candidate_groups = {
        str(row["connectivity_identity_sha256"]) for row in candidate
    }
    baseline_splits = baseline_group_split_map(baseline)
    candidate_splits = baseline_group_split_map(candidate)
    common_groups = baseline_groups & candidate_groups
    changed_groups = sorted(
        group_id
        for group_id in common_groups
        if baseline_splits[group_id] != candidate_splits[group_id]
    )
    return {
        "baseline_member_count": len(baseline_members),
        "candidate_member_count": len(candidate_members),
        "retained_member_count": len(baseline_members & candidate_members),
        "removed_member_count": len(baseline_members - candidate_members),
        "added_member_count": len(candidate_members - baseline_members),
        "baseline_group_count": len(baseline_groups),
        "candidate_group_count": len(candidate_groups),
        "retained_group_count": len(baseline_groups & candidate_groups),
        "removed_group_count": len(baseline_groups - candidate_groups),
        "added_group_count": len(candidate_groups - baseline_groups),
        "removed_connectivity_groups": sorted(baseline_groups - candidate_groups),
        "added_connectivity_groups": sorted(candidate_groups - baseline_groups),
        "added_member_ids": sorted(candidate_members - baseline_members),
        "common_group_split_change_count": len(changed_groups),
        "common_groups_with_changed_split": changed_groups,
    }


def collect_selected_members(
    identity_rows: Path,
    permitted_ids: set[str],
    group_plan: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected_groups = tuple(group_plan["selection_order"])  # type: ignore[arg-type]
    benchmark_groups = set(group_plan["benchmark_groups"])  # type: ignore[arg-type]
    dev_groups = set(group_plan["dev_groups"])  # type: ignore[arg-type]
    selected_set = set(selected_groups)
    group_rank = {group_id: index for index, group_id in enumerate(selected_groups)}
    by_group: dict[str, list[tuple[int, str]]] = {
        group_id: [] for group_id in selected_groups
    }
    for _line_number, row in _read_jsonl(identity_rows):
        member_id = row.get("member_id")
        group_id = row.get("connectivity_identity_sha256")
        if member_id in permitted_ids and group_id in selected_set:
            by_group[group_id].append((member_ordinal(member_id), member_id))

    membership: list[dict[str, object]] = []
    benchmark: list[dict[str, object]] = []
    benchmark_index = 0
    for group_id in selected_groups:
        members = sorted(by_group[group_id])
        if not members:
            raise PF1SelectionError(f"selected group {group_id} has no permitted members")
        split = "dev" if group_id in dev_groups else "train"
        for ordinal, member_id in members:
            row = {
                "schema_version": MEMBERSHIP_SCHEMA,
                "selection_index": len(membership),
                "group_order_index": group_rank[group_id],
                "member_id": member_id,
                "sdf_record_index": ordinal,
                "connectivity_identity_sha256": group_id,
                "split": split,
            }
            membership.append(row)
            if group_id in benchmark_groups:
                benchmark.append(
                    {
                        "schema_version": BENCHMARK_SCHEMA,
                        "benchmark_index": benchmark_index,
                        "pf1_selection_index": row["selection_index"],
                        "group_order_index": row["group_order_index"],
                        "member_id": member_id,
                        "sdf_record_index": ordinal,
                        "connectivity_identity_sha256": group_id,
                        "split": split,
                    }
                )
                benchmark_index += 1
    return membership, benchmark


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def run(args: argparse.Namespace) -> dict[str, object]:
    import numpy as np

    profile_id, target_members, preserve_baseline_splits, target_fraction = (
        resolve_selection_profile(args)
    )
    permitted_path = Path(args.permitted_membership).expanduser().resolve()
    identity_path = Path(args.identity_rows).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ineligible_argument = getattr(args, "ineligible_connectivity_groups", None)
    baseline_argument = getattr(args, "baseline_membership", None)
    support_coverage_status = getattr(args, "support_coverage_status", None)
    research_role = getattr(args, "research_role", "mainline")
    ineligible_path = (
        Path(ineligible_argument).expanduser().resolve()
        if ineligible_argument
        else None
    )
    baseline_path = (
        Path(baseline_argument).expanduser().resolve()
        if baseline_argument
        else None
    )
    if not permitted_path.is_file() or not identity_path.is_file():
        raise PF1SelectionError("permitted membership and identity rows must be files")
    if output_dir.exists():
        raise PF1SelectionError("--output-dir must be a new path")
    if ineligible_path is not None and not ineligible_path.is_file():
        raise PF1SelectionError("--ineligible-connectivity-groups must be a file")
    if baseline_path is not None and not baseline_path.is_file():
        raise PF1SelectionError("--baseline-membership must be a file")

    ineligible_groups: dict[str, dict] = {}
    evidence_groups: dict[str, str] = {}
    if ineligible_path is not None:
        ineligible_groups, evidence_groups = load_ineligible_connectivity_groups(
            ineligible_path
        )

    permitted_ids = load_permitted_member_ids(permitted_path)
    if len(permitted_ids) != args.expected_permitted_members:
        raise PF1SelectionError(
            "expected {:,} permitted members, observed {:,}".format(
                args.expected_permitted_members, len(permitted_ids)
            )
        )
    group_sizes, matched = count_permitted_connectivity_groups(
        identity_path,
        permitted_ids,
        expected_evidence_groups=evidence_groups,
    )
    baseline_rows = (
        load_baseline_membership(baseline_path, group_sizes)
        if baseline_path is not None
        else None
    )
    if preserve_baseline_splits and baseline_rows is None:
        raise PF1SelectionError(
            "--preserve-baseline-splits requires --baseline-membership"
        )
    group_plan = freeze_group_plan(
        group_sizes,
        seed=args.seed,
        target_members=target_members,
        dev_fraction=args.dev_fraction,
        benchmark_target_members=args.benchmark_target_members,
        ineligible_group_ids=ineligible_groups,
        baseline_group_splits=(
            baseline_group_split_map(baseline_rows)
            if baseline_rows is not None
            else None
        ),
        preserve_baseline_splits=preserve_baseline_splits,
        np=np,
    )
    membership, benchmark = collect_selected_members(
        identity_path, permitted_ids, group_plan
    )
    if len(membership) != group_plan["selected_member_count"]:
        raise PF1SelectionError("materialized PF-1 membership count differs from group plan")
    if len(benchmark) != group_plan["benchmark_member_count"]:
        raise PF1SelectionError("materialized benchmark count differs from group plan")

    baseline_comparison = (
        compare_memberships(baseline_rows, membership)
        if baseline_rows is not None
        else None
    )
    if preserve_baseline_splits:
        if baseline_comparison is None:
            raise PF1SelectionError("split preservation lacks a baseline comparison")
        if (
            baseline_comparison["removed_member_count"] != 0
            or baseline_comparison["removed_group_count"] != 0
            or baseline_comparison["common_group_split_change_count"] != 0
        ):
            raise PF1SelectionError(
                "expanded membership is not a split-preserving PF-1 superset"
            )
    if support_coverage_status is None:
        support_coverage_status = "candidate" if ineligible_groups else "complete"
    if support_coverage_status not in {"candidate", "complete"}:
        raise PF1SelectionError(
            "support coverage status must be 'candidate' or 'complete'"
        )
    if research_role not in {"mainline", "fallback_candidate"}:
        raise PF1SelectionError(
            "research role must be 'mainline' or 'fallback_candidate'"
        )

    output_dir.mkdir(parents=True)
    membership_name = "membership.jsonl"
    benchmark_name = "benchmark_membership.jsonl"
    _write_jsonl(output_dir / membership_name, membership)
    _write_jsonl(output_dir / benchmark_name, benchmark)
    split_counts = Counter(row["split"] for row in membership)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": support_coverage_status,
        "source": {
            "permitted_membership": str(permitted_path),
            "identity_rows": str(identity_path),
            "permitted_member_count": matched,
            "permitted_connectivity_group_count": len(group_sizes),
        },
        "selection": {
            "profile_id": profile_id or "explicit-programmatic-target",
            "bit_generator": "PCG64",
            "seed": args.seed,
            "target_fraction": target_fraction,
            "target_rule": (
                "explicit_programmatic_target"
                if target_fraction is None
                else "floor({:.2f} * final_v4_permitted_members)".format(
                    target_fraction
                )
            ),
            "target_member_count": target_members,
            "selected_member_count": len(membership),
            "selected_group_count": len(group_plan["selection_order"]),
            "whole_group_prefix_nearest_target": True,
            "baseline_membership_is_strict_subset": (
                baseline_comparison is not None
                and baseline_comparison["removed_member_count"] == 0
                and baseline_comparison["added_member_count"] > 0
            ),
        },
        "split": {
            "policy": (
                "preserve_baseline_group_roles_then_PCG64_new_group_dev_prefix"
                if preserve_baseline_splits
                else "PCG64_permuted_selected_groups_whole_group_dev_prefix"
            ),
            "target_dev_fraction": args.dev_fraction,
            "train_member_count": split_counts["train"],
            "dev_member_count": split_counts["dev"],
            "train_group_count": len(group_plan["train_groups"]),
            "dev_group_count": len(group_plan["dev_groups"]),
            "connectivity_group_intersection": 0,
            "baseline_split_preservation": group_plan[
                "baseline_split_preservation"
            ],
        },
        "benchmark": {
            "target_member_count": args.benchmark_target_members,
            "selected_member_count": len(benchmark),
            "selected_group_count": len(group_plan["benchmark_groups"]),
            "policy": "PF1_selected_group_order_whole_group_prefix_nearest_target",
        },
        "artifacts": {
            "membership": {"path": membership_name, "row_count": len(membership)},
            "benchmark_membership": {
                "path": benchmark_name,
                "row_count": len(benchmark),
            },
        },
        "method_boundary": {
            "production_release_modified": False,
            "molecule_payload_copied": False,
            "selection_unit": "connectivity_group",
            "shared_four_grid_cohort": ["A0", "A1", "M0", "M1"],
        },
        "research_role": {
            "role": research_role,
            "enters_current_mainline": research_role == "mainline",
        },
    }
    if ineligible_groups:
        excluded_member_count = sum(
            group_sizes[group_id] for group_id in ineligible_groups
        )
        manifest["support_domain"] = {
            "policy": "filter_ineligible_complete_groups_from_original_PCG64_order_then_refill",
            "coverage_status": support_coverage_status,
            "ineligible_connectivity_groups": str(ineligible_path),
            "excluded_group_count": len(ineligible_groups),
            "excluded_permitted_member_count": excluded_member_count,
            "evidence_member_count": len(evidence_groups),
            "excluded_groups": [
                ineligible_groups[group_id] for group_id in sorted(ineligible_groups)
            ],
            "newly_added_members_require_chemistry_screen": (
                support_coverage_status == "candidate"
            ),
        }
    if baseline_comparison is not None:
        manifest["baseline_comparison"] = {
            "baseline_membership": str(baseline_path),
            **baseline_comparison,
        }
    with (output_dir / "manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permitted-membership", required=True)
    parser.add_argument("--identity-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--selection-profile",
        choices=tuple(SELECTION_PROFILES),
        default=PF1_PROFILE_ID,
        help=(
            "named final-v4 fidelity contract; PF-10 fixes target=336006, "
            "seed=20260807 and requires the PF-1 baseline membership"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--expected-permitted-members", type=int, default=EXPECTED_PERMITTED_MEMBERS
    )
    parser.add_argument(
        "--target-members",
        type=int,
        help="legacy explicit target; when supplied it must equal the named profile",
    )
    parser.add_argument("--dev-fraction", type=float, default=DEV_FRACTION)
    parser.add_argument(
        "--benchmark-target-members", type=int, default=BENCHMARK_TARGET_MEMBERS
    )
    parser.add_argument(
        "--ineligible-connectivity-groups",
        help="JSONL of whole connectivity groups outside the representation support domain",
    )
    parser.add_argument(
        "--baseline-membership",
        help=(
            "previous group-complete membership; mandatory for PF-10 and used "
            "to preserve every baseline group split"
        ),
    )
    parser.add_argument(
        "--preserve-baseline-splits",
        action="store_true",
        help="retain every baseline connectivity group's train/dev role in an expanded cohort",
    )
    parser.add_argument(
        "--support-coverage-status",
        choices=("candidate", "complete"),
        help="candidate until every newly admitted member passes the chemistry screen",
    )
    parser.add_argument(
        "--research-role",
        choices=("mainline", "fallback_candidate"),
        default="mainline",
        help="mark a support-filtered cohort as a fallback without promoting it to the mainline",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    numeric_counts = [
        args.expected_permitted_members,
        args.benchmark_target_members,
    ]
    if args.target_members is not None:
        numeric_counts.append(args.target_members)
    if min(numeric_counts) <= 0:
        parser.error("member counts must be positive")
    manifest = run(args)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_SCHEMA",
    "INELIGIBLE_GROUP_SCHEMA",
    "MEMBERSHIP_SCHEMA",
    "PF1_PROFILE_ID",
    "PF10_PROFILE_ID",
    "PF10_TARGET_MEMBERS",
    "PF1SelectionError",
    "SELECTION_PROFILES",
    "closest_complete_prefix",
    "collect_selected_members",
    "count_permitted_connectivity_groups",
    "freeze_group_plan",
    "load_permitted_member_ids",
    "load_ineligible_connectivity_groups",
    "load_baseline_membership",
    "compare_memberships",
    "member_ordinal",
    "resolve_selection_profile",
]
