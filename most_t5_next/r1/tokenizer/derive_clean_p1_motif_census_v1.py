#!/usr/bin/env python3
"""Derive the final-clean P1 exact and attachment-slot motif censuses.

The completed PCQM production release already contains an exact motif census.
Cleaning removes only a small downstream-protected membership.  This tool
therefore decodes only excluded records, subtracts their motif occurrences
from the global census, and then aggregates molecule-local anchor IDs into a
slot-preserving template.  It never decodes the millions of permitted records.

For example, ``O=C(<3*>)N<7*>`` becomes ``O=C(<*>)N<*>``.  Anchor IDs are
removed because they are molecule-local edge labels; attachment positions and
their surrounding bond syntax remain part of motif identity.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

from most_t5_next.r1.adapter.sidecar_v2_codec import decode_record


RELEASE_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-full-release/v2"
CLEAN_SCHEMA = "most-t5-r1/clean-pretrain-membership-manifest/v1"
SUMMARY_SCHEMA = "most-t5-r1/clean-p1-motif-census-summary/v1"
MEMBER_RE = re.compile(r"^ogb_pcqm4mv2_train_row_index:([0-9]+)$")
ANCHOR_RE = re.compile(r"<([0-9]+)\*>")
ANCHOR_LIKE_RE = re.compile(r"<[^>]*\*>")
HEX64 = frozenset("0123456789abcdef")


class CleanMotifCensusError(ValueError):
    """Raised when the source release and clean membership do not close."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def regular_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise CleanMotifCensusError(label + " must be a regular non-symlink file")
    return path


def load_json(path: Path, label: str) -> Mapping[str, object]:
    regular_file(path, label)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CleanMotifCensusError(label + " must contain one JSON object")
    return value


def iter_jsonl(path: Path, label: str) -> Iterator[Mapping[str, object]]:
    regular_file(path, label)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CleanMotifCensusError(
                    "{} line {} is invalid JSON".format(label, line_number)
                ) from exc
            if not isinstance(row, dict):
                raise CleanMotifCensusError(
                    "{} line {} is not an object".format(label, line_number)
                )
            yield row


def project_slot_template(fragment: str) -> Tuple[str, int]:
    if not isinstance(fragment, str) or not fragment:
        raise CleanMotifCensusError("motif fragment must be a non-empty string")
    matches = list(ANCHOR_RE.finditer(fragment))
    if ANCHOR_LIKE_RE.findall(fragment) != [match.group(0) for match in matches]:
        raise CleanMotifCensusError("malformed anchor-like substring in motif fragment")
    for match in matches:
        decimal = match.group(1)
        if len(decimal) > 1 and decimal.startswith("0"):
            raise CleanMotifCensusError("anchor IDs must use canonical decimal")
    template = ANCHOR_RE.sub("<*>", fragment)
    if not template or any(character in template for character in "\x00\t\r\n"):
        raise CleanMotifCensusError("slot template contains a forbidden character")
    return template, len(matches)


def load_global_exact_census(
    path: Path, expected_unique: int, expected_occurrences: int
) -> Dict[str, Tuple[str, int]]:
    result: Dict[str, Tuple[str, int]] = {}
    occurrence_count = 0
    previous = None
    for row in iter_jsonl(path, "global motif census"):
        if set(row) != {"motif_lexeme_sha256", "motif_fragment", "count"}:
            raise CleanMotifCensusError("global motif census row fields differ")
        digest = row["motif_lexeme_sha256"]
        fragment = row["motif_fragment"]
        count = row["count"]
        if not is_sha256(digest) or not isinstance(fragment, str):
            raise CleanMotifCensusError("global motif census identity is malformed")
        if digest != sha256_text(fragment):
            raise CleanMotifCensusError("motif fragment does not match its identity")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise CleanMotifCensusError("global motif count must be positive")
        if previous is not None and digest <= previous:
            raise CleanMotifCensusError("global motif census is not strictly sorted")
        previous = digest
        result[digest] = (fragment, count)
        occurrence_count += count
    if len(result) != expected_unique or occurrence_count != expected_occurrences:
        raise CleanMotifCensusError("global motif census counts disagree with release")
    return result


def validate_shards(release_root: Path, release: Mapping[str, object]) -> List[Mapping[str, int]]:
    raw_shards = release.get("shards")
    configuration = release.get("configuration")
    if not isinstance(raw_shards, list) or not isinstance(configuration, dict):
        raise CleanMotifCensusError("release shard configuration is missing")
    expected_count = configuration.get("shard_count")
    shard_size = configuration.get("shard_size")
    selected_range = configuration.get("selected_ordinal_range")
    if (
        not isinstance(expected_count, int)
        or expected_count != len(raw_shards)
        or not isinstance(shard_size, int)
        or shard_size <= 0
        or not isinstance(selected_range, list)
        or len(selected_range) != 2
        or selected_range[0] != 0
    ):
        raise CleanMotifCensusError("release shard dimensions are malformed")
    shards: List[Mapping[str, int]] = []
    expected_start = 0
    for index, row in enumerate(raw_shards):
        if not isinstance(row, dict):
            raise CleanMotifCensusError("release shard row is malformed")
        start = row.get("range_start")
        end = row.get("range_end")
        shard_index = row.get("shard_index")
        if shard_index != index or start != expected_start or not isinstance(end, int) or end <= start:
            raise CleanMotifCensusError("release shard ranges have a gap or overlap")
        if end - start > shard_size or (index + 1 < len(raw_shards) and end - start != shard_size):
            raise CleanMotifCensusError("release shard range differs from shard size")
        shard_dir = release_root / "shard-{:06d}".format(index)
        if not shard_dir.is_dir() or shard_dir.is_symlink():
            raise CleanMotifCensusError("release shard directory is absent")
        shards.append({"shard_index": index, "range_start": start, "range_end": end})
        expected_start = end
    if expected_start != selected_range[1]:
        raise CleanMotifCensusError("release shards do not cover the selected range")
    return shards


def member_ordinal(member_id: object, upper_bound: int) -> int:
    match = MEMBER_RE.fullmatch(member_id) if isinstance(member_id, str) else None
    if match is None:
        raise CleanMotifCensusError("excluded member ID is outside the PCQM namespace")
    ordinal = int(match.group(1))
    if ordinal < 0 or ordinal >= upper_bound:
        raise CleanMotifCensusError("excluded member ordinal is outside the release")
    return ordinal


def read_excluded_members(
    clean_root: Path, clean: Mapping[str, object], upper_bound: int
) -> List[Tuple[str, int]]:
    artifacts = clean.get("artifacts")
    counts = clean.get("counts")
    if not isinstance(artifacts, dict) or not isinstance(counts, dict):
        raise CleanMotifCensusError("clean membership artifact declarations are missing")
    declaration = artifacts.get("excluded_member_ledger")
    if not isinstance(declaration, dict) or declaration.get("path") != "excluded_member_ledger.jsonl":
        raise CleanMotifCensusError("excluded ledger declaration is malformed")
    expected = counts.get("excluded_member_count")
    if not isinstance(expected, int) or expected < 0 or declaration.get("row_count") != expected:
        raise CleanMotifCensusError("excluded ledger count declaration is malformed")
    members: List[Tuple[str, int]] = []
    seen = set()
    for row in iter_jsonl(clean_root / str(declaration["path"]), "excluded member ledger"):
        member_id = row.get("member_id")
        ordinal = member_ordinal(member_id, upper_bound)
        if member_id in seen:
            raise CleanMotifCensusError("excluded member ledger contains a duplicate")
        seen.add(member_id)
        members.append((str(member_id), ordinal))
    if len(members) != expected:
        raise CleanMotifCensusError("excluded ledger rows disagree with clean manifest")
    return members


def decode_excluded_motifs(
    release_root: Path,
    shards: Sequence[Mapping[str, int]],
    members: Sequence[Tuple[str, int]],
    known_motifs: Mapping[str, Tuple[str, int]],
) -> Tuple[Counter, Mapping[str, int]]:
    try:
        import lmdb
        import numpy as np
    except ImportError as exc:
        raise CleanMotifCensusError("LMDB and NumPy are required to decode excluded records") from exc

    shard_size = int(shards[0]["range_end"]) - int(shards[0]["range_start"])
    grouped: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for member_id, ordinal in members:
        shard_index = ordinal // shard_size
        if shard_index >= len(shards):
            raise CleanMotifCensusError("excluded ordinal maps beyond the release shards")
        shard = shards[shard_index]
        if not (int(shard["range_start"]) <= ordinal < int(shard["range_end"])):
            raise CleanMotifCensusError("excluded ordinal does not map to its declared shard")
        grouped[shard_index].append((member_id, ordinal))

    removed = Counter()
    decoded_records = 0
    affected_shards = 0
    for shard_index in sorted(grouped):
        environment = lmdb.open(
            str(release_root / "shard-{:06d}".format(shard_index) / "geometry_records.lmdb"),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=8,
        )
        try:
            with environment.begin() as transaction:
                for member_id, ordinal in sorted(grouped[shard_index], key=lambda item: item[1]):
                    payload = transaction.get("{:09d}".format(ordinal).encode("ascii"))
                    if payload is None:
                        raise CleanMotifCensusError("excluded member has no production payload")
                    record, _ = decode_record(np, payload)
                    if record.get("member", {}).get("member_id") != member_id:
                        raise CleanMotifCensusError("decoded payload member identity differs")
                    topology = record.get("topology")
                    if not isinstance(topology, dict):
                        raise CleanMotifCensusError("decoded record topology is missing")
                    digests = topology.get("motif_lexeme_sha256")
                    if not isinstance(digests, list) or topology.get("motif_count") != len(digests):
                        raise CleanMotifCensusError("decoded motif count is malformed")
                    for digest in digests:
                        if digest not in known_motifs:
                            raise CleanMotifCensusError("decoded motif is absent from global census")
                        removed[digest] += 1
                    decoded_records += 1
        finally:
            environment.close()
        affected_shards += 1
    return removed, {
        "decoded_excluded_records": decoded_records,
        "affected_shards": affected_shards,
        "removed_motif_occurrences": sum(removed.values()),
        "removed_unique_exact_motifs": len(removed),
    }


def derive_rows(
    global_census: Mapping[str, Tuple[str, int]], removed: Mapping[str, int]
) -> Tuple[List[Mapping[str, object]], List[Mapping[str, object]], List[Mapping[str, object]]]:
    exact_rows: List[Mapping[str, object]] = []
    removed_rows: List[Mapping[str, object]] = []
    slots: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(
        lambda: {"count": 0, "exact_lexeme_count": 0}
    )
    for digest in sorted(global_census):
        fragment, original_count = global_census[digest]
        removed_count = int(removed.get(digest, 0))
        if removed_count < 0 or removed_count > original_count:
            raise CleanMotifCensusError("motif subtraction would produce a negative count")
        clean_count = original_count - removed_count
        if removed_count:
            removed_rows.append(
                {
                    "motif_lexeme_sha256": digest,
                    "motif_fragment": fragment,
                    "original_count": original_count,
                    "removed_count": removed_count,
                    "clean_count": clean_count,
                }
            )
        if not clean_count:
            continue
        exact_rows.append(
            {
                "motif_lexeme_sha256": digest,
                "motif_fragment": fragment,
                "count": clean_count,
            }
        )
        template, slot_count = project_slot_template(fragment)
        aggregate = slots[(template, slot_count)]
        aggregate["count"] += clean_count
        aggregate["exact_lexeme_count"] += 1

    slot_rows = []
    for (template, slot_count), counts in slots.items():
        slot_rows.append(
            {
                "slot_identity_sha256": sha256_text(template),
                "slot_template": template,
                "slot_count": slot_count,
                "exact_lexeme_count": counts["exact_lexeme_count"],
                "count": counts["count"],
            }
        )
    slot_rows.sort(key=lambda row: str(row["slot_identity_sha256"]))
    if len({row["slot_identity_sha256"] for row in slot_rows}) != len(slot_rows):
        raise CleanMotifCensusError("slot-template identity collision was observed")
    if sum(int(row["count"]) for row in exact_rows) != sum(
        int(row["count"]) for row in slot_rows
    ):
        raise CleanMotifCensusError("exact and slot-template occurrence totals differ")
    return exact_rows, slot_rows, removed_rows


def write_jsonl_new(path: Path, rows: Iterable[Mapping[str, object]]) -> Mapping[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    with path.open("xb") as handle:
        for row in rows:
            raw = canonical_json_bytes(row) + b"\n"
            handle.write(raw)
            digest.update(raw)
            byte_count += len(raw)
            row_count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return {"bytes": byte_count, "row_count": row_count, "sha256": digest.hexdigest()}


def write_json_new(path: Path, value: Mapping[str, object]) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def derive_clean_census(release_root: Path, clean_manifest_path: Path, output_dir: Path):
    release_root = release_root.resolve()
    clean_manifest_path = clean_manifest_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise CleanMotifCensusError("output directory already exists")

    release = load_json(release_root / "full_release_manifest.json", "production release")
    clean = load_json(clean_manifest_path, "clean membership manifest")
    if release.get("schema_version") != RELEASE_SCHEMA or release.get("release_status") != "complete":
        raise CleanMotifCensusError("PCQM production release is not complete v2")
    if clean.get("schema_version") != CLEAN_SCHEMA or clean.get("status") != "complete":
        raise CleanMotifCensusError("clean membership is not complete v1")
    if clean.get("pretrain_source", {}).get("release_id") != release.get("release_id"):
        raise CleanMotifCensusError("clean membership and PCQM release IDs differ")
    release_counts = release.get("counts")
    clean_counts = clean.get("counts")
    if not isinstance(release_counts, dict) or not isinstance(clean_counts, dict):
        raise CleanMotifCensusError("release or clean counts are missing")
    if clean_counts.get("pretrain_member_count") != release_counts.get("admitted_record_count"):
        raise CleanMotifCensusError("clean membership is not derived from all admitted records")

    artifact = release.get("global_motif_census")
    if not isinstance(artifact, dict) or artifact.get("relative_path") != "motif_census.jsonl":
        raise CleanMotifCensusError("global motif census declaration is malformed")
    global_census = load_global_exact_census(
        release_root / "motif_census.jsonl",
        int(release_counts["unique_motif_count"]),
        int(release_counts["motif_occurrence_count"]),
    )
    shards = validate_shards(release_root, release)
    upper_bound = int(release["configuration"]["selected_ordinal_range"][1])
    members = read_excluded_members(clean_manifest_path.parent, clean, upper_bound)
    removed, decode_report = decode_excluded_motifs(
        release_root, shards, members, global_census
    )
    if decode_report["decoded_excluded_records"] != clean_counts.get("excluded_member_count"):
        raise CleanMotifCensusError("not every excluded member was decoded exactly once")
    exact_rows, slot_rows, removed_rows = derive_rows(global_census, removed)

    original_occurrences = int(release_counts["motif_occurrence_count"])
    clean_occurrences = sum(int(row["count"]) for row in exact_rows)
    if clean_occurrences + int(decode_report["removed_motif_occurrences"]) != original_occurrences:
        raise CleanMotifCensusError("motif occurrence subtraction does not close")

    output_dir.mkdir(parents=True, exist_ok=False)
    exact_artifact = write_jsonl_new(output_dir / "exact_motif_census.jsonl", exact_rows)
    slot_artifact = write_jsonl_new(output_dir / "slot_identity_census.jsonl", slot_rows)
    removed_artifact = write_jsonl_new(
        output_dir / "excluded_motif_census.jsonl", removed_rows
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "release_id": release["release_id"],
        "clean_derivation_id": clean.get("derivation_id"),
        "method": {
            "global_census_reused": True,
            "permitted_records_decoded": 0,
            "excluded_records_decoded": decode_report["decoded_excluded_records"],
            "anchor_projection": "replace_each_canonical_molecule_local_<N*>_with_<*>_without_deleting_slot",
            "identity_note": "slot_template_preserves_attachment_position_and_surrounding_bond_syntax",
        },
        "counts": {
            "pretrain_members": clean_counts["pretrain_member_count"],
            "permitted_members": clean_counts["permitted_member_count"],
            "excluded_members": clean_counts["excluded_member_count"],
            "global_exact_unique_motifs": len(global_census),
            "clean_exact_unique_motifs": len(exact_rows),
            "clean_slot_unique_motifs": len(slot_rows),
            "global_motif_occurrences": original_occurrences,
            "removed_motif_occurrences": decode_report["removed_motif_occurrences"],
            "clean_motif_occurrences": clean_occurrences,
            "removed_unique_exact_motifs": decode_report["removed_unique_exact_motifs"],
            "affected_shards": decode_report["affected_shards"],
        },
        "proofs": {
            "all_excluded_records_decoded_once": True,
            "all_decoded_motifs_present_in_global_census": True,
            "no_negative_exact_count": True,
            "global_equals_removed_plus_clean_occurrences": True,
            "exact_and_slot_occurrence_totals_equal": True,
            "molecule_local_anchor_ids_not_used_as_macro_identity": True,
            "attachment_slots_not_deleted": True,
        },
        "artifacts": {
            "exact_motif_census": {"path": "exact_motif_census.jsonl", **exact_artifact},
            "slot_identity_census": {"path": "slot_identity_census.jsonl", **slot_artifact},
            "excluded_motif_census": {"path": "excluded_motif_census.jsonl", **removed_artifact},
        },
        "digest_role": "motif_identity_and_artifact_provenance_not_dataset_admission_by_file_hash",
    }
    write_json_new(output_dir / "clean_census_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--clean-membership", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    arguments = parse_args(argv)
    summary = derive_clean_census(
        arguments.release_root, arguments.clean_membership, arguments.output_dir
    )
    print(json.dumps({"status": summary["status"], **summary["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CleanMotifCensusError, OSError, RuntimeError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
