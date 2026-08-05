#!/usr/bin/env python3
"""Extract a candidate-only P2 motif census from a hash-locked legacy LMDB.

The input contains trusted legacy pickle values.  This extractor therefore
verifies the complete source file before the first ``pickle.loads`` call,
requires an explicit command-line acknowledgement, and rehashes the file after
the read transaction.  It never imports RDKit, reruns a linearizer, computes
E3FP, changes a tokenizer, or admits training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import pickle
import platform
import re
import sys
from typing import Any, Iterable, Optional


CONTRACT_SCHEMA = "most-t5-r1/p2-phase2-ready-motif-census-contract/v1"
SOURCE_LOCK_SCHEMA = "most-t5-r1/p2-phase2-ready-source-lock/v1"
RECEIPT_SCHEMA = "most-t5-r1/p2-phase2-ready-motif-census-receipt/v1"
MEMBERSHIP_SCHEMA = "most-t5-r1/p2-phase2-ready-membership/v1"
REJECT_SCHEMA = "most-t5-r1/p2-phase2-ready-reject/v1"
PROJECTION_SCHEMA = "most-t5-r1/p2-phase2-ready-record-projection/v1"
ANCHOR_SUMMARY_SCHEMA = "most-t5-r1/p2-legacy-anchor-summary/v1"
PROJECTION_SPEC_ID = "most-t5-r1/motif-lexeme-projection/v1"
ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_TRUSTED_PICKLE_CAN_EXECUTE_CODE"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PAYLOAD_KEY_RE = re.compile(rb"^[1-9][0-9]*$")
ANCHOR_RE = re.compile(r"<([0-9]+)\*>")

SOURCE_LOCK_FIELDS = {
    "schema_version",
    "source_role",
    "source_format",
    "source_sha256",
    "source_bytes",
    "expected_payload_entry_count",
    "expected_metadata_keys",
    "expected_payload_fields",
    "identity_namespace",
    "membership_status",
    "source_copy_manifest_sha256",
    "pickle_trust_basis_sha256",
    "motif_sequence_producer_status",
    "motif_sequence_producer_sha256",
    "legacy_linearization_spec_sha256",
}
EXPECTED_PAYLOAD_FIELDS = {
    "atom_to_motif_map",
    "atoms",
    "cid",
    "coordinates",
    "description",
    "e3fp",
    "enriched_description",
    "motif_seq",
    "raw_smiles",
    "smiles",
}
REJECT_CODES = {
    "PICKLE_DESERIALIZATION_FAILED",
    "PAYLOAD_NOT_DICT",
    "FIELD_SET_MISMATCH",
    "KEY_CID_MISMATCH",
    "MOTIF_SEQ_TYPE_INVALID",
    "MOTIF_SEQUENCE_PARSE_FAILED",
    "MOTIF_MAPPING_SCHEMA_MISMATCH",
    "MOTIF_MAPPING_CARDINALITY_MISMATCH",
    "MOTIF_FRAGMENT_INVALID",
    "ANCHOR_GRAMMAR_INVALID",
}
DETERMINISTIC_ARTIFACTS = (
    "membership.jsonl",
    "reject_ledger.jsonl",
    "record_projection.jsonl",
    "motif_census.jsonl",
    "pure_motif_census.jsonl",
    "anchor_summary.json",
)


class CensusError(RuntimeError):
    """Fail-closed contract or extraction error."""


class RecordReject(ValueError):
    """A closed, per-record rejection that may be recorded and continued."""

    def __init__(self, code: str) -> None:
        if code not in REJECT_CODES:
            raise CensusError("attempted to emit unknown reject code")
        super().__init__(code)
        self.code = code


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise CensusError("{} must be a lowercase SHA-256".format(label))
    return value


def strict_fields(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise CensusError(
            "{} fields are not closed; missing={}, extra={}".format(
                label, sorted(expected - set(value)), sorted(set(value) - expected)
            )
        )


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CensusError("{} must be a regular non-symlink file: {}".format(label, path))
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CensusError("{} must contain a JSON object".format(label))
    return value


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_row(handle: Any, value: dict[str, Any]) -> None:
    handle.write(canonical_json_bytes(value) + b"\n")


def observe_artifact(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CensusError("artifact is absent or not a regular file: {}".format(path))
    return {
        "relative_path": relative_path,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def validate_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = load_json(path, "P2 census contract")
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise CensusError("P2 census contract schema mismatch")
    if contract.get("legacy_pickle_boundary", {}).get("acknowledgement_literal") != ACKNOWLEDGEMENT:
        raise CensusError("contract pickle acknowledgement differs from extractor")
    if contract.get("projection", {}).get("spec_id") != PROJECTION_SPEC_ID:
        raise CensusError("contract projection spec differs from extractor")
    if set(contract.get("record_validation", {}).get("closed_reject_codes", [])) != REJECT_CODES:
        raise CensusError("contract reject-code set differs from extractor")
    if tuple(contract.get("artifacts", {}).get("required", [])) != DETERMINISTIC_ARTIFACTS + ("derivation_receipt.json",):
        raise CensusError("contract artifact list differs from extractor")
    return contract, observe_artifact(path, path.name)


def validate_source_lock(path: Path, source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = load_json(path, "P2 source lock")
    strict_fields(lock, SOURCE_LOCK_FIELDS, "P2 source lock")
    if lock["schema_version"] != SOURCE_LOCK_SCHEMA:
        raise CensusError("P2 source lock schema mismatch")
    if lock["source_role"] != "phase2_pubchem_ready_lmdb":
        raise CensusError("P2 source role mismatch")
    if lock["source_format"] != "lmdb_single_file_pickle_values":
        raise CensusError("P2 source format mismatch")
    if lock["identity_namespace"] != "pubchem_cid":
        raise CensusError("P2 source identity namespace mismatch")
    if lock["membership_status"] != "candidate_geometry_ready":
        raise CensusError("P2 source membership status is not candidate_geometry_ready")
    require_sha256(lock["source_sha256"], "source SHA-256")
    for field in ("source_copy_manifest_sha256", "pickle_trust_basis_sha256", "legacy_linearization_spec_sha256"):
        require_sha256(lock[field], field)
    producer_status = lock["motif_sequence_producer_status"]
    producer_sha = lock["motif_sequence_producer_sha256"]
    if producer_status == "hash_locked":
        require_sha256(producer_sha, "motif-sequence producer SHA-256")
    elif producer_status == "unknown_legacy_producer":
        if producer_sha is not None:
            raise CensusError("unknown legacy producer must have null producer SHA-256")
    else:
        raise CensusError("motif-sequence producer status is invalid")
    if not isinstance(lock["source_bytes"], int) or isinstance(lock["source_bytes"], bool) or lock["source_bytes"] <= 0:
        raise CensusError("source_bytes must be a positive integer")
    if not isinstance(lock["expected_payload_entry_count"], int) or isinstance(lock["expected_payload_entry_count"], bool) or lock["expected_payload_entry_count"] <= 0:
        raise CensusError("expected_payload_entry_count must be a positive integer")
    metadata = lock["expected_metadata_keys"]
    if not isinstance(metadata, list) or any(not isinstance(item, str) or not item for item in metadata):
        raise CensusError("expected_metadata_keys must be a list of non-empty UTF-8 strings")
    if len(metadata) != len(set(metadata)):
        raise CensusError("expected_metadata_keys contains duplicates")
    if any(PAYLOAD_KEY_RE.fullmatch(item.encode("utf-8")) is not None for item in metadata):
        raise CensusError("metadata keys must not overlap canonical payload keys")
    if lock["expected_payload_fields"] != sorted(EXPECTED_PAYLOAD_FIELDS):
        raise CensusError("source lock payload-field list differs from the closed schema")
    if not source.is_file() or source.is_symlink():
        raise CensusError("source LMDB must be one regular non-symlink file")
    observed_bytes = int(source.stat().st_size)
    observed_sha = sha256_file(source)
    if observed_bytes != lock["source_bytes"] or observed_sha != lock["source_sha256"]:
        raise CensusError("source LMDB byte count or SHA-256 differs from source lock")
    return lock, observe_artifact(path, path.name)


def parse_motif_sequence(value: str) -> dict[str, Any]:
    """Parse the stored envelope using bracket depth, preserving components."""

    if not isinstance(value, str):
        raise RecordReject("MOTIF_SEQ_TYPE_INVALID")
    if "\x00" in value:
        raise RecordReject("MOTIF_FRAGMENT_INVALID")
    left = 0
    while left < len(value) and value[left].isspace():
        left += 1
    right = len(value)
    while right > left and value[right - 1].isspace():
        right -= 1
    trimmed = value[left:right]
    if not trimmed.startswith("<bom>") or not trimmed.endswith("<eom>"):
        raise RecordReject("MOTIF_SEQUENCE_PARSE_FAILED")
    body = trimmed[len("<bom>") : -len("<eom>")]
    fragments: list[str] = []
    component_ranges: list[list[int]] = []
    component_start = 0
    top_level_whitespace = left + (len(value) - right)
    index = 0
    separator_pending = False
    while index < len(body):
        if body[index].isspace():
            top_level_whitespace += 1
            index += 1
            continue
        if body[index] != "[":
            raise RecordReject("MOTIF_SEQUENCE_PARSE_FAILED")
        start = index
        depth = 0
        while index < len(body):
            char = body[index]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth < 0:
                    raise RecordReject("MOTIF_SEQUENCE_PARSE_FAILED")
                if depth == 0:
                    index += 1
                    break
            index += 1
        if depth != 0:
            raise RecordReject("MOTIF_SEQUENCE_PARSE_FAILED")
        fragment = body[start + 1 : index - 1]
        if fragment == ".":
            if not fragments or separator_pending or component_start == len(fragments):
                raise RecordReject("MOTIF_SEQUENCE_PARSE_FAILED")
            component_ranges.append([component_start, len(fragments)])
            component_start = len(fragments)
            separator_pending = True
            continue
        separator_pending = False
        if not fragment:
            raise RecordReject("MOTIF_FRAGMENT_INVALID")
        if any(char in fragment for char in ("\x00", "\t", "\r", "\n")):
            raise RecordReject("MOTIF_FRAGMENT_INVALID")
        fragments.append(fragment)
    if not fragments or separator_pending or component_start == len(fragments):
        raise RecordReject("MOTIF_SEQUENCE_PARSE_FAILED")
    component_ranges.append([component_start, len(fragments)])
    return {
        "fragments": fragments,
        "component_fragment_ranges": component_ranges,
        "top_level_whitespace_count": top_level_whitespace,
    }


def project_fragment(fragment: str) -> tuple[str, list[int]]:
    anchor_ids: list[int] = []
    for match in ANCHOR_RE.finditer(fragment):
        text = match.group(1)
        try:
            anchor_id = int(text)
        except ValueError:
            raise RecordReject("ANCHOR_GRAMMAR_INVALID")
        if str(anchor_id) != text:
            raise RecordReject("ANCHOR_GRAMMAR_INVALID")
        anchor_ids.append(anchor_id)
    core = ANCHOR_RE.sub("", fragment)
    if "<" in core or ">" in core:
        raise RecordReject("ANCHOR_GRAMMAR_INVALID")
    if not core:
        raise RecordReject("MOTIF_FRAGMENT_INVALID")
    return "[{}]".format(core), anchor_ids


def validate_mapping(value: Any) -> list[list[int]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise RecordReject("MOTIF_MAPPING_SCHEMA_MISMATCH")
    normalized: list[list[int]] = []
    for group in value:
        if not isinstance(group, (list, tuple)) or not group:
            raise RecordReject("MOTIF_MAPPING_SCHEMA_MISMATCH")
        row: list[int] = []
        for atom_index in group:
            if not isinstance(atom_index, Integral) or isinstance(atom_index, bool) or atom_index < 0:
                raise RecordReject("MOTIF_MAPPING_SCHEMA_MISMATCH")
            row.append(int(atom_index))
        normalized.append(row)
    return normalized


def _stereo_counts(fragments: Iterable[str]) -> dict[str, int]:
    rows = list(fragments)
    return {
        "fragment_count_with_at": sum("@" in item for item in rows),
        "fragment_count_with_forward_slash": sum("/" in item for item in rows),
        "fragment_count_with_backslash": sum("\\" in item for item in rows),
        "at_marker_count": sum(item.count("@") for item in rows),
        "forward_slash_marker_count": sum(item.count("/") for item in rows),
        "backslash_marker_count": sum(item.count("\\") for item in rows),
    }


def process_payload(source_key: str, value_bytes: bytes) -> dict[str, Any]:
    try:
        payload = pickle.loads(value_bytes)
    except Exception:
        raise RecordReject("PICKLE_DESERIALIZATION_FAILED")
    if not isinstance(payload, dict):
        raise RecordReject("PAYLOAD_NOT_DICT")
    if set(payload) != EXPECTED_PAYLOAD_FIELDS:
        raise RecordReject("FIELD_SET_MISMATCH")
    cid = payload["cid"]
    if isinstance(cid, Integral) and not isinstance(cid, bool):
        normalized_cid = int(cid)
    elif isinstance(cid, str) and cid.isascii() and cid.isdigit() and str(int(cid)) == cid:
        normalized_cid = int(cid)
    else:
        raise RecordReject("KEY_CID_MISMATCH")
    if normalized_cid <= 0 or str(normalized_cid) != source_key:
        raise RecordReject("KEY_CID_MISMATCH")
    parsed = parse_motif_sequence(payload["motif_seq"])
    mapping = validate_mapping(payload["atom_to_motif_map"])
    fragments = parsed["fragments"]
    if len(fragments) != len(mapping):
        raise RecordReject("MOTIF_MAPPING_CARDINALITY_MISMATCH")
    exact_digests: list[str] = []
    pure_tokens: list[str] = []
    fragment_anchor_ids: list[list[int]] = []
    for fragment in fragments:
        token, anchor_ids = project_fragment(fragment)
        exact_digests.append(sha256_bytes(fragment.encode("utf-8")))
        pure_tokens.append(token)
        fragment_anchor_ids.append(anchor_ids)
    component_anchor_multiplicities: list[dict[str, int]] = []
    p1_pair_rule_pass = True
    max_anchor: Optional[int] = None
    for start, stop in parsed["component_fragment_ranges"]:
        counts: Counter[int] = Counter()
        for anchor_ids in fragment_anchor_ids[start:stop]:
            counts.update(anchor_ids)
        component_anchor_multiplicities.append({str(key): int(counts[key]) for key in sorted(counts)})
        if any(value != 2 for value in counts.values()):
            p1_pair_rule_pass = False
        if counts:
            observed_max = max(counts)
            max_anchor = observed_max if max_anchor is None else max(max_anchor, observed_max)
    return {
        "cid": normalized_cid,
        "motif_seq_sha256": sha256_bytes(payload["motif_seq"].encode("utf-8")),
        "motif_mapping_sha256": sha256_json(mapping),
        "fragments": fragments,
        "exact_digests": exact_digests,
        "pure_tokens": pure_tokens,
        "component_fragment_ranges": parsed["component_fragment_ranges"],
        "top_level_whitespace_count": parsed["top_level_whitespace_count"],
        "stereo": _stereo_counts(fragments),
        "component_anchor_multiplicities": component_anchor_multiplicities,
        "p1_pair_rule_pass": p1_pair_rule_pass,
        "max_anchor_id": max_anchor,
    }


def extract_release(
    source_path: Path,
    source_lock_path: Path,
    contract_path: Path,
    output_dir: Path,
    legacy_pickle_acknowledgement: str,
    extractor_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run one immutable extraction.  ``output_dir`` must not exist."""

    if legacy_pickle_acknowledgement != ACKNOWLEDGEMENT:
        raise CensusError("exact legacy-pickle acknowledgement literal is required")
    source_path = source_path.expanduser().resolve()
    source_lock_path = source_lock_path.expanduser().resolve()
    contract_path = contract_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    extractor_path = Path(__file__).resolve() if extractor_path is None else extractor_path.expanduser().resolve()
    contract, contract_artifact = validate_contract(contract_path)
    del contract
    source_lock, source_lock_artifact = validate_source_lock(source_lock_path, source_path)
    if not extractor_path.is_file() or extractor_path.is_symlink():
        raise CensusError("extractor path must be a regular non-symlink file")
    extractor_artifact = observe_artifact(extractor_path, extractor_path.name)
    if output_dir.exists():
        raise FileExistsError("--output-dir must be new")
    output_dir.mkdir(parents=True, exist_ok=False)

    try:
        import lmdb
    except ImportError as exc:
        raise CensusError("the read-only extractor requires the lmdb package") from exc

    expected_metadata_bytes = {item.encode("utf-8") for item in source_lock["expected_metadata_keys"]}
    observed_metadata: set[bytes] = set()
    membership_count = 0
    admitted_count = 0
    rejected_count = 0
    previous_payload_key: Optional[bytes] = None
    exact_counts: Counter[str] = Counter()
    exact_fragments: dict[str, str] = {}
    pure_counts: Counter[str] = Counter()
    pure_exact_digests: dict[str, set[str]] = defaultdict(set)
    pure_stereo_exact_digests: dict[str, set[str]] = defaultdict(set)
    pure_stereo_occurrences: Counter[str] = Counter()
    pure_digest_tokens: dict[str, str] = {}
    anchor_multiplicity_histogram: Counter[int] = Counter()
    records_pair_pass = 0
    records_pair_fail = 0
    component_count = 0
    component_with_anchor_count = 0
    max_anchor_id: Optional[int] = None
    records_with_stereo_at = 0
    fragments_with_stereo_at = 0
    top_level_whitespace_record_count = 0

    membership_path = output_dir / "membership.jsonl"
    reject_path = output_dir / "reject_ledger.jsonl"
    projection_path = output_dir / "record_projection.jsonl"
    with membership_path.open("xb") as membership_handle, reject_path.open("xb") as reject_handle, projection_path.open("xb") as projection_handle:
        env = lmdb.open(
            str(source_path), subdir=False, readonly=True, lock=False,
            readahead=False, meminit=False, max_readers=1,
        )
        try:
            with env.begin(write=False) as txn:
                observed_entries = int(txn.stat()["entries"])
                expected_entries = source_lock["expected_payload_entry_count"] + len(expected_metadata_bytes)
                if observed_entries != expected_entries:
                    raise CensusError("LMDB entry count differs from source lock")
                cursor = txn.cursor()
                for key_bytes, value_bytes in cursor:
                    if key_bytes in expected_metadata_bytes:
                        observed_metadata.add(bytes(key_bytes))
                        continue
                    if PAYLOAD_KEY_RE.fullmatch(key_bytes) is None:
                        raise CensusError("unrecognized LMDB metadata/non-payload key: {!r}".format(bytes(key_bytes)))
                    if previous_payload_key is not None and key_bytes <= previous_payload_key:
                        raise CensusError("LMDB payload keys are not strictly byte-ascending")
                    previous_payload_key = bytes(key_bytes)
                    source_key = key_bytes.decode("ascii")
                    value_sha = sha256_bytes(bytes(value_bytes))
                    membership_count += 1
                    try:
                        result = process_payload(source_key, bytes(value_bytes))
                    except RecordReject as exc:
                        rejected_count += 1
                        membership = {
                            "schema_version": MEMBERSHIP_SCHEMA,
                            "source_key": source_key,
                            "cid": int(source_key),
                            "source_value_sha256": value_sha,
                            "disposition": "rejected",
                            "reason_code": exc.code,
                            "record_projection_sha256": None,
                        }
                        reject = {
                            "schema_version": REJECT_SCHEMA,
                            "source_key": source_key,
                            "cid": int(source_key),
                            "source_value_sha256": value_sha,
                            "reason_code": exc.code,
                        }
                        write_jsonl_row(membership_handle, membership)
                        write_jsonl_row(reject_handle, reject)
                        continue

                    admitted_count += 1
                    projection = {
                        "schema_version": PROJECTION_SCHEMA,
                        "source_key": source_key,
                        "cid": result["cid"],
                        "source_value_sha256": value_sha,
                        "motif_seq_sha256": result["motif_seq_sha256"],
                        "motif_mapping_sha256": result["motif_mapping_sha256"],
                        "ordered_motif_lexeme_sha256": result["exact_digests"],
                        "component_fragment_ranges": result["component_fragment_ranges"],
                        "motif_count": len(result["fragments"]),
                        "component_count": len(result["component_fragment_ranges"]),
                        "top_level_whitespace_count": result["top_level_whitespace_count"],
                        "stereo": result["stereo"],
                        "legacy_anchor": {
                            "component_anchor_multiplicities": result["component_anchor_multiplicities"],
                            "max_anchor_id": result["max_anchor_id"],
                            "p1_pair_rule_pass": result["p1_pair_rule_pass"],
                        },
                    }
                    projection_sha = sha256_json(projection)
                    membership = {
                        "schema_version": MEMBERSHIP_SCHEMA,
                        "source_key": source_key,
                        "cid": result["cid"],
                        "source_value_sha256": value_sha,
                        "disposition": "admitted_candidate_census",
                        "reason_code": None,
                        "record_projection_sha256": projection_sha,
                    }
                    write_jsonl_row(membership_handle, membership)
                    write_jsonl_row(projection_handle, projection)

                    for fragment, digest, token in zip(result["fragments"], result["exact_digests"], result["pure_tokens"]):
                        bound = exact_fragments.setdefault(digest, fragment)
                        if bound != fragment:
                            raise CensusError("exact motif digest collision")
                        exact_counts[digest] += 1
                        pure_digest = sha256_bytes(token.encode("utf-8"))
                        pure_bound = pure_digest_tokens.setdefault(pure_digest, token)
                        if pure_bound != token:
                            raise CensusError("pure motif digest collision")
                        pure_counts[token] += 1
                        pure_exact_digests[token].add(digest)
                        if "@" in fragment:
                            pure_stereo_exact_digests[token].add(digest)
                            pure_stereo_occurrences[token] += 1
                    component_count += len(result["component_fragment_ranges"])
                    for multiplicities in result["component_anchor_multiplicities"]:
                        if multiplicities:
                            component_with_anchor_count += 1
                        for multiplicity in multiplicities.values():
                            anchor_multiplicity_histogram[multiplicity] += 1
                    if result["p1_pair_rule_pass"]:
                        records_pair_pass += 1
                    else:
                        records_pair_fail += 1
                    if result["max_anchor_id"] is not None:
                        max_anchor_id = result["max_anchor_id"] if max_anchor_id is None else max(max_anchor_id, result["max_anchor_id"])
                    if result["stereo"]["fragment_count_with_at"]:
                        records_with_stereo_at += 1
                    fragments_with_stereo_at += result["stereo"]["fragment_count_with_at"]
                    if result["top_level_whitespace_count"]:
                        top_level_whitespace_record_count += 1
        finally:
            env.close()
        for handle in (membership_handle, reject_handle, projection_handle):
            handle.flush()
            os.fsync(handle.fileno())

    if observed_metadata != expected_metadata_bytes:
        raise CensusError("observed LMDB metadata-key set differs from source lock")
    if membership_count != source_lock["expected_payload_entry_count"]:
        raise CensusError("payload membership count differs from source lock")
    if membership_count != admitted_count + rejected_count:
        raise CensusError("membership partition does not balance")
    if int(source_path.stat().st_size) != source_lock["source_bytes"] or sha256_file(source_path) != source_lock["source_sha256"]:
        raise CensusError("source LMDB changed during extraction")

    exact_path = output_dir / "motif_census.jsonl"
    with exact_path.open("xb") as handle:
        for digest in sorted(exact_counts):
            write_jsonl_row(handle, {
                "motif_lexeme_sha256": digest,
                "motif_fragment": exact_fragments[digest],
                "count": int(exact_counts[digest]),
            })
        handle.flush()
        os.fsync(handle.fileno())
    pure_path = output_dir / "pure_motif_census.jsonl"
    with pure_path.open("xb") as handle:
        for token in sorted(pure_counts, key=lambda item: item.encode("utf-8")):
            write_jsonl_row(handle, {
                "pure_motif_token": token,
                "pure_motif_token_sha256": sha256_bytes(token.encode("utf-8")),
                "count": int(pure_counts[token]),
                "exact_lexeme_count": len(pure_exact_digests[token]),
                "stereo_exact_lexeme_count": len(pure_stereo_exact_digests[token]),
                "stereo_occurrence_count": int(pure_stereo_occurrences[token]),
            })
        handle.flush()
        os.fsync(handle.fileno())
    anchor_path = output_dir / "anchor_summary.json"
    anchor_summary = {
        "schema_version": ANCHOR_SUMMARY_SCHEMA,
        "interpretation": "legacy_component_local_labels_measured_not_reinterpreted_as_p1_global_bond_ids",
        "payload_record_count": membership_count,
        "admitted_record_count": admitted_count,
        "rejected_record_count": rejected_count,
        "component_count": component_count,
        "component_with_anchor_count": component_with_anchor_count,
        "record_p1_pair_rule_pass_count": records_pair_pass,
        "record_p1_pair_rule_fail_count": records_pair_fail,
        "anchor_label_multiplicity_histogram": {str(key): int(anchor_multiplicity_histogram[key]) for key in sorted(anchor_multiplicity_histogram)},
        "max_anchor_id": max_anchor_id,
        "records_with_stereo_at_count": records_with_stereo_at,
        "fragments_with_stereo_at_count": fragments_with_stereo_at,
        "records_with_top_level_whitespace_count": top_level_whitespace_record_count,
        "p1_p2_direct_anchor_semantics_claim": False,
        "training_admission": False,
    }
    write_json_new(anchor_path, anchor_summary)

    artifacts = {name.rsplit(".", 1)[0]: observe_artifact(output_dir / name, name) for name in DETERMINISTIC_ARTIFACTS}
    stable_derivation = {
        "source_sha256": source_lock["source_sha256"],
        "source_lock_sha256": source_lock_artifact["sha256"],
        "contract_sha256": contract_artifact["sha256"],
        "extractor_sha256": extractor_artifact["sha256"],
        "motif_sequence_producer_status": source_lock["motif_sequence_producer_status"],
        "motif_sequence_producer_sha256": source_lock["motif_sequence_producer_sha256"],
        "legacy_linearization_spec_sha256": source_lock["legacy_linearization_spec_sha256"],
        "artifact_sha256": {key: artifacts[key]["sha256"] for key in sorted(artifacts)},
        "counts": {
            "payload": membership_count,
            "admitted": admitted_count,
            "rejected": rejected_count,
            "unique_exact_motif": len(exact_counts),
            "exact_motif_occurrence": int(sum(exact_counts.values())),
            "unique_pure_motif": len(pure_counts),
        },
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "created_utc": utc_now(),
        "release_status": "candidate_p2_motif_census_non_release",
        "source": {
            "path": str(source_path),
            "sha256": source_lock["source_sha256"],
            "bytes": source_lock["source_bytes"],
            "expected_payload_entry_count": source_lock["expected_payload_entry_count"],
            "observed_metadata_keys": sorted(item.decode("utf-8") for item in observed_metadata),
        },
        "source_lock": source_lock_artifact,
        "contract": contract_artifact,
        "extractor": extractor_artifact,
        "legacy_pickle_acknowledgement_received": True,
        "provenance": {
            "source_copy_manifest_sha256": source_lock["source_copy_manifest_sha256"],
            "pickle_trust_basis_sha256": source_lock["pickle_trust_basis_sha256"],
            "motif_sequence_producer_status": source_lock["motif_sequence_producer_status"],
            "motif_sequence_producer_sha256": source_lock["motif_sequence_producer_sha256"],
            "legacy_linearization_spec_sha256": source_lock["legacy_linearization_spec_sha256"],
        },
        "counts": stable_derivation["counts"],
        "artifacts": artifacts,
        "logical_derivation_sha256": sha256_json(stable_derivation),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "p1_p2_union_decision_permitted": False,
        "tokenizer_freeze_permitted": False,
        "training_admission": False,
        "training_launcher_permitted": False,
        "next_gate": "Independent byte-identical rerun and P1/P2 projection-domain compatibility audit.",
    }
    receipt["receipt_payload_sha256"] = sha256_json(receipt)
    write_json_new(output_dir / "derivation_receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lmdb", required=True, type=Path)
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--legacy-pickle-acknowledgement", required=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.legacy_pickle_acknowledgement != ACKNOWLEDGEMENT:
        raise CensusError("exact legacy-pickle acknowledgement literal is required")
    receipt = extract_release(
        args.source_lmdb,
        args.source_lock,
        args.contract,
        args.output_dir,
        args.legacy_pickle_acknowledgement,
    )
    print(json.dumps({
        "release_status": receipt["release_status"],
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "logical_derivation_sha256": receipt["logical_derivation_sha256"],
        "counts": receipt["counts"],
        "training_admission": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
