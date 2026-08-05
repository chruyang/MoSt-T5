#!/usr/bin/env python3
"""Independently audit P1/P2 motif projection-domain compatibility.

This auditor reads already-derived exact census artifacts.  It does not import
the P2 extractor, RDKit, a linearizer, E3FP, tokenizer code, Dataset code or a
training launcher.  Its output is descriptive candidate evidence only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Optional


CONTRACT_SCHEMA = "most-t5-r1/p1-p2-motif-projection-compatibility-contract/v1"
P1_MANIFEST_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-full-release/v2"
P1_CONTRACT_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-release-contract/v2"
P2_RECEIPT_SCHEMA = "most-t5-r1/p2-phase2-ready-motif-census-receipt/v1"
P2_CONTRACT_SCHEMA = "most-t5-r1/p2-phase2-ready-motif-census-contract/v1"
P2_SOURCE_LOCK_SCHEMA = "most-t5-r1/p2-phase2-ready-source-lock/v1"
P2_ANCHOR_SCHEMA = "most-t5-r1/p2-legacy-anchor-summary/v1"
REPORT_SCHEMA = "most-t5-r1/p1-p2-motif-projection-compatibility-report/v1"
PROJECTION_SPEC_ID = "most-t5-r1/motif-lexeme-projection/v1"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ANCHOR_RE = re.compile(r"<([0-9]+)\*>")
P2_ARTIFACT_NAMES = (
    "membership.jsonl",
    "reject_ledger.jsonl",
    "record_projection.jsonl",
    "motif_census.jsonl",
    "pure_motif_census.jsonl",
    "anchor_summary.json",
)


class AuditError(RuntimeError):
    pass


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


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise AuditError("{} must be a lowercase SHA-256".format(label))
    return value


def producer_semantic_reasons(provenance: dict[str, Any], p1_linearizer_sha: str) -> list[str]:
    """Return semantic blockers without treating artifact-class SHA drift as proof."""

    require_hash(p1_linearizer_sha, "P1 linearizer hash")
    producer_status = provenance["motif_sequence_producer_status"]
    producer_sha = provenance["motif_sequence_producer_sha256"]
    if producer_status == "unknown_legacy_producer":
        if producer_sha is not None:
            raise AuditError("unknown P2 producer must not claim a producer hash")
        return [
            "P2_MOTIF_SEQUENCE_PRODUCER_UNKNOWN",
            "P1_P2_LINEARIZATION_SEMANTIC_MAPPING_NOT_PROVEN",
        ]
    if producer_status == "hash_locked":
        require_hash(producer_sha, "P2 producer hash")
        if producer_sha != p1_linearizer_sha:
            return ["P1_P2_LINEARIZATION_SEMANTIC_MAPPING_NOT_PROVEN"]
        return []
    raise AuditError("P2 producer status is invalid")


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AuditError("{} must be a regular non-symlink file: {}".format(label, path))
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AuditError("{} must contain a JSON object".format(label))
    return value


def observe(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AuditError("artifact is absent or not regular: {}".format(path))
    return {"relative_path": relative_path, "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def check_observation(root: Path, row: Any, expected_name: str) -> Path:
    if not isinstance(row, dict) or set(row) != {"relative_path", "bytes", "sha256"}:
        raise AuditError("artifact observation fields are not closed: {}".format(expected_name))
    if row["relative_path"] != expected_name:
        raise AuditError("artifact relative path differs: {}".format(expected_name))
    require_hash(row["sha256"], "artifact hash")
    path = root / expected_name
    if not path.is_file() or path.is_symlink():
        raise AuditError("artifact is absent or not regular: {}".format(path))
    if int(path.stat().st_size) != row["bytes"] or sha256_file(path) != row["sha256"]:
        raise AuditError("artifact byte/hash mismatch: {}".format(expected_name))
    return path


def project_exact(fragment: str) -> tuple[str, list[int]]:
    if not isinstance(fragment, str) or not fragment:
        raise AuditError("exact motif fragment must be non-empty text")
    anchor_ids: list[int] = []
    for match in ANCHOR_RE.finditer(fragment):
        decimal = match.group(1)
        try:
            anchor_id = int(decimal)
        except ValueError as exc:
            raise AuditError("invalid motif anchor decimal") from exc
        if str(anchor_id) != decimal:
            raise AuditError("non-canonical motif anchor decimal")
        anchor_ids.append(anchor_id)
    core = ANCHOR_RE.sub("", fragment)
    if "<" in core or ">" in core or not core:
        raise AuditError("motif fragment is outside projection grammar")
    return "[{}]".format(core), anchor_ids


def load_exact_census(path: Path, label: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    previous: Optional[str] = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError("{} line {} is invalid JSON".format(label, line_number)) from exc
            if not isinstance(row, dict) or set(row) != {"motif_lexeme_sha256", "motif_fragment", "count"}:
                raise AuditError("{} row fields are not closed".format(label))
            digest = require_hash(row["motif_lexeme_sha256"], "motif lexeme digest")
            fragment = row["motif_fragment"]
            if not isinstance(fragment, str) or sha256_bytes(fragment.encode("utf-8")) != digest:
                raise AuditError("{} digest-to-fragment binding mismatch".format(label))
            count = row["count"]
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise AuditError("{} count must be a positive integer".format(label))
            if previous is not None and digest <= previous:
                raise AuditError("{} rows are not strictly digest-sorted".format(label))
            previous = digest
            if digest in rows:
                raise AuditError("{} contains a duplicate digest".format(label))
            project_exact(fragment)
            rows[digest] = {"fragment": fragment, "count": count}
    if not rows:
        raise AuditError("{} is empty".format(label))
    return rows


def recompute_pure(exact: dict[str, dict[str, Any]]) -> tuple[dict[str, int], dict[str, set[str]]]:
    counts: Counter[str] = Counter()
    exact_members: dict[str, set[str]] = defaultdict(set)
    digest_bindings: dict[str, str] = {}
    for digest in sorted(exact):
        token, _ = project_exact(exact[digest]["fragment"])
        token_digest = sha256_bytes(token.encode("utf-8"))
        bound = digest_bindings.setdefault(token_digest, token)
        if bound != token:
            raise AuditError("pure motif digest collision")
        counts[token] += exact[digest]["count"]
        exact_members[token].add(digest)
    return dict(counts), exact_members


def validate_p2_pure(path: Path, exact: dict[str, dict[str, Any]]) -> tuple[dict[str, int], dict[str, set[str]]]:
    expected_counts, expected_members = recompute_pure(exact)
    observed_counts: dict[str, int] = {}
    previous: Optional[bytes] = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError("P2 pure census line {} is invalid JSON".format(line_number)) from exc
            expected_fields = {
                "pure_motif_token", "pure_motif_token_sha256", "count", "exact_lexeme_count",
                "stereo_exact_lexeme_count", "stereo_occurrence_count",
            }
            if not isinstance(row, dict) or set(row) != expected_fields:
                raise AuditError("P2 pure-census row fields are not closed")
            token = row["pure_motif_token"]
            if not isinstance(token, str) or not token:
                raise AuditError("P2 pure token is invalid")
            encoded = token.encode("utf-8")
            if previous is not None and encoded <= previous:
                raise AuditError("P2 pure census is not UTF-8-byte sorted")
            previous = encoded
            if sha256_bytes(encoded) != row["pure_motif_token_sha256"]:
                raise AuditError("P2 pure-token digest mismatch")
            if token not in expected_counts or row["count"] != expected_counts[token]:
                raise AuditError("P2 saved pure count differs from independent recomputation")
            if row["exact_lexeme_count"] != len(expected_members[token]):
                raise AuditError("P2 saved exact-to-pure cardinality differs from recomputation")
            stereo_members = {digest for digest in expected_members[token] if "@" in exact[digest]["fragment"]}
            stereo_occurrences = sum(exact[digest]["count"] for digest in stereo_members)
            if row["stereo_exact_lexeme_count"] != len(stereo_members) or row["stereo_occurrence_count"] != stereo_occurrences:
                raise AuditError("P2 saved stereo aggregation differs from recomputation")
            observed_counts[token] = row["count"]
    if observed_counts != expected_counts:
        raise AuditError("P2 saved pure census key/count set differs from recomputation")
    return expected_counts, expected_members


def ppm_floor(numerator: int, denominator: int) -> int:
    return 0 if denominator == 0 else int(numerator * 1_000_000 // denominator)


def overlap_metrics(left: dict[str, int], right: dict[str, int]) -> dict[str, Any]:
    common = set(left) & set(right)
    left_occurrences = int(sum(left.values()))
    right_occurrences = int(sum(right.values()))
    left_covered = int(sum(left[key] for key in common))
    right_covered = int(sum(right[key] for key in common))
    return {
        "p1_unique_count": len(left),
        "p2_unique_count": len(right),
        "shared_unique_count": len(common),
        "p1_occurrence_count": left_occurrences,
        "p2_occurrence_count": right_occurrences,
        "p1_shared_occurrence_count": left_covered,
        "p2_shared_occurrence_count": right_covered,
        "p1_shared_occurrence_coverage_ppm_floor": ppm_floor(left_covered, left_occurrences),
        "p2_shared_occurrence_coverage_ppm_floor": ppm_floor(right_covered, right_occurrences),
    }


def collapse_metrics(exact_members: dict[str, set[str]]) -> dict[str, int]:
    sizes = [len(value) for value in exact_members.values()]
    return {
        "pure_motif_count": len(sizes),
        "pure_motifs_with_multiple_exact_lexemes": sum(value > 1 for value in sizes),
        "maximum_exact_lexemes_per_pure_motif": max(sizes) if sizes else 0,
        "exact_lexeme_count": sum(sizes),
    }


def stereo_metrics(exact: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, marker in (("at", "@"), ("forward_slash", "/"), ("backslash", "\\")):
        matching = [row for row in exact.values() if marker in row["fragment"]]
        result[name] = {
            "unique_exact_lexeme_count": len(matching),
            "weighted_exact_lexeme_occurrence_count": int(sum(row["count"] for row in matching)),
            "weighted_marker_count": int(sum(row["fragment"].count(marker) * row["count"] for row in matching)),
        }
    return result


def anchor_lexeme_metrics(exact: dict[str, dict[str, Any]]) -> dict[str, Any]:
    unique_with_anchor = 0
    occurrence_with_anchor = 0
    weighted_anchor_count = 0
    max_anchor: Optional[int] = None
    for row in exact.values():
        _, anchors = project_exact(row["fragment"])
        if anchors:
            unique_with_anchor += 1
            occurrence_with_anchor += row["count"]
            weighted_anchor_count += len(anchors) * row["count"]
            observed = max(anchors)
            max_anchor = observed if max_anchor is None else max(max_anchor, observed)
    return {
        "unique_exact_lexemes_with_anchor": unique_with_anchor,
        "weighted_exact_lexeme_occurrences_with_anchor": int(occurrence_with_anchor),
        "weighted_anchor_lexeme_count": int(weighted_anchor_count),
        "max_anchor_id": max_anchor,
    }


def validate_p2_receipt(root: Path, contract_path: Path, source_lock_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    receipt_path = root / "derivation_receipt.json"
    receipt = load_json(receipt_path, "P2 derivation receipt")
    if receipt.get("schema_version") != P2_RECEIPT_SCHEMA or receipt.get("release_status") != "candidate_p2_motif_census_non_release":
        raise AuditError("P2 derivation receipt status/schema mismatch")
    payload_hash = require_hash(receipt.get("receipt_payload_sha256"), "P2 receipt payload hash")
    unhashed = dict(receipt)
    del unhashed["receipt_payload_sha256"]
    if sha256_json(unhashed) != payload_hash:
        raise AuditError("P2 derivation receipt self-hash mismatch")
    if receipt.get("training_admission") is not False or receipt.get("p1_p2_union_decision_permitted") is not False:
        raise AuditError("P2 receipt overclaims admission or union permission")
    if receipt.get("contract", {}).get("sha256") != sha256_file(contract_path):
        raise AuditError("P2 receipt binds a different extraction contract")
    if receipt.get("source_lock", {}).get("sha256") != sha256_file(source_lock_path):
        raise AuditError("P2 receipt binds a different source lock")
    artifacts = receipt.get("artifacts")
    expected_keys = {name.rsplit(".", 1)[0] for name in P2_ARTIFACT_NAMES}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_keys:
        raise AuditError("P2 receipt artifact set is not closed")
    paths = {}
    for name in P2_ARTIFACT_NAMES:
        key = name.rsplit(".", 1)[0]
        paths[key] = check_observation(root, artifacts[key], name)
    return receipt, paths


def audit(args: argparse.Namespace) -> dict[str, Any]:
    p1_root = args.p1_release.expanduser().resolve()
    p1_contract_path = args.p1_contract.expanduser().resolve()
    p2_root = args.p2_release.expanduser().resolve()
    p2_contract_path = args.p2_contract.expanduser().resolve()
    p2_source_lock_path = args.p2_source_lock.expanduser().resolve()
    compatibility_contract_path = args.contract.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError("--output must be new")

    compatibility_contract = load_json(compatibility_contract_path, "compatibility contract")
    if compatibility_contract.get("schema_version") != CONTRACT_SCHEMA:
        raise AuditError("compatibility contract schema mismatch")
    if compatibility_contract.get("projection", {}).get("spec_id") != PROJECTION_SPEC_ID:
        raise AuditError("compatibility contract projection spec mismatch")
    p1_contract = load_json(p1_contract_path, "P1 production contract")
    if p1_contract.get("schema_version") != P1_CONTRACT_SCHEMA:
        raise AuditError("P1 production contract schema mismatch")
    p1_manifest_path = p1_root / "full_release_manifest.json"
    p1_manifest = load_json(p1_manifest_path, "P1 full release manifest")
    if p1_manifest.get("schema_version") != P1_MANIFEST_SCHEMA or p1_manifest.get("release_status") != "complete":
        raise AuditError("P1 release is not complete production-v2")
    if p1_manifest.get("configuration", {}).get("production_contract_sha256") != sha256_file(p1_contract_path):
        raise AuditError("P1 manifest binds a different production contract")
    p1_linearizer_sha = require_hash(
        p1_manifest.get("configuration", {}).get("harness", {}).get("components", {}).get("molecule_native_linearizer"),
        "P1 molecule-native linearizer hash",
    )
    p1_census_row = p1_manifest.get("global_motif_census")
    p1_census_path = check_observation(p1_root, p1_census_row, "motif_census.jsonl")

    p2_contract = load_json(p2_contract_path, "P2 extraction contract")
    if p2_contract.get("schema_version") != P2_CONTRACT_SCHEMA:
        raise AuditError("P2 extraction contract schema mismatch")
    p2_source_lock = load_json(p2_source_lock_path, "P2 source lock")
    if p2_source_lock.get("schema_version") != P2_SOURCE_LOCK_SCHEMA:
        raise AuditError("P2 source lock schema mismatch")
    p2_receipt, p2_paths = validate_p2_receipt(p2_root, p2_contract_path, p2_source_lock_path)
    provenance = p2_receipt.get("provenance", {})
    for field in ("source_copy_manifest_sha256", "pickle_trust_basis_sha256", "legacy_linearization_spec_sha256"):
        if provenance.get(field) != p2_source_lock.get(field):
            raise AuditError("P2 receipt/source-lock provenance mismatch: {}".format(field))
        require_hash(provenance[field], "P2 provenance {}".format(field))
    if provenance.get("motif_sequence_producer_status") != p2_source_lock.get("motif_sequence_producer_status"):
        raise AuditError("P2 producer status differs between receipt and source lock")
    if provenance.get("motif_sequence_producer_sha256") != p2_source_lock.get("motif_sequence_producer_sha256"):
        raise AuditError("P2 producer hash differs between receipt and source lock")

    p1_exact = load_exact_census(p1_census_path, "P1 exact census")
    p2_exact = load_exact_census(p2_paths["motif_census"], "P2 exact census")
    for digest in set(p1_exact) & set(p2_exact):
        if p1_exact[digest]["fragment"] != p2_exact[digest]["fragment"]:
            raise AuditError("cross-phase exact motif digest collision")
    p1_pure, p1_members = recompute_pure(p1_exact)
    p2_pure, p2_members = validate_p2_pure(p2_paths["pure_motif_census"], p2_exact)
    p1_exact_counts = {digest: row["count"] for digest, row in p1_exact.items()}
    p2_exact_counts = {digest: row["count"] for digest, row in p2_exact.items()}
    p1_stereo = stereo_metrics(p1_exact)
    p2_stereo = stereo_metrics(p2_exact)
    p1_anchor = anchor_lexeme_metrics(p1_exact)
    p2_anchor = anchor_lexeme_metrics(p2_exact)

    p1_counts = p1_manifest.get("counts")
    if not isinstance(p1_counts, dict):
        raise AuditError("P1 full release manifest counts are absent")
    if p1_counts.get("unique_motif_count") != len(p1_exact) or p1_counts.get("motif_occurrence_count") != sum(row["count"] for row in p1_exact.values()):
        raise AuditError("P1 manifest motif counts differ from its exact census")
    p2_counts = p2_receipt.get("counts")
    if not isinstance(p2_counts, dict):
        raise AuditError("P2 derivation receipt counts are absent")
    if (
        p2_counts.get("unique_exact_motif") != len(p2_exact)
        or p2_counts.get("exact_motif_occurrence") != sum(row["count"] for row in p2_exact.values())
        or p2_counts.get("unique_pure_motif") != len(p2_pure)
    ):
        raise AuditError("P2 receipt motif counts differ from its census artifacts")

    anchor_summary = load_json(p2_paths["anchor_summary"], "P2 anchor summary")
    expected_anchor_fields = {
        "schema_version", "interpretation", "payload_record_count", "admitted_record_count",
        "rejected_record_count", "component_count", "component_with_anchor_count",
        "record_p1_pair_rule_pass_count", "record_p1_pair_rule_fail_count",
        "anchor_label_multiplicity_histogram", "max_anchor_id",
        "records_with_stereo_at_count", "fragments_with_stereo_at_count",
        "records_with_top_level_whitespace_count", "p1_p2_direct_anchor_semantics_claim",
        "training_admission",
    }
    if set(anchor_summary) != expected_anchor_fields:
        raise AuditError("P2 anchor-summary fields are not closed")
    if anchor_summary.get("schema_version") != P2_ANCHOR_SCHEMA:
        raise AuditError("P2 anchor-summary schema mismatch")
    if anchor_summary.get("training_admission") is not False or anchor_summary.get("p1_p2_direct_anchor_semantics_claim") is not False:
        raise AuditError("P2 anchor summary overclaims admission or semantic equivalence")
    if anchor_summary.get("admitted_record_count") != p2_receipt.get("counts", {}).get("admitted"):
        raise AuditError("P2 anchor-summary admitted count differs from receipt")
    if (
        anchor_summary.get("payload_record_count") != p2_counts.get("payload")
        or anchor_summary.get("rejected_record_count") != p2_counts.get("rejected")
        or anchor_summary.get("record_p1_pair_rule_pass_count", 0) + anchor_summary.get("record_p1_pair_rule_fail_count", 0) != p2_counts.get("admitted")
    ):
        raise AuditError("P2 anchor-summary record partition differs from receipt")
    histogram = anchor_summary.get("anchor_label_multiplicity_histogram")
    if not isinstance(histogram, dict) or any(
        not isinstance(key, str) or not key.isdigit() or str(int(key)) != key
        or not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for key, value in histogram.items()
    ):
        raise AuditError("P2 anchor multiplicity histogram is invalid")
    if anchor_summary.get("max_anchor_id") != p2_anchor["max_anchor_id"]:
        raise AuditError("P2 record/lexeme anchor maxima differ")

    reasons: list[str] = []
    producer_status = provenance["motif_sequence_producer_status"]
    producer_sha = provenance["motif_sequence_producer_sha256"]
    reasons.extend(producer_semantic_reasons(provenance, p1_linearizer_sha))
    if anchor_summary.get("record_p1_pair_rule_fail_count", 0) > 0:
        reasons.append("P2_LEGACY_ANCHOR_MULTIPLICITY_VIOLATES_P1_PAIR_RULE")
    if p2_stereo["at"]["weighted_exact_lexeme_occurrence_count"] > 0 and p1_stereo["at"]["weighted_exact_lexeme_occurrence_count"] == 0:
        reasons.append("P2_RETAINS_AT_STEREOCHEMISTRY_WHILE_P1_DOMAIN_OMITS_IT")
    reasons.append("P1_P2_COMPONENT_BOUNDARY_SEMANTICS_NOT_PROVEN_EQUAL")
    reasons = sorted(set(reasons))

    report = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": utc_now(),
        "report_status": "candidate_evidence_non_release",
        "inputs": {
            "compatibility_contract": observe(compatibility_contract_path, compatibility_contract_path.name),
            "p1_contract": observe(p1_contract_path, p1_contract_path.name),
            "p1_manifest": observe(p1_manifest_path, "full_release_manifest.json"),
            "p1_exact_census": observe(p1_census_path, "motif_census.jsonl"),
            "p2_contract": observe(p2_contract_path, p2_contract_path.name),
            "p2_source_lock": observe(p2_source_lock_path, p2_source_lock_path.name),
            "p2_receipt": observe(p2_root / "derivation_receipt.json", "derivation_receipt.json"),
            "p2_exact_census": observe(p2_paths["motif_census"], "motif_census.jsonl"),
            "p2_pure_census": observe(p2_paths["pure_motif_census"], "pure_motif_census.jsonl"),
            "p2_anchor_summary": observe(p2_paths["anchor_summary"], "anchor_summary.json"),
        },
        "projection": {
            "spec_id": PROJECTION_SPEC_ID,
            "p1_independently_recomputed": True,
            "p2_independently_recomputed_and_saved_census_equal": True,
            "normalization_performed": False,
            "rdkit_used": False,
        },
        "overlap": {
            "exact_lexeme": overlap_metrics(p1_exact_counts, p2_exact_counts),
            "pure_motif": overlap_metrics(p1_pure, p2_pure),
        },
        "collapse": {
            "p1": collapse_metrics(p1_members),
            "p2": collapse_metrics(p2_members),
        },
        "stereochemistry": {"p1": p1_stereo, "p2": p2_stereo},
        "anchors": {
            "p1_exact_lexemes": p1_anchor,
            "p2_exact_lexemes": p2_anchor,
            "p2_record_semantics": {
                "interpretation": anchor_summary.get("interpretation"),
                "component_count": anchor_summary.get("component_count"),
                "component_with_anchor_count": anchor_summary.get("component_with_anchor_count"),
                "record_p1_pair_rule_pass_count": anchor_summary.get("record_p1_pair_rule_pass_count"),
                "record_p1_pair_rule_fail_count": anchor_summary.get("record_p1_pair_rule_fail_count"),
                "anchor_label_multiplicity_histogram": anchor_summary.get("anchor_label_multiplicity_histogram"),
                "max_anchor_id": anchor_summary.get("max_anchor_id"),
            },
        },
        "provenance_comparison": {
            "p1_molecule_native_linearizer_sha256": p1_linearizer_sha,
            "p2_motif_sequence_producer_status": producer_status,
            "p2_motif_sequence_producer_sha256": producer_sha,
            "p2_legacy_linearization_spec_sha256": provenance["legacy_linearization_spec_sha256"],
        },
        "direct_projection_domain_compatible": not reasons,
        "incompatibility_reasons": reasons,
        "scientific_interpretation": "Lexical overlap is descriptive coverage evidence only. It does not prove identical stereochemical, anchor-label, component-boundary or producer semantics.",
        "automatic_union_permitted": False,
        "union_decision_permitted": False,
        "cutoff_decision_permitted": False,
        "oov_decision_permitted": False,
        "tokenizer_freeze_permitted": False,
        "training_admission": False,
        "training_launcher_permitted": False,
        "next_gate": "Review this evidence, then explicitly choose either a P2 relinearization study or a scientifically justified domain-bridging policy before any P1+P2 vocabulary experiment.",
    }
    report["report_payload_sha256"] = sha256_json(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-release", required=True, type=Path)
    parser.add_argument("--p1-contract", required=True, type=Path)
    parser.add_argument("--p2-release", required=True, type=Path)
    parser.add_argument("--p2-contract", required=True, type=Path)
    parser.add_argument("--p2-source-lock", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args)
    print(json.dumps({
        "report_status": report["report_status"],
        "direct_projection_domain_compatible": report["direct_projection_domain_compatible"],
        "incompatibility_reasons": report["incompatibility_reasons"],
        "output": str(args.output.expanduser().resolve()),
        "training_admission": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
