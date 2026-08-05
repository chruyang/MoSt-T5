#!/usr/bin/env python3
"""Independently validate an R1 motif-tokenizer/digest-binding release.

This validator deliberately does not import the builder.  It independently
rehashes all supplied inputs, reprojects every exact lexeme, recomputes token
selection and every binding row, reloads the saved tokenizer offline, and can
validate a pre-extracted set of admitted-record digest sequences.  It never
reads SDF, invokes RDKit/linearization, or recomputes E3FP.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
from typing import Any, Iterable


CONTRACT_SCHEMA = "most-t5-r1/motif-tokenizer-binding-release-contract/v1"
PRODUCTION_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-full-release/v2"
BASE_LOCK_SCHEMA = "most-t5-r1/base-model-snapshot-lock/v1"
SCOPE_LOCK_SCHEMA = "most-t5-r1/tokenizer-discovery-scope-lock/v1"
POLICY_SCHEMA = "most-t5-r1/motif-token-selection-policy/v1"
RELEASE_SCHEMA = "most-t5-r1/motif-tokenizer-binding-release/v1"
REPORT_SCHEMA = "most-t5-r1/motif-tokenizer-binding-validation/v1"
SAMPLE_SCHEMA = "most-t5-r1/tokenizer-binding-sample/v1"
SAMPLE_RECEIPT_SCHEMA = "most-t5-r1/tokenizer-binding-sample-extraction-receipt/v1"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ANCHOR_RE = re.compile(r"<([0-9]+)\*>")
ANCHOR_LIKE_RE = re.compile(r"<[^>]*\*>")
INT64_MAX = (1 << 63) - 1
T5_SENTINELS = tuple("<extra_id_{}>".format(index) for index in range(100))
DIGITS = tuple(str(index) for index in range(10))
TASKS = ("<bom>", "<eom>", "[MMM]:", "[Caption]:", "[Text2Mol]:", "[Denoise]:")
STRUCTURAL = ("[.]",)

SCOPE_FIELDS = {
    "schema_version", "phase", "scope_status", "identity_namespace",
    "membership_manifest_sha256", "membership_count",
    "downstream_identity_exclusion_proof_sha256", "census_sha256",
    "census_unique_lexeme_count", "census_occurrence_count", "census_kind",
    "census_derivation_audit_sha256", "source_release_logical_root_sha256",
    "motif_linearization_spec_sha256", "motif_sequence_extraction_spec_sha256",
    "projection_domain_compatibility_audit_sha256",
}
BASE_LOCK_FIELDS = {
    "schema_version", "decision_status", "model_identifier", "revision",
    "expected_tokenizer_class", "tokenizer_and_model_same_revision",
    "snapshot_tree_sha256", "files",
}
POLICY_FIELDS = {
    "schema_version", "decision_status", "discovery_scope", "min_selection_score",
    "max_motif_tokens", "selection_score", "oov_policy", "anchor_policy",
    "reserved_special_token_count", "base_model", "base_vocab_overlap_allowlist",
    "tie_break", "p2_vocab_extension_forbidden",
}
SAMPLE_FIELDS = {
    "schema_version", "member_id", "record_content_sha256", "motif_count",
    "motif_lexeme_sha256", "motif_atom_indices_count", "motif_geometry_valid_count",
}
SAMPLE_RECEIPT_FIELDS = {
    "schema_version", "status", "production_logical_release_root_sha256",
    "production_manifest_sha256", "sample_schedule_sha256", "sample_jsonl_sha256",
    "sample_record_count", "safe_payload_decoder_sha256",
    "payload_index_verification_report_sha256", "component_reference_audit_sha256",
}
BINDING_FIELDS = {
    "motif_lexeme_sha256", "motif_fragment", "anchors", "anchor_ids",
    "pure_motif_token", "pure_motif_token_sha256", "p1_count", "p2_count",
    "selection_score", "selected", "token_id", "binding_disposition",
}
MANIFEST_FIELDS = {
    "schema_version", "created_utc", "release_id", "release_status",
    "independent_validation_status", "p1_training_admission",
    "p1_training_launcher_permitted", "p1_p2_exact_same_mapping",
    "p2_vocab_extension_forbidden", "contract", "builder", "production_release",
    "base_snapshot_lock", "scope_locks", "selection_policy", "projection",
    "statistics", "tokenizer_mapping", "determinism_gate", "runtime", "artifacts",
    "next_gate", "manifest_payload_sha256",
}
MAPPING_FIELDS = {
    "tokenizer_class", "vocab_size", "oov_token", "oov_token_id",
    "id_to_token_sha256", "token_to_id_sha256", "added_token_metadata_sha256",
    "special_token_order", "special_token_order_sha256", "special_token_id_map",
    "special_token_id_map_sha256", "sentinel_token_id_map",
    "sentinel_token_id_map_sha256", "anchor_token_id_map",
    "anchor_token_id_map_sha256", "selected_motif_token_id_map_sha256",
}


class ValidationError(RuntimeError):
    """Raised when a release fails independent validation."""


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


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValidationError("{} is not a lowercase SHA-256".format(label))
    return value


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValidationError("{} is not a regular non-symlink file".format(label))
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValidationError("{} must be a JSON object".format(label))
    return value


def observe_tree(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValidationError("tree root is not a regular directory")
    files = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if candidate.is_symlink():
            raise ValidationError("symlink in immutable tree: {}".format(candidate))
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValidationError("special file in immutable tree: {}".format(candidate))
        files.append({
            "relative_path": candidate.relative_to(root).as_posix(),
            "bytes": int(candidate.stat().st_size),
            "sha256": sha256_file(candidate),
        })
    if not files:
        raise ValidationError("immutable tree contains no regular files")
    return {"files": files, "file_count": len(files), "tree_sha256": sha256_json(files)}


def check_artifact(root: Path, observation: dict[str, Any], expected_relative: str) -> Path:
    if not isinstance(observation, dict) or set(observation) != {"relative_path", "bytes", "sha256"}:
        raise ValidationError("artifact observation is malformed")
    if observation["relative_path"] != expected_relative:
        raise ValidationError("artifact relative path mismatch")
    require_hash(observation["sha256"], "artifact hash")
    path = root / expected_relative
    if not path.is_file() or path.is_symlink():
        raise ValidationError("artifact is absent or not regular: {}".format(path))
    if int(path.stat().st_size) != observation["bytes"] or sha256_file(path) != observation["sha256"]:
        raise ValidationError("artifact byte/hash mismatch: {}".format(path))
    return path


def stable_unique(values: Iterable[str]) -> list[str]:
    seen: dict[str, bool] = {}
    result = []
    for value in values:
        if value not in seen:
            seen[value] = True
            result.append(value)
    return result


def added_metadata(tokenizer: Any) -> list[dict[str, Any]]:
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if decoder is None:
        raise ValidationError("saved tokenizer has no added_tokens_decoder")
    return [
        {
            "id": int(token_id),
            "content": str(getattr(token, "content", token)),
            "single_word": bool(getattr(token, "single_word", False)),
            "lstrip": bool(getattr(token, "lstrip", False)),
            "rstrip": bool(getattr(token, "rstrip", False)),
            "normalized": bool(getattr(token, "normalized", True)),
            "special": bool(getattr(token, "special", False)),
        }
        for token_id, token in sorted(decoder.items(), key=lambda pair: int(pair[0]))
    ]


def id_to_token(token_to_id: dict[str, int]) -> list[str]:
    if not token_to_id:
        raise ValidationError("saved tokenizer vocabulary is empty")
    ids = [int(value) for value in token_to_id.values()]
    result: list[str | None] = [None] * (max(ids) + 1)
    for token, token_id_value in token_to_id.items():
        token_id = int(token_id_value)
        if result[token_id] is not None and result[token_id] != token:
            raise ValidationError("saved tokenizer ID collision")
        result[token_id] = token
    if any(token is None for token in result) or len(ids) != len(set(ids)):
        raise ValidationError("saved tokenizer ID space is not contiguous one-to-one")
    return [str(token) for token in result]


def validate_policy(policy: dict[str, Any]) -> None:
    if set(policy) != POLICY_FIELDS or policy.get("schema_version") != POLICY_SCHEMA:
        raise ValidationError("selection policy fields/schema mismatch")
    if policy["decision_status"] not in {"approved_for_candidate", "approved_for_frozen_release"}:
        raise ValidationError("selection decision is unresolved")
    if policy["discovery_scope"] not in {"p1_only", "p1_p2_permitted_train_union"}:
        raise ValidationError("selection discovery scope is unresolved")
    if isinstance(policy["min_selection_score"], bool) or not isinstance(policy["min_selection_score"], int) or policy["min_selection_score"] <= 0:
        raise ValidationError("minimum selection score is invalid")
    cap = policy["max_motif_tokens"]
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0):
        raise ValidationError("motif token cap is invalid")
    score = policy["selection_score"]
    if not isinstance(score, dict) or set(score) != {"kind", "p1_weight", "p2_weight"} or score["kind"] != "weighted_integer_count":
        raise ValidationError("selection score declaration is invalid")
    for key in ("p1_weight", "p2_weight"):
        if isinstance(score[key], bool) or not isinstance(score[key], int) or score[key] < 0:
            raise ValidationError("selection weight is invalid")
    if score["p1_weight"] <= 0:
        raise ValidationError("P1 weight must be positive")
    if policy["discovery_scope"] == "p1_only" and score["p2_weight"] != 0:
        raise ValidationError("P1-only selection has nonzero P2 weight")
    if policy["discovery_scope"] != "p1_only" and score["p2_weight"] <= 0:
        raise ValidationError("union selection lacks positive P2 weight")
    oov = policy["oov_policy"]
    if not isinstance(oov, dict) or set(oov) != {"kind", "token"} or oov["kind"] not in {"base_unk", "dedicated_motif_unk"}:
        raise ValidationError("OOV policy is invalid")
    if not isinstance(oov["token"], str) or not oov["token"] or any(c in oov["token"] for c in "\x00\t\r\n"):
        raise ValidationError("OOV token is malformed")
    anchor = policy["anchor_policy"]
    if not isinstance(anchor, dict) or set(anchor) != {"max_anchor_id_inclusive", "overflow_action"}:
        raise ValidationError("anchor policy is invalid")
    if isinstance(anchor["max_anchor_id_inclusive"], bool) or not isinstance(anchor["max_anchor_id_inclusive"], int) or anchor["max_anchor_id_inclusive"] < 0 or anchor["overflow_action"] != "fail_closed":
        raise ValidationError("anchor policy is not fail-closed")
    reserve = policy["reserved_special_token_count"]
    if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
        raise ValidationError("reserved token count is invalid")
    if policy["tie_break"] != ["selection_score_desc", "pure_motif_utf8_asc", "pure_motif_sha256_asc"]:
        raise ValidationError("tie-break differs from the contract")
    if policy["p2_vocab_extension_forbidden"] is not True:
        raise ValidationError("P2 vocabulary extension is not forbidden")
    allow = policy["base_vocab_overlap_allowlist"]
    if not isinstance(allow, list) or any(not isinstance(token, str) or not token for token in allow) or allow != sorted(dict.fromkeys(allow), key=lambda token: token.encode("utf-8")):
        raise ValidationError("base-vocabulary overlap allow-list is not canonical")
    base = policy["base_model"]
    if not isinstance(base, dict) or set(base) != {"identifier", "revision"} or not all(isinstance(base[key], str) and base[key] for key in base):
        raise ValidationError("base-model decision is malformed")


def validate_scope(lock: dict[str, Any], phase: str, census: Path, logical_root: str) -> None:
    if set(lock) != SCOPE_FIELDS or lock.get("schema_version") != SCOPE_LOCK_SCHEMA or lock.get("phase") != phase:
        raise ValidationError("{} scope-lock fields/schema mismatch".format(phase))
    if lock["scope_status"] not in {"candidate", "complete"}:
        raise ValidationError("{} scope status is invalid".format(phase))
    if not isinstance(lock["identity_namespace"], str) or not lock["identity_namespace"]:
        raise ValidationError("{} identity namespace is empty".format(phase))
    if isinstance(lock["membership_count"], bool) or not isinstance(lock["membership_count"], int) or lock["membership_count"] <= 0:
        raise ValidationError("{} membership count is invalid".format(phase))
    for field in ("census_unique_lexeme_count", "census_occurrence_count"):
        if isinstance(lock[field], bool) or not isinstance(lock[field], int) or lock[field] <= 0:
            raise ValidationError("{} {} is invalid".format(phase, field))
    for field in (
        "membership_manifest_sha256", "census_sha256", "source_release_logical_root_sha256",
        "motif_linearization_spec_sha256", "motif_sequence_extraction_spec_sha256",
    ):
        require_hash(lock[field], "{} scope {}".format(phase, field))
    if phase == "P1" and lock["source_release_logical_root_sha256"] != logical_root:
        raise ValidationError("P1 scope does not bind supplied production logical root")
    if sha256_file(census) != lock["census_sha256"]:
        raise ValidationError("{} census hash differs from scope lock".format(phase))
    if lock["scope_status"] == "complete":
        if lock["census_kind"] != "permitted_membership_derived":
            raise ValidationError("complete scope census is not permitted-membership-derived")
        require_hash(lock["downstream_identity_exclusion_proof_sha256"], "downstream proof")
        require_hash(lock["census_derivation_audit_sha256"], "census derivation audit")
        if phase == "P2":
            require_hash(lock["projection_domain_compatibility_audit_sha256"], "P1/P2 projection-domain compatibility audit")
        elif lock["projection_domain_compatibility_audit_sha256"] is not None:
            require_hash(lock["projection_domain_compatibility_audit_sha256"], "P1 projection-domain compatibility audit")
    else:
        if lock["census_kind"] not in {"production_global_admitted_candidate", "permitted_membership_derived"}:
            raise ValidationError("candidate census kind is invalid")
        for field in (
            "downstream_identity_exclusion_proof_sha256", "census_derivation_audit_sha256",
            "projection_domain_compatibility_audit_sha256",
        ):
            if lock[field] is not None:
                require_hash(lock[field], "candidate {}".format(field))


def read_census(path: Path, lock: dict[str, Any], phase: str) -> dict[str, tuple[str, int]]:
    rows = {}
    previous = None
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ValidationError("{} census line lacks LF".format(phase))
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {"motif_lexeme_sha256", "motif_fragment", "count"}:
                raise ValidationError("{} census row fields are open".format(phase))
            digest, fragment, count = row["motif_lexeme_sha256"], row["motif_fragment"], row["count"]
            require_hash(digest, "{} digest".format(phase))
            if previous is not None and digest <= previous:
                raise ValidationError("{} census is not digest sorted".format(phase))
            previous = digest
            if not isinstance(fragment, str) or not fragment or any(c in fragment for c in "\x00\t\r\n"):
                raise ValidationError("{} census fragment is malformed".format(phase))
            if sha256_bytes(fragment.encode("utf-8")) != digest:
                raise ValidationError("{} census content address mismatch".format(phase))
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValidationError("{} census count is invalid".format(phase))
            total += count
            if total > INT64_MAX:
                raise ValidationError("{} census count exceeds int64".format(phase))
            rows[digest] = (fragment, count)
    if len(rows) != lock["census_unique_lexeme_count"] or total != lock["census_occurrence_count"]:
        raise ValidationError("{} census counts differ from scope lock".format(phase))
    return rows


def project(fragment: str) -> tuple[list[str], list[int], str]:
    matches = list(ANCHOR_RE.finditer(fragment))
    anchors = [match.group(0) for match in matches]
    if ANCHOR_LIKE_RE.findall(fragment) != anchors:
        raise ValidationError("malformed anchor-like substring")
    anchor_ids = []
    for match in matches:
        value = match.group(1)
        if len(value) > 1 and value.startswith("0"):
            raise ValidationError("noncanonical anchor decimal")
        anchor_ids.append(int(value))
    core = ANCHOR_RE.sub("", fragment)
    if not core:
        raise ValidationError("empty motif core after anchor deletion")
    return anchors, anchor_ids, "[{}]".format(core)


def recompute_rows(
    p1: dict[str, tuple[str, int]], p2: dict[str, tuple[str, int]], policy: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], int]:
    aggregates = defaultdict(lambda: {"p1": 0, "p2": 0, "lexemes": 0})
    bindings = []
    max_anchor = -1
    for digest in sorted(set(p1) | set(p2)):
        left, right = p1.get(digest), p2.get(digest)
        fragments = [entry[0] for entry in (left, right) if entry is not None]
        if not fragments or any(value != fragments[0] for value in fragments):
            raise ValidationError("cross-phase digest collision")
        anchors, anchor_ids, pure = project(fragments[0])
        if anchor_ids:
            max_anchor = max(max_anchor, max(anchor_ids))
        p1_count = 0 if left is None else left[1]
        p2_count = 0 if right is None else right[1]
        aggregates[pure]["p1"] += p1_count
        aggregates[pure]["p2"] += p2_count
        aggregates[pure]["lexemes"] += 1
        bindings.append({
            "motif_lexeme_sha256": digest,
            "motif_fragment": fragments[0],
            "anchors": anchors,
            "anchor_ids": anchor_ids,
            "pure_motif_token": pure,
            "pure_motif_token_sha256": sha256_bytes(pure.encode("utf-8")),
            "p1_count": p1_count,
            "p2_count": p2_count,
        })
    weights = policy["selection_score"]
    pure_rows = []
    for token, counts in aggregates.items():
        score = counts["p1"] * weights["p1_weight"] + counts["p2"] * weights["p2_weight"]
        if score > INT64_MAX:
            raise ValidationError("selection score exceeds int64")
        pure_rows.append({
            "pure_motif_token": token,
            "pure_motif_token_sha256": sha256_bytes(token.encode("utf-8")),
            "p1_count": counts["p1"],
            "p2_count": counts["p2"],
            "total_count": counts["p1"] + counts["p2"],
            "selection_score": score,
            "exact_lexeme_count": counts["lexemes"],
        })
    ranked = sorted(
        (row for row in pure_rows if row["selection_score"] >= policy["min_selection_score"]),
        key=lambda row: (-row["selection_score"], row["pure_motif_token"].encode("utf-8"), row["pure_motif_token_sha256"]),
    )
    cap = policy["max_motif_tokens"]
    selected_rows = ranked if cap is None else ranked[:cap]
    selected = [row["pure_motif_token"] for row in selected_rows]
    ranks = {token: index for index, token in enumerate(selected)}
    for row in pure_rows:
        rank = ranks.get(row["pure_motif_token"])
        row["eligible"] = row["selection_score"] >= policy["min_selection_score"]
        row["selected"] = rank is not None
        row["selection_rank"] = rank
    pure_rows.sort(key=lambda row: row["pure_motif_token"].encode("utf-8"))
    if max_anchor > policy["anchor_policy"]["max_anchor_id_inclusive"]:
        raise ValidationError("observed anchor exceeds frozen reserve")
    return bindings, pure_rows, selected, max_anchor


def compare_jsonl(path: Path, expected: list[dict[str, Any]], label: str) -> None:
    count = 0
    with path.open("rb") as handle:
        for index, line in enumerate(handle):
            if index >= len(expected) or line != canonical_json_bytes(expected[index]) + b"\n":
                raise ValidationError("{} differs at row {}".format(label, index))
            count += 1
    if count != len(expected):
        raise ValidationError("{} row count differs".format(label))


def expected_vocab_bytes(selected: list[str], by_token: dict[str, dict[str, Any]]) -> bytes:
    chunks = []
    for token in selected:
        row = by_token[token]
        chunks.append("\t".join([
            token, str(row["p1_count"]), str(row["p2_count"]), str(row["total_count"]),
            str(row["selection_score"]), row["pure_motif_token_sha256"],
        ]).encode("utf-8") + b"\n")
    return b"".join(chunks)


def validate_samples(
    path: Path | None,
    binding_by_digest: dict[str, dict[str, Any]],
    anchor_max: int,
) -> dict[str, Any]:
    if path is None:
        return {"requested": False, "record_count": 0, "derived_component_boundaries_sha256": None}
    seen_members = set()
    boundaries_payload = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != SAMPLE_FIELDS or row.get("schema_version") != SAMPLE_SCHEMA:
                raise ValidationError("sample row fields/schema mismatch")
            member = row["member_id"]
            if not isinstance(member, str) or not member or member in seen_members:
                raise ValidationError("sample member ID is empty or duplicate")
            seen_members.add(member)
            require_hash(row["record_content_sha256"], "sample record content hash")
            count = row["motif_count"]
            digests = row["motif_lexeme_sha256"]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or not isinstance(digests, list):
                raise ValidationError("sample motif count/digest sequence is invalid")
            if not (count == len(digests) == row["motif_atom_indices_count"] == row["motif_geometry_valid_count"]):
                raise ValidationError("sample motif cardinalities disagree")
            anchor_occurrences = defaultdict(list)
            for ordinal, digest in enumerate(digests):
                require_hash(digest, "sample motif digest")
                binding = binding_by_digest.get(digest)
                if binding is None:
                    raise ValidationError("sample digest is absent from binding table")
                for anchor_id in binding["anchor_ids"]:
                    if anchor_id > anchor_max:
                        raise ValidationError("sample anchor exceeds frozen reserve")
                    anchor_occurrences[anchor_id].append(ordinal)
            adjacency = [set() for _ in range(count)]
            for anchor_id, ordinals in anchor_occurrences.items():
                if len(ordinals) != 2 or ordinals[0] == ordinals[1]:
                    raise ValidationError("anchor {} does not occur twice in distinct motifs".format(anchor_id))
                left, right = ordinals
                adjacency[left].add(right)
                adjacency[right].add(left)
            unseen = set(range(count))
            components = []
            while unseen:
                root = min(unseen)
                stack = [root]
                component = []
                unseen.remove(root)
                while stack:
                    node = stack.pop()
                    component.append(node)
                    for neighbor in sorted(adjacency[node], reverse=True):
                        if neighbor in unseen:
                            unseen.remove(neighbor)
                            stack.append(neighbor)
                component.sort()
                if component != list(range(component[0], component[-1] + 1)):
                    raise ValidationError("anchor-derived component is non-contiguous in motif order")
                components.append(component)
            components.sort(key=lambda component: component[0])
            break_after = [component[-1] for component in components[:-1]]
            token_ids = [binding_by_digest[digest]["token_id"] for digest in digests]
            boundaries_payload.append({
                "member_id": member,
                "component_break_after_motif_ordinals": break_after,
                "motif_token_ids_sha256": sha256_json(token_ids),
            })
    return {
        "requested": True,
        "record_count": len(seen_members),
        "derived_component_boundaries_sha256": sha256_json(boundaries_payload),
    }


def validate_sample_receipt(
    receipt_path: Path | None,
    sample_path: Path | None,
    logical_root: str,
    production_manifest_sha256: str,
) -> dict[str, Any]:
    if sample_path is None:
        if receipt_path is not None:
            raise ValidationError("sample extraction receipt is forbidden without sample digest sequences")
        return {"requested": False, "receipt_sha256": None, "sample_jsonl_sha256": None}
    if receipt_path is None:
        raise ValidationError("sample digest sequences require a provenance extraction receipt")
    receipt = load_json(receipt_path, "sample extraction receipt")
    if set(receipt) != SAMPLE_RECEIPT_FIELDS or receipt.get("schema_version") != SAMPLE_RECEIPT_SCHEMA:
        raise ValidationError("sample extraction receipt fields/schema mismatch")
    if receipt["status"] != "pass":
        raise ValidationError("sample extraction receipt did not pass")
    if receipt["production_logical_release_root_sha256"] != logical_root:
        raise ValidationError("sample extraction receipt binds a different production logical root")
    if receipt["production_manifest_sha256"] != production_manifest_sha256:
        raise ValidationError("sample extraction receipt binds a different production manifest")
    for field in (
        "production_logical_release_root_sha256", "production_manifest_sha256",
        "sample_schedule_sha256", "sample_jsonl_sha256", "safe_payload_decoder_sha256",
        "payload_index_verification_report_sha256", "component_reference_audit_sha256",
    ):
        require_hash(receipt[field], "sample receipt {}".format(field))
    if receipt["sample_jsonl_sha256"] != sha256_file(sample_path):
        raise ValidationError("sample JSONL hash differs from extraction receipt")
    if isinstance(receipt["sample_record_count"], bool) or not isinstance(receipt["sample_record_count"], int) or receipt["sample_record_count"] <= 0:
        raise ValidationError("sample extraction receipt count is invalid")
    return {
        "requested": True,
        "receipt_sha256": sha256_file(receipt_path),
        "sample_jsonl_sha256": receipt["sample_jsonl_sha256"],
        "sample_record_count": receipt["sample_record_count"],
        "sample_schedule_sha256": receipt["sample_schedule_sha256"],
        "safe_payload_decoder_sha256": receipt["safe_payload_decoder_sha256"],
        "payload_index_verification_report_sha256": receipt["payload_index_verification_report_sha256"],
        "component_reference_audit_sha256": receipt["component_reference_audit_sha256"],
    }


def dependency_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_report_new(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--production-release-root", required=True)
    parser.add_argument("--base-snapshot", required=True)
    parser.add_argument("--base-snapshot-lock", required=True)
    parser.add_argument("--selection-policy", required=True)
    parser.add_argument("--p1-scope-lock", required=True)
    parser.add_argument("--p1-census", required=True)
    parser.add_argument("--p2-scope-lock")
    parser.add_argument("--p2-census")
    parser.add_argument("--sample-digest-sequences")
    parser.add_argument("--sample-extraction-receipt")
    parser.add_argument("--require-sample-count", type=int)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    release = Path(args.release_dir).expanduser().resolve()
    manifest_path = release / "tokenizer_release_manifest.json"
    manifest = load_json(manifest_path, "tokenizer release manifest")
    if set(manifest) != MANIFEST_FIELDS or manifest.get("schema_version") != RELEASE_SCHEMA:
        raise ValidationError("tokenizer release manifest schema mismatch")
    payload_hash = manifest.get("manifest_payload_sha256")
    require_hash(payload_hash, "manifest payload hash")
    unhashed = dict(manifest)
    del unhashed["manifest_payload_sha256"]
    if sha256_json(unhashed) != payload_hash:
        raise ValidationError("tokenizer release manifest self-hash mismatch")
    if manifest.get("p1_training_admission") is not False or manifest.get("p1_training_launcher_permitted") is not False:
        raise ValidationError("tokenizer release illegally claims P1 admission")
    if manifest.get("p1_p2_exact_same_mapping") is not True or manifest.get("p2_vocab_extension_forbidden") is not True:
        raise ValidationError("P1/P2 mapping boundary is not frozen")
    if manifest.get("independent_validation_status") != "pending":
        raise ValidationError("immutable builder manifest must remain pending independent validation")

    contract_path = Path(args.contract).resolve()
    contract = load_json(contract_path, "binding contract")
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValidationError("binding contract schema mismatch")
    if manifest["contract"]["sha256"] != sha256_file(contract_path):
        raise ValidationError("release does not bind the supplied contract")
    builder_hash = require_hash(manifest.get("builder", {}).get("sha256"), "builder hash")

    production_root = Path(args.production_release_root).expanduser().resolve()
    production_manifest_path = production_root / "full_release_manifest.json"
    production = load_json(production_manifest_path, "production manifest")
    if production.get("schema_version") != PRODUCTION_SCHEMA or production.get("release_status") != "complete":
        raise ValidationError("production release is not complete v2")
    if sha256_file(production_manifest_path) != manifest["production_release"]["manifest_sha256"]:
        raise ValidationError("production manifest hash differs from tokenizer release")
    logical_root = require_hash(production.get("logical_release_root_sha256"), "production logical root")
    if logical_root != manifest["production_release"]["logical_release_root_sha256"]:
        raise ValidationError("production logical root differs from tokenizer release")
    production_census = check_artifact(production_root, production["global_motif_census"], "motif_census.jsonl")

    base_snapshot = Path(args.base_snapshot).expanduser().resolve()
    base_lock_path = Path(args.base_snapshot_lock).resolve()
    base_lock = load_json(base_lock_path, "base snapshot lock")
    if set(base_lock) != BASE_LOCK_FIELDS or base_lock.get("schema_version") != BASE_LOCK_SCHEMA:
        raise ValidationError("base snapshot lock fields/schema mismatch")
    if base_lock["decision_status"] not in {"approved_for_candidate", "approved_for_frozen_release"}:
        raise ValidationError("base snapshot decision is unresolved")
    if base_lock["tokenizer_and_model_same_revision"] is not True:
        raise ValidationError("base tokenizer/model revision equality is not locked")
    for field in ("model_identifier", "revision", "expected_tokenizer_class"):
        if not isinstance(base_lock[field], str) or not base_lock[field]:
            raise ValidationError("base snapshot {} is empty".format(field))
    base_tree = observe_tree(base_snapshot)
    if base_tree["tree_sha256"] != base_lock["snapshot_tree_sha256"] or base_tree["files"] != base_lock["files"]:
        raise ValidationError("base snapshot differs from its lock")
    if sha256_file(base_lock_path) != manifest["base_snapshot_lock"]["sha256"]:
        raise ValidationError("base snapshot lock hash differs from release")

    policy_path = Path(args.selection_policy).resolve()
    policy = load_json(policy_path, "selection policy")
    validate_policy(policy)
    if sha256_file(policy_path) != manifest["selection_policy"]["sha256"] or policy != manifest["selection_policy"]["decisions"]:
        raise ValidationError("selection policy differs from tokenizer release")
    if policy["base_model"] != {"identifier": base_lock["model_identifier"], "revision": base_lock["revision"]}:
        raise ValidationError("selection policy base model differs from snapshot lock")
    expected_release_status = (
        "frozen_tokenizer_built_non_admission"
        if policy["decision_status"] == "approved_for_frozen_release"
        else "candidate_tokenizer_built_non_release"
    )
    if manifest["release_status"] != expected_release_status:
        raise ValidationError("release status differs from policy decision status")

    p1_census_path = Path(args.p1_census).expanduser().resolve()
    p1_scope_path = Path(args.p1_scope_lock).resolve()
    p1_scope = load_json(p1_scope_path, "P1 scope lock")
    validate_scope(p1_scope, "P1", p1_census_path, logical_root)
    if sha256_file(p1_scope_path) != manifest["scope_locks"]["p1"]["sha256"]:
        raise ValidationError("P1 scope lock differs from tokenizer release")
    if p1_scope["census_kind"] == "production_global_admitted_candidate":
        if p1_census_path != production_census or sha256_file(p1_census_path) != production["global_motif_census"]["sha256"]:
            raise ValidationError("P1 candidate census is not the production manifest census")
    p2_scope = None
    p2_census_path = None
    if policy["discovery_scope"] == "p1_only":
        if args.p2_scope_lock or args.p2_census or manifest["scope_locks"]["p2"] is not None:
            raise ValidationError("P2 input exists under P1-only policy")
    else:
        if not args.p2_scope_lock or not args.p2_census or manifest["scope_locks"]["p2"] is None:
            raise ValidationError("P2 union inputs are incomplete")
        p2_scope_path = Path(args.p2_scope_lock).resolve()
        p2_census_path = Path(args.p2_census).expanduser().resolve()
        p2_scope = load_json(p2_scope_path, "P2 scope lock")
        validate_scope(p2_scope, "P2", p2_census_path, logical_root)
        if sha256_file(p2_scope_path) != manifest["scope_locks"]["p2"]["sha256"]:
            raise ValidationError("P2 scope lock differs from tokenizer release")
    if policy["decision_status"] == "approved_for_frozen_release":
        if base_lock["decision_status"] != "approved_for_frozen_release":
            raise ValidationError("frozen release uses a candidate-only base snapshot lock")
        if p1_scope["scope_status"] != "complete" or (p2_scope is not None and p2_scope["scope_status"] != "complete"):
            raise ValidationError("frozen release uses an incomplete discovery scope")

    p1 = read_census(p1_census_path, p1_scope, "P1")
    p2 = {} if p2_census_path is None else read_census(p2_census_path, p2_scope, "P2")
    expected_bindings, expected_pure, selected, max_anchor = recompute_rows(p1, p2, policy)
    selected_set = set(selected)
    expected_statistics = {
        "exact_lexeme_count": len(expected_bindings),
        "pure_motif_count": len(expected_pure),
        "eligible_pure_motif_count": sum(1 for row in expected_pure if row["eligible"]),
        "selected_pure_motif_count": len(selected),
        "observed_max_anchor_id": max_anchor,
        "p1_occurrence_count": sum(row["p1_count"] for row in expected_pure),
        "p1_selected_occurrence_count": sum(row["p1_count"] for row in expected_pure if row["pure_motif_token"] in selected_set),
        "p2_occurrence_count": sum(row["p2_count"] for row in expected_pure),
        "p2_selected_occurrence_count": sum(row["p2_count"] for row in expected_pure if row["pure_motif_token"] in selected_set),
    }
    expected_statistics["p1_oov_occurrence_count"] = expected_statistics["p1_occurrence_count"] - expected_statistics["p1_selected_occurrence_count"]
    expected_statistics["p2_oov_occurrence_count"] = expected_statistics["p2_occurrence_count"] - expected_statistics["p2_selected_occurrence_count"]
    if manifest.get("statistics") != expected_statistics:
        raise ValidationError("manifest statistics differ from independent census aggregation")
    expected_projection_hash = sha256_json({
        "anchor_regex": ANCHOR_RE.pattern,
        "anchor_decimal_policy": "canonical_no_leading_zero",
        "pure_motif_formula": "'[' + anchor_regex.sub('', exact_lexeme) + ']'",
        "normalization": "none",
    })
    if manifest.get("projection") != {
        "spec_id": "most-t5-r1/motif-lexeme-projection/v1",
        "spec_sha256": expected_projection_hash,
    }:
        raise ValidationError("projection spec binding differs from independent implementation")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "pure_motif_census", "ordered_motif_vocab", "motif_digest_binding",
        "id_to_token", "token_to_id", "tokenizer_snapshot",
    }:
        raise ValidationError("release artifact set is not closed")
    pure_path = check_artifact(release, artifacts["pure_motif_census"], "pure_motif_census.jsonl")
    ordered_path = check_artifact(release, artifacts["ordered_motif_vocab"], "ordered_motif_vocab.tsv")
    binding_path = check_artifact(release, artifacts["motif_digest_binding"], "motif_digest_binding.jsonl")
    id_path = check_artifact(release, artifacts["id_to_token"], "id_to_token.json")
    token_path = check_artifact(release, artifacts["token_to_id"], "token_to_id.json")
    compare_jsonl(pure_path, expected_pure, "pure motif census")
    pure_by_token = {row["pure_motif_token"]: row for row in expected_pure}
    with ordered_path.open("rb") as handle:
        if handle.read() != expected_vocab_bytes(selected, pure_by_token):
            raise ValidationError("ordered motif vocabulary differs from independent selection")

    os.environ.update({
        "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
    })
    from transformers import AutoTokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(
        str(base_snapshot), use_fast=False, local_files_only=True, trust_remote_code=False
    )
    if base_tokenizer.__class__.__name__ != base_lock["expected_tokenizer_class"]:
        raise ValidationError("base tokenizer class differs from lock")
    base_vocab = {str(token): int(token_id) for token, token_id in base_tokenizer.get_vocab().items()}
    saved_snapshot = release / "tokenizer_snapshot"
    observed_saved_tree = observe_tree(saved_snapshot)
    if observed_saved_tree != artifacts["tokenizer_snapshot"]:
        raise ValidationError("saved tokenizer tree differs from release manifest")
    saved = AutoTokenizer.from_pretrained(
        str(saved_snapshot), use_fast=False, local_files_only=True, trust_remote_code=False
    )
    token_to_id_map = {str(token): int(token_id) for token, token_id in saved.get_vocab().items()}
    id_to_token_map = id_to_token(token_to_id_map)
    with id_path.open("r", encoding="utf-8") as handle:
        if json.load(handle) != id_to_token_map:
            raise ValidationError("id_to_token artifact differs from saved tokenizer")
    with token_path.open("r", encoding="utf-8") as handle:
        if json.load(handle) != token_to_id_map:
            raise ValidationError("token_to_id artifact differs from saved tokenizer")
    mapping_manifest = manifest["tokenizer_mapping"]
    if not isinstance(mapping_manifest, dict) or set(mapping_manifest) != MAPPING_FIELDS:
        raise ValidationError("tokenizer mapping manifest fields are not closed")
    if mapping_manifest["vocab_size"] != len(id_to_token_map):
        raise ValidationError("vocabulary size differs from release manifest")
    mapping_hashes = {
        "id_to_token_sha256": sha256_json(id_to_token_map),
        "token_to_id_sha256": sha256_json(token_to_id_map),
        "added_token_metadata_sha256": sha256_json(added_metadata(saved)),
    }
    for field, value in mapping_hashes.items():
        if mapping_manifest[field] != value:
            raise ValidationError("{} differs from saved tokenizer".format(field))
    if any(token not in set(saved.all_special_tokens or []) for token in T5_SENTINELS):
        raise ValidationError("saved tokenizer lacks a T5 sentinel")
    if mapping_manifest["sentinel_token_id_map"] != {token: int(saved.convert_tokens_to_ids(token)) for token in T5_SENTINELS}:
        raise ValidationError("sentinel token-ID map differs from saved tokenizer")
    anchor_tokens = ["<{}*>".format(index) for index in range(policy["anchor_policy"]["max_anchor_id_inclusive"] + 1)]
    if mapping_manifest["anchor_token_id_map"] != {token: int(saved.convert_tokens_to_ids(token)) for token in anchor_tokens}:
        raise ValidationError("anchor token-ID map differs from saved tokenizer")
    base_specials = [str(token) for token in (base_tokenizer.additional_special_tokens or [])]
    optional_oov = [policy["oov_policy"]["token"]] if policy["oov_policy"]["kind"] == "dedicated_motif_unk" else []
    expected_special_order = stable_unique(
        base_specials + list(DIGITS) + list(TASKS) + list(STRUCTURAL) + optional_oov
        + ["[RESERVED_{}]".format(index) for index in range(policy["reserved_special_token_count"])]
        + anchor_tokens
    )
    if mapping_manifest.get("special_token_order") != expected_special_order:
        raise ValidationError("special-token order differs from contract")
    if mapping_manifest.get("special_token_order_sha256") != sha256_json(expected_special_order):
        raise ValidationError("special-token order hash differs from contract")
    if mapping_manifest["special_token_id_map"] != {token: int(saved.convert_tokens_to_ids(token)) for token in expected_special_order}:
        raise ValidationError("special token-ID map differs from saved tokenizer")
    if [str(token) for token in (saved.additional_special_tokens or [])] != expected_special_order:
        raise ValidationError("saved tokenizer additional-special order differs from contract")
    overlap = sorted((token for token in selected if token in base_vocab), key=lambda token: token.encode("utf-8"))
    if overlap != policy["base_vocab_overlap_allowlist"]:
        raise ValidationError("selected/base vocabulary overlap differs from allow-list")
    selected_id_map = {token: int(saved.convert_tokens_to_ids(token)) for token in selected}
    if sha256_json(selected_id_map) != mapping_manifest["selected_motif_token_id_map_sha256"]:
        raise ValidationError("selected motif token-ID map hash differs")
    oov_token = policy["oov_policy"]["token"]
    oov_id = int(saved.convert_tokens_to_ids(oov_token))
    if policy["oov_policy"]["kind"] == "base_unk" and oov_token != str(base_tokenizer.unk_token):
        raise ValidationError("base OOV token differs from base tokenizer unk token")
    if mapping_manifest["oov_token"] != oov_token or mapping_manifest["oov_token_id"] != oov_id:
        raise ValidationError("OOV binding differs from saved tokenizer")

    for row in expected_bindings:
        chosen = row["pure_motif_token"] in selected_id_map
        pure = pure_by_token[row["pure_motif_token"]]
        row["selection_score"] = pure["selection_score"]
        row["selected"] = chosen
        row["token_id"] = selected_id_map[row["pure_motif_token"]] if chosen else oov_id
        row["binding_disposition"] = "selected_motif_token" if chosen else "frozen_oov"
    compare_jsonl(binding_path, expected_bindings, "motif digest binding")
    binding_by_digest = {row["motif_lexeme_sha256"]: row for row in expected_bindings}

    gate = manifest.get("determinism_gate")
    if not isinstance(gate, dict) or gate.get("pass") is not True:
        raise ValidationError("builder determinism gate did not pass")
    probes = gate.get("probes")
    if not isinstance(probes, list) or len(probes) < 3 or len({str(row.get("pythonhashseed")) for row in probes}) != len(probes):
        raise ValidationError("fewer than three distinct determinism probes")
    expected_probe = {
        "id_to_token_sha256": mapping_hashes["id_to_token_sha256"],
        "token_to_id_sha256": mapping_hashes["token_to_id_sha256"],
        "added_token_metadata_sha256": mapping_hashes["added_token_metadata_sha256"],
        "special_token_order_sha256": sha256_json(expected_special_order),
        "special_token_id_map_sha256": sha256_json(mapping_manifest["special_token_id_map"]),
        "sentinel_token_id_map_sha256": sha256_json(mapping_manifest["sentinel_token_id_map"]),
        "anchor_token_id_map_sha256": sha256_json(mapping_manifest["anchor_token_id_map"]),
        "vocab_size": len(id_to_token_map),
    }
    if any(row.get("mapping") != expected_probe for row in probes):
        raise ValidationError("determinism probe mapping differs from saved tokenizer")

    sample_path = None if not args.sample_digest_sequences else Path(args.sample_digest_sequences).resolve()
    sample_receipt_path = None if not args.sample_extraction_receipt else Path(args.sample_extraction_receipt).resolve()
    sample_receipt = validate_sample_receipt(
        sample_receipt_path,
        sample_path,
        logical_root,
        sha256_file(production_manifest_path),
    )
    sample_result = validate_samples(sample_path, binding_by_digest, policy["anchor_policy"]["max_anchor_id_inclusive"])
    if sample_receipt["requested"] and sample_result["record_count"] != sample_receipt["sample_record_count"]:
        raise ValidationError("sample row count differs from extraction receipt")
    if args.require_sample_count is not None and sample_result["record_count"] != args.require_sample_count:
        raise ValidationError("validated sample count differs from --require-sample-count")

    report = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass",
        "release_id": manifest["release_id"],
        "release_manifest_sha256": sha256_file(manifest_path),
        "release_manifest_payload_sha256": payload_hash,
        "contract_sha256": sha256_file(contract_path),
        "builder_sha256": builder_hash,
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "production_logical_release_root_sha256": logical_root,
        "base_snapshot_tree_sha256": base_tree["tree_sha256"],
        "selection_policy_sha256": sha256_file(policy_path),
        "counts": {
            "exact_lexeme_bindings": len(expected_bindings),
            "pure_motifs": len(expected_pure),
            "selected_motifs": len(selected),
            "vocab_size": len(id_to_token_map),
            "observed_max_anchor_id": max_anchor,
        },
        "sample_digest_sequence_validation": sample_result,
        "sample_extraction_receipt": sample_receipt,
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "transformers": dependency_version("transformers"),
            "tokenizers": dependency_version("tokenizers"),
            "sentencepiece": dependency_version("sentencepiece"),
        },
        "p1_training_admission": False,
        "p1_training_launcher_permitted": False,
        "next_gate": "This validation proves deterministic tokenizer/digest binding only; overlap, semantic review, Dataset/Collator, GPU backward, and admission gates remain separate.",
    }
    report["report_payload_sha256"] = sha256_json(report)
    output_report = Path(args.output_report).expanduser().resolve()
    write_report_new(output_report, report)
    print(json.dumps({
        "status": "pass", "report": str(output_report),
        "release_id": manifest["release_id"], "selected_motifs": len(selected),
        "sample_records": sample_result["record_count"], "p1_training_admission": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
