#!/usr/bin/env python3
"""Build a deterministic motif-tokenizer and digest-binding release.

The builder consumes only completed/candidate motif census artifacts and
explicit, hash-locked decisions.  It never reads SDF, imports RDKit, invokes a
linearizer, or computes E3FP.  Final token selection is intentionally blocked
until the caller supplies the scientific choices in a versioned policy.

An output directory is created once and is never overwritten.  The final
manifest is written last; a directory without it is an incomplete attempt and
must not be relabelled as a release.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Iterable


CONTRACT_SCHEMA = "most-t5-r1/motif-tokenizer-binding-release-contract/v1"
PRODUCTION_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-full-release/v2"
BASE_LOCK_SCHEMA = "most-t5-r1/base-model-snapshot-lock/v1"
SCOPE_LOCK_SCHEMA = "most-t5-r1/tokenizer-discovery-scope-lock/v1"
POLICY_SCHEMA = "most-t5-r1/motif-token-selection-policy/v1"
RELEASE_SCHEMA = "most-t5-r1/motif-tokenizer-binding-release/v1"
PROJECTION_SPEC_ID = "most-t5-r1/motif-lexeme-projection/v1"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ANCHOR_RE = re.compile(r"<([0-9]+)\*>")
ANCHOR_LIKE_RE = re.compile(r"<[^>]*\*>")
INT64_MAX = (1 << 63) - 1

DIGIT_SPECIAL_TOKENS = tuple(str(value) for value in range(10))
TASK_SPECIAL_TOKENS = (
    "<bom>",
    "<eom>",
    "[MMM]:",
    "[Caption]:",
    "[Text2Mol]:",
    "[Denoise]:",
)
STRUCTURAL_SPECIAL_TOKENS = ("[.]",)
T5_SENTINELS = tuple("<extra_id_{}>".format(index) for index in range(100))

SCOPE_FIELDS = {
    "schema_version",
    "phase",
    "scope_status",
    "identity_namespace",
    "membership_manifest_sha256",
    "membership_count",
    "downstream_identity_exclusion_proof_sha256",
    "census_sha256",
    "census_unique_lexeme_count",
    "census_occurrence_count",
    "census_kind",
    "census_derivation_audit_sha256",
    "source_release_logical_root_sha256",
    "motif_linearization_spec_sha256",
    "motif_sequence_extraction_spec_sha256",
    "projection_domain_compatibility_audit_sha256",
}
POLICY_FIELDS = {
    "schema_version",
    "decision_status",
    "discovery_scope",
    "min_selection_score",
    "max_motif_tokens",
    "selection_score",
    "oov_policy",
    "anchor_policy",
    "reserved_special_token_count",
    "base_model",
    "base_vocab_overlap_allowlist",
    "tie_break",
    "p2_vocab_extension_forbidden",
}
BASE_LOCK_FIELDS = {
    "schema_version",
    "decision_status",
    "model_identifier",
    "revision",
    "expected_tokenizer_class",
    "tokenizer_and_model_same_revision",
    "snapshot_tree_sha256",
    "files",
}


class ContractError(RuntimeError):
    """Raised when an immutable input or invariant is not satisfied."""


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


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ContractError("{} must be a lowercase SHA-256 hex string".format(label))
    return value


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContractError("{} must be a regular, non-symlink file: {}".format(label, path))
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError("{} must contain a JSON object".format(label))
    return value


def write_json_new(path: Path, value: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        if pretty:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        else:
            handle.write(canonical_json_bytes(value).decode("utf-8"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def observe_tree(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ContractError("snapshot must be a regular local directory without symlink indirection")
    rows: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if candidate.is_symlink():
            raise ContractError("snapshot symlink is forbidden: {}".format(candidate))
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ContractError("snapshot special file is forbidden: {}".format(candidate))
        rows.append(
            {
                "relative_path": candidate.relative_to(root).as_posix(),
                "bytes": int(candidate.stat().st_size),
                "sha256": sha256_file(candidate),
            }
        )
    if not rows:
        raise ContractError("base snapshot contains no regular files")
    return {"files": rows, "file_count": len(rows), "tree_sha256": sha256_json(rows)}


def observe_artifact(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContractError("release artifact is absent or not a regular file: {}".format(path))
    return {
        "relative_path": relative_path,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _strict_fields(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ContractError(
            "{} fields are not closed; missing={}, extra={}".format(
                label, sorted(expected - set(value)), sorted(set(value) - expected)
            )
        )


def validate_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = load_json(path, "binding contract")
    if value.get("schema_version") != CONTRACT_SCHEMA:
        raise ContractError("binding contract schema mismatch")
    return value, observe_artifact(path, path.name)


def validate_base_lock(path: Path, snapshot: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = load_json(path, "base snapshot lock")
    _strict_fields(lock, BASE_LOCK_FIELDS, "base snapshot lock")
    if lock["schema_version"] != BASE_LOCK_SCHEMA:
        raise ContractError("base snapshot lock schema mismatch")
    if lock["decision_status"] not in {"approved_for_candidate", "approved_for_frozen_release"}:
        raise ContractError("base snapshot decision_status is unresolved")
    for field in ("model_identifier", "revision", "expected_tokenizer_class"):
        if not isinstance(lock[field], str) or not lock[field]:
            raise ContractError("base snapshot lock {} must be non-empty".format(field))
    if lock["tokenizer_and_model_same_revision"] is not True:
        raise ContractError("base tokenizer and model weights must be locked to the same revision")
    require_sha256(lock["snapshot_tree_sha256"], "base snapshot tree hash")
    observed = observe_tree(snapshot)
    if lock["snapshot_tree_sha256"] != observed["tree_sha256"]:
        raise ContractError("base snapshot tree SHA-256 mismatch")
    if lock["files"] != observed["files"]:
        raise ContractError("base snapshot per-file observation mismatch")
    return lock, {"sha256": sha256_file(path), "bytes": int(path.stat().st_size)}


def validate_production_release(root: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest_path = root / "full_release_manifest.json"
    manifest = load_json(manifest_path, "production full-release manifest")
    if manifest.get("schema_version") != PRODUCTION_SCHEMA or manifest.get("release_status") != "complete":
        raise ContractError("production release is not a complete v2 release")
    if manifest.get("tokenizer_binding") != "absent_and_forbidden":
        raise ContractError("production release tokenizer boundary is not the locked v2 value")
    if manifest.get("p1_training_admission") is not False:
        raise ContractError("production pretokenizer manifest unexpectedly claims P1 admission")
    require_sha256(manifest.get("logical_release_root_sha256"), "production logical root")
    artifact = manifest.get("global_motif_census")
    if not isinstance(artifact, dict) or set(artifact) != {"relative_path", "bytes", "sha256"}:
        raise ContractError("production global motif census observation is malformed")
    if artifact["relative_path"] != "motif_census.jsonl":
        raise ContractError("production global motif census relative path is not canonical")
    census_path = root / artifact["relative_path"]
    observed = observe_artifact(census_path, artifact["relative_path"])
    if observed != artifact:
        raise ContractError("production global motif census bytes or SHA-256 mismatch")
    return manifest, census_path, {
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_bytes": int(manifest_path.stat().st_size),
        "logical_release_root_sha256": manifest["logical_release_root_sha256"],
        "release_id": manifest.get("release_id"),
        "global_motif_census": observed,
    }


def validate_scope_lock(
    path: Path,
    phase: str,
    census_path: Path,
    production_logical_root: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = load_json(path, "{} discovery scope lock".format(phase))
    _strict_fields(lock, SCOPE_FIELDS, "{} discovery scope lock".format(phase))
    if lock["schema_version"] != SCOPE_LOCK_SCHEMA or lock["phase"] != phase:
        raise ContractError("{} discovery scope schema/phase mismatch".format(phase))
    if lock["scope_status"] not in {"candidate", "complete"}:
        raise ContractError("{} discovery scope status is unresolved".format(phase))
    if not isinstance(lock["identity_namespace"], str) or not lock["identity_namespace"]:
        raise ContractError("{} identity namespace must be non-empty".format(phase))
    if isinstance(lock["membership_count"], bool) or not isinstance(lock["membership_count"], int) or lock["membership_count"] <= 0:
        raise ContractError("{} membership_count must be a positive integer".format(phase))
    for field in (
        "membership_manifest_sha256", "census_sha256", "source_release_logical_root_sha256",
        "motif_linearization_spec_sha256", "motif_sequence_extraction_spec_sha256",
    ):
        require_sha256(lock[field], "{} {}".format(phase, field))
    if lock["source_release_logical_root_sha256"] != production_logical_root and phase == "P1":
        raise ContractError("P1 scope lock does not bind the supplied production release")
    if lock["census_sha256"] != sha256_file(census_path):
        raise ContractError("{} scope-lock census SHA-256 mismatch".format(phase))
    for field in ("census_unique_lexeme_count", "census_occurrence_count"):
        value = lock[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContractError("{} {} must be a positive integer".format(phase, field))
    if lock["census_kind"] not in {"production_global_admitted_candidate", "permitted_membership_derived"}:
        raise ContractError("{} census_kind is unsupported".format(phase))
    if lock["scope_status"] == "complete":
        if lock["census_kind"] != "permitted_membership_derived":
            raise ContractError("a complete discovery scope requires a permitted-membership-derived census")
        require_sha256(lock["downstream_identity_exclusion_proof_sha256"], "downstream exclusion proof")
        require_sha256(lock["census_derivation_audit_sha256"], "census derivation audit")
        if phase == "P2":
            require_sha256(
                lock["projection_domain_compatibility_audit_sha256"],
                "P1/P2 projection-domain compatibility audit",
            )
    else:
        for field in (
            "downstream_identity_exclusion_proof_sha256", "census_derivation_audit_sha256",
            "projection_domain_compatibility_audit_sha256",
        ):
            value = lock[field]
            if value is not None:
                require_sha256(value, "candidate {}".format(field))
    if lock["scope_status"] == "complete" and phase == "P1":
        value = lock["projection_domain_compatibility_audit_sha256"]
        if value is not None:
            require_sha256(value, "P1 projection-domain compatibility audit")
    return lock, {"sha256": sha256_file(path), "bytes": int(path.stat().st_size)}


def validate_policy(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = load_json(path, "motif selection policy")
    _strict_fields(policy, POLICY_FIELDS, "motif selection policy")
    if policy["schema_version"] != POLICY_SCHEMA:
        raise ContractError("selection policy schema mismatch")
    if policy["decision_status"] not in {"approved_for_candidate", "approved_for_frozen_release"}:
        raise ContractError("selection policy decision_status is unresolved")
    if policy["discovery_scope"] not in {"p1_only", "p1_p2_permitted_train_union"}:
        raise ContractError("discovery_scope must be explicitly p1_only or p1_p2_permitted_train_union")
    minimum = policy["min_selection_score"]
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
        raise ContractError("min_selection_score must be a positive integer")
    cap = policy["max_motif_tokens"]
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0):
        raise ContractError("max_motif_tokens must be a positive integer or null")
    score = policy["selection_score"]
    if not isinstance(score, dict) or set(score) != {"kind", "p1_weight", "p2_weight"}:
        raise ContractError("selection_score fields are not closed")
    if score["kind"] != "weighted_integer_count":
        raise ContractError("only exact weighted_integer_count selection is supported")
    for phase in ("p1", "p2"):
        weight = score["{}_weight".format(phase)]
        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 0:
            raise ContractError("{} weight must be a non-negative integer".format(phase))
    if score["p1_weight"] <= 0:
        raise ContractError("P1 weight must be positive")
    if policy["discovery_scope"] == "p1_only" and score["p2_weight"] != 0:
        raise ContractError("P1-only discovery requires p2_weight=0")
    if policy["discovery_scope"] == "p1_p2_permitted_train_union" and score["p2_weight"] <= 0:
        raise ContractError("P1+P2 union discovery requires a positive p2_weight")
    oov = policy["oov_policy"]
    if not isinstance(oov, dict) or set(oov) != {"kind", "token"}:
        raise ContractError("oov_policy fields are not closed")
    if oov["kind"] not in {"base_unk", "dedicated_motif_unk"}:
        raise ContractError("OOV policy is unresolved")
    if not isinstance(oov["token"], str) or not oov["token"] or any(c in oov["token"] for c in "\x00\t\r\n"):
        raise ContractError("OOV token is malformed")
    anchor = policy["anchor_policy"]
    if not isinstance(anchor, dict) or set(anchor) != {"max_anchor_id_inclusive", "overflow_action"}:
        raise ContractError("anchor_policy fields are not closed")
    maximum = anchor["max_anchor_id_inclusive"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise ContractError("max_anchor_id_inclusive must be a non-negative integer")
    if anchor["overflow_action"] != "fail_closed":
        raise ContractError("anchor overflow must fail closed")
    reserve = policy["reserved_special_token_count"]
    if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
        raise ContractError("reserved_special_token_count must be a non-negative integer")
    base = policy["base_model"]
    if not isinstance(base, dict) or set(base) != {"identifier", "revision"}:
        raise ContractError("base_model fields are not closed")
    if not all(isinstance(base[field], str) and base[field] for field in ("identifier", "revision")):
        raise ContractError("base_model identifier and revision are mandatory")
    allowlist = policy["base_vocab_overlap_allowlist"]
    if not isinstance(allowlist, list) or any(not isinstance(token, str) or not token for token in allowlist):
        raise ContractError("base vocabulary overlap allow-list must be a string list")
    if len(allowlist) != len(dict.fromkeys(allowlist)) or allowlist != sorted(allowlist, key=lambda x: x.encode("utf-8")):
        raise ContractError("base vocabulary overlap allow-list must be unique and UTF-8 sorted")
    expected_tie = [
        "selection_score_desc",
        "pure_motif_utf8_asc",
        "pure_motif_sha256_asc",
    ]
    if policy["tie_break"] != expected_tie:
        raise ContractError("tie-break must equal the contract-defined total order")
    if policy["p2_vocab_extension_forbidden"] is not True:
        raise ContractError("P2 vocabulary extension must be explicitly forbidden")
    return policy, {"sha256": sha256_file(path), "bytes": int(path.stat().st_size)}


def read_census(path: Path, lock: dict[str, Any], phase: str) -> dict[str, tuple[str, int]]:
    rows: dict[str, tuple[str, int]] = {}
    previous_digest: str | None = None
    occurrence_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ContractError("{} census line {} lacks LF terminator".format(phase, line_number))
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {"motif_lexeme_sha256", "motif_fragment", "count"}:
                raise ContractError("{} census row {} fields are not closed".format(phase, line_number))
            digest = require_sha256(row["motif_lexeme_sha256"], "{} census digest".format(phase))
            fragment = row["motif_fragment"]
            count = row["count"]
            if previous_digest is not None and digest <= previous_digest:
                raise ContractError("{} census digest order is not strictly ascending".format(phase))
            previous_digest = digest
            if not isinstance(fragment, str) or not fragment or any(c in fragment for c in "\x00\t\r\n"):
                raise ContractError("{} census fragment is empty or contains a forbidden character".format(phase))
            if sha256_bytes(fragment.encode("utf-8")) != digest:
                raise ContractError("{} census digest/fragment mismatch".format(phase))
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ContractError("{} census count must be a positive integer".format(phase))
            occurrence_count += count
            if occurrence_count > INT64_MAX:
                raise ContractError("{} census occurrence sum exceeds int64".format(phase))
            rows[digest] = (fragment, count)
    if len(rows) != lock["census_unique_lexeme_count"]:
        raise ContractError("{} census unique count differs from scope lock".format(phase))
    if occurrence_count != lock["census_occurrence_count"]:
        raise ContractError("{} census occurrence count differs from scope lock".format(phase))
    return rows


def project_lexeme(fragment: str) -> tuple[list[str], list[int], str]:
    matches = list(ANCHOR_RE.finditer(fragment))
    matched_strings = [match.group(0) for match in matches]
    anchor_like = ANCHOR_LIKE_RE.findall(fragment)
    if anchor_like != matched_strings:
        raise ContractError("malformed anchor-like substring in exact motif lexeme")
    anchor_ids: list[int] = []
    for match in matches:
        decimal = match.group(1)
        if len(decimal) > 1 and decimal.startswith("0"):
            raise ContractError("anchor IDs must use canonical decimal without leading zero")
        anchor_ids.append(int(decimal))
    core = ANCHOR_RE.sub("", fragment)
    if not core:
        raise ContractError("anchor removal produced an empty motif core")
    pure = "[{}]".format(core)
    if any(c in pure for c in "\x00\t\r\n"):
        raise ContractError("projected pure motif contains a forbidden character")
    return matched_strings, anchor_ids, pure


def aggregate_projection(
    p1: dict[str, tuple[str, int]],
    p2: dict[str, tuple[str, int]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, Any]]:
    all_digests = sorted(set(p1) | set(p2))
    pure_aggregate: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"p1_count": 0, "p2_count": 0, "lexeme_count": 0}
    )
    lexeme_rows: list[dict[str, Any]] = []
    max_anchor = -1
    for digest in all_digests:
        left = p1.get(digest)
        right = p2.get(digest)
        fragments = [entry[0] for entry in (left, right) if entry is not None]
        if not fragments or any(fragment != fragments[0] for fragment in fragments[1:]):
            raise ContractError("digest maps to distinct exact fragments across phase census inputs")
        fragment = fragments[0]
        anchors, anchor_ids, pure = project_lexeme(fragment)
        if anchor_ids:
            max_anchor = max(max_anchor, max(anchor_ids))
        p1_count = 0 if left is None else int(left[1])
        p2_count = 0 if right is None else int(right[1])
        aggregate = pure_aggregate[pure]
        aggregate["p1_count"] += p1_count
        aggregate["p2_count"] += p2_count
        aggregate["lexeme_count"] += 1
        lexeme_rows.append(
            {
                "motif_lexeme_sha256": digest,
                "motif_fragment": fragment,
                "anchors": anchors,
                "anchor_ids": anchor_ids,
                "pure_motif_token": pure,
                "pure_motif_token_sha256": sha256_bytes(pure.encode("utf-8")),
                "p1_count": p1_count,
                "p2_count": p2_count,
            }
        )

    weights = policy["selection_score"]
    pure_rows: list[dict[str, Any]] = []
    for pure, counts in pure_aggregate.items():
        p1_count = int(counts["p1_count"])
        p2_count = int(counts["p2_count"])
        score = p1_count * weights["p1_weight"] + p2_count * weights["p2_weight"]
        if score > INT64_MAX:
            raise ContractError("weighted selection score exceeds int64")
        pure_rows.append(
            {
                "pure_motif_token": pure,
                "pure_motif_token_sha256": sha256_bytes(pure.encode("utf-8")),
                "p1_count": p1_count,
                "p2_count": p2_count,
                "total_count": p1_count + p2_count,
                "selection_score": score,
                "exact_lexeme_count": int(counts["lexeme_count"]),
            }
        )
    ranked = sorted(
        (row for row in pure_rows if row["selection_score"] >= policy["min_selection_score"]),
        key=lambda row: (
            -row["selection_score"],
            row["pure_motif_token"].encode("utf-8"),
            row["pure_motif_token_sha256"],
        ),
    )
    cap = policy["max_motif_tokens"]
    selected_ranked = ranked if cap is None else ranked[:cap]
    selected_tokens = [row["pure_motif_token"] for row in selected_ranked]
    selected_rank = {token: index for index, token in enumerate(selected_tokens)}
    for row in pure_rows:
        rank = selected_rank.get(row["pure_motif_token"])
        row["eligible"] = row["selection_score"] >= policy["min_selection_score"]
        row["selected"] = rank is not None
        row["selection_rank"] = rank
    pure_rows.sort(key=lambda row: row["pure_motif_token"].encode("utf-8"))
    anchor_limit = policy["anchor_policy"]["max_anchor_id_inclusive"]
    if max_anchor > anchor_limit:
        raise ContractError(
            "observed anchor ID {} exceeds explicitly frozen maximum {}".format(max_anchor, anchor_limit)
        )
    selected_set = set(selected_tokens)
    p1_total = sum(row["p1_count"] for row in pure_rows)
    p2_total = sum(row["p2_count"] for row in pure_rows)
    p1_selected = sum(row["p1_count"] for row in pure_rows if row["pure_motif_token"] in selected_set)
    p2_selected = sum(row["p2_count"] for row in pure_rows if row["pure_motif_token"] in selected_set)
    stats = {
        "exact_lexeme_count": len(lexeme_rows),
        "pure_motif_count": len(pure_rows),
        "eligible_pure_motif_count": len(ranked),
        "selected_pure_motif_count": len(selected_tokens),
        "observed_max_anchor_id": max_anchor,
        "p1_occurrence_count": p1_total,
        "p1_selected_occurrence_count": p1_selected,
        "p1_oov_occurrence_count": p1_total - p1_selected,
        "p2_occurrence_count": p2_total,
        "p2_selected_occurrence_count": p2_selected,
        "p2_oov_occurrence_count": p2_total - p2_selected,
    }
    return lexeme_rows, pure_rows, selected_tokens, stats


def stable_unique(values: Iterable[str]) -> list[str]:
    observed: dict[str, bool] = {}
    result: list[str] = []
    for value in values:
        if value not in observed:
            observed[value] = True
            result.append(value)
    return result


def id_to_token_from_vocab(token_to_id: dict[str, int]) -> list[str]:
    if not token_to_id:
        raise ContractError("tokenizer returned an empty vocabulary")
    ids = [int(value) for value in token_to_id.values()]
    result: list[str | None] = [None] * (max(ids) + 1)
    for token, token_id_value in token_to_id.items():
        token_id = int(token_id_value)
        if result[token_id] is not None and result[token_id] != token:
            raise ContractError("token ID collision at {}".format(token_id))
        result[token_id] = token
    if any(token is None for token in result):
        raise ContractError("tokenizer ID domain is not contiguous")
    if len(ids) != len(set(ids)):
        raise ContractError("tokenizer vocabulary is not one-to-one")
    return [str(token) for token in result]


def added_token_metadata(tokenizer: Any) -> list[dict[str, Any]]:
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if decoder is None:
        raise ContractError("tokenizer lacks added_tokens_decoder")
    rows = []
    for token_id, token in sorted(decoder.items(), key=lambda pair: int(pair[0])):
        rows.append(
            {
                "id": int(token_id),
                "content": str(getattr(token, "content", token)),
                "single_word": bool(getattr(token, "single_word", False)),
                "lstrip": bool(getattr(token, "lstrip", False)),
                "rstrip": bool(getattr(token, "rstrip", False)),
                "normalized": bool(getattr(token, "normalized", True)),
                "special": bool(getattr(token, "special", False)),
            }
        )
    return rows


def _dependency_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_tokenizer_mapping(
    base_snapshot: Path,
    base_lock: dict[str, Any],
    policy: dict[str, Any],
    selected_tokens: list[str],
    *,
    save_directory: Path | None,
) -> tuple[Any, dict[str, Any]]:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from transformers import AddedToken, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_snapshot), use_fast=False, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.__class__.__name__ != base_lock["expected_tokenizer_class"]:
        raise ContractError(
            "base tokenizer class {} differs from locked {}".format(
                tokenizer.__class__.__name__, base_lock["expected_tokenizer_class"]
            )
        )
    base_vocab = {str(token): int(token_id) for token, token_id in tokenizer.get_vocab().items()}
    base_special_objects = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    base_special_tokens = [str(token) for token in base_special_objects]
    base_decoder_by_content = {
        str(getattr(token, "content", token)): token
        for token in (getattr(tokenizer, "added_tokens_decoder", {}) or {}).values()
    }
    base_special_by_content = {
        str(token): base_decoder_by_content.get(str(token), token) for token in base_special_objects
    }
    missing_sentinels = [token for token in T5_SENTINELS if token not in set(tokenizer.all_special_tokens or [])]
    if missing_sentinels:
        raise ContractError("base tokenizer lacks T5 sentinels; first missing={}".format(missing_sentinels[:5]))

    oov = policy["oov_policy"]
    optional_oov = [oov["token"]] if oov["kind"] == "dedicated_motif_unk" else []
    if oov["kind"] == "base_unk" and oov["token"] != str(tokenizer.unk_token):
        raise ContractError("base_unk policy token differs from the frozen base tokenizer unk token")
    if oov["kind"] == "dedicated_motif_unk" and oov["token"] in base_vocab:
        raise ContractError("dedicated motif OOV token already exists in base vocabulary")
    reserved = tuple(
        "[RESERVED_{}]".format(index) for index in range(policy["reserved_special_token_count"])
    )
    anchors = tuple(
        "<{}*>".format(index)
        for index in range(policy["anchor_policy"]["max_anchor_id_inclusive"] + 1)
    )
    declared_specials = stable_unique(
        base_special_tokens
        + list(DIGIT_SPECIAL_TOKENS)
        + list(TASK_SPECIAL_TOKENS)
        + list(STRUCTURAL_SPECIAL_TOKENS)
        + optional_oov
        + list(reserved)
        + list(anchors)
    )
    special_objects = [
        base_special_by_content.get(
            token,
            AddedToken(token, lstrip=False, rstrip=False, normalized=False, special=True),
        )
        for token in declared_specials
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_objects})
    all_special_ids = set(int(value) for value in tokenizer.all_special_ids)
    special_map = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in declared_specials}
    if any(token_id not in all_special_ids for token_id in special_map.values()):
        raise ContractError("one or more declared special tokens are not special after registration")
    special_set = set(declared_specials)
    special_overlap = [token for token in selected_tokens if token in special_set]
    if special_overlap:
        raise ContractError("selected motif overlaps special-token domain: {}".format(special_overlap[:5]))
    observed_base_overlaps = sorted(
        (token for token in selected_tokens if token in base_vocab), key=lambda token: token.encode("utf-8")
    )
    if observed_base_overlaps != policy["base_vocab_overlap_allowlist"]:
        raise ContractError(
            "selected motif/base-vocabulary overlaps differ from the explicit allow-list; observed={}".format(
                observed_base_overlaps[:10]
            )
        )
    motif_objects = [
        AddedToken(token, lstrip=False, rstrip=False, normalized=False, special=False)
        for token in selected_tokens
    ]
    tokenizer.add_tokens(motif_objects, special_tokens=False)
    token_to_id = {str(token): int(token_id) for token, token_id in tokenizer.get_vocab().items()}
    id_to_token = id_to_token_from_vocab(token_to_id)
    if len(tokenizer) != len(id_to_token):
        raise ContractError("tokenizer length differs from complete contiguous ID mapping")
    selected_id_map = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in selected_tokens}
    if len(set(selected_id_map.values())) != len(selected_id_map):
        raise ContractError("selected motif tokens do not map one-to-one to token IDs")
    if any(token_to_id.get(token) != token_id for token, token_id in selected_id_map.items()):
        raise ContractError("selected motif token lookup disagrees with complete vocabulary")
    oov_id = int(tokenizer.convert_tokens_to_ids(oov["token"]))
    if oov_id < 0 or token_to_id.get(oov["token"]) != oov_id:
        raise ContractError("frozen OOV token does not have an exact vocabulary ID")
    metadata = added_token_metadata(tokenizer)
    summary = {
        "tokenizer_class": tokenizer.__class__.__name__,
        "base_additional_special_tokens": base_special_tokens,
        "declared_special_tokens": declared_specials,
        "special_token_order": declared_specials,
        "special_token_id_map": special_map,
        "sentinel_token_id_map": {
            token: int(tokenizer.convert_tokens_to_ids(token)) for token in T5_SENTINELS
        },
        "anchor_token_id_map": {token: special_map[token] for token in anchors},
        "selected_motif_token_id_map": selected_id_map,
        "oov_token": oov["token"],
        "oov_token_id": oov_id,
        "id_to_token": id_to_token,
        "token_to_id": token_to_id,
        "added_token_metadata": metadata,
        "vocab_size": len(id_to_token),
    }
    for name in (
        "special_token_order",
        "special_token_id_map",
        "sentinel_token_id_map",
        "anchor_token_id_map",
        "selected_motif_token_id_map",
        "id_to_token",
        "token_to_id",
        "added_token_metadata",
    ):
        summary["{}_sha256".format(name)] = sha256_json(summary[name])

    if save_directory is not None:
        if save_directory.exists():
            raise FileExistsError("refusing to overwrite tokenizer snapshot: {}".format(save_directory))
        tokenizer.save_pretrained(str(save_directory))
        reloaded = AutoTokenizer.from_pretrained(
            str(save_directory), use_fast=False, local_files_only=True, trust_remote_code=False
        )
        reloaded_vocab = {str(token): int(token_id) for token, token_id in reloaded.get_vocab().items()}
        reloaded_id_to_token = id_to_token_from_vocab(reloaded_vocab)
        reloaded_metadata = added_token_metadata(reloaded)
        if reloaded_id_to_token != id_to_token or reloaded_vocab != token_to_id:
            raise ContractError("saved tokenizer mapping differs after offline reload")
        if reloaded_metadata != metadata:
            raise ContractError("saved tokenizer AddedToken metadata differs after offline reload")
        if list(reloaded.additional_special_tokens or []) != list(tokenizer.additional_special_tokens or []):
            raise ContractError("saved tokenizer special-token order differs after offline reload")
        summary["saved_snapshot"] = observe_tree(save_directory)
    return tokenizer, summary


def _read_ordered_vocab(path: Path) -> list[str]:
    tokens: list[str] = []
    seen: dict[str, bool] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ContractError("ordered vocab line {} lacks LF terminator".format(line_number))
            fields = line[:-1].split("\t")
            if len(fields) != 6:
                raise ContractError("ordered vocab line {} does not have six fields".format(line_number))
            token = fields[0]
            if not token or token in seen:
                raise ContractError("ordered vocabulary contains an empty or duplicate token")
            seen[token] = True
            tokens.append(token)
    return tokens


def internal_probe(args: argparse.Namespace) -> None:
    base_snapshot = Path(args.base_snapshot).expanduser().resolve()
    base_lock, _ = validate_base_lock(Path(args.base_snapshot_lock).resolve(), base_snapshot)
    policy, _ = validate_policy(Path(args.selection_policy).resolve())
    if policy["base_model"] != {
        "identifier": base_lock["model_identifier"],
        "revision": base_lock["revision"],
    }:
        raise ContractError("policy base model differs from base snapshot lock")
    selected = _read_ordered_vocab(Path(args.ordered_vocab).resolve())
    _, summary = build_tokenizer_mapping(base_snapshot, base_lock, policy, selected, save_directory=None)
    fields = [
        "id_to_token_sha256",
        "token_to_id_sha256",
        "added_token_metadata_sha256",
        "special_token_order_sha256",
        "special_token_id_map_sha256",
        "sentinel_token_id_map_sha256",
        "anchor_token_id_map_sha256",
        "vocab_size",
    ]
    print(json.dumps({field: summary[field] for field in fields}, sort_keys=True, separators=(",", ":")))


def run_hashseed_probes(
    seeds: list[str],
    base_snapshot: Path,
    base_lock_path: Path,
    policy_path: Path,
    ordered_vocab_path: Path,
    expected_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ContractError("at least three distinct PYTHONHASHSEED values are required")
    fields = [
        "id_to_token_sha256",
        "token_to_id_sha256",
        "added_token_metadata_sha256",
        "special_token_order_sha256",
        "special_token_id_map_sha256",
        "sentinel_token_id_map_sha256",
        "anchor_token_id_map_sha256",
        "vocab_size",
    ]
    expected = {field: expected_summary[field] for field in fields}
    observations: list[dict[str, Any]] = []
    for seed in seeds:
        if not seed.isdigit():
            raise ContractError("PYTHONHASHSEED values must be non-negative decimal integers")
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONHASHSEED": seed,
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            }
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--internal-probe",
            "--base-snapshot",
            str(base_snapshot),
            "--base-snapshot-lock",
            str(base_lock_path),
            "--selection-policy",
            str(policy_path),
            "--ordered-vocab",
            str(ordered_vocab_path),
        ]
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise ContractError(
                "PYTHONHASHSEED={} probe failed with closed stderr hash {}".format(
                    seed, sha256_bytes(completed.stderr.encode("utf-8"))
                )
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise ContractError("PYTHONHASHSEED={} probe emitted an unexpected stdout shape".format(seed))
        observed = json.loads(lines[0])
        if observed != expected:
            raise ContractError("PYTHONHASHSEED={} tokenizer mapping differs".format(seed))
        observations.append({"pythonhashseed": seed, "mapping": observed})
    return observations


def _write_ordered_vocab(path: Path, selected_tokens: list[str], by_token: dict[str, dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for token in selected_tokens:
            row = by_token[token]
            values = [
                token,
                str(row["p1_count"]),
                str(row["p2_count"]),
                str(row["total_count"]),
                str(row["selection_score"]),
                row["pure_motif_token_sha256"],
            ]
            handle.write("\t".join(values) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--base-snapshot")
    parser.add_argument("--base-snapshot-lock")
    parser.add_argument("--selection-policy")
    parser.add_argument("--ordered-vocab", help=argparse.SUPPRESS)
    parser.add_argument("--contract")
    parser.add_argument("--production-release-root")
    parser.add_argument("--p1-scope-lock")
    parser.add_argument("--p1-census")
    parser.add_argument("--p2-scope-lock")
    parser.add_argument("--p2-census")
    parser.add_argument("--release-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--hash-seeds", nargs="+", default=["0", "1", "271828"])
    args = parser.parse_args()
    if args.internal_probe:
        for field in ("base_snapshot", "base_snapshot_lock", "selection_policy", "ordered_vocab"):
            if not getattr(args, field):
                parser.error("--{} is required for internal probe".format(field.replace("_", "-")))
        internal_probe(args)
        return
    required = (
        "base_snapshot",
        "base_snapshot_lock",
        "selection_policy",
        "contract",
        "production_release_root",
        "p1_scope_lock",
        "p1_census",
        "release_id",
        "output_dir",
    )
    for field in required:
        if not getattr(args, field):
            parser.error("--{} is required".format(field.replace("_", "-")))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.release_id):
        raise ContractError("release ID is malformed")

    contract_path = Path(args.contract).resolve()
    _, contract_observation = validate_contract(contract_path)
    production_root = Path(args.production_release_root).expanduser().resolve()
    production_manifest, production_census, production_observation = validate_production_release(production_root)
    p1_census_path = Path(args.p1_census).expanduser().resolve()
    p1_scope_path = Path(args.p1_scope_lock).expanduser().resolve()
    p1_scope, p1_scope_observation = validate_scope_lock(
        p1_scope_path,
        "P1",
        p1_census_path,
        production_manifest["logical_release_root_sha256"],
    )
    if p1_scope["census_kind"] == "production_global_admitted_candidate":
        if p1_census_path != production_census or p1_scope["census_sha256"] != production_observation["global_motif_census"]["sha256"]:
            raise ContractError("P1 production-global candidate census must be the manifest-bound global census")

    policy_path = Path(args.selection_policy).resolve()
    policy, policy_observation = validate_policy(policy_path)
    p2_scope = None
    p2_scope_observation = None
    p2_census_path = None
    if policy["discovery_scope"] == "p1_only":
        if args.p2_scope_lock or args.p2_census:
            raise ContractError("P2 inputs are forbidden for an explicitly P1-only policy")
    else:
        if not args.p2_scope_lock or not args.p2_census:
            raise ContractError("P1+P2 discovery requires both P2 scope lock and P2 census")
        p2_census_path = Path(args.p2_census).expanduser().resolve()
        p2_scope, p2_scope_observation = validate_scope_lock(
            Path(args.p2_scope_lock).resolve(),
            "P2",
            p2_census_path,
            production_manifest["logical_release_root_sha256"],
        )

    base_snapshot = Path(args.base_snapshot).expanduser().resolve()
    base_lock_path = Path(args.base_snapshot_lock).resolve()
    base_lock, base_lock_observation = validate_base_lock(base_lock_path, base_snapshot)
    if policy["base_model"] != {
        "identifier": base_lock["model_identifier"],
        "revision": base_lock["revision"],
    }:
        raise ContractError("selection policy base-model decision differs from base snapshot lock")
    if policy["decision_status"] == "approved_for_frozen_release":
        if base_lock["decision_status"] != "approved_for_frozen_release":
            raise ContractError("frozen release requires a frozen base snapshot decision")
        if p1_scope["scope_status"] != "complete" or (p2_scope and p2_scope["scope_status"] != "complete"):
            raise ContractError("frozen release requires complete phase discovery scopes")

    p1_rows = read_census(p1_census_path, p1_scope, "P1")
    p2_rows = {} if p2_census_path is None else read_census(p2_census_path, p2_scope, "P2")
    lexeme_rows, pure_rows, selected_tokens, statistics = aggregate_projection(p1_rows, p2_rows, policy)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    pure_path = output / "pure_motif_census.jsonl"
    ordered_path = output / "ordered_motif_vocab.tsv"
    binding_path = output / "motif_digest_binding.jsonl"
    id_path = output / "id_to_token.json"
    token_path = output / "token_to_id.json"
    snapshot_path = output / "tokenizer_snapshot"
    manifest_path = output / "tokenizer_release_manifest.json"

    write_jsonl_new(pure_path, pure_rows)
    pure_by_token = {row["pure_motif_token"]: row for row in pure_rows}
    _write_ordered_vocab(ordered_path, selected_tokens, pure_by_token)
    tokenizer, mapping = build_tokenizer_mapping(
        base_snapshot, base_lock, policy, selected_tokens, save_directory=snapshot_path
    )
    del tokenizer
    selected_ids = mapping["selected_motif_token_id_map"]
    oov_id = mapping["oov_token_id"]
    for row in lexeme_rows:
        selected = row["pure_motif_token"] in selected_ids
        pure = pure_by_token[row["pure_motif_token"]]
        row["selection_score"] = pure["selection_score"]
        row["selected"] = selected
        row["token_id"] = selected_ids[row["pure_motif_token"]] if selected else oov_id
        row["binding_disposition"] = "selected_motif_token" if selected else "frozen_oov"
    write_jsonl_new(binding_path, lexeme_rows)
    write_json_new(id_path, mapping["id_to_token"], pretty=False)
    write_json_new(token_path, mapping["token_to_id"], pretty=False)

    probes = run_hashseed_probes(
        [str(seed) for seed in args.hash_seeds],
        base_snapshot,
        base_lock_path,
        policy_path,
        ordered_path,
        mapping,
    )
    release_status = (
        "frozen_tokenizer_built_non_admission"
        if policy["decision_status"] == "approved_for_frozen_release"
        else "candidate_tokenizer_built_non_release"
    )
    artifacts = {
        "pure_motif_census": observe_artifact(pure_path, pure_path.name),
        "ordered_motif_vocab": observe_artifact(ordered_path, ordered_path.name),
        "motif_digest_binding": observe_artifact(binding_path, binding_path.name),
        "id_to_token": observe_artifact(id_path, id_path.name),
        "token_to_id": observe_artifact(token_path, token_path.name),
        "tokenizer_snapshot": mapping["saved_snapshot"],
    }
    runtime = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "transformers": _dependency_version("transformers"),
        "tokenizers": _dependency_version("tokenizers"),
        "sentencepiece": _dependency_version("sentencepiece"),
    }
    manifest = {
        "schema_version": RELEASE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "release_id": args.release_id,
        "release_status": release_status,
        "independent_validation_status": "pending",
        "p1_training_admission": False,
        "p1_training_launcher_permitted": False,
        "p1_p2_exact_same_mapping": True,
        "p2_vocab_extension_forbidden": True,
        "contract": contract_observation,
        "builder": {
            "sha256": sha256_file(Path(__file__).resolve()),
            "schema_implementation": "build_motif_tokenizer_binding_release_v1.py",
        },
        "production_release": production_observation,
        "base_snapshot_lock": {
            "sha256": base_lock_observation["sha256"],
            "bytes": base_lock_observation["bytes"],
            "model_identifier": base_lock["model_identifier"],
            "revision": base_lock["revision"],
            "snapshot_tree_sha256": base_lock["snapshot_tree_sha256"],
        },
        "scope_locks": {
            "p1": {
                "sha256": p1_scope_observation["sha256"],
                "bytes": p1_scope_observation["bytes"],
                "scope_status": p1_scope["scope_status"],
                "membership_manifest_sha256": p1_scope["membership_manifest_sha256"],
                "census_sha256": p1_scope["census_sha256"],
            },
            "p2": None if p2_scope is None else {
                "sha256": p2_scope_observation["sha256"],
                "bytes": p2_scope_observation["bytes"],
                "scope_status": p2_scope["scope_status"],
                "membership_manifest_sha256": p2_scope["membership_manifest_sha256"],
                "census_sha256": p2_scope["census_sha256"],
            },
        },
        "selection_policy": {
            "sha256": policy_observation["sha256"],
            "bytes": policy_observation["bytes"],
            "decisions": policy,
        },
        "projection": {
            "spec_id": PROJECTION_SPEC_ID,
            "spec_sha256": sha256_json(
                {
                    "anchor_regex": ANCHOR_RE.pattern,
                    "anchor_decimal_policy": "canonical_no_leading_zero",
                    "pure_motif_formula": "'[' + anchor_regex.sub('', exact_lexeme) + ']'",
                    "normalization": "none",
                }
            ),
        },
        "statistics": statistics,
        "tokenizer_mapping": {
            "tokenizer_class": mapping["tokenizer_class"],
            "vocab_size": mapping["vocab_size"],
            "oov_token": mapping["oov_token"],
            "oov_token_id": mapping["oov_token_id"],
            "id_to_token_sha256": mapping["id_to_token_sha256"],
            "token_to_id_sha256": mapping["token_to_id_sha256"],
            "added_token_metadata_sha256": mapping["added_token_metadata_sha256"],
            "special_token_order": mapping["special_token_order"],
            "special_token_order_sha256": mapping["special_token_order_sha256"],
            "special_token_id_map": mapping["special_token_id_map"],
            "special_token_id_map_sha256": mapping["special_token_id_map_sha256"],
            "sentinel_token_id_map": mapping["sentinel_token_id_map"],
            "sentinel_token_id_map_sha256": mapping["sentinel_token_id_map_sha256"],
            "anchor_token_id_map": mapping["anchor_token_id_map"],
            "anchor_token_id_map_sha256": mapping["anchor_token_id_map_sha256"],
            "selected_motif_token_id_map_sha256": mapping["selected_motif_token_id_map_sha256"],
        },
        "determinism_gate": {
            "pass": True,
            "distinct_pythonhashseed_count": len(probes),
            "probes": probes,
        },
        "runtime": runtime,
        "artifacts": artifacts,
        "next_gate": "Independent validator pass and the remaining P0 overlap/semantic gates are mandatory before any P1 admission decision.",
    }
    manifest["manifest_payload_sha256"] = sha256_json(manifest)
    write_json_new(manifest_path, manifest)
    print(json.dumps({
        "release_id": args.release_id,
        "release_status": release_status,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selected_motif_tokens": len(selected_tokens),
        "exact_lexeme_bindings": len(lexeme_rows),
        "vocab_size": mapping["vocab_size"],
        "p1_training_admission": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
