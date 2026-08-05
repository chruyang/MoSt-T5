#!/usr/bin/env python3
"""Validate the logical-motif CE-first vNext record and P1 decision artifacts.

This gate is intentionally standard-library only.  It validates the contract
between a future tokenizer-bound Dataset/Collator implementation and the model;
it does not tokenize molecules, load a dataset, run a model, or mutate the
historical v2/v3 contracts.

Two independent artifact kinds are supported:

* ``logical_motif_training_record``: an unpadded tokenizer-bound record plus
  one logical-motif mask realization.  ``identity_recovery_mask`` is mandatory
  for every profile.  C3 state masks and EMA targets are forbidden on the
  CE-first profile and mandatory on the C3 profile.
* ``p1_admission_decision``: a separately hashed decision that binds external
  evidence and can authorize PF-1 only.  It does not rewrite a candidate data
  release manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath


HERE = Path(__file__).resolve().parent
CONTRACT_DIR = HERE.parent / "contracts"
RECORD_CONTRACT_PATH = CONTRACT_DIR / "p1_logical_motif_ce_first_contract_vnext1.json"
ADMISSION_CONTRACT_PATH = CONTRACT_DIR / "p1_admission_decision_contract_vnext1.json"


def _load_contract(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


RECORD_CONTRACT = _load_contract(RECORD_CONTRACT_PATH)
ADMISSION_CONTRACT = _load_contract(ADMISSION_CONTRACT_PATH)

RECORD_SCHEMA = RECORD_CONTRACT["record_schema_version"]
ADMISSION_SCHEMA = ADMISSION_CONTRACT["artifact_schema_version"]
RECORD_KIND = RECORD_CONTRACT["document_kind"]
ADMISSION_KIND = ADMISSION_CONTRACT["document_kind"]
REPORT_SCHEMA = "most-t5-r1/p1-logical-motif-validation-report/vnext1"

CE_PROFILE = "ce_first"
C3_PROFILE = "c3_ema_state_prediction"
TRAINING_PROFILES = frozenset((CE_PROFILE, C3_PROFILE))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_TOKEN_ROLES = frozenset(RECORD_CONTRACT["domains"]["token"]["allowed_token_roles"])
ALLOWED_BOND_TYPES = frozenset(("single", "double", "triple", "aromatic", "dative", "other"))
FORBIDDEN_LEGACY_FIELDS = frozenset(RECORD_CONTRACT["forbidden_legacy_field_names"])

RECORD_TOP_BASE = frozenset(RECORD_CONTRACT["required_top_level_fields"])
RECORD_BINDINGS = frozenset(RECORD_CONTRACT["bindings_required_fields"])
TOKEN_FIELDS = frozenset(RECORD_CONTRACT["domains"]["token"]["required_fields"])
MOTIF_FIELDS = frozenset(RECORD_CONTRACT["domains"]["logical_motif"]["required_fields"])
ATOM_FIELDS = frozenset(RECORD_CONTRACT["domains"]["atom"]["required_fields"])
C3_TEACHER_FIELDS = frozenset(RECORD_CONTRACT["c3_teacher_contract"]["required_fields"])
MASK_DECISION_FIELDS = frozenset(RECORD_CONTRACT["mask_decision_required_fields"])

ADMISSION_TOP_FIELDS = frozenset(ADMISSION_CONTRACT["required_top_level_fields"])
ADMISSION_RELEASE_FIELDS = frozenset(ADMISSION_CONTRACT["candidate_release_required_fields"])
ADMISSION_REFERENCE_FIELDS = frozenset(ADMISSION_CONTRACT["artifact_reference_required_fields"])
ADMISSION_REFERENCE_KINDS = ADMISSION_CONTRACT["candidate_release_reference_artifact_kinds"]
EVIDENCE_RECEIPT_FIELDS = frozenset(ADMISSION_CONTRACT["evidence_receipt_required_fields"])
EVIDENCE_SCHEMA = ADMISSION_CONTRACT["evidence_schema_version"]
EVIDENCE_KIND_TEMPLATE = ADMISSION_CONTRACT["evidence_artifact_kind_template"]
EVIDENCE_STATUS_VALUES = frozenset(ADMISSION_CONTRACT["evidence_status_values"])
BASE_GATE_FIELDS = frozenset(ADMISSION_CONTRACT["base_gate_fields"])
C3_GATE_FIELDS = frozenset(ADMISSION_CONTRACT["c3_conditional_gate_fields"])


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_nonnegative_int(value):
    return _is_int(value) and value >= 0


def _is_positive_int(value):
    return _is_int(value) and value > 0


def _is_sha256(value):
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _canonical_sha256(value):
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError("duplicate JSON object key {!r}".format(key))
        result[key] = value
    return result


def _load_json_evidence(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)


def _error(errors, path, message):
    errors.append({"path": path, "message": message})


def _require_object(value, errors, path):
    if not isinstance(value, dict):
        _error(errors, path, "must be an object")
        return {}
    return value


def _require_array(value, errors, path):
    if not isinstance(value, list):
        _error(errors, path, "must be an array")
        return []
    return value


def _require_exact_keys(mapping, expected, errors, path):
    if not isinstance(mapping, dict):
        return
    observed = set(mapping)
    for key in sorted(expected - observed):
        _error(errors, "{}.{}".format(path, key), "is required")
    for key in sorted(observed - expected):
        _error(errors, "{}.{}".format(path, key), "is not permitted by this contract")


def _require_string(mapping, key, errors, path):
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        _error(errors, "{}.{}".format(path, key), "must be a non-empty string")
        return None
    return value


def _require_sha(mapping, key, errors, path):
    value = mapping.get(key)
    if not _is_sha256(value):
        _error(errors, "{}.{}".format(path, key), "must be a lower-case 64-hex SHA-256")
        return None
    return value


def _require_nonzero_sha(mapping, key, errors, path):
    value = _require_sha(mapping, key, errors, path)
    if value == "0" * 64:
        _error(errors, "{}.{}".format(path, key), "must not be the all-zero placeholder SHA-256")
        return None
    return value


def _require_bool(mapping, key, errors, path, expected=None):
    value = mapping.get(key)
    if not isinstance(value, bool):
        _error(errors, "{}.{}".format(path, key), "must be boolean")
        return None
    if expected is not None and value is not expected:
        _error(errors, "{}.{}".format(path, key), "must be {!r}".format(expected))
    return value


def _require_positive_int(mapping, key, errors, path):
    value = mapping.get(key)
    if not _is_positive_int(value):
        _error(errors, "{}.{}".format(path, key), "must be a positive integer")
        return None
    return value


def _require_length(array, length, errors, path):
    if len(array) != length:
        _error(errors, path, "must have length {}; observed {}".format(length, len(array)))
        return False
    return True


def _relative_path_parts(value, errors, path):
    """Return canonical relative components safe on both Windows and POSIX."""

    if not isinstance(value, str) or not value:
        _error(errors, path, "must be a non-empty relative path string")
        return None
    if value != value.strip():
        _error(errors, path, "must not have leading or trailing whitespace")
        return None
    if "\x00" in value:
        _error(errors, path, "must not contain a NUL byte")
        return None

    # Treat either separator as a separator on every host.  This prevents a
    # Windows traversal string from becoming an innocent filename on POSIX (or
    # vice versa) and makes admission artifacts portable across both systems.
    normalized = value.replace("\\", "/")
    windows_path = PureWindowsPath(value)
    posix_path = PurePosixPath(normalized)
    if windows_path.is_absolute() or windows_path.drive or posix_path.is_absolute():
        _error(errors, path, "must be relative to artifact_root; absolute and drive-relative paths are forbidden")
        return None
    raw_parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in raw_parts):
        _error(errors, path, "must be canonical and must not contain empty, dot, or parent-traversal components")
        return None
    if any(":" in part for part in raw_parts):
        _error(errors, path, "must not contain a colon (Windows drive/alternate-stream syntax is forbidden)")
        return None
    return tuple(raw_parts)


def _is_link_like(path):
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction()) if callable(is_junction) else False
    except OSError:
        # An unreadable/reparse path is never acceptable evidence.
        return True


def _prepare_artifact_root(artifact_root, errors, required):
    if artifact_root is None:
        if required:
            _error(errors, "$.artifact_root", "is required to authorize an admit decision")
        return None
    try:
        root = Path(artifact_root)
    except (TypeError, ValueError) as exc:
        _error(errors, "$.artifact_root", "is not a valid filesystem path: {}".format(exc))
        return None
    if _is_link_like(root):
        _error(errors, "$.artifact_root", "must not itself be a symbolic link or junction")
        return None
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        _error(errors, "$.artifact_root", "cannot be resolved: {}".format(exc))
        return None
    if not resolved.is_dir():
        _error(errors, "$.artifact_root", "must resolve to an existing directory")
        return None
    return resolved


def _resolve_confined_regular_file(root, parts, errors, path):
    current = root
    for part in parts:
        current = current / part
        if not current.exists():
            _error(errors, path, "referenced artifact does not exist beneath artifact_root")
            return None
        if _is_link_like(current):
            _error(errors, path, "symbolic links and junctions are forbidden in referenced artifact paths")
            return None
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        _error(errors, path, "referenced artifact escapes artifact_root")
        return None
    if not resolved.is_file():
        _error(errors, path, "referenced artifact must be a regular file")
        return None
    return resolved


def _validate_reference_structure(raw, errors, path, release_id, expected_kind, evidence_gate=None):
    reference = _require_object(raw, errors, path)
    expected_fields = EVIDENCE_RECEIPT_FIELDS if evidence_gate is not None else ADMISSION_REFERENCE_FIELDS
    _require_exact_keys(reference, expected_fields, errors, path)
    parts = _relative_path_parts(reference.get("path"), errors, "{}.path".format(path))
    _require_nonzero_sha(reference, "sha256", errors, path)
    schema_version = _require_string(reference, "schema_version", errors, path)
    artifact_kind = _require_string(reference, "artifact_kind", errors, path)
    subject_release_id = _require_string(reference, "subject_release_id", errors, path)
    if release_id is not None and subject_release_id is not None and subject_release_id != release_id:
        _error(errors, "{}.subject_release_id".format(path), "must equal candidate_release.release_id")
    if artifact_kind is not None and artifact_kind != expected_kind:
        _error(errors, "{}.artifact_kind".format(path), "must be {!r}".format(expected_kind))
    if evidence_gate is not None:
        if schema_version is not None and schema_version != EVIDENCE_SCHEMA:
            _error(errors, "{}.schema_version".format(path), "must be {!r}".format(EVIDENCE_SCHEMA))
        status = reference.get("status")
        if status not in EVIDENCE_STATUS_VALUES:
            _error(errors, "{}.status".format(path), "must be one of {}".format(sorted(EVIDENCE_STATUS_VALUES)))
    return reference, parts


def _resolve_and_bind_json_reference(reference, parts, root, errors, path, extra_bindings=None):
    if root is None or parts is None:
        return None
    artifact_path = _resolve_confined_regular_file(root, parts, errors, "{}.path".format(path))
    if artifact_path is None:
        return None
    expected_sha = reference.get("sha256")
    try:
        observed_sha = _sha256_file(artifact_path)
    except OSError as exc:
        _error(errors, "{}.path".format(path), "cannot hash referenced artifact: {}".format(exc))
        return None
    if not _is_sha256(expected_sha) or observed_sha != expected_sha:
        _error(errors, "{}.sha256".format(path), "does not match the referenced file SHA-256")
        return None
    try:
        content = _load_json_evidence(artifact_path)
    except (OSError, UnicodeError, ValueError) as exc:
        _error(errors, "{}.path".format(path), "referenced artifact is not unambiguous UTF-8 JSON: {}".format(exc))
        return None
    if not isinstance(content, dict):
        _error(errors, "{}.path".format(path), "referenced JSON artifact must be an object")
        return None
    for key in ("schema_version", "artifact_kind", "subject_release_id"):
        if content.get(key) != reference.get(key):
            _error(errors, "{}.{}".format(path, key), "does not match referenced JSON metadata")
    if "status" in reference and content.get("status") != reference.get("status"):
        _error(errors, "{}.status".format(path), "does not match referenced JSON metadata")
    for key, expected in (extra_bindings or {}).items():
        if content.get(key) != expected:
            _error(errors, "{}.path".format(path), "referenced JSON field {!r} must equal {!r}".format(key, expected))
    return content


def _find_forbidden_fields(value, errors, path="$"):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = "{}.{}".format(path, key)
            if key in FORBIDDEN_LEGACY_FIELDS:
                _error(errors, child_path, "legacy merged-mask/MSE field is forbidden in vNext")
            _find_forbidden_fields(child, errors, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _find_forbidden_fields(child, errors, "{}[{}]".format(path, index))


def _validate_bindings(raw, errors):
    path = "bindings"
    bindings = _require_object(raw, errors, path)
    _require_exact_keys(bindings, RECORD_BINDINGS, errors, path)
    _require_string(bindings, "release_id", errors, path)
    for key in sorted(RECORD_BINDINGS - {"release_id"}):
        _require_sha(bindings, key, errors, path)


def _validate_member(raw, errors):
    path = "member"
    member = _require_object(raw, errors, path)
    expected = frozenset(("member_id", "storage_key"))
    _require_exact_keys(member, expected, errors, path)
    _require_string(member, "member_id", errors, path)
    _require_string(member, "storage_key", errors, path)
    return member


def _validate_dimensions(raw, errors):
    path = "dimensions"
    dimensions = _require_object(raw, errors, path)
    expected = frozenset(
        (
            "token_count",
            "logical_motif_count",
            "atom_count",
            "source_atom_count",
            "e3fp_level_count",
        )
    )
    _require_exact_keys(dimensions, expected, errors, path)
    values = {}
    for key in sorted(expected):
        values[key] = _require_positive_int(dimensions, key, errors, path)
    return values


def _validate_token_domain(raw, token_count, motif_count, errors):
    path = "token_domain"
    domain = _require_object(raw, errors, path)
    _require_exact_keys(domain, TOKEN_FIELDS, errors, path)
    input_ids = _require_array(domain.get("input_ids"), errors, "{}.input_ids".format(path))
    attention = _require_array(domain.get("attention_mask"), errors, "{}.attention_mask".format(path))
    token_to_motif = _require_array(
        domain.get("token_to_logical_motif"), errors, "{}.token_to_logical_motif".format(path)
    )
    roles = _require_array(domain.get("token_role"), errors, "{}.token_role".format(path))

    for name, array in (
        ("input_ids", input_ids),
        ("attention_mask", attention),
        ("token_to_logical_motif", token_to_motif),
        ("token_role", roles),
    ):
        _require_length(array, token_count, errors, "{}.{}".format(path, name))

    for index, value in enumerate(input_ids):
        if not _is_nonnegative_int(value):
            _error(errors, "{}.input_ids[{}]".format(path, index), "must be a non-negative integer token id")
    for index, value in enumerate(attention):
        if value is not True:
            _error(errors, "{}.attention_mask[{}]".format(path, index), "must be true in an unpadded record")
    for index, value in enumerate(token_to_motif):
        if not _is_int(value) or value < -1 or value >= motif_count:
            _error(
                errors,
                "{}.token_to_logical_motif[{}]".format(path, index),
                "must be -1 or an in-range logical motif index",
            )
    for index, role in enumerate(roles):
        if role not in ALLOWED_TOKEN_ROLES:
            _error(errors, "{}.token_role[{}]".format(path, index), "unknown token role {!r}".format(role))
            continue
        mapped = token_to_motif[index] if index < len(token_to_motif) else None
        if role in ("identity", "connection") and not (_is_int(mapped) and 0 <= mapped < motif_count):
            _error(errors, "{}.token_to_logical_motif[{}]".format(path, index), "{} tokens must map to one motif".format(role))
        if role in ("boundary", "text") and mapped != -1:
            _error(errors, "{}.token_to_logical_motif[{}]".format(path, index), "{} tokens must map to -1".format(role))
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "token_to_motif": token_to_motif,
        "roles": roles,
    }


def _validate_span(raw_span, token_count, errors, path):
    if not isinstance(raw_span, list) or len(raw_span) != 2:
        _error(errors, path, "must be a two-integer half-open interval [start,end]")
        return None
    start, end = raw_span
    if not _is_int(start) or not _is_int(end) or not (0 <= start < end <= token_count):
        _error(errors, path, "must satisfy 0 <= start < end <= token_count")
        return None
    return start, end


def _validate_endpoint(raw, motif_count, atom_count, slot_counts, motif_atoms, errors, path):
    endpoint = _require_object(raw, errors, path)
    expected = frozenset(("logical_motif_index", "atom_index", "slot_ordinal"))
    _require_exact_keys(endpoint, expected, errors, path)
    motif_index = endpoint.get("logical_motif_index")
    atom_index = endpoint.get("atom_index")
    slot_ordinal = endpoint.get("slot_ordinal")
    if not _is_int(motif_index) or not (0 <= motif_index < motif_count):
        _error(errors, "{}.logical_motif_index".format(path), "must be an in-range motif index")
        motif_index = None
    if not _is_int(atom_index) or not (0 <= atom_index < atom_count):
        _error(errors, "{}.atom_index".format(path), "must be an in-range atom index")
        atom_index = None
    if motif_index is not None and atom_index is not None and atom_index not in motif_atoms[motif_index]:
        _error(errors, "{}.atom_index".format(path), "atom does not belong to the declared logical motif")
    if not _is_nonnegative_int(slot_ordinal):
        _error(errors, "{}.slot_ordinal".format(path), "must be a non-negative integer")
        slot_ordinal = None
    elif motif_index is not None and motif_index < len(slot_counts) and _is_nonnegative_int(slot_counts[motif_index]):
        if slot_ordinal >= slot_counts[motif_index]:
            _error(errors, "{}.slot_ordinal".format(path), "is outside the motif slot_count")
    return motif_index, atom_index, slot_ordinal


def _validate_motif_domain(raw, token_info, token_count, motif_count, atom_count, errors):
    path = "logical_motif_domain"
    domain = _require_object(raw, errors, path)
    _require_exact_keys(domain, MOTIF_FIELDS, errors, path)

    raw_spans = _require_array(domain.get("identity_spans"), errors, "{}.identity_spans".format(path))
    raw_connections = _require_array(
        domain.get("connection_token_indices"), errors, "{}.connection_token_indices".format(path)
    )
    carriers = _require_array(domain.get("logical_to_carrier"), errors, "{}.logical_to_carrier".format(path))
    digests = _require_array(domain.get("exact_identity_sha256"), errors, "{}.exact_identity_sha256".format(path))
    geometry_valid = _require_array(domain.get("motif_geometry_valid"), errors, "{}.motif_geometry_valid".format(path))
    raw_motif_atoms = _require_array(domain.get("motif_atom_indices"), errors, "{}.motif_atom_indices".format(path))
    raw_slot_atoms = _require_array(
        domain.get("motif_slot_atom_indices"),
        errors,
        "{}.motif_slot_atom_indices".format(path),
    )
    slot_counts = _require_array(domain.get("slot_count"), errors, "{}.slot_count".format(path))
    raw_bonds = _require_array(domain.get("cross_motif_bonds"), errors, "{}.cross_motif_bonds".format(path))

    for name, array in (
        ("identity_spans", raw_spans),
        ("connection_token_indices", raw_connections),
        ("logical_to_carrier", carriers),
        ("exact_identity_sha256", digests),
        ("motif_geometry_valid", geometry_valid),
        ("motif_atom_indices", raw_motif_atoms),
        ("motif_slot_atom_indices", raw_slot_atoms),
        ("slot_count", slot_counts),
    ):
        _require_length(array, motif_count, errors, "{}.{}".format(path, name))

    spans = []
    identity_coverage = {}
    for motif_index in range(min(motif_count, len(raw_spans))):
        span = _validate_span(raw_spans[motif_index], token_count, errors, "{}.identity_spans[{}]".format(path, motif_index))
        spans.append(span)
        if span is None:
            continue
        start, end = span
        for token_index in range(start, end):
            if token_index in identity_coverage:
                _error(errors, "{}.identity_spans[{}]".format(path, motif_index), "overlaps another identity span")
            identity_coverage[token_index] = motif_index
            if token_index < len(token_info["roles"]) and token_info["roles"][token_index] != "identity":
                _error(errors, "{}.identity_spans[{}]".format(path, motif_index), "covers a non-identity token")
            if token_index < len(token_info["token_to_motif"]) and token_info["token_to_motif"][token_index] != motif_index:
                _error(errors, "{}.identity_spans[{}]".format(path, motif_index), "token mapping disagrees with motif index")
    expected_identity = {i for i, role in enumerate(token_info["roles"]) if role == "identity"}
    if set(identity_coverage) != expected_identity:
        _error(errors, "{}.identity_spans".format(path), "must cover every and only identity-role token exactly once")

    connection_coverage = {}
    for motif_index in range(min(motif_count, len(raw_connections))):
        indices = _require_array(
            raw_connections[motif_index], errors, "{}.connection_token_indices[{}]".format(path, motif_index)
        )
        if any(not _is_int(value) for value in indices) or indices != sorted(set(indices)):
            _error(
                errors,
                "{}.connection_token_indices[{}]".format(path, motif_index),
                "must be strictly increasing unique integer indices",
            )
        for token_index in indices:
            if not _is_int(token_index) or not (0 <= token_index < token_count):
                _error(errors, "{}.connection_token_indices[{}]".format(path, motif_index), "contains an out-of-range index")
                continue
            if token_index in connection_coverage:
                _error(errors, "{}.connection_token_indices[{}]".format(path, motif_index), "reuses a connection token")
            connection_coverage[token_index] = motif_index
            if token_info["roles"][token_index] != "connection":
                _error(errors, "{}.connection_token_indices[{}]".format(path, motif_index), "names a non-connection token")
            if token_info["token_to_motif"][token_index] != motif_index:
                _error(errors, "{}.connection_token_indices[{}]".format(path, motif_index), "token mapping disagrees with motif index")
    expected_connections = {i for i, role in enumerate(token_info["roles"]) if role == "connection"}
    if set(connection_coverage) != expected_connections:
        _error(errors, "{}.connection_token_indices".format(path), "must partition all connection-role tokens")

    seen_carriers = set()
    for motif_index, carrier in enumerate(carriers[:motif_count]):
        carrier_path = "{}.logical_to_carrier[{}]".format(path, motif_index)
        if not _is_int(carrier) or not (0 <= carrier < token_count):
            _error(errors, carrier_path, "must be an in-range token index")
            continue
        if carrier in seen_carriers:
            _error(errors, carrier_path, "carrier token must be unique across motifs")
        seen_carriers.add(carrier)
        span = spans[motif_index] if motif_index < len(spans) else None
        if span is not None and carrier != span[0]:
            _error(errors, carrier_path, "must equal the first token of the motif identity span")
        if carrier < len(token_info["roles"]) and token_info["roles"][carrier] != "identity":
            _error(errors, carrier_path, "must name an identity-role token")
        if carrier < len(token_info["token_to_motif"]) and token_info["token_to_motif"][carrier] != motif_index:
            _error(errors, carrier_path, "token mapping disagrees with motif index")

    for index, digest in enumerate(digests):
        if not _is_sha256(digest):
            _error(errors, "{}.exact_identity_sha256[{}]".format(path, index), "must be a lower-case SHA-256")
    for index, value in enumerate(geometry_valid):
        if not isinstance(value, bool):
            _error(errors, "{}.motif_geometry_valid[{}]".format(path, index), "must be boolean")
    for index, value in enumerate(slot_counts):
        if not _is_nonnegative_int(value):
            _error(errors, "{}.slot_count[{}]".format(path, index), "must be a non-negative integer")

    motif_atoms = []
    atom_owner = {}
    for motif_index in range(min(motif_count, len(raw_motif_atoms))):
        values = _require_array(raw_motif_atoms[motif_index], errors, "{}.motif_atom_indices[{}]".format(path, motif_index))
        valid_values = []
        if not values:
            _error(errors, "{}.motif_atom_indices[{}]".format(path, motif_index), "must be nonempty")
        if any(not _is_int(value) for value in values) or values != sorted(set(values)):
            _error(errors, "{}.motif_atom_indices[{}]".format(path, motif_index), "must be strictly increasing unique integers")
        for atom_index in values:
            if not _is_int(atom_index) or not (0 <= atom_index < atom_count):
                _error(errors, "{}.motif_atom_indices[{}]".format(path, motif_index), "contains an out-of-range atom")
                continue
            valid_values.append(atom_index)
            if atom_index in atom_owner:
                _error(errors, "{}.motif_atom_indices[{}]".format(path, motif_index), "atom is assigned to multiple motifs")
            atom_owner[atom_index] = motif_index
        motif_atoms.append(set(valid_values))
    while len(motif_atoms) < motif_count:
        motif_atoms.append(set())
    if set(atom_owner) != set(range(atom_count)):
        _error(errors, "{}.motif_atom_indices".format(path), "motif groups must cover every atom exactly once")

    motif_slot_atoms = []
    for motif_index in range(min(motif_count, len(raw_slot_atoms))):
        values = _require_array(
            raw_slot_atoms[motif_index],
            errors,
            "{}.motif_slot_atom_indices[{}]".format(path, motif_index),
        )
        motif_slot_atoms.append(values)
        expected_count = slot_counts[motif_index] if motif_index < len(slot_counts) else None
        if _is_nonnegative_int(expected_count) and len(values) != expected_count:
            _error(
                errors,
                "{}.motif_slot_atom_indices[{}]".format(path, motif_index),
                "length must equal slot_count",
            )
        motif_atoms_for_id = motif_atoms[motif_index] if motif_index < len(motif_atoms) else set()
        for slot_ordinal, atom_index in enumerate(values):
            if not _is_int(atom_index) or atom_index not in motif_atoms_for_id:
                _error(
                    errors,
                    "{}.motif_slot_atom_indices[{}][{}]".format(path, motif_index, slot_ordinal),
                    "must be an atom in the declared logical motif",
                )
    while len(motif_slot_atoms) < motif_count:
        motif_slot_atoms.append([])

    edge_ids = set()
    endpoint_atoms = set()
    slot_keys = set()
    incidence_counts = [0] * motif_count
    observed_edge_keys = []
    for edge_index, raw_bond in enumerate(raw_bonds):
        bond_path = "{}.cross_motif_bonds[{}]".format(path, edge_index)
        bond = _require_object(raw_bond, errors, bond_path)
        expected = frozenset(("edge_id", "left", "right", "bond_type"))
        _require_exact_keys(bond, expected, errors, bond_path)
        edge_id = bond.get("edge_id")
        if not _is_nonnegative_int(edge_id):
            _error(errors, "{}.edge_id".format(bond_path), "must be a non-negative integer")
        elif edge_id != edge_index:
            _error(errors, "{}.edge_id".format(bond_path), "must be dense and equal to its record index")
        elif edge_id in edge_ids:
            _error(errors, "{}.edge_id".format(bond_path), "must be unique")
        else:
            edge_ids.add(edge_id)
        bond_type = bond.get("bond_type")
        if bond_type not in ALLOWED_BOND_TYPES:
            _error(errors, "{}.bond_type".format(bond_path), "must be one of {}".format(sorted(ALLOWED_BOND_TYPES)))
        endpoints = []
        for side in ("left", "right"):
            endpoint = _validate_endpoint(
                bond.get(side), motif_count, atom_count, slot_counts, motif_atoms, errors, "{}.{}".format(bond_path, side)
            )
            endpoints.append(endpoint)
            motif_index, atom_index, slot_ordinal = endpoint
            if atom_index is not None:
                endpoint_atoms.add(atom_index)
            if motif_index is not None:
                incidence_counts[motif_index] += 1
            if motif_index is not None and slot_ordinal is not None:
                key = (motif_index, slot_ordinal)
                if key in slot_keys:
                    _error(errors, "{}.{}.slot_ordinal".format(bond_path, side), "motif slot is reused")
                slot_keys.add(key)
                if (
                    motif_index < len(motif_slot_atoms)
                    and slot_ordinal < len(motif_slot_atoms[motif_index])
                    and atom_index is not None
                    and atom_index != motif_slot_atoms[motif_index][slot_ordinal]
                ):
                    _error(
                        errors,
                        "{}.{}.atom_index".format(bond_path, side),
                        "must equal motif_slot_atom_indices[motif][slot_ordinal]",
                    )
        if endpoints[0][0] is not None and endpoints[0][0] == endpoints[1][0]:
            _error(errors, bond_path, "cross-motif edge endpoints must belong to distinct motifs")
        if all(value is not None for endpoint in endpoints for value in endpoint):
            left_key = (endpoints[0][0], endpoints[0][2], endpoints[0][1])
            right_key = (endpoints[1][0], endpoints[1][2], endpoints[1][1])
            if not left_key < right_key:
                _error(errors, bond_path, "left/right endpoints must be in canonical increasing order")
            observed_edge_keys.append((left_key, right_key, bond_type))
    if observed_edge_keys != sorted(observed_edge_keys):
        _error(errors, "{}.cross_motif_bonds".format(path), "edge records must be in canonical endpoint/bond order")
    if len(slot_counts) == motif_count and all(_is_nonnegative_int(value) for value in slot_counts):
        if incidence_counts != slot_counts:
            _error(errors, "{}.slot_count".format(path), "must equal cross-motif endpoint incidence counts")
        expected_slot_keys = {(motif, slot) for motif, count in enumerate(slot_counts) for slot in range(count)}
        if slot_keys != expected_slot_keys:
            _error(errors, "{}.cross_motif_bonds".format(path), "must reference every declared motif slot exactly once")

    return {
        "spans": spans,
        "geometry_valid": geometry_valid,
        "motif_atoms": motif_atoms,
        "atom_owner": atom_owner,
        "attachment_atoms": endpoint_atoms,
        "motif_slot_atoms": motif_slot_atoms,
    }


def _validate_atom_domain(raw, dimensions, motif_info, errors):
    path = "atom_domain"
    domain = _require_object(raw, errors, path)
    _require_exact_keys(domain, ATOM_FIELDS, errors, path)
    atom_count = dimensions["atom_count"]
    source_atom_count = dimensions["source_atom_count"]
    motif_count = dimensions["logical_motif_count"]
    level_count = dimensions["e3fp_level_count"]
    atom_to_motif = _require_array(domain.get("atom_to_logical_motif"), errors, "{}.atom_to_logical_motif".format(path))
    source_indices = _require_array(
        domain.get("model_to_source_atom_index"), errors, "{}.model_to_source_atom_index".format(path)
    )
    valid_mask = _require_array(domain.get("atom_valid_mask"), errors, "{}.atom_valid_mask".format(path))
    attachment_mask = _require_array(domain.get("atom_is_attachment"), errors, "{}.atom_is_attachment".format(path))
    e3fp = _require_array(domain.get("full_e3fp_ids"), errors, "{}.full_e3fp_ids".format(path))
    for name, array in (
        ("atom_to_logical_motif", atom_to_motif),
        ("model_to_source_atom_index", source_indices),
        ("atom_valid_mask", valid_mask),
        ("atom_is_attachment", attachment_mask),
        ("full_e3fp_ids", e3fp),
    ):
        _require_length(array, atom_count, errors, "{}.{}".format(path, name))

    for atom_index, motif_index in enumerate(atom_to_motif):
        if not _is_int(motif_index) or not (0 <= motif_index < motif_count):
            _error(errors, "{}.atom_to_logical_motif[{}]".format(path, atom_index), "must be an in-range motif index")
            continue
        expected_owner = motif_info["atom_owner"].get(atom_index)
        if expected_owner != motif_index:
            _error(errors, "{}.atom_to_logical_motif[{}]".format(path, atom_index), "disagrees with motif_atom_indices")
    if any(not _is_int(value) or not (0 <= value < source_atom_count) for value in source_indices):
        _error(
            errors,
            "{}.model_to_source_atom_index".format(path),
            "must contain values in [0,source_atom_count)",
        )
    elif source_indices != sorted(set(source_indices)):
        _error(
            errors,
            "{}.model_to_source_atom_index".format(path),
            "must be strictly increasing and unique",
        )

    for atom_index, value in enumerate(valid_mask):
        if not isinstance(value, bool):
            _error(errors, "{}.atom_valid_mask[{}]".format(path, atom_index), "must be boolean")
        elif value is not True:
            _error(errors, "{}.atom_valid_mask[{}]".format(path, atom_index), "must be true in narrow P1 geometry")
    for atom_index, value in enumerate(attachment_mask):
        if not isinstance(value, bool):
            _error(errors, "{}.atom_is_attachment[{}]".format(path, atom_index), "must be boolean")

    for atom_index, row in enumerate(e3fp):
        row_path = "{}.full_e3fp_ids[{}]".format(path, atom_index)
        if not isinstance(row, list) or len(row) != level_count:
            _error(errors, row_path, "must have length e3fp_level_count={}".format(level_count))
            continue
        if any(not _is_int(value) or value < -1 or value > 4095 for value in row):
            _error(errors, row_path, "values must be integer E3FP ids in [-1,4095]")
            continue
        if row[0] == -1:
            _error(errors, row_path, "narrow P1 requires a non-padding level-0 E3FP id")

    for motif_index in range(motif_count):
        atoms = motif_info["motif_atoms"][motif_index]
        declared = motif_info["geometry_valid"][motif_index] if motif_index < len(motif_info["geometry_valid"]) else None
        if declared is not True:
            _error(errors, "logical_motif_domain.motif_geometry_valid[{}]".format(motif_index), "must be true in narrow P1 geometry")

    if len(attachment_mask) == atom_count and all(isinstance(value, bool) for value in attachment_mask):
        observed_attachment_atoms = {index for index, value in enumerate(attachment_mask) if value}
        if observed_attachment_atoms != motif_info["attachment_atoms"]:
            _error(errors, "{}.atom_is_attachment".format(path), "must exactly match cross_motif_bonds endpoint atoms")


def _validate_boolean_mask(raw, length, errors, path, require_true=True):
    values = _require_array(raw, errors, path)
    _require_length(values, length, errors, path)
    for index, value in enumerate(values):
        if not isinstance(value, bool):
            _error(errors, "{}[{}]".format(path, index), "must be boolean")
    if require_true and not any(value is True for value in values):
        _error(errors, path, "must select at least one logical motif")
    return values


def _validate_c3_teacher(raw, state_mask, motif_count, errors):
    path = "c3_teacher"
    teacher = _require_object(raw, errors, path)
    _require_exact_keys(teacher, C3_TEACHER_FIELDS, errors, path)
    _require_sha(teacher, "contract_sha256", errors, path)
    required = RECORD_CONTRACT["c3_teacher_contract"]["required_values"]
    for key, expected in required.items():
        value = teacher.get(key)
        if value != expected:
            _error(errors, "{}.{}".format(path, key), "must be {!r}".format(expected))
    target_dim = _require_positive_int(teacher, "target_dim", errors, path)
    indices = _require_array(
        teacher.get("target_logical_motif_indices"), errors, "{}.target_logical_motif_indices".format(path)
    )
    vectors = _require_array(teacher.get("target_vectors"), errors, "{}.target_vectors".format(path))
    expected_indices = [index for index, selected in enumerate(state_mask) if selected is True]
    if indices != expected_indices:
        _error(errors, "{}.target_logical_motif_indices".format(path), "must exactly equal true state_prediction_mask indices")
    if len(vectors) != len(indices):
        _error(errors, "{}.target_vectors".format(path), "must contain one vector per target index")
    if target_dim is not None:
        for vector_index, vector in enumerate(vectors):
            vector_path = "{}.target_vectors[{}]".format(path, vector_index)
            if not isinstance(vector, list) or len(vector) != target_dim:
                _error(errors, vector_path, "must have length target_dim={}".format(target_dim))
                continue
            if any(not _is_number(value) for value in vector):
                _error(errors, vector_path, "must contain finite numeric values")
                continue
            norm = math.sqrt(sum(float(value) * float(value) for value in vector))
            if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                _error(errors, vector_path, "must be unit-L2 normalized; observed norm {:.8g}".format(norm))


def _validate_masks(raw, profile, motif_info, motif_count, errors):
    path = "masks"
    masks = _require_object(raw, errors, path)
    expected = {"identity_recovery_mask"}
    if profile == C3_PROFILE:
        expected.add("state_prediction_mask")
    _require_exact_keys(masks, frozenset(expected), errors, path)
    identity_mask = _validate_boolean_mask(
        masks.get("identity_recovery_mask"), motif_count, errors, "{}.identity_recovery_mask".format(path)
    )
    state_mask = []
    if profile == C3_PROFILE:
        state_mask = _validate_boolean_mask(
            masks.get("state_prediction_mask"), motif_count, errors, "{}.state_prediction_mask".format(path)
        )
        for motif_index in range(min(len(identity_mask), len(state_mask), motif_count)):
            if identity_mask[motif_index] is True and state_mask[motif_index] is True:
                _error(errors, "{}.state_prediction_mask[{}]".format(path, motif_index), "must be disjoint from identity_recovery_mask")
            geometry_valid = motif_info["geometry_valid"][motif_index] if motif_index < len(motif_info["geometry_valid"]) else None
            if state_mask[motif_index] is True and geometry_valid is not True:
                _error(errors, "{}.state_prediction_mask[{}]".format(path, motif_index), "may select only geometry-valid motifs")
    return identity_mask, state_mask


def _stateless_motif_score(seed, epoch, member_id, objective, logical_motif_id):
    preimage = json.dumps(
        [seed, epoch, member_id, objective, logical_motif_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big")
    return integer / float(1 << 64)


def _select_identity_mask(seed, epoch, member_id, objective, mask_probability, motif_count):
    scores = tuple(
        _stateless_motif_score(seed, epoch, member_id, objective, motif_id)
        for motif_id in range(motif_count)
    )
    selected = [score < mask_probability for score in scores]
    if motif_count and not any(selected):
        selected[min(range(motif_count), key=lambda motif_id: (scores[motif_id], motif_id))] = True
    return selected


def _mask_decision_sha256(seed, epoch, member_id, objective, mask_probability, selected_indices):
    payload = {
        "epoch": epoch,
        "mask_probability": mask_probability,
        "objective": objective,
        "record_id": member_id,
        "schema": "most-t5-next/ce-first-mask-decision/v1",
        "seed": seed,
        "selected_logical_motif_ids": list(selected_indices),
    }
    return _canonical_sha256(payload)


def _validate_mask_decision(raw, member_id, identity_mask, motif_count, errors):
    path = "mask_decision"
    decision = _require_object(raw, errors, path)
    _require_exact_keys(decision, MASK_DECISION_FIELDS, errors, path)
    objective = decision.get("objective")
    if objective != "identity_recovery_ce":
        _error(errors, "{}.objective".format(path), "must be identity_recovery_ce")
    seed = decision.get("seed")
    if not _is_nonnegative_int(seed):
        _error(errors, "{}.seed".format(path), "must be a nonnegative integer")
    epoch = decision.get("epoch")
    if not _is_nonnegative_int(epoch):
        _error(errors, "{}.epoch".format(path), "must be a nonnegative integer")
    probability = decision.get("mask_probability")
    if not _is_number(probability) or not 0.0 < float(probability) <= 1.0:
        _error(errors, "{}.mask_probability".format(path), "must be finite and in (0,1]")
    indices = _require_array(
        decision.get("selected_logical_motif_indices"),
        errors,
        "{}.selected_logical_motif_indices".format(path),
    )
    if any(not _is_nonnegative_int(value) or value >= motif_count for value in indices):
        _error(
            errors,
            "{}.selected_logical_motif_indices".format(path),
            "must contain only logical motif indices",
        )
    if indices != sorted(set(value for value in indices if _is_nonnegative_int(value))):
        _error(
            errors,
            "{}.selected_logical_motif_indices".format(path),
            "must be strictly increasing and unique",
        )
    expected_indices = [
        motif_id for motif_id, selected in enumerate(identity_mask) if selected is True
    ]
    if indices != expected_indices:
        _error(
            errors,
            "{}.selected_logical_motif_indices".format(path),
            "must exactly equal true identity_recovery_mask indices",
        )
    declared_sha = decision.get("decision_sha256")
    if not _is_sha256(declared_sha):
        _error(errors, "{}.decision_sha256".format(path), "must be a lower-case SHA-256")
    if (
        _is_nonnegative_int(seed)
        and _is_nonnegative_int(epoch)
        and isinstance(member_id, str)
        and member_id
        and objective == "identity_recovery_ce"
        and _is_number(probability)
        and 0.0 < float(probability) <= 1.0
    ):
        expected_mask = _select_identity_mask(
            seed, epoch, member_id, objective, float(probability), motif_count
        )
        if list(identity_mask) != expected_mask:
            _error(
                errors,
                "masks.identity_recovery_mask",
                "does not match the reproducible stateless mask decision",
            )
        expected_sha = _mask_decision_sha256(
            seed, epoch, member_id, objective, float(probability), expected_indices
        )
        if declared_sha != expected_sha:
            _error(
                errors,
                "{}.decision_sha256".format(path),
                "does not match the canonical mask decision payload",
            )


def validate_training_record(document):
    errors = []
    document = _require_object(document, errors, "$")
    _find_forbidden_fields(document, errors)
    profile = document.get("training_profile")
    expected_top = set(RECORD_TOP_BASE)
    if profile == C3_PROFILE:
        expected_top.add("c3_teacher")
    _require_exact_keys(document, frozenset(expected_top), errors, "$")

    if document.get("schema_version") != RECORD_SCHEMA:
        _error(errors, "$.schema_version", "must be {}".format(RECORD_SCHEMA))
    if document.get("document_kind") != RECORD_KIND:
        _error(errors, "$.document_kind", "must be {}".format(RECORD_KIND))
    if profile not in TRAINING_PROFILES:
        _error(errors, "$.training_profile", "must be one of {}".format(sorted(TRAINING_PROFILES)))

    _validate_bindings(document.get("bindings"), errors)
    member = _validate_member(document.get("member"), errors)
    dimensions = _validate_dimensions(document.get("dimensions"), errors)
    if not all(
        _is_positive_int(dimensions.get(key))
        for key in (
            "token_count",
            "logical_motif_count",
            "atom_count",
            "source_atom_count",
            "e3fp_level_count",
        )
    ):
        return _make_report(RECORD_KIND, document, errors)

    token_info = _validate_token_domain(
        document.get("token_domain"), dimensions["token_count"], dimensions["logical_motif_count"], errors
    )
    motif_info = _validate_motif_domain(
        document.get("logical_motif_domain"),
        token_info,
        dimensions["token_count"],
        dimensions["logical_motif_count"],
        dimensions["atom_count"],
        errors,
    )
    _validate_atom_domain(document.get("atom_domain"), dimensions, motif_info, errors)
    identity_mask, state_mask = _validate_masks(
        document.get("masks"), profile, motif_info, dimensions["logical_motif_count"], errors
    )
    _validate_mask_decision(
        document.get("mask_decision"),
        member.get("member_id"),
        identity_mask,
        dimensions["logical_motif_count"],
        errors,
    )
    if profile == C3_PROFILE:
        _validate_c3_teacher(document.get("c3_teacher"), state_mask, dimensions["logical_motif_count"], errors)
    elif "c3_teacher" in document:
        _error(errors, "$.c3_teacher", "is forbidden in ce_first")
    return _make_report(RECORD_KIND, document, errors)


def _validate_timestamp(value, errors, path):
    if not isinstance(value, str) or not value:
        _error(errors, path, "must be a non-empty ISO-8601 timestamp")
        return
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _error(errors, path, "must be a valid ISO-8601 timestamp")
        return
    if parsed.tzinfo is None:
        _error(errors, path, "must include an explicit timezone")


def validate_admission_decision(document, artifact_root=None):
    errors = []
    document = _require_object(document, errors, "$")
    _require_exact_keys(document, ADMISSION_TOP_FIELDS, errors, "$")
    if document.get("schema_version") != ADMISSION_SCHEMA:
        _error(errors, "$.schema_version", "must be {}".format(ADMISSION_SCHEMA))
    if document.get("document_kind") != ADMISSION_KIND:
        _error(errors, "$.document_kind", "must be {}".format(ADMISSION_KIND))
    _require_string(document, "decision_id", errors, "$")
    _validate_timestamp(document.get("created_at_utc"), errors, "$.created_at_utc")
    profile = document.get("training_profile")
    if profile not in TRAINING_PROFILES:
        _error(errors, "$.training_profile", "must be one of {}".format(sorted(TRAINING_PROFILES)))

    decision = document.get("decision")
    if decision not in ("admit", "reject"):
        _error(errors, "$.decision", "must be admit or reject")
    admitted = document.get("p1_admitted")
    if not isinstance(admitted, bool):
        _error(errors, "$.p1_admitted", "must be boolean")
    authorized_stage = document.get("authorized_stage")
    if authorized_stage not in ("none", "pf_1"):
        _error(errors, "$.authorized_stage", "must be none or pf_1")
    reason_codes = _require_array(document.get("reason_codes"), errors, "$.reason_codes")
    for index, code in enumerate(reason_codes):
        if not isinstance(code, str) or not code.strip():
            _error(errors, "$.reason_codes[{}]".format(index), "must be a non-empty string")
    if len(reason_codes) != len(set(code for code in reason_codes if isinstance(code, str))):
        _error(errors, "$.reason_codes", "must not contain duplicate codes")

    release = _require_object(document.get("candidate_release"), errors, "candidate_release")
    _require_exact_keys(release, ADMISSION_RELEASE_FIELDS, errors, "candidate_release")
    release_id = _require_string(release, "release_id", errors, "candidate_release")
    if release.get("release_status") != "candidate":
        _error(errors, "candidate_release.release_status", "must remain candidate; admission is this separate artifact")

    contract_sha = _require_nonzero_sha(
        release, "training_record_contract_sha256", errors, "candidate_release"
    )
    actual_contract_sha = _sha256_file(RECORD_CONTRACT_PATH)
    if contract_sha is not None and contract_sha != actual_contract_sha:
        _error(
            errors,
            "candidate_release.training_record_contract_sha256",
            "must equal the validator's current logical-motif training-record contract SHA-256",
        )

    root = _prepare_artifact_root(artifact_root, errors, required=(decision == "admit"))
    referenced_paths = set()
    for reference_name in ("release_manifest", "membership_manifest", "tokenizer_contract"):
        reference_path = "candidate_release.{}".format(reference_name)
        reference, parts = _validate_reference_structure(
            release.get(reference_name),
            errors,
            reference_path,
            release_id,
            ADMISSION_REFERENCE_KINDS[reference_name],
        )
        if parts is not None:
            normalized = "/".join(parts)
            if normalized in referenced_paths:
                _error(errors, "{}.path".format(reference_path), "must name a distinct referenced artifact")
            referenced_paths.add(normalized)
        extra_bindings = {}
        if reference_name == "release_manifest":
            extra_bindings = {"release_id": release_id, "release_status": "candidate"}
        _resolve_and_bind_json_reference(
            reference,
            parts,
            root,
            errors,
            reference_path,
            extra_bindings=extra_bindings,
        )

    expected_gates = set(BASE_GATE_FIELDS)
    if profile == C3_PROFILE:
        expected_gates.update(C3_GATE_FIELDS)
    evidence = _require_object(document.get("evidence_receipts"), errors, "evidence_receipts")
    _require_exact_keys(evidence, frozenset(expected_gates), errors, "evidence_receipts")
    for gate_name in sorted(expected_gates):
        evidence_path = "evidence_receipts.{}".format(gate_name)
        expected_kind = EVIDENCE_KIND_TEMPLATE.format(gate_name=gate_name)
        receipt, parts = _validate_reference_structure(
            evidence.get(gate_name),
            errors,
            evidence_path,
            release_id,
            expected_kind,
            evidence_gate=gate_name,
        )
        if parts is not None:
            normalized = "/".join(parts)
            if normalized in referenced_paths:
                _error(errors, "{}.path".format(evidence_path), "must name a distinct referenced artifact")
            referenced_paths.add(normalized)
        if decision == "admit" and receipt.get("status") != "pass":
            _error(errors, "{}.status".format(evidence_path), "must be pass for an admit decision")
        _resolve_and_bind_json_reference(receipt, parts, root, errors, evidence_path)

    if decision == "admit":
        if admitted is not True:
            _error(errors, "$.p1_admitted", "must be true for an admit decision")
        if authorized_stage != "pf_1":
            _error(errors, "$.authorized_stage", "an admit decision authorizes pf_1 only")
        if reason_codes:
            _error(errors, "$.reason_codes", "must be empty for an admit decision")
    elif decision == "reject":
        if admitted is not False:
            _error(errors, "$.p1_admitted", "must be false for a reject decision")
        if authorized_stage != "none":
            _error(errors, "$.authorized_stage", "must be none for a reject decision")
        if not reason_codes:
            _error(errors, "$.reason_codes", "must contain at least one reason for a reject decision")
    return _make_report(ADMISSION_KIND, document, errors)


def _make_report(kind, document, errors):
    contract_path = RECORD_CONTRACT_PATH if kind == RECORD_KIND else ADMISSION_CONTRACT_PATH
    return {
        "schema_version": REPORT_SCHEMA,
        "artifact_kind": kind,
        "artifact_sha256": _canonical_sha256(document),
        "contract_path": str(contract_path),
        "contract_sha256": _sha256_file(contract_path),
        "pass": not errors,
        "error_count": len(errors),
        "errors": errors,
    }


def validate_artifact(document, artifact_root=None):
    if isinstance(document, dict) and document.get("document_kind") == RECORD_KIND:
        return validate_training_record(document)
    if isinstance(document, dict) and document.get("document_kind") == ADMISSION_KIND:
        return validate_admission_decision(document, artifact_root=artifact_root)
    errors = []
    _error(errors, "$.document_kind", "must be {} or {}".format(RECORD_KIND, ADMISSION_KIND))
    return _make_report("unknown", document, errors)


def _write_json(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path is None:
        sys.stdout.write(encoded)
    else:
        Path(path).write_text(encoded, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="JSON record or admission-decision artifact")
    parser.add_argument(
        "--artifact-root",
        help="trusted root containing admission references; mandatory for an admit decision",
    )
    parser.add_argument("--output", help="optional validation report path")
    args = parser.parse_args(argv)
    try:
        document = _load_json_evidence(Path(args.artifact))
    except (OSError, UnicodeError, ValueError) as exc:
        report = {
            "schema_version": REPORT_SCHEMA,
            "artifact_kind": "unreadable",
            "pass": False,
            "error_count": 1,
            "errors": [{"path": "$", "message": "cannot read JSON artifact: {}".format(exc)}],
        }
        _write_json(args.output, report)
        return 2
    report = validate_artifact(document, artifact_root=args.artifact_root)
    _write_json(args.output, report)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
