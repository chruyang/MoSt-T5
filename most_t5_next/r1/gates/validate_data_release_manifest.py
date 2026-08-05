#!/usr/bin/env python3
"""Validate a frozen MoSt-T5 R1 P1/P2 data-release manifest.

This is deliberately a small, standard-library-only sidecar gate.  It does
not read an LMDB, launch training, alter the historical project, or copy any
dataset.  With ``--verify-source-locks`` it reads only the already-declared
source paths to recompute their byte counts and hashes; use that option on the
remote host that owns the frozen inputs.

The required contract is documented in
``../contracts/data_release_manifest_contract.json``.  A passing *candidate*
manifest proves that the required evidence has been declared consistently.  A
training launcher must additionally require the corresponding admission flag
and a successful remote source-lock verification report.
"""

from __future__ import print_function

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path


MANIFEST_SCHEMA = "most-t5-r1/data-release-manifest/v3"
REPORT_SCHEMA = "most-t5-r1/data-release-validation-report/v3"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
P1_SAME_MOL_ADAPTER_SCHEMA = "most-t5-r1/p1-same-mol-adapter/v2"
P1_SAME_MOL_DERIVATION = "single_post_projection_geometry_mol"
P1_SOURCE_ATOM_INDEX_TAG = "_r1_source_atom_index"
P1_REQUIRED_MAINLINE_BATCH_FIELDS = (
    "joint_mask_positions",
    "geo_only_mask_positions",
    "geometry_input_mask",
    "unmasked_input_ids",
    "unmasked_atom_attention_mask",
)
P1_GEOMETRY_INPUT_MASK_DEFINITION = "joint_mask_positions OR geo_only_mask_positions"
P1_GEOMETRY_TARGET_MASK_DEFINITION = "geo_only_mask_positions AND token_geometry_valid_mask"

SOURCE_ROLES = frozenset(
    (
        "p1_membership_source",
        "p1_geometry_source",
        "p2_membership_source",
        "p2_geometry_source",
        "p1_adapter_harness",
        "p1_record_schema",
        "pcqm_identity_normalization_contract",
        "tokenizer_base_snapshot",
        "tokenizer_builder_harness",
        "stable_tokenizer_contract",
        "geometry_policy_spec",
        "identity_exclusion_method",
    )
)
DATA_SOURCE_ROLES = frozenset(
    (
        "p1_membership_source",
        "p1_geometry_source",
        "p2_membership_source",
        "p2_geometry_source",
    )
)
# The adapter and serialization schema are source-locked for every P1 release.
# The PCQM identity contract is profile-specific because the legacy 3D-MolM
# control has a different identity namespace and provenance mechanism.
REQUIRED_SOURCE_ROLES = frozenset(
    (
        "p1_membership_source",
        "p1_geometry_source",
        "p2_membership_source",
        "p2_geometry_source",
        "p1_adapter_harness",
        "p1_record_schema",
        "tokenizer_base_snapshot",
        "tokenizer_builder_harness",
        "stable_tokenizer_contract",
        "geometry_policy_spec",
        "identity_exclusion_method",
    )
)
PROFILE_REQUIRED_SOURCE_ROLES = {
    "pcqm4mv2_candidate": frozenset(("pcqm_identity_normalization_contract",)),
}
ALLOWED_LOCATION_SCOPES = frozenset(("remote_shared", "remote_ephemeral", "local_code_only"))
ALLOWED_LOCK_KINDS = frozenset(("file", "directory"))
ALLOWED_REASON_ACTIONS = frozenset(
    (
        "exclude_from_geometry_release",
        "keep_text_or_2d_only_mask_geometry",
        "dedicated_nonpadding_geometry_sentinel",
    )
)


# The fixed controls reflect the audited R0 membership evidence.  PCQM4Mv2 is
# intentionally only a candidate profile: 3D-MolT5's exact post-processing
# member list was not published, so this gate forbids force-matching it.
PROFILE_SPECS = {
    "legacy_3dmolm_control": {
        "p1": {
            "identity_namespace": "pubchemqc_id",
            "source_record_count": 3119717,
            "geometry_admitted_record_count": 3119714,
            "reason_counts": {"E3FP_UNSUPPORTED_H2": 3},
        },
        "p2": {
            "identity_namespace": "pubchem_cid",
            "source_record_count": 301658,
            "geometry_admitted_record_count": 301655,
            "reason_counts": {"E3FP_SINGLE_ATOM_NO_DISTANCE_PAIRS": 3},
        },
    },
    "pcqm4mv2_candidate": {
        "p1": {
            "identity_namespace": "ogb_pcqm4mv2_train_row_index",
            "source_record_count": 3378606,
        },
        "p2": {
            "identity_namespace": "pubchem_cid",
            "source_record_count": 301658,
            "geometry_admitted_record_count": 301655,
            "reason_counts": {"E3FP_SINGLE_ATOM_NO_DISTANCE_PAIRS": 3},
        },
    },
}


def required_source_roles_for_profile(source_profile):
    """Return the immutable source-lock roles required by one profile."""
    return REQUIRED_SOURCE_ROLES.union(PROFILE_REQUIRED_SOURCE_ROLES.get(source_profile, frozenset()))


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def is_nonnegative_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_positive_int(value):
    return is_nonnegative_int(value) and value > 0


def is_sha256(value):
    return isinstance(value, str) and bool(SHA256_RE.match(value))


def sha256_file(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    size = 0
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return {"kind": "file", "bytes": size, "sha256": digest.hexdigest(), "files": 1}


def sha256_directory_tree(path):
    """Hash a directory without following symlinks.

    The returned digest is the SHA-256 of canonical JSON file records, sorted
    by POSIX relative path.  It intentionally differs from a tar checksum: a
    release lock is about immutable file contents, not archive metadata.
    """
    root = Path(path)
    records = []
    total_bytes = 0
    for candidate in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if candidate.is_symlink():
            raise RuntimeError("symlink is forbidden in a directory source lock: {}".format(candidate))
        if candidate.is_file():
            observed = sha256_file(candidate)
            records.append(
                {
                    "relative_path": candidate.relative_to(root).as_posix(),
                    "bytes": observed["bytes"],
                    "sha256": observed["sha256"],
                }
            )
            total_bytes += observed["bytes"]
    encoded = json.dumps(records, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "kind": "directory",
        "bytes": total_bytes,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": len(records),
    }


def hash_locked_path(path, declared_kind):
    target = Path(path)
    if declared_kind == "file":
        if not target.is_file():
            raise RuntimeError("expected regular file, observed {}".format("missing" if not target.exists() else "non-file"))
        return sha256_file(target)
    if declared_kind == "directory":
        if not target.is_dir():
            raise RuntimeError("expected directory, observed {}".format("missing" if not target.exists() else "non-directory"))
        return sha256_directory_tree(target)
    raise RuntimeError("unsupported declared kind: {}".format(declared_kind))


def manifest_hash(manifest_path, manifest):
    if manifest_path and Path(manifest_path).is_file():
        return sha256_file(manifest_path)["sha256"]
    encoded = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def add_error(errors, path, message):
    errors.append({"path": path, "message": message})


def add_warning(warnings, path, message):
    warnings.append({"path": path, "message": message})


def require_mapping(value, errors, path):
    if not isinstance(value, dict):
        add_error(errors, path, "must be an object")
        return {}
    return value


def require_list(value, errors, path):
    if not isinstance(value, list):
        add_error(errors, path, "must be an array")
        return []
    return value


def require_string(mapping, key, errors, path, allow_empty=False):
    value = mapping.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        add_error(errors, "{}.{}".format(path, key), "must be a non-empty string")
        return None
    return value


def require_sha256(mapping, key, errors, path):
    value = mapping.get(key)
    if not is_sha256(value):
        add_error(errors, "{}.{}".format(path, key), "must be a lower-case 64-hex SHA-256")
        return None
    return value


def require_nonnegative_int(mapping, key, errors, path):
    value = mapping.get(key)
    if not is_nonnegative_int(value):
        add_error(errors, "{}.{}".format(path, key), "must be a non-negative integer")
        return None
    return value


def require_bool(mapping, key, expected, errors, path):
    value = mapping.get(key)
    if not isinstance(value, bool):
        add_error(errors, "{}.{}".format(path, key), "must be boolean")
        return None
    if value is not expected:
        add_error(errors, "{}.{}".format(path, key), "must be {!r}".format(expected))
    return value


def resolved_lock_path(raw_path, manifest_path):
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute() or manifest_path is None:
        return candidate
    return Path(manifest_path).resolve().parent / candidate


def validate_release(release, errors, require_p1_admission, require_p2_admission):
    release = require_mapping(release, errors, "release")
    require_string(release, "release_id", errors, "release")
    status = require_string(release, "status", errors, "release")
    if status is not None and status not in ("candidate", "admitted"):
        add_error(errors, "release.status", "must be candidate or admitted")
    scope = require_string(release, "data_transfer_scope", errors, "release")
    if scope is not None and scope != "remote_only":
        add_error(errors, "release.data_transfer_scope", "must be remote_only for dataset artifacts")
    p1_admitted = release.get("p1_admitted")
    p2_admitted = release.get("p2_admitted")
    if not isinstance(p1_admitted, bool):
        add_error(errors, "release.p1_admitted", "must be boolean")
    if not isinstance(p2_admitted, bool):
        add_error(errors, "release.p2_admitted", "must be boolean")
    if p2_admitted is True and p1_admitted is not True:
        add_error(errors, "release.p2_admitted", "cannot be true unless p1_admitted is true")
    if status == "candidate" and (p1_admitted is True or p2_admitted is True):
        add_error(errors, "release.status", "candidate releases cannot authorize P1 or P2")
    if status == "admitted":
        require_sha256(release, "admission_decision_sha256", errors, "release")
        if p1_admitted is not True:
            add_error(errors, "release.p1_admitted", "must be true when release.status is admitted")
    if require_p1_admission and p1_admitted is not True:
        add_error(errors, "release.p1_admitted", "P1 admission was required by this gate invocation")
    if require_p2_admission and p2_admitted is not True:
        add_error(errors, "release.p2_admitted", "P2 admission was required by this gate invocation")
    return release


def validate_source_locks(source_locks, errors, warnings, manifest_path, verify_source_locks, source_profile):
    source_locks = require_list(source_locks, errors, "source_locks")
    by_name = {}
    observed_locks = []
    roles_seen = set()
    for index, raw_lock in enumerate(source_locks):
        path = "source_locks[{}]".format(index)
        lock = require_mapping(raw_lock, errors, path)
        name = require_string(lock, "name", errors, path)
        role = require_string(lock, "role", errors, path)
        kind = require_string(lock, "kind", errors, path)
        raw_location = require_string(lock, "location_scope", errors, path)
        raw_path = require_string(lock, "path", errors, path)
        expected_bytes = require_nonnegative_int(lock, "bytes", errors, path)
        expected_sha = require_sha256(lock, "sha256", errors, path)
        immutable = lock.get("immutable")
        if immutable is not True:
            add_error(errors, "{}.immutable".format(path), "must be true")
        if name is not None:
            if name in by_name:
                add_error(errors, "{}.name".format(path), "duplicates source lock {!r}".format(name))
            else:
                by_name[name] = lock
        if role is not None:
            if role not in SOURCE_ROLES:
                add_error(errors, "{}.role".format(path), "unknown source-lock role {!r}".format(role))
            else:
                roles_seen.add(role)
                if role in DATA_SOURCE_ROLES and raw_location not in ("remote_shared", "remote_ephemeral"):
                    add_error(errors, "{}.location_scope".format(path), "dataset source locks must be remote_shared or remote_ephemeral")
                # Adapter harnesses and record-schema contracts are deliberately
                # allowed to live in remote_shared: the release gate must lock
                # the code artifact actually used beside the remote data.
        if kind is not None and kind not in ALLOWED_LOCK_KINDS:
            add_error(errors, "{}.kind".format(path), "must be file or directory")
        if raw_location is not None and raw_location not in ALLOWED_LOCATION_SCOPES:
            add_error(errors, "{}.location_scope".format(path), "unknown location scope")

        if verify_source_locks and raw_path is not None and kind in ALLOWED_LOCK_KINDS:
            resolved = resolved_lock_path(raw_path, manifest_path)
            observation = {"name": name, "declared_path": raw_path, "resolved_path": str(resolved)}
            try:
                measured = hash_locked_path(resolved, kind)
            except Exception as exc:
                add_error(errors, "{}.path".format(path), "source-lock verification failed: {}".format(exc))
                observation["verified"] = False
                observation["error"] = str(exc)
            else:
                observation.update(measured)
                observation["verified"] = measured["bytes"] == expected_bytes and measured["sha256"] == expected_sha
                if measured["bytes"] != expected_bytes:
                    add_error(
                        errors,
                        "{}.bytes".format(path),
                        "declared {} differs from observed {}".format(expected_bytes, measured["bytes"]),
                    )
                if measured["sha256"] != expected_sha:
                    add_error(errors, "{}.sha256".format(path), "declared hash differs from observed hash")
            observed_locks.append(observation)
        elif not verify_source_locks:
            add_warning(
                warnings,
                path,
                "source bytes and hashes were declared but not recomputed; rerun remotely with --verify-source-locks before admission",
            )

    missing_roles = sorted(required_source_roles_for_profile(source_profile).difference(roles_seen))
    if missing_roles:
        add_error(errors, "source_locks", "missing required roles: {}".format(", ".join(missing_roles)))
    return by_name, observed_locks


def required_phase_source_roles(phase_name, source_profile):
    """Return roles that one phase must explicitly reference."""
    roles = ["{}_membership_source".format(phase_name), "{}_geometry_source".format(phase_name)]
    if phase_name == "p1":
        roles.extend(("p1_adapter_harness", "p1_record_schema"))
        if source_profile == "pcqm4mv2_candidate":
            roles.append("pcqm_identity_normalization_contract")
    return tuple(roles)


def validate_phase_source_refs(phase_name, phase, source_profile, locks_by_name, errors):
    refs = require_list(phase.get("source_lock_names"), errors, "phases.{}.source_lock_names".format(phase_name))
    ref_locks = []
    for index, name in enumerate(refs):
        ref_path = "phases.{}.source_lock_names[{}]".format(phase_name, index)
        if not isinstance(name, str) or not name:
            add_error(errors, ref_path, "must be a non-empty source-lock name")
        elif name not in locks_by_name:
            add_error(errors, ref_path, "does not refer to a declared source lock")
        else:
            ref_locks.append(locks_by_name[name])
    required_roles = required_phase_source_roles(phase_name, source_profile)
    referred_roles = set(lock.get("role") for lock in ref_locks)
    for role in required_roles:
        if role not in referred_roles:
            add_error(errors, "phases.{}.source_lock_names".format(phase_name), "must include a lock with role {}".format(role))
    return ref_locks


def require_referenced_lock_hash(referenced_locks, role, declared_sha, errors, path):
    """Require one phase-referenced immutable lock with a matching role/hash."""
    if declared_sha is None:
        return
    if not any(lock.get("role") == role and lock.get("sha256") == declared_sha for lock in referenced_locks):
        add_error(
            errors,
            path,
            "must match a P1-referenced immutable source lock with role {}".format(role),
        )


def validate_p1_same_mol_adapter(phase, source_profile, referenced_locks, errors):
    """Validate the P1 sidecar's one-RDKit-Mol alignment declaration.

    This gate verifies the immutable code/schema bindings and the declared
    invariants.  Record-level enforcement remains the responsibility of the
    separately locked adapter harness and sidecar validator.
    """
    path = "phases.p1.same_mol_adapter"
    adapter = require_mapping(phase.get("same_mol_adapter"), errors, path)

    schema_version = require_string(adapter, "schema_version", errors, path)
    if schema_version is not None and schema_version != P1_SAME_MOL_ADAPTER_SCHEMA:
        add_error(errors, "{}.schema_version".format(path), "must be {}".format(P1_SAME_MOL_ADAPTER_SCHEMA))

    harness_sha = require_sha256(adapter, "adapter_harness_sha256", errors, path)
    record_schema_sha = require_sha256(adapter, "record_schema_sha256", errors, path)
    require_referenced_lock_hash(referenced_locks, "p1_adapter_harness", harness_sha, errors, "{}.adapter_harness_sha256".format(path))
    require_referenced_lock_hash(referenced_locks, "p1_record_schema", record_schema_sha, errors, "{}.record_schema_sha256".format(path))

    same_mol_derivation = require_string(adapter, "same_mol_derivation", errors, path)
    if same_mol_derivation is not None and same_mol_derivation != P1_SAME_MOL_DERIVATION:
        add_error(errors, "{}.same_mol_derivation".format(path), "must be {}".format(P1_SAME_MOL_DERIVATION))
    source_index_tag = require_string(adapter, "source_atom_index_tag", errors, path)
    if source_index_tag is not None and source_index_tag != P1_SOURCE_ATOM_INDEX_TAG:
        add_error(errors, "{}.source_atom_index_tag".format(path), "must be {}".format(P1_SOURCE_ATOM_INDEX_TAG))

    for key in (
        "source_atom_index_tagged_pre_removehs",
        "retained_source_index_validation_required",
        "compacted_index_inference_forbidden",
        "smiles_rebuild_for_alignment_forbidden",
        "frozen_mol_linearizer_required",
    ):
        require_bool(adapter, key, True, errors, path)

    required_batch_fields = require_list(adapter.get("required_batch_fields"), errors, "{}.required_batch_fields".format(path))
    if required_batch_fields != list(P1_REQUIRED_MAINLINE_BATCH_FIELDS):
        add_error(
            errors,
            "{}.required_batch_fields".format(path),
            "must be exactly {}".format(", ".join(P1_REQUIRED_MAINLINE_BATCH_FIELDS)),
        )
    require_bool(adapter, "legacy_mask_positions_forbidden", True, errors, path)
    require_bool(adapter, "geometry_target_mask_required", True, errors, path)

    geometry_input_definition = require_string(adapter, "geometry_input_mask_definition", errors, path)
    if geometry_input_definition is not None and geometry_input_definition != P1_GEOMETRY_INPUT_MASK_DEFINITION:
        add_error(
            errors,
            "{}.geometry_input_mask_definition".format(path),
            "must be {}".format(P1_GEOMETRY_INPUT_MASK_DEFINITION),
        )

    geometry_target_definition = require_string(adapter, "geometry_target_mask_definition", errors, path)
    if geometry_target_definition is not None and geometry_target_definition != P1_GEOMETRY_TARGET_MASK_DEFINITION:
        add_error(
            errors,
            "{}.geometry_target_mask_definition".format(path),
            "must be {}".format(P1_GEOMETRY_TARGET_MASK_DEFINITION),
        )

    if "mask_positions" in adapter:
        add_error(
            errors,
            "{}.mask_positions".format(path),
            "legacy mask_positions field is forbidden; declare the V1 split masks instead",
        )

    if source_profile == "pcqm4mv2_candidate":
        identity_sha = require_sha256(adapter, "identity_normalization_contract_sha256", errors, path)
        require_referenced_lock_hash(
            referenced_locks,
            "pcqm_identity_normalization_contract",
            identity_sha,
            errors,
            "{}.identity_normalization_contract_sha256".format(path),
        )


def validate_reason_counts(raw_reasons, errors, path):
    reasons = require_mapping(raw_reasons, errors, path)
    clean = {}
    for reason, count in reasons.items():
        if not isinstance(reason, str) or not reason:
            add_error(errors, path, "reason names must be non-empty strings")
            continue
        if not is_positive_int(count):
            add_error(errors, "{}.{}".format(path, reason), "must be a positive integer")
            continue
        clean[reason] = count
    return clean


def validate_phase_membership(phase_name, phase, profile_spec, errors):
    membership_path = "phases.{}.membership".format(phase_name)
    membership = require_mapping(phase.get("membership"), errors, membership_path)
    namespace = require_string(membership, "identity_namespace", errors, membership_path)
    source_count = require_nonnegative_int(membership, "source_record_count", errors, membership_path)
    admitted_count = require_nonnegative_int(membership, "geometry_admitted_record_count", errors, membership_path)
    require_sha256(membership, "member_ids_sha256", errors, membership_path)
    manifest_sha = require_sha256(membership, "manifest_file_sha256", errors, membership_path)
    text_only_count = membership.get("text_or_2d_only_record_count", 0)
    if not is_nonnegative_int(text_only_count):
        add_error(errors, "{}.text_or_2d_only_record_count".format(membership_path), "must be a non-negative integer")
        text_only_count = 0
    if phase_name == "p1" and text_only_count != 0:
        add_error(errors, "{}.text_or_2d_only_record_count".format(membership_path), "must be zero for structural-only P1")

    ledger_path = "{}.reject_ledger".format(membership_path)
    ledger = require_mapping(membership.get("reject_ledger"), errors, ledger_path)
    require_sha256(ledger, "file_sha256", errors, ledger_path)
    reject_count = require_nonnegative_int(ledger, "record_count", errors, ledger_path)
    require_sha256(ledger, "ids_sha256", errors, ledger_path)
    reason_counts = validate_reason_counts(ledger.get("reason_counts"), errors, "{}.reason_counts".format(ledger_path))
    if reject_count is not None and sum(reason_counts.values()) != reject_count:
        add_error(errors, "{}.record_count".format(ledger_path), "must equal the sum of reason_counts")
    if source_count is not None and admitted_count is not None and reject_count is not None:
        if source_count != admitted_count + reject_count:
            add_error(errors, membership_path, "source_record_count must equal geometry_admitted_record_count plus reject_ledger.record_count")
    if reject_count is not None and text_only_count > reject_count:
        add_error(errors, "{}.text_or_2d_only_record_count".format(membership_path), "cannot exceed reject_ledger.record_count")

    if namespace is not None and namespace != profile_spec["identity_namespace"]:
        add_error(errors, "{}.identity_namespace".format(membership_path), "must be {} for the selected source profile".format(profile_spec["identity_namespace"]))
    if source_count is not None and source_count != profile_spec["source_record_count"]:
        add_error(errors, "{}.source_record_count".format(membership_path), "must be {} for the selected source profile".format(profile_spec["source_record_count"]))
    expected_admitted = profile_spec.get("geometry_admitted_record_count")
    if expected_admitted is not None and admitted_count is not None and admitted_count != expected_admitted:
        add_error(errors, "{}.geometry_admitted_record_count".format(membership_path), "must be {} for the selected source profile".format(expected_admitted))
    expected_reasons = profile_spec.get("reason_counts")
    if expected_reasons is not None and reason_counts != expected_reasons:
        add_error(errors, "{}.reason_counts".format(ledger_path), "must exactly match the R0-audited reason counts for this source profile")

    return {
        "membership_manifest_sha256": manifest_sha,
        "reason_counts": reason_counts,
        "text_or_2d_only_record_count": text_only_count,
    }


def validate_phases(phases, source_profile, locks_by_name, errors):
    phases = require_mapping(phases, errors, "phases")
    result = {}
    for phase_name in ("p1", "p2"):
        phase_path = "phases.{}".format(phase_name)
        phase = require_mapping(phases.get(phase_name), errors, phase_path)
        referenced_locks = validate_phase_source_refs(phase_name, phase, source_profile, locks_by_name, errors)
        if phase_name == "p1":
            validate_p1_same_mol_adapter(phase, source_profile, referenced_locks, errors)
        result[phase_name] = validate_phase_membership(phase_name, phase, PROFILE_SPECS[source_profile][phase_name], errors)
    return result


def validate_geometry_policy(policy, phase_state, locks_by_name, errors):
    path = "geometry_mask_policy"
    policy = require_mapping(policy, errors, path)
    require_string(policy, "schema_version", errors, path)
    policy_sha = require_sha256(policy, "policy_file_sha256", errors, path)
    if policy_sha is not None:
        matching_locks = [
            lock
            for lock in locks_by_name.values()
            if lock.get("role") == "geometry_policy_spec" and lock.get("sha256") == policy_sha
        ]
        if not matching_locks:
            add_error(errors, "{}.policy_file_sha256".format(path), "must match a geometry_policy_spec source lock")
    scope = require_list(policy.get("mse_task_scope"), errors, "{}.mse_task_scope".format(path))
    if scope != ["mmm"]:
        add_error(errors, "{}.mse_task_scope".format(path), "must be exactly ['mmm'] for this R1 contract")
    require_bool(policy, "per_example_geometry_validity_mask", True, errors, path)
    require_bool(policy, "zero_vector_as_geometry_target_forbidden", True, errors, path)
    require_bool(policy, "mask_statistics_required", True, errors, path)
    if policy.get("padding_mapped_atom_action") != "mask_or_exclude":
        add_error(errors, "{}.padding_mapped_atom_action".format(path), "must be mask_or_exclude")
    if policy.get("geometryless_motif_action") != "mask_or_exclude":
        add_error(errors, "{}.geometryless_motif_action".format(path), "must be mask_or_exclude")
    require_sha256(policy, "mask_statistics_manifest_sha256", errors, path)
    known_counts = require_mapping(policy.get("known_invalid_geometry_case_counts"), errors, "{}.known_invalid_geometry_case_counts".format(path))
    require_nonnegative_int(known_counts, "padding_mapped_atom_count", errors, "{}.known_invalid_geometry_case_counts".format(path))
    require_nonnegative_int(known_counts, "geometryless_motif_group_count", errors, "{}.known_invalid_geometry_case_counts".format(path))

    actions = require_mapping(policy.get("reject_reason_actions"), errors, "{}.reject_reason_actions".format(path))
    action_by_reason = {}
    all_reasons = {}
    for phase_name, state in phase_state.items():
        for reason, count in state["reason_counts"].items():
            all_reasons[reason] = all_reasons.get(reason, 0) + count
    for reason in sorted(all_reasons):
        action_path = "{}.reject_reason_actions.{}".format(path, reason)
        action = require_mapping(actions.get(reason), errors, action_path)
        release_action = require_string(action, "release_action", errors, action_path)
        mse_enabled = action.get("geometry_mse_enabled")
        if release_action is not None and release_action not in ALLOWED_REASON_ACTIONS:
            add_error(errors, "{}.release_action".format(action_path), "is not an allowed explicit geometry disposition")
        if mse_enabled is not False:
            add_error(errors, "{}.geometry_mse_enabled".format(action_path), "must be false for every rejected geometry record")
        if release_action == "dedicated_nonpadding_geometry_sentinel":
            require_sha256(action, "sentinel_validation_sha256", errors, action_path)
        action_by_reason[reason] = release_action

    h2_action = action_by_reason.get("E3FP_UNSUPPORTED_H2")
    if h2_action is not None and h2_action not in (
        "exclude_from_geometry_release",
        "dedicated_nonpadding_geometry_sentinel",
    ):
        add_error(errors, "{}.reject_reason_actions.E3FP_UNSUPPORTED_H2.release_action".format(path), "cannot silently retain H2 in a geometry branch")
    singleton_action = action_by_reason.get("E3FP_SINGLE_ATOM_NO_DISTANCE_PAIRS")
    if singleton_action is not None and singleton_action not in ALLOWED_REASON_ACTIONS:
        add_error(errors, "{}.reject_reason_actions.E3FP_SINGLE_ATOM_NO_DISTANCE_PAIRS.release_action".format(path), "must explicitly exclude, 2D/text-route, or use a validated sentinel")

    p2_text_only = phase_state["p2"]["text_or_2d_only_record_count"]
    if p2_text_only:
        permitted = sum(
            count
            for reason, count in phase_state["p2"]["reason_counts"].items()
            if action_by_reason.get(reason) == "keep_text_or_2d_only_mask_geometry"
        )
        if p2_text_only > permitted:
            add_error(errors, "phases.p2.membership.text_or_2d_only_record_count", "requires enough reject records with keep_text_or_2d_only_mask_geometry action")


def validate_identity_exclusion(exclusion, locks_by_name, errors):
    path = "downstream_identity_exclusion"
    exclusion = require_mapping(exclusion, errors, path)
    method_sha = require_sha256(exclusion, "method_sha256", errors, path)
    if method_sha is not None:
        matching_locks = [
            lock
            for lock in locks_by_name.values()
            if lock.get("role") == "identity_exclusion_method" and lock.get("sha256") == method_sha
        ]
        if not matching_locks:
            add_error(errors, "{}.method_sha256".format(path), "must match an identity_exclusion_method source lock")
    require_sha256(exclusion, "downstream_validation_test_ids_sha256", errors, path)
    for phase_name in ("p1", "p2"):
        key = "{}_overlap_count".format(phase_name)
        count = require_nonnegative_int(exclusion, key, errors, path)
        if count is not None and count != 0:
            add_error(errors, "{}.{}".format(path, key), "must be zero before P1/P2 admission")


def validate_cross_phase_overlap(overlap, errors):
    path = "cross_phase_overlap"
    overlap = require_mapping(overlap, errors, path)
    policy = require_string(overlap, "policy", errors, path)
    if policy is not None and policy not in ("explicitly_declared", "disjoint_required", "replay_permitted"):
        add_error(errors, "{}.policy".format(path), "must explicitly declare the P1/P2 overlap policy")
    count = require_nonnegative_int(overlap, "overlap_count", errors, path)
    if policy == "disjoint_required" and count is not None and count != 0:
        add_error(errors, "{}.overlap_count".format(path), "must be zero for disjoint_required")
    require_sha256(overlap, "evidence_sha256", errors, path)


def validate_tokenizer(tokenizer, phase_state, locks_by_name, errors):
    path = "tokenizer"
    tokenizer = require_mapping(tokenizer, errors, path)
    contract_sha = require_sha256(tokenizer, "contract_manifest_sha256", errors, path)
    base_sha = require_sha256(tokenizer, "base_snapshot_tree_sha256", errors, path)
    require_sha256(tokenizer, "builder_harness_sha256", errors, path)
    require_sha256(tokenizer, "special_token_spec_sha256", errors, path)
    require_sha256(tokenizer, "ordered_motif_vocab_sha256", errors, path)
    id_to_token_sha = require_sha256(tokenizer, "id_to_token_sha256", errors, path)
    vocab_size = tokenizer.get("vocab_size")
    if not is_positive_int(vocab_size):
        add_error(errors, "{}.vocab_size".format(path), "must be a positive integer")
    require_bool(tokenizer, "p1_p2_exact_same_mapping", True, errors, path)
    require_bool(tokenizer, "p2_vocab_extension_forbidden", True, errors, path)
    permitted = require_mapping(tokenizer.get("permitted_membership_manifest_sha256"), errors, "{}.permitted_membership_manifest_sha256".format(path))
    for phase_name in ("p1", "p2"):
        actual = permitted.get(phase_name)
        if not is_sha256(actual):
            add_error(errors, "{}.permitted_membership_manifest_sha256.{}".format(path, phase_name), "must be a lower-case 64-hex SHA-256")
        elif actual != phase_state[phase_name]["membership_manifest_sha256"]:
            add_error(errors, "{}.permitted_membership_manifest_sha256.{}".format(path, phase_name), "must match the frozen phase membership manifest")

    deterministic = require_mapping(tokenizer.get("determinism_gate"), errors, "{}.determinism_gate".format(path))
    if deterministic.get("passed") is not True:
        add_error(errors, "{}.determinism_gate.passed".format(path), "must be true")
    seeds = require_list(deterministic.get("pythonhashseeds"), errors, "{}.determinism_gate.pythonhashseeds".format(path))
    if len(seeds) < 3 or len(set(str(seed) for seed in seeds)) < 3:
        add_error(errors, "{}.determinism_gate.pythonhashseeds".format(path), "must record at least three distinct hash-seed values")
    observed_mapping = require_sha256(deterministic, "id_to_token_sha256", errors, "{}.determinism_gate".format(path))
    if observed_mapping is not None and id_to_token_sha is not None and observed_mapping != id_to_token_sha:
        add_error(errors, "{}.determinism_gate.id_to_token_sha256".format(path), "must equal tokenizer.id_to_token_sha256")

    base_locks = [lock for lock in locks_by_name.values() if lock.get("role") == "tokenizer_base_snapshot"]
    if base_sha is not None and not any(lock.get("sha256") == base_sha for lock in base_locks):
        add_error(errors, "{}.base_snapshot_tree_sha256".format(path), "must match a tokenizer_base_snapshot source lock")
    builder_sha = tokenizer.get("builder_harness_sha256")
    builder_locks = [lock for lock in locks_by_name.values() if lock.get("role") == "tokenizer_builder_harness"]
    if is_sha256(builder_sha) and not any(lock.get("sha256") == builder_sha for lock in builder_locks):
        add_error(errors, "{}.builder_harness_sha256".format(path), "must match a tokenizer_builder_harness source lock")
    contract_locks = [lock for lock in locks_by_name.values() if lock.get("role") == "stable_tokenizer_contract"]
    if contract_sha is not None and not any(lock.get("sha256") == contract_sha for lock in contract_locks):
        add_error(errors, "{}.contract_manifest_sha256".format(path), "must match a stable_tokenizer_contract source lock")


def validate_checkpoint_prerequisites(value, errors):
    path = "checkpoint_prerequisites.p1_to_p2"
    root = require_mapping(value, errors, "checkpoint_prerequisites")
    contract = require_mapping(root.get("p1_to_p2"), errors, path)
    expected = {
        "legacy_p1_checkpoint_allowed": False,
        "same_tokenizer_mapping_required": True,
        "strict_load_required": True,
        "ignore_mismatched_sizes_forbidden": True,
        "equal_vocab_size_required": True,
        "embedding_shape_match_required": True,
        "checkpoint_tokenizer_snapshot_required": True,
        "checkpoint_release_manifest_required": True,
    }
    for key, wanted in expected.items():
        require_bool(contract, key, wanted, errors, path)


def validate_release_manifest(
    manifest,
    manifest_path=None,
    verify_source_locks=False,
    require_p1_admission=False,
    require_p2_admission=False,
):
    """Return a deterministic report dictionary for a parsed manifest."""
    errors = []
    warnings = []
    manifest = require_mapping(manifest, errors, "manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        add_error(errors, "schema_version", "must be {}".format(MANIFEST_SCHEMA))
    release = validate_release(
        manifest.get("release"), errors, require_p1_admission=require_p1_admission, require_p2_admission=require_p2_admission
    )
    source_profile = manifest.get("source_profile")
    if source_profile not in PROFILE_SPECS:
        add_error(errors, "source_profile", "must be one of {}".format(", ".join(sorted(PROFILE_SPECS))))
        source_profile = "legacy_3dmolm_control"
    locks_by_name, observed_locks = validate_source_locks(
        manifest.get("source_locks"),
        errors,
        warnings,
        manifest_path=manifest_path,
        verify_source_locks=verify_source_locks,
        source_profile=source_profile,
    )
    phase_state = validate_phases(manifest.get("phases"), source_profile, locks_by_name, errors)
    validate_geometry_policy(manifest.get("geometry_mask_policy"), phase_state, locks_by_name, errors)
    validate_identity_exclusion(manifest.get("downstream_identity_exclusion"), locks_by_name, errors)
    validate_cross_phase_overlap(manifest.get("cross_phase_overlap"), errors)
    validate_tokenizer(manifest.get("tokenizer"), phase_state, locks_by_name, errors)
    validate_checkpoint_prerequisites(manifest.get("checkpoint_prerequisites"), errors)

    return {
        "schema_version": REPORT_SCHEMA,
        "created_utc": utc_now(),
        "manifest_path": str(Path(manifest_path).resolve()) if manifest_path else None,
        "manifest_sha256": manifest_hash(manifest_path, manifest),
        "source_profile": source_profile,
        "release": {
            "release_id": release.get("release_id"),
            "status": release.get("status"),
            "p1_admitted": release.get("p1_admitted"),
            "p2_admitted": release.get("p2_admitted"),
        },
        "source_lock_verification_requested": bool(verify_source_locks),
        "source_lock_observations": observed_locks,
        "pass": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "next_action": (
            "A candidate passed structural checks. Run this validator remotely with --verify-source-locks and obtain an explicit admission decision before P1."
            if not errors
            else "Repair every listed contract error; do not launch P1 or P2 from this manifest."
        ),
    }


def write_json_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite existing validation report: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    if temporary.exists():
        raise FileExistsError("temporary validation report already exists: {}".format(temporary))
    try:
        with open(str(temporary), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(str(temporary), str(path))
    except Exception:
        # Preserve a failed temporary report for inspection rather than
        # deleting it automatically.  A later release attempt must use a
        # distinct explicit output path.
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="frozen candidate/admitted release manifest JSON")
    parser.add_argument("--output", default="", help="new sidecar validation report JSON; never overwritten")
    parser.add_argument(
        "--verify-source-locks",
        action="store_true",
        help="recompute declared file/directory locks; run this on the remote host owning data",
    )
    parser.add_argument("--require-p1-admission", action="store_true")
    parser.add_argument("--require-p2-admission", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).expanduser().resolve()
    try:
        with open(str(manifest_path), "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:
        parser.error("cannot read manifest {}: {}".format(manifest_path, exc))
    report = validate_release_manifest(
        manifest,
        manifest_path=manifest_path,
        verify_source_locks=args.verify_source_locks,
        require_p1_admission=args.require_p1_admission,
        require_p2_admission=args.require_p2_admission,
    )
    if args.output:
        try:
            write_json_new(Path(args.output).expanduser(), report)
        except Exception as exc:
            parser.error(str(exc))
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "errors": report["error_count"],
                "warnings": report["warning_count"],
                "output": str(Path(args.output).expanduser().resolve()) if args.output else None,
            },
            sort_keys=True,
        )
    )
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
