#!/usr/bin/env python3
"""Prove P1/P2/downstream identity overlap from immutable hash-only manifests.

The gate deliberately consumes normalized identity rows rather than molecule
payloads.  It never compares dataset-local IDs across namespaces, and it keeps
connectivity, stereo, conformer, text, and molecule-text-pair evidence
separate.  SQLite is used as an exact set engine so a full PCQM manifest does
not need to be materialized as Python sets.

This script is a proof consumer, not a molecule-identity extractor.  An
extractor must independently bind its source release, code, normalization
specification, excluded metadata keys, and resulting canonical JSONL files in
one collection manifest.
"""

from __future__ import print_function

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sqlite3
import sys
from pathlib import Path


CONTRACT_SCHEMA = "most-t5-r1/p1-p2-downstream-overlap-proof-contract/v1"
REQUEST_SCHEMA = "most-t5-r1/p1-p2-downstream-overlap-proof-request/v1"
COLLECTION_SCHEMA = "most-t5-r1/identity-collection-manifest/v1"
MOLECULE_ROW_SCHEMA = "most-t5-r1/molecule-identity-row/v1"
TEXT_ROW_SCHEMA = "most-t5-r1/text-pair-identity-row/v1"
REPORT_SCHEMA = "most-t5-r1/p1-p2-downstream-overlap-proof-report/v1"

SHA256_HEX = frozenset("0123456789abcdef")
MOLECULE_FIELDS = frozenset(
    (
        "schema_version",
        "collection_id",
        "member_id",
        "connectivity_identity_sha256",
        "stereo_identity_sha256",
        "conformer_identity_sha256",
    )
)
TEXT_FIELDS = frozenset(
    (
        "schema_version",
        "collection_id",
        "pair_id",
        "member_id",
        "task_family",
        "text_exact_sha256",
        "text_normalized_sha256",
        "connectivity_text_pair_sha256",
        "stereo_text_pair_sha256",
    )
)
COLLECTION_FIELDS = frozenset(
    (
        "schema_version",
        "collection_id",
        "dataset_id",
        "release_id",
        "phase",
        "split",
        "role",
        "task_family",
        "identity_specs",
        "molecule_rows",
        "text_pair_rows",
        "provenance",
    )
)
ARTIFACT_FIELDS = frozenset(("path", "bytes", "sha256", "row_count", "key_lf_sha256"))
PROVENANCE_FIELDS = frozenset(
    (
        "source_identity_namespace",
        "source_release_manifest_sha256",
        "extractor_sha256",
        "excluded_source_metadata_keys",
    )
)
IDENTITY_SPEC_FIELDS = frozenset(
    (
        "connectivity_identity_spec_sha256",
        "stereo_identity_spec_sha256",
        "conformer_identity",
        "text_identity",
    )
)
OPTIONAL_SPEC_FIELDS = frozenset(("status", "spec_sha256"))
TEXT_SPEC_FIELDS = frozenset(("status", "exact_spec_sha256", "normalized_spec_sha256"))
REQUEST_FIELDS = frozenset(
    ("schema_version", "request_id", "contract_sha256", "collections", "comparisons", "coverage")
)
COLLECTION_REF_FIELDS = frozenset(("manifest_path", "manifest_sha256"))
COMPARISON_FIELDS = frozenset(
    (
        "comparison_id",
        "left_collection_id",
        "right_collection_id",
        "relationship",
        "policy",
        "required_zero",
        "report_only",
    )
)
COVERAGE_FIELDS = frozenset(
    (
        "required_collection_roles",
        "required_downstream_task_splits",
        "downstream_eval_splits",
        "require_p1_p2_comparison",
        "require_each_pretrain_vs_each_downstream_eval",
        "require_within_task_split_comparisons",
    )
)
TASK_SPLIT_FIELDS = frozenset(("task_family", "splits"))

DIMENSIONS = (
    "connectivity_identity",
    "stereo_identity",
    "conformer_identity",
    "text_exact",
    "text_normalized",
    "connectivity_text_pair",
    "stereo_text_pair",
)
DIMENSION_TABLE_COLUMN = {
    "connectivity_identity": ("molecules", "connectivity_sha256"),
    "stereo_identity": ("molecules", "stereo_sha256"),
    "conformer_identity": ("molecules", "conformer_sha256"),
    "text_exact": ("text_pairs", "text_exact_sha256"),
    "text_normalized": ("text_pairs", "text_normalized_sha256"),
    "connectivity_text_pair": ("text_pairs", "connectivity_pair_sha256"),
    "stereo_text_pair": ("text_pairs", "stereo_pair_sha256"),
}
ROLES = frozenset(
    (
        "p1_structure_train",
        "p2_permitted_train_membership",
        "p2_alignment_train",
        "p2_geometry_replay_train",
        "downstream_train",
        "downstream_validation",
        "downstream_test",
    )
)
RELATIONSHIPS = frozenset(
    (
        "p1_to_p2",
        "pretrain_to_downstream_eval",
        "downstream_within_task_split",
        "declared_additional",
    )
)
POLICIES = frozenset(("disjoint_required", "explicitly_declared", "replay_permitted"))
PRETRAIN_ROLES = frozenset(
    (
        "p1_structure_train",
        "p2_permitted_train_membership",
        "p2_alignment_train",
        "p2_geometry_replay_train",
    )
)
DOWNSTREAM_ROLES = frozenset(("downstream_train", "downstream_validation", "downstream_test"))


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def canonical_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    size = 0
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and not (set(value) - SHA256_HEX)


def require_sha256(value, label, nullable=False):
    if value is None and nullable:
        return
    if not is_sha256(value):
        raise ValueError("{} must be a lowercase SHA-256{}".format(label, " or null" if nullable else ""))


def require_string(value, label):
    if not isinstance(value, str) or not value or any(character in value for character in "\x00\r\n\t"):
        raise ValueError("{} must be a non-empty control-free string".format(label))
    return value


def require_exact_fields(value, expected, label):
    if not isinstance(value, dict):
        raise ValueError("{} must be an object".format(label))
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            "{} fields differ; missing={}, extra={}".format(
                label, sorted(expected - actual), sorted(actual - expected)
            )
        )


def regular_nonsymlink(path, label):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("{} is not a regular non-symlink file: {}".format(label, path))
    return path


def resolve_path(raw_path, parent, label):
    require_string(raw_path, label)
    result = Path(raw_path)
    if not result.is_absolute():
        result = parent / result
    regular_nonsymlink(result, label)
    return result.resolve()


def reject_duplicate_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key: {}".format(key))
        value[key] = item
    return value


def reject_nonfinite_json_constant(value):
    raise ValueError("non-finite JSON constant is forbidden: {}".format(value))


def load_json(path, label):
    path = regular_nonsymlink(path, label)
    with open(str(path), "r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_nonfinite_json_constant,
        )
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(label))
    return value


def validate_artifact_declaration(value, label):
    require_exact_fields(value, ARTIFACT_FIELDS, label)
    require_string(value["path"], "{}.path".format(label))
    if not isinstance(value["bytes"], int) or isinstance(value["bytes"], bool) or value["bytes"] < 0:
        raise ValueError("{}.bytes must be a non-negative integer".format(label))
    if not isinstance(value["row_count"], int) or isinstance(value["row_count"], bool) or value["row_count"] < 0:
        raise ValueError("{}.row_count must be a non-negative integer".format(label))
    require_sha256(value["sha256"], "{}.sha256".format(label))
    require_sha256(value["key_lf_sha256"], "{}.key_lf_sha256".format(label))


def pair_digest(domain, molecule_identity_sha256, text_normalized_sha256):
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": domain,
                "molecule_identity_sha256": molecule_identity_sha256,
                "text_normalized_sha256": text_normalized_sha256,
            }
        )
    )


def create_database(path):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.executescript(
        """
        CREATE TABLE molecules (
            collection_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            connectivity_sha256 TEXT NOT NULL,
            stereo_sha256 TEXT NOT NULL,
            conformer_sha256 TEXT,
            PRIMARY KEY (collection_id, member_id)
        ) WITHOUT ROWID;
        CREATE TABLE text_pairs (
            collection_id TEXT NOT NULL,
            pair_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            task_family TEXT NOT NULL,
            text_exact_sha256 TEXT NOT NULL,
            text_normalized_sha256 TEXT NOT NULL,
            connectivity_pair_sha256 TEXT NOT NULL,
            stereo_pair_sha256 TEXT NOT NULL,
            PRIMARY KEY (collection_id, pair_id),
            FOREIGN KEY (collection_id, member_id)
                REFERENCES molecules(collection_id, member_id)
        ) WITHOUT ROWID;
        """
    )
    return connection


def iter_canonical_jsonl(path, label):
    with open(str(path), "rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw or raw == b"\n" or not raw.endswith(b"\n"):
                raise ValueError("{} line {} is blank or lacks one terminal LF".format(label, line_number))
            try:
                value = json.loads(
                    raw[:-1].decode("utf-8"),
                    object_pairs_hook=reject_duplicate_pairs,
                    parse_constant=reject_nonfinite_json_constant,
                )
            except Exception as exc:
                raise ValueError("{} line {} is not UTF-8 JSON: {}".format(label, line_number, exc))
            if not isinstance(value, dict):
                raise ValueError("{} line {} is not an object".format(label, line_number))
            if canonical_json_bytes(value) + b"\n" != raw:
                raise ValueError("{} line {} is not canonical JSONL".format(label, line_number))
            yield value, raw


def load_molecule_rows(connection, collection, manifest_path):
    declaration = collection["molecule_rows"]
    validate_artifact_declaration(declaration, "molecule_rows")
    path = resolve_path(declaration["path"], manifest_path.parent, "molecule_rows.path")
    file_digest = hashlib.sha256()
    key_digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    previous_key = None
    collection_id = collection["collection_id"]
    conformer_status = collection["identity_specs"]["conformer_identity"]["status"]
    batch = []
    for row, raw in iter_canonical_jsonl(path, "molecule rows"):
        file_digest.update(raw)
        byte_count += len(raw)
        require_exact_fields(row, MOLECULE_FIELDS, "molecule row")
        if row["schema_version"] != MOLECULE_ROW_SCHEMA or row["collection_id"] != collection_id:
            raise ValueError("molecule row schema/collection binding mismatch")
        member_id = require_string(row["member_id"], "molecule row member_id")
        encoded_key = member_id.encode("utf-8")
        if previous_key is not None and encoded_key <= previous_key:
            raise ValueError("molecule rows are not strictly UTF-8-key sorted or contain duplicates")
        previous_key = encoded_key
        key_digest.update(encoded_key + b"\n")
        connectivity = row["connectivity_identity_sha256"]
        stereo = row["stereo_identity_sha256"]
        conformer = row["conformer_identity_sha256"]
        require_sha256(connectivity, "molecule connectivity identity")
        require_sha256(stereo, "molecule stereo identity")
        require_sha256(conformer, "molecule conformer identity", nullable=True)
        if conformer_status == "available" and conformer is None:
            raise ValueError("available conformer specification requires every molecule row to carry a conformer identity")
        if conformer_status == "unavailable" and conformer is not None:
            raise ValueError("unavailable conformer specification forbids row-level conformer identities")
        batch.append((collection_id, member_id, connectivity, stereo, conformer))
        if len(batch) >= 10000:
            connection.executemany("INSERT INTO molecules VALUES (?,?,?,?,?)", batch)
            batch = []
        row_count += 1
    if batch:
        connection.executemany("INSERT INTO molecules VALUES (?,?,?,?,?)", batch)
    observed = {
        "path": str(path),
        "bytes": byte_count,
        "sha256": file_digest.hexdigest(),
        "row_count": row_count,
        "key_lf_sha256": key_digest.hexdigest(),
    }
    for key in ("bytes", "sha256", "row_count", "key_lf_sha256"):
        if observed[key] != declaration[key]:
            raise ValueError("molecule row artifact {} differs from its collection manifest".format(key))
    return observed


def load_text_rows(connection, collection, manifest_path):
    declaration = collection["text_pair_rows"]
    status = collection["identity_specs"]["text_identity"]["status"]
    if declaration is None:
        if status != "unavailable":
            raise ValueError("available text identity requires text_pair_rows")
        return None
    if status != "available":
        raise ValueError("text_pair_rows require an available text identity specification")
    validate_artifact_declaration(declaration, "text_pair_rows")
    path = resolve_path(declaration["path"], manifest_path.parent, "text_pair_rows.path")
    file_digest = hashlib.sha256()
    key_digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    previous_key = None
    collection_id = collection["collection_id"]
    task_family = collection["task_family"]
    batch = []
    for row, raw in iter_canonical_jsonl(path, "text-pair rows"):
        file_digest.update(raw)
        byte_count += len(raw)
        require_exact_fields(row, TEXT_FIELDS, "text-pair row")
        if row["schema_version"] != TEXT_ROW_SCHEMA or row["collection_id"] != collection_id:
            raise ValueError("text-pair row schema/collection binding mismatch")
        pair_id = require_string(row["pair_id"], "text-pair row pair_id")
        member_id = require_string(row["member_id"], "text-pair row member_id")
        if row["task_family"] != task_family:
            raise ValueError("text-pair task_family differs from its collection manifest")
        encoded_key = pair_id.encode("utf-8")
        if previous_key is not None and encoded_key <= previous_key:
            raise ValueError("text-pair rows are not strictly UTF-8-key sorted or contain duplicates")
        previous_key = encoded_key
        key_digest.update(encoded_key + b"\n")
        for key in (
            "text_exact_sha256",
            "text_normalized_sha256",
            "connectivity_text_pair_sha256",
            "stereo_text_pair_sha256",
        ):
            require_sha256(row[key], "text-pair row {}".format(key))
        molecule = connection.execute(
            "SELECT connectivity_sha256, stereo_sha256 FROM molecules WHERE collection_id=? AND member_id=?",
            (collection_id, member_id),
        ).fetchone()
        if molecule is None:
            raise ValueError("text-pair row references a missing molecule member")
        expected_connectivity_pair = pair_digest(
            "most-t5-r1/connectivity-text-pair/v1", molecule[0], row["text_normalized_sha256"]
        )
        expected_stereo_pair = pair_digest(
            "most-t5-r1/stereo-text-pair/v1", molecule[1], row["text_normalized_sha256"]
        )
        if row["connectivity_text_pair_sha256"] != expected_connectivity_pair:
            raise ValueError("connectivity-text pair digest is not derivable from its member and text identity")
        if row["stereo_text_pair_sha256"] != expected_stereo_pair:
            raise ValueError("stereo-text pair digest is not derivable from its member and text identity")
        batch.append(
            (
                collection_id,
                pair_id,
                member_id,
                task_family,
                row["text_exact_sha256"],
                row["text_normalized_sha256"],
                row["connectivity_text_pair_sha256"],
                row["stereo_text_pair_sha256"],
            )
        )
        if len(batch) >= 10000:
            connection.executemany("INSERT INTO text_pairs VALUES (?,?,?,?,?,?,?,?)", batch)
            batch = []
        row_count += 1
    if batch:
        connection.executemany("INSERT INTO text_pairs VALUES (?,?,?,?,?,?,?,?)", batch)
    observed = {
        "path": str(path),
        "bytes": byte_count,
        "sha256": file_digest.hexdigest(),
        "row_count": row_count,
        "key_lf_sha256": key_digest.hexdigest(),
    }
    for key in ("bytes", "sha256", "row_count", "key_lf_sha256"):
        if observed[key] != declaration[key]:
            raise ValueError("text-pair row artifact {} differs from its collection manifest".format(key))
    return observed


def validate_optional_spec(value, label):
    require_exact_fields(value, OPTIONAL_SPEC_FIELDS, label)
    status = value["status"]
    if status not in ("available", "unavailable"):
        raise ValueError("{}.status must be available or unavailable".format(label))
    require_sha256(value["spec_sha256"], "{}.spec_sha256".format(label), nullable=True)
    if (status == "available") != (value["spec_sha256"] is not None):
        raise ValueError("{} status/spec_sha256 availability mismatch".format(label))


def validate_text_spec(value):
    require_exact_fields(value, TEXT_SPEC_FIELDS, "identity_specs.text_identity")
    status = value["status"]
    if status not in ("available", "unavailable"):
        raise ValueError("text identity status must be available or unavailable")
    require_sha256(value["exact_spec_sha256"], "text exact spec", nullable=True)
    require_sha256(value["normalized_spec_sha256"], "text normalized spec", nullable=True)
    expected_present = status == "available"
    if expected_present != (value["exact_spec_sha256"] is not None and value["normalized_spec_sha256"] is not None):
        raise ValueError("text identity status/spec availability mismatch")


def validate_collection_manifest(value):
    require_exact_fields(value, COLLECTION_FIELDS, "collection manifest")
    if value["schema_version"] != COLLECTION_SCHEMA:
        raise ValueError("collection manifest schema mismatch")
    for key in ("collection_id", "dataset_id", "release_id", "phase", "split", "role", "task_family"):
        require_string(value[key], "collection manifest {}".format(key))
    if value["role"] not in ROLES:
        raise ValueError("unknown collection role")
    role = value["role"]
    expected_phase_split = {
        "p1_structure_train": ("p1", "train"),
        "p2_permitted_train_membership": ("p2", "train"),
        "p2_alignment_train": ("p2", "train"),
        "p2_geometry_replay_train": ("p2", "train"),
        "downstream_train": ("downstream", "train"),
        "downstream_validation": ("downstream", "validation"),
        "downstream_test": ("downstream", "test"),
    }[role]
    if (value["phase"], value["split"]) != expected_phase_split:
        raise ValueError("collection role is inconsistent with phase/split")
    if role == "p1_structure_train" and value["task_family"] != "none":
        raise ValueError("structural-only P1 collection task_family must be none")
    if role == "p2_permitted_train_membership" and value["task_family"] != "none":
        raise ValueError("P2 permitted-membership collection task_family must be none")
    if role in DOWNSTREAM_ROLES and value["task_family"] == "none":
        raise ValueError("downstream collection must name its task family")
    specs = value["identity_specs"]
    require_exact_fields(specs, IDENTITY_SPEC_FIELDS, "identity_specs")
    require_sha256(specs["connectivity_identity_spec_sha256"], "connectivity identity spec")
    require_sha256(specs["stereo_identity_spec_sha256"], "stereo identity spec")
    validate_optional_spec(specs["conformer_identity"], "identity_specs.conformer_identity")
    validate_text_spec(specs["text_identity"])
    validate_artifact_declaration(value["molecule_rows"], "molecule_rows")
    if value["molecule_rows"]["row_count"] <= 0:
        raise ValueError("a collection cannot prove overlap from an empty molecule-row artifact")
    if value["text_pair_rows"] is not None:
        validate_artifact_declaration(value["text_pair_rows"], "text_pair_rows")
        if value["text_pair_rows"]["row_count"] <= 0:
            raise ValueError("an available text-pair artifact cannot be empty")
    if role == "p1_structure_train" and specs["text_identity"]["status"] != "unavailable":
        raise ValueError("structural-only P1 must not carry text-pair identity rows")
    if role == "p2_permitted_train_membership" and specs["text_identity"]["status"] != "unavailable":
        raise ValueError("P2 permitted-membership collection must not carry task text rows")
    if role == "p2_alignment_train" and specs["text_identity"]["status"] != "available":
        raise ValueError("P2 alignment train must bind its molecule-text pairs")
    provenance = value["provenance"]
    require_exact_fields(provenance, PROVENANCE_FIELDS, "provenance")
    require_string(provenance["source_identity_namespace"], "source identity namespace")
    require_sha256(provenance["source_release_manifest_sha256"], "source release manifest")
    require_sha256(provenance["extractor_sha256"], "identity extractor")
    metadata_keys = provenance["excluded_source_metadata_keys"]
    if not isinstance(metadata_keys, list) or any(not isinstance(item, str) for item in metadata_keys):
        raise ValueError("excluded_source_metadata_keys must be a string array")
    if len(metadata_keys) != len(set(metadata_keys)):
        raise ValueError("excluded_source_metadata_keys contains duplicates")


def load_collection(connection, manifest_path, expected_sha256):
    manifest_path = regular_nonsymlink(manifest_path, "collection manifest")
    size, observed_sha = sha256_file(manifest_path)
    require_sha256(expected_sha256, "collection manifest reference SHA-256")
    if observed_sha != expected_sha256:
        raise ValueError("collection manifest SHA-256 differs from request binding")
    collection = load_json(manifest_path, "collection manifest")
    validate_collection_manifest(collection)
    molecule_observation = load_molecule_rows(connection, collection, manifest_path)
    text_observation = load_text_rows(connection, collection, manifest_path)
    return collection, {
        "manifest_path": str(manifest_path),
        "manifest_bytes": size,
        "manifest_sha256": observed_sha,
        "molecule_rows": molecule_observation,
        "text_pair_rows": text_observation,
    }


def scalar(connection, query, parameters):
    row = connection.execute(query, parameters).fetchone()
    return int(row[0])


def collection_summary(connection, collection):
    collection_id = collection["collection_id"]
    summary = {
        "collection_id": collection_id,
        "dataset_id": collection["dataset_id"],
        "release_id": collection["release_id"],
        "phase": collection["phase"],
        "split": collection["split"],
        "role": collection["role"],
        "task_family": collection["task_family"],
        "molecule_member_count": scalar(
            connection, "SELECT COUNT(*) FROM molecules WHERE collection_id=?", (collection_id,)
        ),
        "unique_connectivity_count": scalar(
            connection,
            "SELECT COUNT(DISTINCT connectivity_sha256) FROM molecules WHERE collection_id=?",
            (collection_id,),
        ),
        "unique_stereo_count": scalar(
            connection,
            "SELECT COUNT(DISTINCT stereo_sha256) FROM molecules WHERE collection_id=?",
            (collection_id,),
        ),
        "connectivity_duplicate_group_count": scalar(
            connection,
            "SELECT COUNT(*) FROM (SELECT connectivity_sha256 FROM molecules WHERE collection_id=? GROUP BY connectivity_sha256 HAVING COUNT(*)>1)",
            (collection_id,),
        ),
        "stereo_duplicate_group_count": scalar(
            connection,
            "SELECT COUNT(*) FROM (SELECT stereo_sha256 FROM molecules WHERE collection_id=? GROUP BY stereo_sha256 HAVING COUNT(*)>1)",
            (collection_id,),
        ),
        "text_pair_row_count": scalar(
            connection, "SELECT COUNT(*) FROM text_pairs WHERE collection_id=?", (collection_id,)
        ),
    }
    if collection["identity_specs"]["conformer_identity"]["status"] == "available":
        summary["unique_conformer_count"] = scalar(
            connection,
            "SELECT COUNT(DISTINCT conformer_sha256) FROM molecules WHERE collection_id=?",
            (collection_id,),
        )
        summary["multi_conformer_stereo_group_count"] = scalar(
            connection,
            "SELECT COUNT(*) FROM (SELECT stereo_sha256 FROM molecules WHERE collection_id=? GROUP BY stereo_sha256 HAVING COUNT(DISTINCT conformer_sha256)>1)",
            (collection_id,),
        )
    else:
        summary["unique_conformer_count"] = None
        summary["multi_conformer_stereo_group_count"] = None
    return summary


def dimension_availability(left, right, dimension):
    left_specs = left["identity_specs"]
    right_specs = right["identity_specs"]
    if dimension == "connectivity_identity":
        return left_specs["connectivity_identity_spec_sha256"] == right_specs["connectivity_identity_spec_sha256"], "identity_spec_sha256_mismatch"
    if dimension == "stereo_identity":
        return left_specs["stereo_identity_spec_sha256"] == right_specs["stereo_identity_spec_sha256"], "identity_spec_sha256_mismatch"
    if dimension == "conformer_identity":
        left_spec = left_specs["conformer_identity"]
        right_spec = right_specs["conformer_identity"]
        if left_spec["status"] != "available" or right_spec["status"] != "available":
            return False, "conformer_identity_unavailable"
        return left_spec["spec_sha256"] == right_spec["spec_sha256"], "conformer_identity_spec_sha256_mismatch"
    left_text = left_specs["text_identity"]
    right_text = right_specs["text_identity"]
    if left_text["status"] != "available" or right_text["status"] != "available":
        return False, "text_identity_unavailable"
    if dimension == "text_exact":
        return left_text["exact_spec_sha256"] == right_text["exact_spec_sha256"], "text_exact_spec_sha256_mismatch"
    if left_text["normalized_spec_sha256"] != right_text["normalized_spec_sha256"]:
        return False, "text_normalized_spec_sha256_mismatch"
    if dimension == "connectivity_text_pair":
        return left_specs["connectivity_identity_spec_sha256"] == right_specs["connectivity_identity_spec_sha256"], "connectivity_identity_spec_sha256_mismatch"
    if dimension == "stereo_text_pair":
        return left_specs["stereo_identity_spec_sha256"] == right_specs["stereo_identity_spec_sha256"], "stereo_identity_spec_sha256_mismatch"
    return True, None


def dimension_unique_count(connection, collection_id, dimension):
    table, column = DIMENSION_TABLE_COLUMN[dimension]
    nonnull = " AND {} IS NOT NULL".format(column)
    unique_query = "SELECT COUNT(DISTINCT {0}) FROM {1} WHERE collection_id=?{2}".format(column, table, nonnull)
    return scalar(connection, unique_query, (collection_id,))


def dimension_counts(connection, left_id, right_id, dimension, left_unique_count=None):
    table, column = DIMENSION_TABLE_COLUMN[dimension]
    nonnull = " AND {} IS NOT NULL".format(column)
    overlap_query = (
        "SELECT COUNT(*) FROM ("
        "SELECT {0} FROM {1} WHERE collection_id=?{2} GROUP BY {0} "
        "INTERSECT SELECT {0} FROM {1} WHERE collection_id=?{2} GROUP BY {0})"
    ).format(column, table, nonnull)
    impacted_query = (
        "SELECT COUNT(*) FROM {0} AS left_rows WHERE left_rows.collection_id=? "
        "AND left_rows.{1} IS NOT NULL AND EXISTS (SELECT 1 FROM {0} AS right_rows "
        "WHERE right_rows.collection_id=? AND right_rows.{1}=left_rows.{1})"
    ).format(table, column)
    return {
        "left_unique_count": (
            dimension_unique_count(connection, left_id, dimension)
            if left_unique_count is None
            else left_unique_count
        ),
        "right_unique_count": dimension_unique_count(connection, right_id, dimension),
        "overlap_unique_count": scalar(connection, overlap_query, (left_id, right_id)),
        "left_rows_impacted": scalar(connection, impacted_query, (left_id, right_id)),
        "right_rows_impacted": scalar(connection, impacted_query, (right_id, left_id)),
    }


def cross_resolution_counts(connection, left_id, right_id, conformer_available):
    result = {
        "left_members_connectivity_overlap_without_stereo_match": scalar(
            connection,
            """
            SELECT COUNT(*) FROM molecules AS l
            WHERE l.collection_id=?
              AND EXISTS (SELECT 1 FROM molecules AS r WHERE r.collection_id=? AND r.connectivity_sha256=l.connectivity_sha256)
              AND NOT EXISTS (SELECT 1 FROM molecules AS r WHERE r.collection_id=? AND r.stereo_sha256=l.stereo_sha256)
            """,
            (left_id, right_id, right_id),
        ),
        "right_members_connectivity_overlap_without_stereo_match": scalar(
            connection,
            """
            SELECT COUNT(*) FROM molecules AS r
            WHERE r.collection_id=?
              AND EXISTS (SELECT 1 FROM molecules AS l WHERE l.collection_id=? AND l.connectivity_sha256=r.connectivity_sha256)
              AND NOT EXISTS (SELECT 1 FROM molecules AS l WHERE l.collection_id=? AND l.stereo_sha256=r.stereo_sha256)
            """,
            (right_id, left_id, left_id),
        ),
    }
    if conformer_available:
        result["left_members_molecule_overlap_without_exact_conformer_match"] = scalar(
            connection,
            """
            SELECT COUNT(*) FROM molecules AS l
            WHERE l.collection_id=?
              AND EXISTS (SELECT 1 FROM molecules AS r WHERE r.collection_id=? AND r.stereo_sha256=l.stereo_sha256)
              AND NOT EXISTS (SELECT 1 FROM molecules AS r WHERE r.collection_id=? AND r.conformer_sha256=l.conformer_sha256)
            """,
            (left_id, right_id, right_id),
        )
        result["right_members_molecule_overlap_without_exact_conformer_match"] = scalar(
            connection,
            """
            SELECT COUNT(*) FROM molecules AS r
            WHERE r.collection_id=?
              AND EXISTS (SELECT 1 FROM molecules AS l WHERE l.collection_id=? AND l.stereo_sha256=r.stereo_sha256)
              AND NOT EXISTS (SELECT 1 FROM molecules AS l WHERE l.collection_id=? AND l.conformer_sha256=r.conformer_sha256)
            """,
            (right_id, left_id, left_id),
        )
    else:
        result["left_members_molecule_overlap_without_exact_conformer_match"] = None
        result["right_members_molecule_overlap_without_exact_conformer_match"] = None
    return result


def validate_comparison_shape(comparison):
    require_exact_fields(comparison, COMPARISON_FIELDS, "comparison")
    for key in ("comparison_id", "left_collection_id", "right_collection_id", "relationship", "policy"):
        require_string(comparison[key], "comparison {}".format(key))
    if comparison["left_collection_id"] == comparison["right_collection_id"]:
        raise ValueError("comparison cannot compare a collection with itself")
    if comparison["relationship"] not in RELATIONSHIPS:
        raise ValueError("unknown comparison relationship")
    if comparison["policy"] not in POLICIES:
        raise ValueError("unknown comparison policy")
    for key in ("required_zero", "report_only"):
        values = comparison[key]
        if not isinstance(values, list) or len(values) != len(set(values)) or any(item not in DIMENSIONS for item in values):
            raise ValueError("comparison {} must be a duplicate-free identity-dimension array".format(key))
    if set(comparison["required_zero"]) & set(comparison["report_only"]):
        raise ValueError("comparison required_zero and report_only overlap")
    if "connectivity_identity" not in set(comparison["required_zero"]) | set(comparison["report_only"]):
        raise ValueError("every comparison must explicitly classify connectivity_identity")
    relationship = comparison["relationship"]
    if relationship in ("pretrain_to_downstream_eval", "downstream_within_task_split"):
        if comparison["policy"] != "disjoint_required" or "connectivity_identity" not in comparison["required_zero"]:
            raise ValueError("evaluation/split isolation requires zero connectivity overlap")
    if relationship == "p1_to_p2" and comparison["policy"] == "disjoint_required":
        if "connectivity_identity" not in comparison["required_zero"]:
            raise ValueError("disjoint P1/P2 policy requires zero connectivity overlap")
    if comparison["policy"] == "disjoint_required" and "connectivity_identity" not in comparison["required_zero"]:
        raise ValueError("every disjoint_required comparison must require zero connectivity overlap")


def pair_key(left_id, right_id):
    return tuple(sorted((left_id, right_id)))


def validate_coverage(request, collections, comparisons):
    coverage = request["coverage"]
    require_exact_fields(coverage, COVERAGE_FIELDS, "coverage")
    errors = []
    roles = coverage["required_collection_roles"]
    if not isinstance(roles, list) or len(roles) != len(set(roles)) or any(role not in ROLES for role in roles):
        raise ValueError("required_collection_roles is invalid")
    present_roles = set(collection["role"] for collection in collections.values())
    for role in roles:
        if role not in present_roles:
            errors.append("missing required collection role {}".format(role))
    task_splits = coverage["required_downstream_task_splits"]
    if not isinstance(task_splits, list) or not task_splits:
        raise ValueError("required_downstream_task_splits must explicitly enumerate at least one downstream task")
    declared_task_splits = set()
    for item in task_splits:
        require_exact_fields(item, TASK_SPLIT_FIELDS, "required downstream task split")
        task = require_string(item["task_family"], "required downstream task family")
        splits = item["splits"]
        if not isinstance(splits, list) or not splits or len(splits) != len(set(splits)):
            raise ValueError("required downstream task splits must be a non-empty duplicate-free array")
        for split in splits:
            require_string(split, "required downstream split")
            declared_task_splits.add((task, split))
    observed_task_split_counts = {}
    for collection in collections.values():
        if collection["role"] in DOWNSTREAM_ROLES:
            key = (collection["task_family"], collection["split"])
            observed_task_split_counts[key] = observed_task_split_counts.get(key, 0) + 1
    observed_task_splits = set(observed_task_split_counts)
    for item in sorted(declared_task_splits - observed_task_splits):
        errors.append("missing downstream collection for task/split {}/{}".format(item[0], item[1]))
    for item in sorted(observed_task_splits - declared_task_splits):
        errors.append("downstream collection task/split {}/{} is not declared by coverage".format(item[0], item[1]))
    for item in sorted(declared_task_splits & observed_task_splits):
        if observed_task_split_counts[item] != 1:
            errors.append(
                "task/split {}/{} resolves to {} collections; use a unique task_family per audited collection"
                .format(item[0], item[1], observed_task_split_counts[item])
            )
    eval_splits = coverage["downstream_eval_splits"]
    if not isinstance(eval_splits, list) or not eval_splits or len(eval_splits) != len(set(eval_splits)):
        raise ValueError("downstream_eval_splits must be a non-empty duplicate-free array")
    for split in eval_splits:
        require_string(split, "downstream evaluation split")
    for key in (
        "require_p1_p2_comparison",
        "require_each_pretrain_vs_each_downstream_eval",
        "require_within_task_split_comparisons",
    ):
        if not isinstance(coverage[key], bool):
            raise ValueError("coverage.{} must be boolean".format(key))
    by_pair = {}
    for comparison in comparisons:
        key = pair_key(comparison["left_collection_id"], comparison["right_collection_id"])
        if key in by_pair:
            errors.append("duplicate comparison for unordered collection pair {}".format(key))
        by_pair[key] = comparison
    if coverage["require_p1_p2_comparison"]:
        p1 = [item for item in collections.values() if item["role"] == "p1_structure_train"]
        p2 = [item for item in collections.values() if item["role"] == "p2_permitted_train_membership"]
        if not p1 or not p2:
            errors.append("P1/P2 comparison requested but one role is absent")
        for left in p1:
            for right in p2:
                comparison = by_pair.get(pair_key(left["collection_id"], right["collection_id"]))
                if comparison is None or comparison["relationship"] != "p1_to_p2":
                    errors.append("missing p1_to_p2 comparison {} vs {}".format(left["collection_id"], right["collection_id"]))
    if coverage["require_each_pretrain_vs_each_downstream_eval"]:
        pretrain = [item for item in collections.values() if item["role"] in PRETRAIN_ROLES]
        downstream_eval = [
            item
            for item in collections.values()
            if item["role"] in DOWNSTREAM_ROLES and item["split"] in eval_splits
        ]
        if not pretrain or not downstream_eval:
            errors.append("pretrain-vs-downstream coverage requested but one side is empty")
        for left in pretrain:
            for right in downstream_eval:
                comparison = by_pair.get(pair_key(left["collection_id"], right["collection_id"]))
                if comparison is None or comparison["relationship"] != "pretrain_to_downstream_eval":
                    errors.append("missing pretrain_to_downstream_eval comparison {} vs {}".format(left["collection_id"], right["collection_id"]))
    if coverage["require_within_task_split_comparisons"]:
        by_task = {}
        for collection in collections.values():
            if collection["role"] in DOWNSTREAM_ROLES:
                by_task.setdefault(collection["task_family"], []).append(collection)
        for task, task_collections in sorted(by_task.items()):
            for index, left in enumerate(task_collections):
                for right in task_collections[index + 1 :]:
                    if left["split"] == right["split"]:
                        continue
                    comparison = by_pair.get(pair_key(left["collection_id"], right["collection_id"]))
                    if comparison is None or comparison["relationship"] != "downstream_within_task_split":
                        errors.append("missing within-task split comparison {} vs {} for {}".format(left["collection_id"], right["collection_id"], task))
    return {"passed": not errors, "errors": errors}


def compare_collections(connection, comparison, collections):
    validate_comparison_shape(comparison)
    left_id = comparison["left_collection_id"]
    right_id = comparison["right_collection_id"]
    if left_id not in collections or right_id not in collections:
        raise ValueError("comparison references an unknown collection")
    left = collections[left_id]
    right = collections[right_id]
    left_role = left["role"]
    right_role = right["role"]
    relationship = comparison["relationship"]
    role_pair = frozenset((left_role, right_role))
    if relationship == "p1_to_p2" and role_pair != frozenset(("p1_structure_train", "p2_permitted_train_membership")):
        raise ValueError("p1_to_p2 comparison must bind P1 structure train to P2 permitted membership")
    if relationship == "pretrain_to_downstream_eval":
        if not (
            (left_role in PRETRAIN_ROLES and right_role in ("downstream_validation", "downstream_test"))
            or (right_role in PRETRAIN_ROLES and left_role in ("downstream_validation", "downstream_test"))
        ):
            raise ValueError("pretrain_to_downstream_eval comparison has incompatible collection roles")
    if relationship == "downstream_within_task_split":
        if left_role not in DOWNSTREAM_ROLES or right_role not in DOWNSTREAM_ROLES:
            raise ValueError("downstream split comparison requires two downstream collections")
        if left["task_family"] != right["task_family"] or left["split"] == right["split"]:
            raise ValueError("downstream split comparison requires one task family and two distinct splits")
    required_zero = set(comparison["required_zero"])
    report_only = set(comparison["report_only"])
    dimensions = {}
    violations = []
    conformer_available = False
    availability = {}
    for dimension in DIMENSIONS:
        available, reason = dimension_availability(left, right, dimension)
        availability[dimension] = available
        if not available:
            dimensions[dimension] = {
                "status": "unavailable",
                "reason": reason,
                "policy": "required_zero" if dimension in required_zero else ("report_only" if dimension in report_only else "informational"),
            }
            if dimension in required_zero:
                violations.append("required dimension {} is unavailable: {}".format(dimension, reason))
            continue
        counts = dimension_counts(connection, left_id, right_id, dimension)
        policy = "required_zero" if dimension in required_zero else ("report_only" if dimension in report_only else "informational")
        dimensions[dimension] = {"status": "available", "policy": policy, "counts": counts}
        if dimension == "conformer_identity":
            conformer_available = True
        if dimension in required_zero and counts["overlap_unique_count"] != 0:
            violations.append("{} overlap is {}, required zero".format(dimension, counts["overlap_unique_count"]))
    both_text = (
        left["identity_specs"]["text_identity"]["status"] == "available"
        and right["identity_specs"]["text_identity"]["status"] == "available"
    )
    if comparison["relationship"] in ("pretrain_to_downstream_eval", "downstream_within_task_split") and both_text:
        if "connectivity_text_pair" not in required_zero:
            violations.append("text-bearing evaluation comparison must require zero connectivity_text_pair overlap")
        if "text_normalized" not in required_zero | report_only:
            violations.append("text-bearing evaluation comparison must at least report normalized-text overlap")
    return {
        "comparison_id": comparison["comparison_id"],
        "left_collection_id": left_id,
        "right_collection_id": right_id,
        "relationship": comparison["relationship"],
        "policy": comparison["policy"],
        "required_zero": comparison["required_zero"],
        "report_only": comparison["report_only"],
        "dimensions": dimensions,
        "cross_resolution": (
            cross_resolution_counts(connection, left_id, right_id, conformer_available)
            if availability["connectivity_identity"] and availability["stereo_identity"]
            else {
                "status": "unavailable",
                "reason": "connectivity_or_stereo_identity_spec_sha256_mismatch",
            }
        ),
        "passed": not violations,
        "violations": violations,
    }


def validate_contract(contract):
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("overlap contract schema mismatch")
    if tuple(contract.get("identity_dimensions", [])) != DIMENSIONS:
        raise ValueError("overlap contract identity dimensions differ from the gate")
    if set(contract.get("collection_roles", [])) != set(ROLES):
        raise ValueError("overlap contract collection roles differ from the gate")
    if set(contract.get("comparison_relationships", [])) != set(RELATIONSHIPS):
        raise ValueError("overlap contract comparison relationships differ from the gate")
    if set(contract.get("comparison_policies", [])) != set(POLICIES):
        raise ValueError("overlap contract comparison policies differ from the gate")


def report_payload_sha256(report):
    payload = dict(report)
    payload.pop("report_canonical_payload_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def write_json_new(path, value):
    path = Path(path)
    with open(str(path), "xb") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_proof(contract_path, request_path, output_dir, database_path=":memory:"):
    contract_path = Path(contract_path)
    request_path = Path(request_path)
    regular_nonsymlink(contract_path, "overlap contract")
    regular_nonsymlink(request_path, "proof request")
    contract_path = contract_path.resolve()
    request_path = request_path.resolve()
    contract_bytes, contract_sha = sha256_file(contract_path)
    request_bytes, request_sha = sha256_file(request_path)
    contract = load_json(contract_path, "overlap contract")
    validate_contract(contract)
    request = load_json(request_path, "proof request")
    require_exact_fields(request, REQUEST_FIELDS, "proof request")
    if request["schema_version"] != REQUEST_SCHEMA:
        raise ValueError("proof request schema mismatch")
    require_string(request["request_id"], "proof request ID")
    require_sha256(request["contract_sha256"], "proof request contract SHA-256")
    if request["contract_sha256"] != contract_sha:
        raise ValueError("proof request does not bind the supplied overlap contract")
    refs = request["collections"]
    if not isinstance(refs, list) or not refs:
        raise ValueError("proof request must reference at least one collection")
    comparisons = request["comparisons"]
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("proof request must contain at least one comparison")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if database_path != ":memory:":
        database_file = Path(database_path)
        if database_file.exists():
            raise FileExistsError("refusing to reuse an existing overlap database: {}".format(database_file))
        database_file.parent.mkdir(parents=True, exist_ok=True)
        database_path = str(database_file.resolve())
    connection = create_database(database_path)
    collections = {}
    observations = {}
    try:
        for index, ref in enumerate(refs):
            require_exact_fields(ref, COLLECTION_REF_FIELDS, "collections[{}]".format(index))
            require_sha256(ref["manifest_sha256"], "collection manifest SHA-256")
            path = resolve_path(ref["manifest_path"], request_path.parent, "collection manifest path")
            collection, observation = load_collection(connection, path, ref["manifest_sha256"])
            collection_id = collection["collection_id"]
            if collection_id in collections:
                raise ValueError("duplicate collection ID")
            collections[collection_id] = collection
            observations[collection_id] = observation
        connection.commit()
        connection.executescript(
            """
            CREATE INDEX molecules_connectivity ON molecules(collection_id, connectivity_sha256);
            CREATE INDEX molecules_stereo ON molecules(collection_id, stereo_sha256);
            CREATE INDEX molecules_conformer ON molecules(collection_id, conformer_sha256);
            CREATE INDEX text_exact ON text_pairs(collection_id, text_exact_sha256);
            CREATE INDEX text_normalized ON text_pairs(collection_id, text_normalized_sha256);
            CREATE INDEX text_connectivity_pair ON text_pairs(collection_id, connectivity_pair_sha256);
            CREATE INDEX text_stereo_pair ON text_pairs(collection_id, stereo_pair_sha256);
            """
        )
        comparison_ids = set()
        for comparison in comparisons:
            validate_comparison_shape(comparison)
            if comparison["comparison_id"] in comparison_ids:
                raise ValueError("duplicate comparison ID")
            comparison_ids.add(comparison["comparison_id"])
        coverage = validate_coverage(request, collections, comparisons)
        comparison_reports = [compare_collections(connection, item, collections) for item in comparisons]
        report = {
            "schema_version": REPORT_SCHEMA,
            "request_id": request["request_id"],
            "generated_at_utc": utc_now(),
            "status": "pass" if coverage["passed"] and all(item["passed"] for item in comparison_reports) else "fail",
            "p1_training_admission": False,
            "p2_training_admission": False,
            "provenance": {
                "contract_path": str(contract_path),
                "contract_bytes": contract_bytes,
                "contract_sha256": contract_sha,
                "request_path": str(request_path),
                "request_bytes": request_bytes,
                "request_sha256": request_sha,
                "gate_path": str(Path(__file__).resolve()),
                "gate_sha256": sha256_file(Path(__file__).resolve())[1],
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "sqlite": sqlite3.sqlite_version,
            },
            "collections": [
                {
                    "summary": collection_summary(connection, collections[collection_id]),
                    "input_observation": observations[collection_id],
                    "identity_specs": collections[collection_id]["identity_specs"],
                    "source_identity_namespace": collections[collection_id]["provenance"]["source_identity_namespace"],
                    "excluded_source_metadata_keys": collections[collection_id]["provenance"]["excluded_source_metadata_keys"],
                }
                for collection_id in sorted(collections, key=lambda value: value.encode("utf-8"))
            ],
            "coverage": coverage,
            "comparisons": comparison_reports,
            "scope_warning": "This report proves only its explicitly bound collections, task families, splits, identity specifications, and dimensions; it is not a training admission.",
        }
        report["report_canonical_payload_sha256"] = report_payload_sha256(report)
        write_json_new(output_dir / "overlap_proof_report.json", report)
        return report
    finally:
        connection.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--database",
        default=":memory:",
        help="SQLite database path for low-RAM/disk-backed execution; defaults to :memory:. Existing files are rejected.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    database = args.database
    if database != ":memory:":
        database_path = Path(database).resolve()
        if database_path.exists():
            raise FileExistsError("refusing to reuse an existing overlap database: {}".format(database_path))
        database_path.parent.mkdir(parents=True, exist_ok=True)
        database = str(database_path)
    report = run_proof(args.contract, args.request, args.output_dir, database)
    print(json.dumps({"status": report["status"], "report": str(Path(args.output_dir).resolve() / "overlap_proof_report.json")}, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
