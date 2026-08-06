#!/usr/bin/env python3
"""Derive a clean pretraining membership from downstream evaluation identities.

The derivation reads the hash-only molecule rows already bound by collection
manifests.  A pretraining member is excluded if, and only if, its connectivity
identity occurs in any protected downstream validation/test collection.
Stereo identity is summarized separately; text identity is reported from the
manifests and is never used as an exclusion key.  No molecule payload is read
or copied.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    from . import prove_membership_identity_overlap_v1 as proof
except ImportError:  # Direct execution from this directory.
    import prove_membership_identity_overlap_v1 as proof


PERMITTED_ROW_SCHEMA = "most-t5-r1/permitted-pretrain-member/v1"
EXCLUDED_ROW_SCHEMA = "most-t5-r1/excluded-pretrain-member/v1"
MANIFEST_SCHEMA = "most-t5-r1/clean-pretrain-membership-manifest/v1"
PERMITTED_FILENAME = "permitted_member_ids.jsonl"
EXCLUDED_FILENAME = "excluded_member_ledger.jsonl"
MANIFEST_FILENAME = "clean_membership_manifest.json"
PRETRAIN_ROLES = frozenset(proof.PRETRAIN_ROLES)
PROTECTED_ROLES = frozenset(("downstream_validation", "downstream_test"))


class CanonicalJsonlWriter(object):
    def __init__(self, path):
        self.path = Path(path)
        self.handle = open(str(self.path), "xb")
        self.file_digest = hashlib.sha256()
        self.key_digest = hashlib.sha256()
        self.byte_count = 0
        self.row_count = 0

    def write(self, row, key):
        raw = proof.canonical_json_bytes(row) + b"\n"
        self.handle.write(raw)
        self.file_digest.update(raw)
        encoded_key = key.encode("utf-8")
        self.key_digest.update(encoded_key + b"\n")
        self.byte_count += len(raw)
        self.row_count += 1

    def finish(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        return {
            "path": self.path.name,
            "bytes": self.byte_count,
            "sha256": self.file_digest.hexdigest(),
            "row_count": self.row_count,
            "key_lf_sha256": self.key_digest.hexdigest(),
        }

    def close(self):
        if not self.handle.closed:
            self.handle.close()


def write_canonical_json_new(path, value):
    raw = proof.canonical_json_bytes(value) + b"\n"
    with open(str(path), "xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def load_molecule_collection(connection, manifest_path):
    """Load one validated manifest and its molecule rows without reading text rows."""
    manifest_path = proof.regular_nonsymlink(Path(manifest_path), "collection manifest").resolve()
    manifest_bytes, observed_sha256 = proof.sha256_file(manifest_path)
    collection = proof.load_json(manifest_path, "collection manifest")
    proof.validate_collection_manifest(collection)
    molecule_observation = proof.load_molecule_rows(connection, collection, manifest_path)
    return collection, {
        "manifest_bytes": manifest_bytes,
        "manifest_sha256": observed_sha256,
        "molecule_rows": molecule_observation,
    }


def require_collection_roles_and_specs(pretrain, protected):
    if pretrain["role"] not in PRETRAIN_ROLES:
        raise ValueError("pretrain collection role is not a pretraining role")
    seen_ids = {pretrain["collection_id"]}
    connectivity_spec = pretrain["identity_specs"]["connectivity_identity_spec_sha256"]
    for collection in protected:
        if collection["role"] not in PROTECTED_ROLES:
            raise ValueError("protected collection role must be downstream_validation or downstream_test")
        if collection["collection_id"] in seen_ids:
            raise ValueError("collection IDs must be unique")
        seen_ids.add(collection["collection_id"])
        if collection["identity_specs"]["connectivity_identity_spec_sha256"] != connectivity_spec:
            raise ValueError("connectivity identity specifications must match before exclusion")


def compact_artifact_observation(observation):
    return {
        key: observation[key]
        for key in ("bytes", "sha256", "row_count", "key_lf_sha256")
    }


def source_binding(collection, observation):
    text_rows = collection["text_pair_rows"]
    return {
        "collection_id": collection["collection_id"],
        "dataset_id": collection["dataset_id"],
        "release_id": collection["release_id"],
        "role": collection["role"],
        "split": collection["split"],
        "task_family": collection["task_family"],
        "collection_manifest": {
            "bytes": observation["manifest_bytes"],
            "sha256": observation["manifest_sha256"],
        },
        "molecule_rows": compact_artifact_observation(observation["molecule_rows"]),
        "text_identity": {
            "status": collection["identity_specs"]["text_identity"]["status"],
            "text_pair_row_count": text_rows["row_count"] if text_rows is not None else 0,
            "used_for_exclusion": False,
        },
    }


def named_dimension_counts(counts):
    return {
        "pretrain_unique_count": counts["left_unique_count"],
        "protected_unique_count": counts["right_unique_count"],
        "overlap_unique_count": counts["overlap_unique_count"],
        "pretrain_members_impacted": counts["left_rows_impacted"],
        "protected_members_impacted": counts["right_rows_impacted"],
    }


def report_stereo(connection, pretrain, protected, pretrain_unique_count):
    comparable, reason = proof.dimension_availability(pretrain, protected, "stereo_identity")
    result = {"used_for_exclusion": False}
    if comparable:
        result.update(
            {
                "status": "compared_report_only",
                "counts": named_dimension_counts(
                    proof.dimension_counts(
                        connection,
                        pretrain["collection_id"],
                        protected["collection_id"],
                        "stereo_identity",
                        left_unique_count=pretrain_unique_count,
                    )
                ),
            }
        )
    else:
        result.update({"status": "not_comparable", "reason": reason, "counts": None})
    return result


def report_text(pretrain, protected):
    left = pretrain["identity_specs"]["text_identity"]
    right = protected["identity_specs"]["text_identity"]
    return {
        "status": "manifest_metadata_only",
        "reason": "text rows are not read by a connectivity-membership derivation",
        "used_for_exclusion": False,
        "pretrain_status": left["status"],
        "protected_status": right["status"],
        "exact_spec_sha256_equal": (
            left["status"] == "available"
            and right["status"] == "available"
            and left["exact_spec_sha256"] == right["exact_spec_sha256"]
        ),
        "normalized_spec_sha256_equal": (
            left["status"] == "available"
            and right["status"] == "available"
            and left["normalized_spec_sha256"] == right["normalized_spec_sha256"]
        ),
    }


def protected_sort_key(collection):
    return tuple(item.encode("utf-8") for item in (collection["task_family"], collection["split"], collection["collection_id"]))


def write_membership_artifacts(connection, pretrain, protected, output_dir):
    protected_ids = [collection["collection_id"] for collection in protected]
    placeholders = ",".join("?" for _ in protected_ids)
    query = """
        SELECT p.member_id, p.connectivity_sha256, d.collection_id
        FROM molecules AS p
        LEFT JOIN (
            SELECT collection_id, connectivity_sha256
            FROM molecules
            WHERE collection_id IN ({})
            GROUP BY collection_id, connectivity_sha256
        ) AS d ON d.connectivity_sha256=p.connectivity_sha256
        WHERE p.collection_id=?
        ORDER BY p.member_id COLLATE BINARY, d.collection_id COLLATE BINARY
    """.format(placeholders)
    rows = connection.execute(query, tuple(protected_ids) + (pretrain["collection_id"],))
    by_id = {collection["collection_id"]: collection for collection in protected}
    permitted = CanonicalJsonlWriter(output_dir / PERMITTED_FILENAME)
    excluded = CanonicalJsonlWriter(output_dir / EXCLUDED_FILENAME)
    excluded_multiple = 0
    current_member = None
    current_connectivity = None
    current_collection_ids = []

    def emit_current():
        nonlocal excluded_multiple
        if current_member is None:
            return
        if not current_collection_ids:
            permitted.write(
                {"schema_version": PERMITTED_ROW_SCHEMA, "member_id": current_member},
                current_member,
            )
            return
        matches = [by_id[collection_id] for collection_id in current_collection_ids]
        matches.sort(key=protected_sort_key)
        if len(matches) > 1:
            excluded_multiple += 1
        excluded.write(
            {
                "schema_version": EXCLUDED_ROW_SCHEMA,
                "member_id": current_member,
                "connectivity_identity_sha256": current_connectivity,
                "matched_protected_collections": [
                    {
                        "task_family": collection["task_family"],
                        "split": collection["split"],
                        "collection_id": collection["collection_id"],
                    }
                    for collection in matches
                ],
            },
            current_member,
        )

    try:
        for member_id, connectivity_sha256, collection_id in rows:
            if member_id != current_member:
                emit_current()
                current_member = member_id
                current_connectivity = connectivity_sha256
                current_collection_ids = []
            if collection_id is not None and collection_id not in current_collection_ids:
                current_collection_ids.append(collection_id)
        emit_current()
        permitted_artifact = permitted.finish()
        excluded_artifact = excluded.finish()
    except Exception:
        permitted.close()
        excluded.close()
        raise
    return permitted_artifact, excluded_artifact, excluded_multiple


def derive_clean_membership(
    pretrain_manifest_path,
    protected_manifest_paths,
    output_dir,
):
    if not protected_manifest_paths:
        raise ValueError("at least one protected validation/test manifest is required")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to reuse an existing output directory: {}".format(output_dir))

    connection = proof.create_database(":memory:")
    try:
        pretrain, pretrain_observation = load_molecule_collection(connection, pretrain_manifest_path)
        protected_pairs = [
            load_molecule_collection(connection, path)
            for path in protected_manifest_paths
        ]
        protected_pairs.sort(key=lambda pair: protected_sort_key(pair[0]))
        protected = [pair[0] for pair in protected_pairs]
        require_collection_roles_and_specs(pretrain, protected)
        connection.commit()
        connection.executescript(
            """
            CREATE INDEX clean_molecules_connectivity
                ON molecules(collection_id, connectivity_sha256);
            CREATE INDEX clean_molecules_stereo
                ON molecules(collection_id, stereo_sha256);
            """
        )
        pretrain_unique_counts = {
            dimension: proof.dimension_unique_count(
                connection, pretrain["collection_id"], dimension
            )
            for dimension in ("connectivity_identity", "stereo_identity")
        }

        output_dir.mkdir(parents=True, exist_ok=False)
        permitted_artifact, excluded_artifact, excluded_multiple = write_membership_artifacts(
            connection, pretrain, protected, output_dir
        )
        pretrain_member_count = proof.scalar(
            connection,
            "SELECT COUNT(*) FROM molecules WHERE collection_id=?",
            (pretrain["collection_id"],),
        )
        excluded_unique_connectivity_count = proof.scalar(
            connection,
            """
            SELECT COUNT(DISTINCT p.connectivity_sha256)
            FROM molecules AS p
            WHERE p.collection_id=? AND EXISTS (
                SELECT 1 FROM molecules AS d
                WHERE d.collection_id IN ({})
                  AND d.connectivity_sha256=p.connectivity_sha256
            )
            """.format(",".join("?" for _ in protected)),
            (pretrain["collection_id"],) + tuple(item["collection_id"] for item in protected),
        )
        protected_reports = []
        for collection, observation in protected_pairs:
            connectivity_counts = named_dimension_counts(
                proof.dimension_counts(
                    connection,
                    pretrain["collection_id"],
                    collection["collection_id"],
                    "connectivity_identity",
                    left_unique_count=pretrain_unique_counts["connectivity_identity"],
                )
            )
            protected_reports.append(
                {
                    "source": source_binding(collection, observation),
                    "hard_exclusion_connectivity": {
                        "used_for_exclusion": True,
                        "counts": connectivity_counts,
                    },
                    "report_only_stereo": report_stereo(
                        connection,
                        pretrain,
                        collection,
                        pretrain_unique_counts["stereo_identity"],
                    ),
                    "report_only_text": report_text(pretrain, collection),
                }
            )

        source_observations = {
            "pretrain_manifest_sha256": pretrain_observation["manifest_sha256"],
            "protected_manifest_sha256": [
                observation["manifest_sha256"] for _, observation in protected_pairs
            ],
            "policy": "exclude_on_connectivity_identity_sha256_union_only",
        }
        binding_sha256 = proof.sha256_bytes(proof.canonical_json_bytes(source_observations))
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "derivation_id": "clean-pretrain-membership-{}".format(binding_sha256[:20]),
            "derivation_binding_sha256": binding_sha256,
            "status": "complete",
            "policy": {
                "hard_exclusion_key": "connectivity_identity_sha256",
                "protected_roles": sorted(PROTECTED_ROLES),
                "match_semantics": "exclude_if_present_in_any_protected_collection",
                "report_only_dimensions": ["stereo_identity_sha256", "text_identity"],
            },
            "release_handling": {
                "source_releases_preserved": True,
                "molecule_payload_copied": False,
                "derived_membership_only": True,
            },
            "provenance_observation_policy": {
                "caller_supplied_digest_required": False,
                "observed_artifact_digests_are_scientific_admission_gates": False,
                "semantic_admission_basis": [
                    "collection schema and role",
                    "referenced molecule-row closure",
                    "compatible connectivity identity specification",
                ],
            },
            "pretrain_source": source_binding(pretrain, pretrain_observation),
            "protected_collections": protected_reports,
            "counts": {
                "pretrain_member_count": pretrain_member_count,
                "permitted_member_count": permitted_artifact["row_count"],
                "excluded_member_count": excluded_artifact["row_count"],
                "excluded_unique_connectivity_count": excluded_unique_connectivity_count,
                "excluded_members_matching_multiple_protected_collections": excluded_multiple,
            },
            "artifacts": {
                "permitted_member_ids": permitted_artifact,
                "excluded_member_ledger": excluded_artifact,
            },
            "implementation_sha256": proof.sha256_file(Path(__file__).resolve())[1],
        }
        if permitted_artifact["row_count"] + excluded_artifact["row_count"] != pretrain_member_count:
            raise RuntimeError("permitted and excluded rows do not partition pretrain membership")
        manifest["manifest_canonical_payload_sha256"] = proof.sha256_bytes(
            proof.canonical_json_bytes(manifest)
        )
        write_canonical_json_new(output_dir / MANIFEST_FILENAME, manifest)
        return manifest
    finally:
        connection.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-manifest", required=True)
    parser.add_argument(
        "--protected-manifest",
        action="append",
        metavar="PATH",
        required=True,
        help="Repeat for each protected downstream validation/test collection.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = derive_clean_membership(
        args.pretrain_manifest,
        args.protected_manifest,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "permitted_member_count": manifest["counts"]["permitted_member_count"],
                "excluded_member_count": manifest["counts"]["excluded_member_count"],
                "manifest": str(Path(args.output_dir).resolve() / MANIFEST_FILENAME),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
