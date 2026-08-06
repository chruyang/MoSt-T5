#!/usr/bin/env python3
"""Derive a connectivity-group-disjoint view of one reported downstream split.

The input train/validation/test collections remain immutable.  Connectivity
groups are assigned once with the fixed priority ``test > validation >
train``.  Every molecule member in the owning split is retained, together
with every text-pair row that references that retained member; occurrences of
the same connectivity in lower-priority splits are represented in an explicit
disposition ledger instead of being silently discarded.

Caller-supplied digests are deliberately not part of this interface.  Input
manifest and implementation digests are recorded as provenance observations,
not scientific-admission claims.  The referenced row artifacts still have to
close under the standard ``identity-collection-manifest/v1`` contract: their
declared rows must exist, be canonical, match their declarations, and text
pairs must reference members in the same collection.
"""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

try:
    from . import prove_membership_identity_overlap_v1 as proof
except ImportError:  # Direct execution from this directory.
    import prove_membership_identity_overlap_v1 as proof


SUMMARY_SCHEMA = "most-t5-r1/downstream-connectivity-clean-view-summary/v1"
DISPOSITION_SCHEMA = "most-t5-r1/downstream-connectivity-member-disposition/v1"
SUMMARY_FILENAME = "clean_view_summary.json"
DISPOSITION_FILENAME = "member_disposition.jsonl"
SPLITS = ("train", "validation", "test")
PRIORITY = {"train": 0, "validation": 1, "test": 2}
EXPECTED_ROLES = {
    "train": "downstream_train",
    "validation": "downstream_validation",
    "test": "downstream_test",
}


class CleanViewError(ValueError):
    """Raised when reported collections cannot define one clean view."""


def canonical_json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_observation(path):
    byte_count, digest = proof.sha256_file(path)
    return {"bytes": byte_count, "sha256": digest}


def compact_artifact_observation(observation):
    return {
        key: observation[key]
        for key in ("bytes", "sha256", "row_count", "key_lf_sha256")
    }


def write_json_new(path, value):
    payload = canonical_json_bytes(value) + b"\n"
    with open(str(path), "xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {"bytes": len(payload), "sha256": sha256_bytes(payload)}


def write_rows_new(path, rows):
    """Write ``(row, stable_key)`` pairs and return a standard declaration."""
    file_digest = hashlib.sha256()
    key_digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    with open(str(path), "xb") as handle:
        for row, stable_key in rows:
            raw = canonical_json_bytes(row) + b"\n"
            encoded_key = stable_key.encode("utf-8")
            handle.write(raw)
            file_digest.update(raw)
            key_digest.update(encoded_key + b"\n")
            byte_count += len(raw)
            row_count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": path.name,
        "bytes": byte_count,
        "sha256": file_digest.hexdigest(),
        "row_count": row_count,
        "key_lf_sha256": key_digest.hexdigest(),
    }


def read_and_validate_manifest(path, split):
    path = proof.regular_nonsymlink(Path(path), "{} collection manifest".format(split)).resolve()
    manifest = proof.load_json(path, "{} collection manifest".format(split))
    proof.validate_collection_manifest(manifest)
    if manifest["phase"] != "downstream":
        raise CleanViewError("{} collection phase must be downstream".format(split))
    if manifest["split"] != split:
        raise CleanViewError(
            "{} input declares split {}".format(split, manifest["split"])
        )
    if manifest["role"] != EXPECTED_ROLES[split]:
        raise CleanViewError(
            "{} input role must be {}".format(split, EXPECTED_ROLES[split])
        )
    observation = file_observation(path)
    observation["path"] = str(path)
    return manifest, path, observation


def require_one_reported_family(manifests):
    collection_ids = [manifests[split]["collection_id"] for split in SPLITS]
    if len(set(collection_ids)) != len(collection_ids):
        raise CleanViewError("reported collection IDs must be distinct")
    reference = manifests["train"]
    for split in ("validation", "test"):
        candidate = manifests[split]
        for field in ("dataset_id", "release_id", "task_family"):
            if candidate[field] != reference[field]:
                raise CleanViewError(
                    "reported collections differ in {}".format(field)
                )
        if candidate["identity_specs"] != reference["identity_specs"]:
            raise CleanViewError("reported collections differ in identity_specs")


def create_work_database(path):
    connection = proof.create_database(path)
    connection.execute(
        """
        CREATE TABLE source_collections (
            collection_id TEXT PRIMARY KEY,
            split TEXT NOT NULL UNIQUE,
            priority INTEGER NOT NULL UNIQUE
        ) WITHOUT ROWID
        """
    )
    return connection


def load_reported_rows(connection, manifests, paths, observations):
    for split in SPLITS:
        manifest = manifests[split]
        connection.execute(
            "INSERT INTO source_collections VALUES (?,?,?)",
            (manifest["collection_id"], split, PRIORITY[split]),
        )
        observations[split]["molecule_rows"] = proof.load_molecule_rows(
            connection, manifest, paths[split]
        )
        observations[split]["text_pair_rows"] = proof.load_text_rows(
            connection, manifest, paths[split]
        )
    connection.execute(
        """
        CREATE TEMP TABLE connectivity_owners AS
        SELECT m.connectivity_sha256 AS connectivity_sha256,
               MAX(s.priority) AS owner_priority
        FROM molecules AS m
        JOIN source_collections AS s ON s.collection_id=m.collection_id
        GROUP BY m.connectivity_sha256
        """
    )
    connection.execute(
        "CREATE UNIQUE INDEX connectivity_owners_key "
        "ON connectivity_owners(connectivity_sha256)"
    )
    connection.commit()


def scalar(connection, query, parameters=()):
    row = connection.execute(query, parameters).fetchone()
    return int(row[0])


def source_counts(connection, collection_id):
    return {
        "member_count": scalar(
            connection,
            "SELECT COUNT(*) FROM molecules WHERE collection_id=?",
            (collection_id,),
        ),
        "unique_connectivity_count": scalar(
            connection,
            "SELECT COUNT(DISTINCT connectivity_sha256) "
            "FROM molecules WHERE collection_id=?",
            (collection_id,),
        ),
        "text_pair_count": scalar(
            connection,
            "SELECT COUNT(*) FROM text_pairs WHERE collection_id=?",
            (collection_id,),
        ),
    }


def retained_counts(connection, split):
    parameters = (split,)
    member_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM molecules AS m
        JOIN source_collections AS s ON s.collection_id=m.collection_id
        JOIN connectivity_owners AS o
          ON o.connectivity_sha256=m.connectivity_sha256
        WHERE s.split=? AND s.priority=o.owner_priority
        """,
        parameters,
    )
    unique_connectivity_count = scalar(
        connection,
        """
        SELECT COUNT(DISTINCT m.connectivity_sha256)
        FROM molecules AS m
        JOIN source_collections AS s ON s.collection_id=m.collection_id
        JOIN connectivity_owners AS o
          ON o.connectivity_sha256=m.connectivity_sha256
        WHERE s.split=? AND s.priority=o.owner_priority
        """,
        parameters,
    )
    text_pair_count = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM text_pairs AS t
        JOIN source_collections AS s ON s.collection_id=t.collection_id
        JOIN molecules AS m
          ON m.collection_id=t.collection_id AND m.member_id=t.member_id
        JOIN connectivity_owners AS o
          ON o.connectivity_sha256=m.connectivity_sha256
        WHERE s.split=? AND s.priority=o.owner_priority
        """,
        parameters,
    )
    return {
        "member_count": member_count,
        "unique_connectivity_count": unique_connectivity_count,
        "text_pair_count": text_pair_count,
    }


def removed_text_pair_count(connection, split):
    return scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM text_pairs AS t
        JOIN source_collections AS s ON s.collection_id=t.collection_id
        JOIN molecules AS m
          ON m.collection_id=t.collection_id AND m.member_id=t.member_id
        JOIN connectivity_owners AS o
          ON o.connectivity_sha256=m.connectivity_sha256
        WHERE s.split=? AND s.priority<o.owner_priority
        """,
        (split,),
    )


def molecule_output_rows(connection, split, output_collection_id):
    rows = connection.execute(
        """
        SELECT m.member_id, m.connectivity_sha256, m.stereo_sha256,
               m.conformer_sha256
        FROM molecules AS m
        JOIN source_collections AS s ON s.collection_id=m.collection_id
        JOIN connectivity_owners AS o
          ON o.connectivity_sha256=m.connectivity_sha256
        WHERE s.split=? AND s.priority=o.owner_priority
        ORDER BY m.member_id COLLATE BINARY
        """,
        (split,),
    )
    for member_id, connectivity, stereo, conformer in rows:
        yield (
            {
                "schema_version": proof.MOLECULE_ROW_SCHEMA,
                "collection_id": output_collection_id,
                "member_id": member_id,
                "connectivity_identity_sha256": connectivity,
                "stereo_identity_sha256": stereo,
                "conformer_identity_sha256": conformer,
            },
            member_id,
        )


def text_output_rows(connection, split, output_collection_id):
    rows = connection.execute(
        """
        SELECT t.pair_id, t.member_id, t.task_family, t.text_exact_sha256,
               t.text_normalized_sha256, t.connectivity_pair_sha256,
               t.stereo_pair_sha256
        FROM text_pairs AS t
        JOIN source_collections AS s ON s.collection_id=t.collection_id
        JOIN molecules AS m
          ON m.collection_id=t.collection_id AND m.member_id=t.member_id
        JOIN connectivity_owners AS o
          ON o.connectivity_sha256=m.connectivity_sha256
        WHERE s.split=? AND s.priority=o.owner_priority
        ORDER BY t.pair_id COLLATE BINARY
        """,
        (split,),
    )
    for values in rows:
        pair_id, member_id, task_family, exact, normalized, connectivity_pair, stereo_pair = values
        yield (
            {
                "schema_version": proof.TEXT_ROW_SCHEMA,
                "collection_id": output_collection_id,
                "pair_id": pair_id,
                "member_id": member_id,
                "task_family": task_family,
                "text_exact_sha256": exact,
                "text_normalized_sha256": normalized,
                "connectivity_text_pair_sha256": connectivity_pair,
                "stereo_text_pair_sha256": stereo_pair,
            },
            pair_id,
        )


def disposition_rows(connection):
    rows = connection.execute(
        """
        SELECT s.collection_id, s.split, m.member_id, m.connectivity_sha256,
               s.priority, o.owner_priority
        FROM molecules AS m
        JOIN source_collections AS s ON s.collection_id=m.collection_id
        JOIN connectivity_owners AS o
          ON o.connectivity_sha256=m.connectivity_sha256
        ORDER BY s.split COLLATE BINARY, m.member_id COLLATE BINARY
        """
    )
    split_by_priority = {value: key for key, value in PRIORITY.items()}
    for collection_id, split, member_id, connectivity, priority, owner_priority in rows:
        assigned_split = split_by_priority[owner_priority]
        retained = priority == owner_priority
        row = {
            "schema_version": DISPOSITION_SCHEMA,
            "source_collection_id": collection_id,
            "source_split": split,
            "member_id": member_id,
            "connectivity_identity_sha256": connectivity,
            "disposition": "retained" if retained else "removed",
            "assigned_split": assigned_split,
            "removal_reason": (
                None
                if retained
                else "connectivity_owned_by_higher_priority_{}".format(assigned_split)
            ),
        }
        stable_key = canonical_json_bytes([collection_id, member_id]).decode("utf-8")
        yield row, stable_key


def projected_digest(connection, table, collection_id):
    digest = hashlib.sha256()
    count = 0
    if table == "molecules":
        rows = connection.execute(
            """
            SELECT member_id, connectivity_sha256, stereo_sha256, conformer_sha256
            FROM molecules WHERE collection_id=?
            ORDER BY member_id COLLATE BINARY
            """,
            (collection_id,),
        )
        keys = ("member_id", "connectivity", "stereo", "conformer")
    elif table == "text_pairs":
        rows = connection.execute(
            """
            SELECT pair_id, member_id, task_family, text_exact_sha256,
                   text_normalized_sha256, connectivity_pair_sha256,
                   stereo_pair_sha256
            FROM text_pairs WHERE collection_id=?
            ORDER BY pair_id COLLATE BINARY
            """,
            (collection_id,),
        )
        keys = (
            "pair_id",
            "member_id",
            "task_family",
            "text_exact",
            "text_normalized",
            "connectivity_pair",
            "stereo_pair",
        )
    else:
        raise ValueError("unknown projection table")
    for values in rows:
        digest.update(canonical_json_bytes(dict(zip(keys, values))) + b"\n")
        count += 1
    return {"row_count": count, "projection_sha256": digest.hexdigest()}


def pairwise_connectivity_intersections(connection, output_ids):
    result = {}
    for left_index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[left_index + 1 :]:
            result["{}__{}".format(left_split, right_split)] = scalar(
                connection,
                """
                SELECT COUNT(DISTINCT left_rows.connectivity_sha256)
                FROM molecules AS left_rows
                JOIN molecules AS right_rows
                  ON right_rows.connectivity_sha256=left_rows.connectivity_sha256
                WHERE left_rows.collection_id=? AND right_rows.collection_id=?
                """,
                (output_ids[left_split], output_ids[right_split]),
            )
    return result


def write_clean_collection(
    connection,
    output_dir,
    split,
    source_manifest,
    source_bundle_sha256,
    implementation_sha256,
    excluded_metadata_keys,
):
    output_collection_id = source_manifest["collection_id"] + "-connectivity-clean-v1"
    collection_dir = output_dir / "collections" / split
    collection_dir.mkdir(parents=True, exist_ok=False)
    molecule_declaration = write_rows_new(
        collection_dir / "molecule_identity_rows.jsonl",
        molecule_output_rows(connection, split, output_collection_id),
    )
    if molecule_declaration["row_count"] <= 0:
        raise CleanViewError("clean {} collection would be empty".format(split))

    text_spec = source_manifest["identity_specs"]["text_identity"]
    text_declaration = None
    if text_spec["status"] == "available":
        text_declaration = write_rows_new(
            collection_dir / "text_pair_identity_rows.jsonl",
            text_output_rows(connection, split, output_collection_id),
        )
        if text_declaration["row_count"] <= 0:
            raise CleanViewError(
                "clean {} collection has available text identity but no retained text pair".format(split)
            )

    manifest = {
        "schema_version": proof.COLLECTION_SCHEMA,
        "collection_id": output_collection_id,
        "dataset_id": source_manifest["dataset_id"],
        "release_id": source_manifest["release_id"],
        "phase": "downstream",
        "split": split,
        "role": EXPECTED_ROLES[split],
        "task_family": source_manifest["task_family"],
        "identity_specs": copy.deepcopy(source_manifest["identity_specs"]),
        "molecule_rows": molecule_declaration,
        "text_pair_rows": text_declaration,
        "provenance": {
            "source_identity_namespace": "connectivity-clean-view:{}:{}".format(
                source_manifest["dataset_id"], source_manifest["release_id"]
            ),
            "source_release_manifest_sha256": source_bundle_sha256,
            "extractor_sha256": implementation_sha256,
            "excluded_source_metadata_keys": excluded_metadata_keys,
        },
    }
    proof.validate_collection_manifest(manifest)
    manifest_path = collection_dir / "collection_manifest.json"
    manifest_observation = write_json_new(manifest_path, manifest)
    return manifest, manifest_path, {
        "collection_id": output_collection_id,
        "manifest": {
            "relative_path": manifest_path.relative_to(output_dir).as_posix(),
            **manifest_observation,
        },
        "molecule_rows": {
            "relative_path": (
                collection_dir / "molecule_identity_rows.jsonl"
            ).relative_to(output_dir).as_posix(),
            **{key: molecule_declaration[key] for key in ("bytes", "sha256", "row_count", "key_lf_sha256")},
        },
        "text_pair_rows": (
            None
            if text_declaration is None
            else {
                "relative_path": (
                    collection_dir / "text_pair_identity_rows.jsonl"
                ).relative_to(output_dir).as_posix(),
                **{key: text_declaration[key] for key in ("bytes", "sha256", "row_count", "key_lf_sha256")},
            }
        ),
    }


def derive_clean_view(train_manifest, validation_manifest, test_manifest, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("output directory already exists: {}".format(output_dir))

    supplied = {
        "train": train_manifest,
        "validation": validation_manifest,
        "test": test_manifest,
    }
    manifests = {}
    paths = {}
    observations = {}
    for split in SPLITS:
        manifests[split], paths[split], observations[split] = read_and_validate_manifest(
            supplied[split], split
        )
    require_one_reported_family(manifests)

    temporary = tempfile.NamedTemporaryFile(
        prefix="most_t5_downstream_clean_", suffix=".sqlite", delete=False
    )
    database_path = Path(temporary.name)
    temporary.close()
    connection = None
    try:
        connection = create_work_database(database_path)
        load_reported_rows(connection, manifests, paths, observations)

        input_counts = {
            split: source_counts(connection, manifests[split]["collection_id"])
            for split in SPLITS
        }
        clean_counts = {split: retained_counts(connection, split) for split in SPLITS}
        text_available = (
            manifests["train"]["identity_specs"]["text_identity"]["status"]
            == "available"
        )
        for split in SPLITS:
            if clean_counts[split]["member_count"] <= 0:
                raise CleanViewError("clean {} membership would be empty".format(split))
            if text_available and clean_counts[split]["text_pair_count"] <= 0:
                raise CleanViewError(
                    "clean {} view would have no text-pair rows".format(split)
                )

        input_binding = [
            {
                "split": split,
                "collection_id": manifests[split]["collection_id"],
                "observed_manifest_sha256": observations[split]["sha256"],
            }
            for split in SPLITS
        ]
        source_bundle_sha256 = sha256_bytes(canonical_json_bytes(input_binding))
        implementation_sha256 = file_observation(Path(__file__).resolve())["sha256"]
        excluded_metadata_keys = sorted(
            {
                key
                for split in SPLITS
                for key in manifests[split]["provenance"]["excluded_source_metadata_keys"]
            },
            key=lambda value: value.encode("utf-8"),
        )

        output_dir.mkdir(parents=True, exist_ok=False)
        output_manifests = {}
        output_paths = {}
        output_reports = {}
        output_ids = {}
        for split in SPLITS:
            manifest, manifest_path, report = write_clean_collection(
                connection,
                output_dir,
                split,
                manifests[split],
                source_bundle_sha256,
                implementation_sha256,
                excluded_metadata_keys,
            )
            output_manifests[split] = manifest
            output_paths[split] = manifest_path
            output_reports[split] = report
            output_ids[split] = manifest["collection_id"]

        # Re-read the emitted standard artifacts through the shared validator.
        for split in SPLITS:
            proof.load_molecule_rows(
                connection, output_manifests[split], output_paths[split]
            )
            proof.load_text_rows(
                connection, output_manifests[split], output_paths[split]
            )
        connection.commit()

        emitted_counts = {
            split: source_counts(connection, output_ids[split]) for split in SPLITS
        }
        if emitted_counts != clean_counts:
            raise RuntimeError(
                "emitted collection counts differ from the priority-derived view"
            )
        removed_text_counts = {
            split: removed_text_pair_count(connection, split) for split in SPLITS
        }
        all_source_text_pairs_accounted = all(
            input_counts[split]["text_pair_count"]
            == clean_counts[split]["text_pair_count"] + removed_text_counts[split]
            for split in SPLITS
        )
        if not all_source_text_pairs_accounted:
            raise RuntimeError("source text-pair rows are not fully accounted")

        disposition_declaration = write_rows_new(
            output_dir / DISPOSITION_FILENAME, disposition_rows(connection)
        )
        total_input_members = sum(item["member_count"] for item in input_counts.values())
        total_retained_members = sum(item["member_count"] for item in clean_counts.values())
        if disposition_declaration["row_count"] != total_input_members:
            raise RuntimeError("member disposition ledger does not cover every reported member")

        intersections = pairwise_connectivity_intersections(connection, output_ids)
        if any(intersections.values()):
            raise RuntimeError("derived connectivity groups are not split-disjoint")

        test_source_id = manifests["test"]["collection_id"]
        test_output_id = output_ids["test"]
        test_molecule_source = projected_digest(connection, "molecules", test_source_id)
        test_molecule_output = projected_digest(connection, "molecules", test_output_id)
        test_text_source = projected_digest(connection, "text_pairs", test_source_id)
        test_text_output = projected_digest(connection, "text_pairs", test_output_id)
        test_unchanged = (
            test_molecule_source == test_molecule_output
            and test_text_source == test_text_output
        )
        if not test_unchanged:
            raise RuntimeError("highest-priority test collection was not preserved")

        removal_reasons = {}
        split_by_priority = {value: key for key, value in PRIORITY.items()}
        reason_rows = connection.execute(
            """
            SELECT s.priority, o.owner_priority, COUNT(*)
            FROM molecules AS m
            JOIN source_collections AS s ON s.collection_id=m.collection_id
            JOIN connectivity_owners AS o
              ON o.connectivity_sha256=m.connectivity_sha256
            WHERE s.priority<o.owner_priority
            GROUP BY s.priority, o.owner_priority
            ORDER BY s.priority, o.owner_priority
            """
        )
        for source_priority, owner_priority, count in reason_rows:
            key = "{}_removed_for_{}".format(
                split_by_priority[source_priority], split_by_priority[owner_priority]
            )
            removal_reasons[key] = int(count)

        source_reports = []
        for split in SPLITS:
            text_observation = observations[split]["text_pair_rows"]
            source_reports.append(
                {
                    "split": split,
                    "collection_id": manifests[split]["collection_id"],
                    "manifest_observation": {
                        "path": observations[split]["path"],
                        "bytes": observations[split]["bytes"],
                        "sha256": observations[split]["sha256"],
                    },
                    "molecule_rows": compact_artifact_observation(
                        observations[split]["molecule_rows"]
                    ),
                    "text_pair_rows": (
                        None
                        if text_observation is None
                        else compact_artifact_observation(text_observation)
                    ),
                    "counts": input_counts[split],
                }
            )

        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "status": "complete",
            "view_id": "{}-connectivity-group-disjoint-clean-v1".format(
                manifests["train"]["dataset_id"]
            ),
            "dataset_id": manifests["train"]["dataset_id"],
            "release_id": manifests["train"]["release_id"],
            "task_family": manifests["train"]["task_family"],
            "policy": {
                "group_key": "connectivity_identity_sha256",
                "split_priority_high_to_low": ["test", "validation", "train"],
                "within_owner_split_semantics": "retain_all_members_and_referencing_text_pairs",
                "reported_inputs_modified": False,
            },
            "provenance_observation_policy": {
                "caller_supplied_digest_required": False,
                "observed_digests_are_scientific_admission_gates": False,
                "semantic_admission_basis": [
                    "one reported dataset release and task family",
                    "standard collection roles and split labels",
                    "identical identity specifications",
                    "referenced molecule and text row closure",
                ],
                "source_bundle_observation_sha256": source_bundle_sha256,
                "implementation_observation_sha256": implementation_sha256,
            },
            "reported_sources": source_reports,
            "clean_collections": [output_reports[split] for split in SPLITS],
            "counts": {
                "input_members": total_input_members,
                "retained_members": total_retained_members,
                "removed_members": total_input_members - total_retained_members,
                "by_split": {
                    split: {
                        "input": input_counts[split],
                        "retained": clean_counts[split],
                        "removed_member_count": (
                            input_counts[split]["member_count"]
                            - clean_counts[split]["member_count"]
                        ),
                        "removed_text_pair_count": (
                            removed_text_counts[split]
                        ),
                    }
                    for split in SPLITS
                },
                "removal_reasons": removal_reasons,
            },
            "member_disposition": {
                "relative_path": DISPOSITION_FILENAME,
                **{key: disposition_declaration[key] for key in ("bytes", "sha256", "row_count", "key_lf_sha256")},
            },
            "proofs": {
                "pairwise_clean_connectivity_intersections": intersections,
                "connectivity_split_disjoint": not any(intersections.values()),
                "test_molecules_unchanged": test_molecule_source == test_molecule_output,
                "test_text_pairs_unchanged": test_text_source == test_text_output,
                "test_unchanged": test_unchanged,
                "all_reported_members_have_one_disposition": (
                    disposition_declaration["row_count"] == total_input_members
                ),
                "every_removed_member_has_reason": (
                    sum(removal_reasons.values())
                    == total_input_members - total_retained_members
                ),
                "emitted_counts_match_priority_view": emitted_counts == clean_counts,
                "all_source_text_pairs_accounted": all_source_text_pairs_accounted,
                "output_text_pairs_reference_retained_members": True,
                "test_projection_observations": {
                    "reported_molecules": test_molecule_source,
                    "clean_molecules": test_molecule_output,
                    "reported_text_pairs": test_text_source,
                    "clean_text_pairs": test_text_output,
                },
            },
        }
        write_json_new(output_dir / SUMMARY_FILENAME, summary)
        return summary
    finally:
        if connection is not None:
            connection.close()
        if database_path.is_file():
            database_path.unlink()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = derive_clean_view(
        args.train_manifest,
        args.validation_manifest,
        args.test_manifest,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "retained_members": summary["counts"]["retained_members"],
                "removed_members": summary["counts"]["removed_members"],
                "summary": str(Path(args.output_dir).resolve() / SUMMARY_FILENAME),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
