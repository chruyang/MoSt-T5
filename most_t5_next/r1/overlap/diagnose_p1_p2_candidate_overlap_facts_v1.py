#!/usr/bin/env python3
"""Report P1-vs-candidate-P2 identity-overlap facts without policy admission.

The diagnostic is deliberately narrower than the all-downstream proof.  It
loads both collection manifests through the production proof consumer's
strict ``load_collection`` path, computes only connectivity/stereo facts, and
never selects or evaluates an overlap policy.
"""

from __future__ import print_function

import argparse
import json
import os
import platform
import sqlite3
import sys
from pathlib import Path

try:
    from . import prove_membership_identity_overlap_v1 as proof
except ImportError:  # Direct execution from this directory.
    import prove_membership_identity_overlap_v1 as proof


CONTRACT_SCHEMA = "most-t5-r1/p1-p2-candidate-overlap-fact-contract/v1"
REPORT_SCHEMA = "most-t5-r1/p1-p2-candidate-overlap-fact-report/v1"
REPORT_FILENAME = "p1_p2_candidate_overlap_fact_report.json"
P1_ROLE = "p1_structure_train"
P2_ROLE = "p2_permitted_train_membership"
SHARED_SPEC_FIELDS = (
    "connectivity_identity_spec_sha256",
    "stereo_identity_spec_sha256",
)
ADMISSION_FIELDS = (
    "p1_training_admission",
    "p2_training_admission",
    "all_downstream_exclusion_proven",
    "p1_p2_policy_compliance_proven",
)
REPORTED_FACTS = (
    "connectivity_identity_unique_overlap",
    "stereo_identity_unique_overlap",
    "members_impacted_on_each_side",
    "connectivity_overlap_without_stereo_match",
)
CONTRACT_FIELDS = frozenset(
    (
        "schema_version",
        "purpose",
        "report_schema_version",
        "accepted_input_roles",
        "required_shared_identity_specs",
        "reported_facts",
        "unavailable_dimensions",
        "output_filename",
        "admission_fields_fixed_false",
    )
)


def validate_contract(contract):
    proof.require_exact_fields(contract, CONTRACT_FIELDS, "candidate-overlap fact contract")
    if contract["schema_version"] != CONTRACT_SCHEMA:
        raise ValueError("candidate-overlap fact contract schema mismatch")
    if contract["report_schema_version"] != REPORT_SCHEMA:
        raise ValueError("candidate-overlap fact report schema mismatch")
    if contract["accepted_input_roles"] != {"p1": P1_ROLE, "p2": P2_ROLE}:
        raise ValueError("candidate-overlap fact contract roles differ from the diagnostic")
    if contract["required_shared_identity_specs"] != list(SHARED_SPEC_FIELDS):
        raise ValueError("candidate-overlap shared identity specs differ from the diagnostic")
    if contract["unavailable_dimensions"] != ["conformer_identity", "text_identity"]:
        raise ValueError("candidate-overlap unavailable dimensions differ from the diagnostic")
    if contract["output_filename"] != REPORT_FILENAME:
        raise ValueError("candidate-overlap output filename differs from the diagnostic")
    if contract["admission_fields_fixed_false"] != list(ADMISSION_FIELDS):
        raise ValueError("candidate-overlap admission semantics differ from the diagnostic")
    proof.require_string(contract["purpose"], "candidate-overlap fact contract purpose")
    if contract["reported_facts"] != list(REPORTED_FACTS):
        raise ValueError("candidate-overlap reported facts differ from the diagnostic")


def require_roles_and_shared_specs(p1_collection, p2_collection):
    if p1_collection["role"] != P1_ROLE:
        raise ValueError("p1 manifest role must be {}".format(P1_ROLE))
    if p2_collection["role"] != P2_ROLE:
        raise ValueError("p2 manifest role must be {}".format(P2_ROLE))
    if p1_collection["collection_id"] == p2_collection["collection_id"]:
        raise ValueError("p1 and p2 collection IDs must differ")
    for field in SHARED_SPEC_FIELDS:
        left = p1_collection["identity_specs"][field]
        right = p2_collection["identity_specs"][field]
        if left != right:
            raise ValueError("p1 and p2 {} values differ".format(field))


def named_counts(counts):
    return {
        "p1_unique_count": counts["left_unique_count"],
        "p2_unique_count": counts["right_unique_count"],
        "overlap_unique_count": counts["overlap_unique_count"],
        "p1_members_impacted": counts["left_rows_impacted"],
        "p2_members_impacted": counts["right_rows_impacted"],
    }


def named_cross_resolution(counts):
    return {
        "p1_members_connectivity_overlap_without_stereo_match": counts[
            "left_members_connectivity_overlap_without_stereo_match"
        ],
        "p2_members_connectivity_overlap_without_stereo_match": counts[
            "right_members_connectivity_overlap_without_stereo_match"
        ],
        "p1_members_molecule_overlap_without_exact_conformer_match": None,
        "p2_members_molecule_overlap_without_exact_conformer_match": None,
    }


def canonical_payload_sha256(report):
    payload = dict(report)
    payload.pop("report_canonical_payload_sha256", None)
    return proof.sha256_bytes(proof.canonical_json_bytes(payload))


def write_canonical_json_new(path, value):
    raw = proof.canonical_json_bytes(value) + b"\n"
    with open(str(path), "xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def make_input_binding(slot, expected_manifest_sha256, collection, observation):
    return {
        "slot": slot,
        "collection_id": collection["collection_id"],
        "role": collection["role"],
        "expected_manifest_sha256": expected_manifest_sha256,
        "strict_load_observation": observation,
    }


def run_diagnostic(
    contract_path,
    p1_manifest_path,
    p1_manifest_sha256,
    p2_manifest_path,
    p2_manifest_sha256,
    output_dir,
    database_path=":memory:",
):
    contract_path = proof.regular_nonsymlink(Path(contract_path), "candidate-overlap fact contract").resolve()
    contract_bytes, contract_sha256 = proof.sha256_file(contract_path)
    contract = proof.load_json(contract_path, "candidate-overlap fact contract")
    validate_contract(contract)
    proof.require_sha256(p1_manifest_sha256, "p1 external manifest SHA-256")
    proof.require_sha256(p2_manifest_sha256, "p2 external manifest SHA-256")
    p1_manifest_path = proof.regular_nonsymlink(Path(p1_manifest_path), "p1 collection manifest").resolve()
    p2_manifest_path = proof.regular_nonsymlink(Path(p2_manifest_path), "p2 collection manifest").resolve()

    if database_path != ":memory:":
        database_file = Path(database_path)
        if database_file.exists():
            raise FileExistsError("refusing to reuse an existing diagnostic database: {}".format(database_file))
        database_file.parent.mkdir(parents=True, exist_ok=True)
        database_path = str(database_file.resolve())

    connection = proof.create_database(database_path)
    try:
        p1_collection, p1_observation = proof.load_collection(
            connection, p1_manifest_path, p1_manifest_sha256
        )
        p2_collection, p2_observation = proof.load_collection(
            connection, p2_manifest_path, p2_manifest_sha256
        )
        require_roles_and_shared_specs(p1_collection, p2_collection)
        connection.commit()
        connection.executescript(
            """
            CREATE INDEX candidate_molecules_connectivity
                ON molecules(collection_id, connectivity_sha256);
            CREATE INDEX candidate_molecules_stereo
                ON molecules(collection_id, stereo_sha256);
            """
        )
        p1_id = p1_collection["collection_id"]
        p2_id = p2_collection["collection_id"]
        connectivity = named_counts(
            proof.dimension_counts(connection, p1_id, p2_id, "connectivity_identity")
        )
        stereo = named_counts(proof.dimension_counts(connection, p1_id, p2_id, "stereo_identity"))
        cross_resolution = named_cross_resolution(
            proof.cross_resolution_counts(connection, p1_id, p2_id, conformer_available=False)
        )
        diagnostic_binding_sha256 = proof.sha256_bytes(
            proof.canonical_json_bytes(
                {
                    "contract_sha256": contract_sha256,
                    "p1_manifest_sha256": p1_manifest_sha256,
                    "p2_manifest_sha256": p2_manifest_sha256,
                }
            )
        )
        report = {
            "schema_version": REPORT_SCHEMA,
            "diagnostic_id": "p1-p2-candidate-overlap-facts-{}".format(
                diagnostic_binding_sha256[:20]
            ),
            "diagnostic_completion": "facts_reported",
            "diagnostic_only": True,
            "generated_at_utc": proof.utc_now(),
            "admissions": {field: False for field in ADMISSION_FIELDS},
            "provenance": {
                "contract_path": str(contract_path),
                "contract_bytes": contract_bytes,
                "contract_sha256": contract_sha256,
                "diagnostic_path": str(Path(__file__).resolve()),
                "diagnostic_sha256": proof.sha256_file(Path(__file__).resolve())[1],
                "strict_loader_path": str(Path(proof.__file__).resolve()),
                "strict_loader_sha256": proof.sha256_file(Path(proof.__file__).resolve())[1],
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "sqlite": sqlite3.sqlite_version,
            },
            "input_artifact_bindings": [
                make_input_binding("p1", p1_manifest_sha256, p1_collection, p1_observation),
                make_input_binding("p2", p2_manifest_sha256, p2_collection, p2_observation),
            ],
            "collections": {
                "p1": {
                    "summary": proof.collection_summary(connection, p1_collection),
                    "identity_specs": p1_collection["identity_specs"],
                },
                "p2": {
                    "summary": proof.collection_summary(connection, p2_collection),
                    "identity_specs": p2_collection["identity_specs"],
                },
            },
            "facts": {
                "connectivity_identity": connectivity,
                "stereo_identity": stereo,
                "cross_resolution": cross_resolution,
                "unavailable_dimensions": {
                    "conformer_identity": {
                        "status": "unavailable_for_comparison",
                        "reason": "excluded_from_this_candidate_fact_contract_no_shared_conformer_spec_is_required",
                        "p1_manifest_status": p1_collection["identity_specs"]["conformer_identity"]["status"],
                        "p2_manifest_status": p2_collection["identity_specs"]["conformer_identity"]["status"],
                    },
                    "text_identity": {
                        "status": "unavailable_for_comparison",
                        "reason": "both_accepted_roles_forbid_text_identity_rows",
                        "p1_manifest_status": p1_collection["identity_specs"]["text_identity"]["status"],
                        "p2_manifest_status": p2_collection["identity_specs"]["text_identity"]["status"],
                    },
                },
            },
            "scope_warning": (
                "This diagnostic reports facts for exactly two bound candidate collections. "
                "It does not select an overlap policy, prove all-downstream exclusion, or admit P1/P2 training."
            ),
        }
        report["report_canonical_payload_sha256"] = canonical_payload_sha256(report)
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        write_canonical_json_new(output_dir / REPORT_FILENAME, report)
        return report
    finally:
        connection.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--p1-manifest", required=True)
    parser.add_argument("--p1-manifest-sha256", required=True)
    parser.add_argument("--p2-manifest", required=True)
    parser.add_argument("--p2-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--database",
        default=":memory:",
        help="SQLite path for a disk-backed exact-set diagnosis; an existing file is rejected.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_diagnostic(
        args.contract,
        args.p1_manifest,
        args.p1_manifest_sha256,
        args.p2_manifest,
        args.p2_manifest_sha256,
        args.output_dir,
        args.database,
    )
    print(
        json.dumps(
            {
                "diagnostic_completion": report["diagnostic_completion"],
                "report": str(Path(args.output_dir).resolve() / REPORT_FILENAME),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
