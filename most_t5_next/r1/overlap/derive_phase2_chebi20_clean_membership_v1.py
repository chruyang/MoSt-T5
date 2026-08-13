#!/usr/bin/env python3
"""Derive Phase-II membership protected against ChEBI-20 test connectivity.

This is a deliberately narrow policy wrapper around
``derive_clean_pretrain_membership_v1``.  It prevents a caller from silently
adding another downstream task, substituting the validation split, or
using a different Phase-II source collection.  The underlying exclusion rule
is canonical non-stereo connectivity identity: every matching Phase-II record
is excluded in full, independent of its represented stereochemistry, training
objective, or text.
"""

from __future__ import print_function

import argparse
import json
import sys
from pathlib import Path

try:
    from . import derive_clean_pretrain_membership_v1 as derive
    from . import prove_membership_identity_overlap_v1 as proof
except ImportError:  # Direct execution from this directory.
    import derive_clean_pretrain_membership_v1 as derive
    import prove_membership_identity_overlap_v1 as proof


POLICY_RECEIPT_SCHEMA = "most-t5-r1/phase2-chebi20-decontamination-receipt/v1"
POLICY_RECEIPT_FILENAME = "phase2_chebi20_policy_receipt.json"
POLICY_ID = "phase2-chebi20-test-connectivity-exclusion-v1"

EXPECTED_PRETRAIN = {
    "collection_id": "p2-pubchem-motif-ready-301655-identity-v1",
    "dataset_id": "3dmolt5-processed-3dmolm-pubchem-pretrain",
    "release_id": "p2-pubchem-evidence-r0-v1",
    "phase": "p2",
    "split": "train",
    "role": "p2_permitted_train_membership",
    "task_family": "none",
}

EXPECTED_PROTECTED = {
    "test": {
        "collection_id": "downstream-chebi20-test-identity-20260806-v1",
        "dataset_id": "3dmolt5-chebi20-hf",
        "release_id": "3dmolt5-chebi20-hf-9949fae8860ffd7c7dbca8e9848ad37842f1c279",
        "phase": "downstream",
        "split": "test",
        "role": "downstream_test",
        "task_family": "text_to_molecule_generation",
    },
}


def _load_manifest(path):
    resolved = proof.regular_nonsymlink(Path(path), "collection manifest").resolve()
    manifest = proof.load_json(resolved, "collection manifest")
    proof.validate_collection_manifest(manifest)
    return resolved, manifest


def _require_exact_fields(label, observed, expected):
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "{} does not match the frozen Phase-II/ChEBI-20 policy: {}".format(
                label, json.dumps(mismatches, sort_keys=True)
            )
        )


def validate_policy_inputs(pretrain_manifest_path, protected_manifest_paths):
    """Validate and return the exact ChEBI-20 test protected manifest."""
    if len(protected_manifest_paths) != 1:
        raise ValueError(
            "the frozen policy requires exactly one protected manifest: "
            "ChEBI-20 test"
        )
    pretrain_path, pretrain = _load_manifest(pretrain_manifest_path)
    _require_exact_fields("Phase-II source", pretrain, EXPECTED_PRETRAIN)

    protected_by_split = {}
    for path in protected_manifest_paths:
        resolved, manifest = _load_manifest(path)
        split = manifest.get("split")
        if split not in EXPECTED_PROTECTED:
            raise ValueError("protected manifest is not ChEBI-20 test")
        if split in protected_by_split:
            raise ValueError("duplicate protected split: {}".format(split))
        _require_exact_fields("ChEBI-20 {}".format(split), manifest, EXPECTED_PROTECTED[split])
        protected_by_split[split] = (resolved, manifest)
    if set(protected_by_split) != {"test"}:
        raise ValueError("the ChEBI-20 test manifest is required")
    return pretrain_path, [protected_by_split["test"][0]]


def derive_phase2_chebi20_clean_membership(
    pretrain_manifest_path,
    protected_manifest_paths,
    output_dir,
):
    pretrain_path, protected_paths = validate_policy_inputs(
        pretrain_manifest_path, protected_manifest_paths
    )
    output_dir = Path(output_dir).resolve()
    manifest = derive.derive_clean_membership(
        pretrain_path,
        protected_paths,
        output_dir,
        exclusion_dimension="connectivity_identity",
    )
    clean_manifest_path = output_dir / derive.MANIFEST_FILENAME
    clean_manifest_bytes, clean_manifest_sha256 = proof.sha256_file(clean_manifest_path)
    receipt = {
        "schema_version": POLICY_RECEIPT_SCHEMA,
        "policy_id": POLICY_ID,
        "status": "complete",
        "scope": {
            "pretraining_phase": "phase_ii",
            "protected_dataset": "ChEBI-20",
            "protected_splits": ["test"],
            "other_downstream_datasets_used_for_exclusion": [],
            "objective_specific_exceptions": False,
        },
        "exclusion": {
            "key": "connectivity_identity_sha256",
            "action": "exclude_entire_phase2_record",
            "stereo_used_for_exclusion": False,
            "connectivity_used_for_exclusion": True,
            "text_used_for_exclusion": False,
            "source_payload_mutated": False,
        },
        "clean_membership_manifest": {
            "path": derive.MANIFEST_FILENAME,
            "bytes": clean_manifest_bytes,
            "sha256": clean_manifest_sha256,
            "derivation_binding_sha256": manifest["derivation_binding_sha256"],
        },
        "counts": manifest["counts"],
        "implementation_sha256": proof.sha256_file(Path(__file__).resolve())[1],
    }
    receipt["canonical_payload_sha256"] = proof.sha256_bytes(
        proof.canonical_json_bytes(receipt)
    )
    derive.write_canonical_json_new(output_dir / POLICY_RECEIPT_FILENAME, receipt)
    return manifest, receipt


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-manifest", required=True)
    parser.add_argument("--chebi-test-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest, receipt = derive_phase2_chebi20_clean_membership(
        args.pretrain_manifest,
        [args.chebi_test_manifest],
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "policy_id": receipt["policy_id"],
                "counts": manifest["counts"],
                "output_dir": str(Path(args.output_dir).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
