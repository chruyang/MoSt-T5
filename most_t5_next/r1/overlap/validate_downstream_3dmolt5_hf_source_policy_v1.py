#!/usr/bin/env python3
"""Validate the frozen 3D-MolT5 Hugging Face downstream source policy."""

from __future__ import print_function

import argparse
import json
import sys
from pathlib import Path


SCHEMA = "most-t5-r1/downstream-3dmolt5-hf-source-policy/v1"
REFERENCE_REPOSITORY = "QizhiPei/3D-MolT5"
REFERENCE_COMMIT = "82dbe088e424f19fa713dbd657f5235990bd324f"
EXPECTED_DATASETS = {
    "pubchemqc_computed_property_prediction": ("QizhiPei/e3fp-pubchemqc-prop", "e436b3c039a54bbca3beae6a4343b9474c13045c"),
    "pubchem_computed_property_prediction": ("QizhiPei/e3fp-pubchem-com", "22daef54a096ccb6a5c4e366898dafd055438b89"),
    "pubchem_descriptive_property_prediction": ("QizhiPei/e3fp-pubchem-des", "e703538ec77f2af0338b869c175338bfc29b2013"),
    "pubchem_3d_molecule_captioning": ("QizhiPei/e3fp-pubchem-cap", "14802c87730b5fd1846aa69b9e83fd7365fe1f49"),
    "qm9_computed_property_prediction": ("QizhiPei/e3fp-mol-instructions-qm9", "bfe55090be9ebf1c9cbbe6687a5796711ac0edd8"),
    "mol_instructions_reagent_prediction": ("QizhiPei/e3fp-mol-instructions-reagent-prediction", "05ea2015c15ccbe44ad1c69c0635e2a9bdaad5de"),
    "mol_instructions_forward_reaction_prediction": ("QizhiPei/e3fp-mol-instructions-forward-reaction-prediction", "11d2c189c13542b592b0015bb1376497f2a64248"),
    "mol_instructions_retrosynthesis": ("QizhiPei/e3fp-mol-instructions-retrosynthesis", "f45e74b53d94099ce99349bee8b2e498b719176f"),
    "mol_instructions_reaction_all": ("QizhiPei/e3fp-mol-instructions-react-all", "754cc256aa2ed31979408118b28c4eaa7eecafda"),
    "chebi20_text_to_molecule_generation": ("QizhiPei/e3fp-chebi-molgen", "9949fae8860ffd7c7dbca8e9848ad37842f1c279"),
    "uspto50k_retrosynthesis_optional": ("QizhiPei/e3fp-uspto-50k", "1b9a5673d6d42167c65df689ff8f44b969d94187"),
}
EXPECTED_MOLECULENET_TASKS = ["BACE", "BBBP", "HIV", "ClinTox"]


def load_and_validate(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA or document.get("status") != "frozen":
        raise ValueError("downstream source policy schema/status mismatch")
    reference = document.get("official_3dmolt5_reference", {})
    if reference.get("repository") != REFERENCE_REPOSITORY or reference.get("commit") != REFERENCE_COMMIT:
        raise ValueError("official 3D-MolT5 reference binding mismatch")

    observed = {}
    for row in document.get("huggingface_datasets", []):
        task_id = row.get("task_id")
        if task_id in observed:
            raise ValueError("duplicate Hugging Face task_id: {}".format(task_id))
        observed[task_id] = (row.get("repository_id"), row.get("revision"))
    if observed != EXPECTED_DATASETS:
        raise ValueError("official 3D-MolT5 Hugging Face dataset map mismatch")
    for repository_id, revision in observed.values():
        if not repository_id.startswith("QizhiPei/e3fp-"):
            raise ValueError("non-official Hugging Face dataset owner/name")
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError("Hugging Face dataset revision must be a full commit")

    exception = document.get("moleculenet_exception", {})
    if exception.get("is_only_source_family_exception") is not True:
        raise ValueError("MoleculeNet must be the only source-family exception")
    if exception.get("tasks") != EXPECTED_MOLECULENET_TASKS:
        raise ValueError("MoleculeNet task list mismatch")
    if "KPGT must not be cited as the source of HIV" not in exception.get("provenance_boundary", ""):
        raise ValueError("HIV provenance boundary is missing")

    deferred = document.get("deferred_without_3dmolt5_hf_source", [])
    if deferred != [{
        "task_id": "zero_shot_molecule_text_retrieval",
        "status": "deferred_no_official_3dmolt5_hf_dataset",
        "rule": "Do not choose an alternative dataset silently. Promotion requires a separate versioned source and evaluation protocol decision.",
    }]:
        raise ValueError("zero-shot retrieval must remain explicitly deferred")
    admission = document.get("admission", {})
    for key in (
        "revision_argument_required",
        "record_source_manifest_required",
        "split_names_and_counts_must_be_observed",
        "dataset_schema_must_be_observed",
        "legacy_local_source_fallback_forbidden",
    ):
        if admission.get(key) is not True:
            raise ValueError("missing source admission gate: {}".format(key))
    return document


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    document = load_and_validate(args.policy)
    print(json.dumps({
        "status": "pass",
        "policy_id": document["policy_id"],
        "official_hf_dataset_count": len(document["huggingface_datasets"]),
        "moleculenet_exception_task_count": len(document["moleculenet_exception"]["tasks"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
