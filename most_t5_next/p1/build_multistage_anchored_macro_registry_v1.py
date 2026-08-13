#!/usr/bin/env python3
"""Publish the frozen Phase-I/II 18k base plus all ChEBI-train motif registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from most_t5_next.p1.analyze_multistage_anchored_vocab_v1 import (
    _equal_stage_ranking,
    _load_chebi,
    _load_registry,
    _sha256_file,
)


SCHEMA_VERSION = "most-t5-next/multistage-anchored-macro-registry/v1"
REGISTRY_NAME = "macro_registry.phase1_phase2_equal_stage_plus_all_chebi_train.jsonl"
MANIFEST_NAME = "manifest.json"


class MultistageMacroRegistryError(RuntimeError):
    """The frozen registry inputs or deterministic output are inconsistent."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _identity_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _count(source: Mapping[str, tuple[int, int]], pure: str) -> int:
    return source.get(pure, (-1, 0))[1]


def build(args: argparse.Namespace) -> dict[str, object]:
    phase1_path = Path(args.phase1_registry).expanduser().resolve()
    phase2_path = Path(args.phase2_registry).expanduser().resolve()
    chebi_path = Path(args.chebi_train_census).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise MultistageMacroRegistryError("output directory must be absent")
    phase1, _ = _load_registry(phase1_path)
    phase2, _ = _load_registry(phase2_path)
    chebi = _load_chebi(chebi_path)
    ranking = _equal_stage_ranking(phase1, phase2)
    if args.general_base_budget <= 0 or args.general_base_budget > len(ranking):
        raise MultistageMacroRegistryError("general base budget is invalid")
    base = tuple(ranking[: args.general_base_budget])
    base_set = set(base)
    additions = tuple(
        sorted(
            (pure for pure in chebi if pure not in base_set),
            key=lambda pure: (-chebi[pure], pure.encode("utf-8")),
        )
    )
    selected = base + additions
    if len(selected) != len(set(selected)):
        raise MultistageMacroRegistryError("selected registry contains duplicates")

    output.mkdir(parents=True)
    registry_path = output / REGISTRY_NAME
    with registry_path.open("x", encoding="utf-8", newline="\n") as handle:
        for rank, pure in enumerate(selected):
            p1_count = _count(phase1, pure)
            p2_count = _count(phase2, pure)
            row = {
                "rank": rank,
                "pure_motif": pure,
                "pure_motif_sha256": _identity_sha256(pure),
                "surface_token": f"<MOST:MACRO:{rank:06d}>",
                "selection_role": (
                    "phase1_phase2_equal_stage_base"
                    if rank < len(base)
                    else "chebi20_train_all_extension"
                ),
                "phase1_train_occurrences": p1_count,
                "phase2_train_occurrences": p2_count,
                "pretraining_train_occurrences": p1_count + p2_count,
                "chebi20_train_occurrences": chebi.get(pure, 0),
            }
            handle.write(_canonical_json(row) + "\n")

    absent_pretraining = sum(
        _count(phase1, pure) + _count(phase2, pure) == 0 for pure in additions
    )
    singleton_additions = sum(chebi[pure] == 1 for pure in additions)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate",
        "inputs": {
            "phase1_registry": {
                "path": str(phase1_path),
                "sha256": _sha256_file(phase1_path),
            },
            "phase2_registry": {
                "path": str(phase2_path),
                "sha256": _sha256_file(phase2_path),
            },
            "chebi20_train_census": {
                "path": str(chebi_path),
                "sha256": _sha256_file(chebi_path),
            },
        },
        "selection": {
            "general_ranking": "phase1_phase2_equal_stage_mass_then_utf8_identity",
            "general_base_budget": len(base),
            "chebi20_train_extension_policy": "all_absent_from_general_base",
            "chebi20_train_unique_motifs": len(chebi),
            "chebi20_train_additions": len(additions),
            "total_macro_rows": len(selected),
            "extension_singletons": singleton_additions,
            "extension_absent_from_both_pretraining_stages": absent_pretraining,
        },
        "artifacts": {
            REGISTRY_NAME: {
                "bytes": registry_path.stat().st_size,
                "sha256": _sha256_file(registry_path),
                "rows": len(selected),
            }
        },
        "contracts": {
            "registry_is_frozen_before_phase_i": True,
            "phase1_and_phase2_train_select_general_base": True,
            "all_chebi20_train_motifs_are_registered": True,
            "validation_or_test_selects_tokens": False,
            "lossless_chemical_lexer_remains_mandatory": True,
            "cold_extension_rows_require_compositional_initialization_and_exposure": True,
            "training_admission": False,
        },
    }
    manifest_path = output / MANIFEST_NAME
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-registry", required=True)
    parser.add_argument("--phase2-registry", required=True)
    parser.add_argument("--chebi-train-census", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--general-base-budget", type=int, default=18000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        manifest = build(_parser().parse_args(argv))
    except Exception as exc:
        print(
            f"multistage macro registry failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
