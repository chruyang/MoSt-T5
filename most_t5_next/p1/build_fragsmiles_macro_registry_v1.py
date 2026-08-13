"""Freeze the joint Phase-I/Phase-II fragSMILES macro registry.

The general rows are ranked by equal stage mass without floating-point
arithmetic.  ChEBI-20 train may append specialist rows, but validation and test
data are never accepted by this builder.  Fragment macro coverage is reported
separately from the always-lossless Smirk-glyph fragment phrases and
whole-molecule fallback routing.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Mapping

from most_t5_next.p1.fragsmiles_compact_stereo_codec_v1 import (
    compact_stereo_token_universe,
)
from most_t5_next.p1.fragsmiles_lossless_fallback_v1 import fallback_token_universe
from most_t5_next.p1.fragsmiles_macro_fallback_surface_v1 import fixed_surface_tokens
from most_t5_next.r1.tokenizer.smirk_smiles_vocabulary_v1 import (
    SCHEMA_VERSION as SMIRK_VOCABULARY_SCHEMA,
    UPSTREAM_DISTRIBUTION_VERSION,
    UPSTREAM_PROJECT,
    UPSTREAM_SOURCE_COMMIT,
    smiles_added_token_universe,
    smiles_glyph_token_map,
)


SCHEMA_VERSION = "most-t5-next/fragsmiles-macro-registry/v1"
BASE_T5_VOCAB_SIZE = 32100
NON_MACRO_ADDITIONS = 166
SHARED_MOLECULE_BOUNDARY_TOKENS = ("<bom>", "<eom>")


class FragSmilesMacroRegistryError(RuntimeError):
    """A census or registry violates the frozen vocabulary policy."""


def non_macro_token_universe() -> tuple[str, ...]:
    tokens = (
        smiles_added_token_universe()
        + fixed_surface_tokens()
        + compact_stereo_token_universe()
    )
    # The whole-molecule fallback and fragment macro misses deliberately share
    # the same molecular glyph rows; no byte or fallback-class namespace exists.
    if fallback_token_universe() != tuple(
        surface for _glyph, surface in smiles_glyph_token_map()
    ):
        raise FragSmilesMacroRegistryError("fallback SMILES vocabulary drifted")
    if len(tokens) != NON_MACRO_ADDITIONS or len(set(tokens)) != len(tokens):
        raise FragSmilesMacroRegistryError("non-macro token universe drifted")
    return tokens


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_census(root: Path) -> tuple[dict[str, int], dict[str, object]]:
    manifest_path = root / "manifest.json"
    census_path = root / "fragment_census.jsonl"
    cache_path = root / "molecule_fragments.jsonl.gz"
    if not all(path.is_file() for path in (manifest_path, census_path, cache_path)):
        raise FragSmilesMacroRegistryError(f"census artifacts are incomplete: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != "most-t5-next/fragsmiles-fragment-census/v1"
        or manifest.get("status") != "pass"
        or manifest.get("training_admission") is not False
    ):
        raise FragSmilesMacroRegistryError(f"census manifest is not admissible: {root}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise FragSmilesMacroRegistryError(f"census artifact lock is absent: {root}")
    for path in (census_path, cache_path):
        lock = artifacts.get(path.name)
        if (
            not isinstance(lock, dict)
            or lock.get("bytes") != path.stat().st_size
            or lock.get("sha256") != _sha256_file(path)
        ):
            raise FragSmilesMacroRegistryError(
                f"census artifact differs from manifest: {path}"
            )
    counts: dict[str, int] = {}
    with census_path.open("r", encoding="utf-8") as handle:
        for expected_rank, line in enumerate(handle):
            row = json.loads(line)
            identity = row.get("fragment_identity")
            count = row.get("occurrences")
            if (
                row.get("rank") != expected_rank
                or not isinstance(identity, str)
                or not identity
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                or identity in counts
            ):
                raise FragSmilesMacroRegistryError(
                    f"invalid fragment census row {expected_rank}: {root}"
                )
            counts[identity] = count
    if not counts or sum(counts.values()) != manifest["counts"]["fragment_occurrences"]:
        raise FragSmilesMacroRegistryError(f"census occurrence total differs: {root}")
    return counts, manifest


def equal_stage_ranking(
    phase1: Mapping[str, int], phase2: Mapping[str, int]
) -> tuple[str, ...]:
    """Rank by count/P1_total + count/P2_total using exact integer products."""

    total1 = sum(phase1.values())
    total2 = sum(phase2.values())
    if total1 <= 0 or total2 <= 0:
        raise FragSmilesMacroRegistryError("pretraining stage census is empty")
    return tuple(
        sorted(
            set(phase1) | set(phase2),
            key=lambda identity: (
                -(phase1.get(identity, 0) * total2 + phase2.get(identity, 0) * total1),
                identity.encode("utf-8"),
            ),
        )
    )


def _coverage(
    cache_path: Path,
    selected: set[str],
    expected_occurrences: Mapping[str, int],
) -> dict[str, object]:
    counts = Counter()
    replayed_occurrences: Counter[str] = Counter()
    legacy_eligibility_rows = 0
    with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
        for expected_index, line in enumerate(handle):
            row = json.loads(line)
            if row.get("selection_index") != expected_index:
                raise FragSmilesMacroRegistryError(
                    f"molecule cache order drifted: {cache_path}"
                )
            mode = row.get("mode")
            identities = row.get("fragment_identities")
            eligible = row.get("fragment_macro_eligible")
            if eligible is None:
                eligible = [True] * len(identities) if isinstance(identities, list) else None
                legacy_eligibility_rows += 1
            if mode not in ("compact", "whole_molecule_fallback") or not isinstance(
                identities, list
            ) or not isinstance(eligible, list) or len(eligible) != len(identities):
                raise FragSmilesMacroRegistryError(f"invalid molecule cache row: {cache_path}")
            counts["records"] += 1
            counts[f"mode:{mode}"] += 1
            if mode == "whole_molecule_fallback":
                if identities:
                    raise FragSmilesMacroRegistryError(
                        "whole-molecule fallback unexpectedly carries fragment identities"
                    )
                continue
            misses = sum(
                (not admitted) or identity not in selected
                for identity, admitted in zip(identities, eligible)
            )
            counts["fragment_occurrences"] += len(identities)
            replayed_occurrences.update(
                identity for identity, admitted in zip(identities, eligible) if admitted
            )
            counts["macro_occurrences"] += len(identities) - misses
            counts["semantic_fragment_fallback_occurrences"] += misses
            counts["fully_macro_records"] += int(misses == 0)
            for threshold in (1, 2, 5):
                counts[f"fallback_le_{threshold}"] += int(misses <= threshold)
    if counts["records"] <= 0 or counts["fragment_occurrences"] <= 0:
        raise FragSmilesMacroRegistryError(f"molecule cache is empty: {cache_path}")
    if replayed_occurrences != Counter(expected_occurrences):
        raise FragSmilesMacroRegistryError(
            f"molecule cache fragment counts differ from registry: {cache_path}"
        )
    compact = counts["mode:compact"]
    return {
        "records": counts["records"],
        "compact_records": compact,
        "whole_molecule_fallback_records": counts["mode:whole_molecule_fallback"],
        "legacy_rows_without_explicit_fragment_eligibility": legacy_eligibility_rows,
        "fragment_occurrences": counts["fragment_occurrences"],
        "macro_occurrences": counts["macro_occurrences"],
        "semantic_fragment_fallback_occurrences": counts[
            "semantic_fragment_fallback_occurrences"
        ],
        "macro_occurrence_coverage": counts["macro_occurrences"]
        / counts["fragment_occurrences"],
        "fully_macro_tokenized_rate_among_compact": counts["fully_macro_records"]
        / compact,
        "fully_macro_tokenized_rate_overall": counts["fully_macro_records"]
        / counts["records"],
        "compact_records_with_at_most_1_semantic_fallback_rate": counts[
            "fallback_le_1"
        ]
        / compact,
        "compact_records_with_at_most_2_semantic_fallback_rate": counts[
            "fallback_le_2"
        ]
        / compact,
        "compact_records_with_at_most_5_semantic_fallback_rate": counts[
            "fallback_le_5"
        ]
        / compact,
    }


def build_registry(
    *,
    phase1_census: Path,
    phase2_census: Path,
    chebi_train_census: Path,
    output_dir: Path,
    general_budget: int = 18000,
) -> dict[str, object]:
    if general_budget <= 0:
        raise ValueError("general_budget must be positive")
    non_macro_token_universe()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    phase1, p1_manifest = _load_census(phase1_census)
    phase2, p2_manifest = _load_census(phase2_census)
    chebi, chebi_manifest = _load_census(chebi_train_census)
    ranking = equal_stage_ranking(phase1, phase2)
    if len(ranking) < general_budget:
        raise FragSmilesMacroRegistryError("general budget exceeds observed identity domain")
    general = ranking[:general_budget]
    general_set = set(general)
    chebi_additions = tuple(
        sorted(
            (identity for identity in chebi if identity not in general_set),
            key=lambda identity: (-chebi[identity], identity.encode("utf-8")),
        )
    )
    selected = general + chebi_additions
    selected_set = set(selected)
    if len(selected_set) != len(selected) or not set(chebi).issubset(selected_set):
        raise FragSmilesMacroRegistryError("selected registry is not a strict union")
    output_dir.mkdir(parents=True)
    registry_path = output_dir / "macro_registry.jsonl"
    with registry_path.open("w", encoding="utf-8", newline="\n") as handle:
        for rank, identity in enumerate(selected):
            handle.write(
                json.dumps(
                    {
                        "rank": rank,
                        "surface_token": f"<MOST:FM:{rank:06d}>",
                        "fragment_identity": identity,
                        "fragment_smiles": identity,
                        "fragment_identity_sha256": hashlib.sha256(
                            identity.encode("utf-8")
                        ).hexdigest(),
                        "selection_role": (
                            "phase1_phase2_equal_stage_base"
                            if rank < general_budget
                            else "chebi20_train_extension"
                        ),
                        "phase1_train_occurrences": phase1.get(identity, 0),
                        "phase2_train_occurrences": phase2.get(identity, 0),
                        "chebi20_train_occurrences": chebi.get(identity, 0),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
    coverage = {
        "phase1_train": _coverage(
            phase1_census / "molecule_fragments.jsonl.gz", selected_set, phase1
        ),
        "phase2_train": _coverage(
            phase2_census / "molecule_fragments.jsonl.gz", selected_set, phase2
        ),
        "chebi20_train": _coverage(
            chebi_train_census / "molecule_fragments.jsonl.gz", selected_set, chebi
        ),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate",
        "training_admission": False,
        "policy": {
            "general_ranking": "phase1_phase2_equal_stage_mass_then_utf8_identity",
            "general_budget": general_budget,
            "chebi20_extension": "all_train_identities_absent_from_general_base",
            "validation_or_test_used_for_vocabulary": False,
            "no_tokens_added_after_pretraining_begins": True,
        },
        "counts": {
            "general_macros": general_budget,
            "chebi20_train_extension_macros": len(chebi_additions),
            "total_macros": len(selected),
            "non_macro_additions": NON_MACRO_ADDITIONS,
            "base_t5_vocabulary": BASE_T5_VOCAB_SIZE,
            "projected_union_vocabulary": BASE_T5_VOCAB_SIZE
            + NON_MACRO_ADDITIONS
            + len(SHARED_MOLECULE_BOUNDARY_TOKENS)
            + len(selected),
            "shared_molecule_boundary_additions": len(
                SHARED_MOLECULE_BOUNDARY_TOKENS
            ),
        },
        "smiles_vocabulary": {
            "schema_version": SMIRK_VOCABULARY_SCHEMA,
            "upstream_project": UPSTREAM_PROJECT,
            "upstream_distribution_version": UPSTREAM_DISTRIBUTION_VERSION,
            "upstream_source_commit": UPSTREAM_SOURCE_COMMIT,
            "core_glyphs_excluding_upstream_unk": 158,
            "rdkit_2024_03_5_extensions": ["si", "te"],
            "total_molecular_glyphs": 160,
            "new_molecular_glyph_rows": 150,
            "base_t5_digit_rows_reused": 10,
            "whole_molecule_fallback_shares_glyph_rows": True,
            "utf8_byte_token_rows": 0,
            "fallback_class_token_rows": 0,
            "natural_text_segmentation_uses_base_t5_sentencepiece": True,
            "molecular_segmentation_is_bounded_by_bom_eom": True,
            "shared_molecule_boundary_tokens": list(
                SHARED_MOLECULE_BOUNDARY_TOKENS
            ),
        },
        "coverage": coverage,
        "inputs": {
            name: {
                "path": str(root.resolve()),
                "manifest_sha256": _sha256_file(root / "manifest.json"),
                "fragment_census_sha256": _sha256_file(root / "fragment_census.jsonl"),
                "molecule_cache_sha256": _sha256_file(
                    root / "molecule_fragments.jsonl.gz"
                ),
                "processed_records": manifest["counts"]["processed_records"],
            }
            for name, root, manifest in (
                ("phase1_train", phase1_census, p1_manifest),
                ("phase2_train", phase2_census, p2_manifest),
                ("chebi20_train", chebi_train_census, chebi_manifest),
            )
        },
        "artifacts": {
            "macro_registry.jsonl": {
                "bytes": registry_path.stat().st_size,
                "sha256": _sha256_file(registry_path),
            }
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-census", required=True)
    parser.add_argument("--phase2-census", required=True)
    parser.add_argument("--chebi-train-census", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--general-budget", type=int, default=18000)
    args = parser.parse_args(argv)
    manifest = build_registry(
        phase1_census=Path(args.phase1_census),
        phase2_census=Path(args.phase2_census),
        chebi_train_census=Path(args.chebi_train_census),
        output_dir=Path(args.output_dir),
        general_budget=args.general_budget,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASE_T5_VOCAB_SIZE",
    "FragSmilesMacroRegistryError",
    "NON_MACRO_ADDITIONS",
    "SCHEMA_VERSION",
    "SHARED_MOLECULE_BOUNDARY_TOKENS",
    "build_registry",
    "equal_stage_ranking",
    "non_macro_token_universe",
]
