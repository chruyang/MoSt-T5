"""Measure locality, coverage, and parameter cost of fragSMILES macro caps.

This is a selection analysis, not a second fragmenter.  Every rejected macro
remains losslessly representable by the pinned molecular glyph vocabulary.
The tool therefore reports the exact trade between shorter macro phrases and
the extra open-vocabulary glyph tokens introduced by each cap.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

from rdkit import Chem

from most_t5_next.r1.tokenizer.smirk_smiles_vocabulary_v1 import (
    encode_smiles_glyphs,
)


SCHEMA_VERSION = "most-t5-next/fragsmiles-macro-locality-analysis/v1"
DEFAULT_HIDDEN_SIZE = 768
_DOMAINS = (
    ("phase1", "phase1_train_occurrences"),
    ("phase2", "phase2_train_occurrences"),
    ("chebi20", "chebi20_train_occurrences"),
    ("uspto50k", "uspto50k_train_reaction_component_occurrences"),
)


class MacroLocalityAnalysisError(RuntimeError):
    """The candidate registry or cap policy is malformed."""


@dataclass(frozen=True)
class MacroShape:
    row: Mapping[str, object]
    atom_count: int
    glyph_count: int
    canonical_fixed_point: bool


@dataclass(frozen=True)
class CapPolicy:
    name: str
    max_atoms: int | None
    max_glyphs: int | None

    def accepts(self, shape: MacroShape) -> bool:
        return (
            (self.max_atoms is None or shape.atom_count <= self.max_atoms)
            and (self.max_glyphs is None or shape.glyph_count <= self.max_glyphs)
        )


DEFAULT_POLICIES = (
    CapPolicy("unbounded", None, None),
    CapPolicy("atoms_32", 32, None),
    CapPolicy("glyphs_64", None, 64),
    CapPolicy("atoms_32_and_glyphs_64", 32, 64),
    CapPolicy("atoms_40_and_glyphs_80", 40, 80),
    CapPolicy("atoms_48_and_glyphs_96", 48, 96),
    CapPolicy("atoms_64_and_glyphs_128", 64, 128),
)


def _positive_int(row: Mapping[str, object], field: str) -> int:
    value = row.get(field, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MacroLocalityAnalysisError(f"invalid non-negative count: {field}")
    return value


def characterize_registry(rows: Iterable[Mapping[str, object]]) -> tuple[MacroShape, ...]:
    result: list[MacroShape] = []
    seen: set[str] = set()
    for expected_rank, row in enumerate(rows):
        identity = row.get("fragment_identity")
        if (
            row.get("rank") != expected_rank
            or not isinstance(identity, str)
            or not identity
            or identity in seen
        ):
            raise MacroLocalityAnalysisError(f"invalid registry row {expected_rank}")
        seen.add(identity)
        mol = Chem.MolFromSmiles(identity)
        if mol is None:
            raise MacroLocalityAnalysisError(f"RDKit rejected registry row {expected_rank}")
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        glyph_count = len(encode_smiles_glyphs(identity).glyphs)
        for _domain, field in _DOMAINS:
            _positive_int(row, field)
        result.append(
            MacroShape(
                row=row,
                atom_count=mol.GetNumAtoms(),
                glyph_count=glyph_count,
                canonical_fixed_point=canonical == identity,
            )
        )
    if not result:
        raise MacroLocalityAnalysisError("registry is empty")
    return tuple(result)


def analyze_policy(
    shapes: tuple[MacroShape, ...], policy: CapPolicy, *, hidden_size: int
) -> dict[str, object]:
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    kept = tuple(shape for shape in shapes if policy.accepts(shape))
    removed = tuple(shape for shape in shapes if not policy.accepts(shape))
    role_kept = Counter(str(shape.row.get("selection_role", "missing")) for shape in kept)
    role_removed = Counter(str(shape.row.get("selection_role", "missing")) for shape in removed)
    domains: dict[str, dict[str, int | float]] = {}
    for domain, field in _DOMAINS:
        total = sum(_positive_int(shape.row, field) for shape in shapes)
        kept_occurrences = sum(_positive_int(shape.row, field) for shape in kept)
        removed_extra_glyph_tokens = sum(
            _positive_int(shape.row, field) * (shape.glyph_count - 1)
            for shape in removed
        )
        domains[domain] = {
            "candidate_occurrences": total,
            "kept_macro_occurrences": kept_occurrences,
            "kept_occurrence_rate": kept_occurrences / total if total else 1.0,
            "removed_macro_occurrences": total - kept_occurrences,
            "fallback_extra_glyph_tokens": removed_extra_glyph_tokens,
        }
    return {
        "name": policy.name,
        "max_atoms": policy.max_atoms,
        "max_glyphs": policy.max_glyphs,
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "kept_by_role": dict(sorted(role_kept.items())),
        "removed_by_role": dict(sorted(role_removed.items())),
        "removed_canonical_non_fixed_point_rows": sum(
            not shape.canonical_fixed_point for shape in removed
        ),
        "embedding_parameters": len(kept) * hidden_size,
        "untied_embedding_plus_lm_head_parameters": len(kept) * hidden_size * 2,
        "parameter_savings_vs_unbounded_tied": len(removed) * hidden_size,
        "domains": domains,
        "largest_removed": [
            {
                "rank": shape.row["rank"],
                "fragment_identity": shape.row["fragment_identity"],
                "selection_role": shape.row.get("selection_role"),
                "atom_count": shape.atom_count,
                "glyph_count": shape.glyph_count,
            }
            for shape in sorted(
                removed,
                key=lambda item: (-item.glyph_count, -item.atom_count, int(item.row["rank"])),
            )[:20]
        ],
    }


def analyze_registry(
    rows: Iterable[Mapping[str, object]],
    *,
    policies: tuple[CapPolicy, ...] = DEFAULT_POLICIES,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
) -> dict[str, object]:
    shapes = characterize_registry(rows)
    if len({policy.name for policy in policies}) != len(policies):
        raise MacroLocalityAnalysisError("cap policy names must be unique")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "training_admission": False,
        "runtime": {
            "rdkit_version": Chem.rdBase.rdkitVersion,
            "hidden_size": hidden_size,
        },
        "candidate": {
            "rows": len(shapes),
            "canonical_fixed_point_rows": sum(
                shape.canonical_fixed_point for shape in shapes
            ),
            "canonical_non_fixed_point_rows": sum(
                not shape.canonical_fixed_point for shape in shapes
            ),
        },
        "policies": [
            analyze_policy(shapes, policy, hidden_size=hidden_size) for policy in policies
        ],
        "interpretation": {
            "removed_macros_remain_lossless": True,
            "fallback_cost_definition": "occurrences * (Smirk glyph count - one macro token)",
            "coverage_scope": "candidate-registry occurrences in each train domain",
            "record_level_coverage_not_claimed": True,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = analyze_registry(_read_jsonl(args.registry), hidden_size=args.hidden_size)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "candidate": report["candidate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
