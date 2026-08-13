"""Freeze a dense fragSMILES macro candidate after the locality/cost gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rdkit import Chem

from most_t5_next.p1.analyze_fragsmiles_macro_locality_v1 import (
    CapPolicy,
    analyze_policy,
    characterize_registry,
)
from most_t5_next.p1.build_fragsmiles_macro_registry_v1 import _coverage


SCHEMA_VERSION = "most-t5-next/fragsmiles-macro-locality-freeze/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _census_counts(root: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with (root / "fragment_census.jsonl").open("r", encoding="utf-8") as handle:
        for expected_rank, line in enumerate(handle):
            row = json.loads(line)
            identity = row.get("fragment_identity")
            count = row.get("occurrences")
            if (
                row.get("rank") != expected_rank
                or not isinstance(identity, str)
                or identity in result
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
            ):
                raise ValueError(f"invalid census row {expected_rank}: {root}")
            result[identity] = count
    return result


def freeze_registry(
    *,
    candidate_registry: Path,
    phase1_census: Path,
    phase2_census: Path,
    chebi20_census: Path,
    output_dir: Path,
    max_atoms: int = 32,
    max_glyphs: int = 64,
    hidden_size: int = 768,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if max_atoms <= 0 or max_glyphs <= 0:
        raise ValueError("locality caps must be positive")
    rows = _read_jsonl(candidate_registry)
    shapes = characterize_registry(rows)
    policy = CapPolicy("atoms_32_and_glyphs_64", max_atoms, max_glyphs)
    kept = tuple(shape for shape in shapes if policy.accepts(shape))
    removed = tuple(shape for shape in shapes if not policy.accepts(shape))
    if not kept or not removed:
        raise ValueError("locality gate must retain and remove at least one candidate")
    if any(not shape.canonical_fixed_point for shape in shapes):
        raise ValueError("candidate registry is not canonical under the freeze runtime")

    output_dir.mkdir(parents=True)
    registry_path = output_dir / "macro_registry.jsonl"
    removed_path = output_dir / "removed_macros.jsonl"
    selected: set[str] = set()
    with registry_path.open("w", encoding="utf-8", newline="\n") as handle:
        for rank, shape in enumerate(kept):
            row = dict(shape.row)
            identity = str(row["fragment_identity"])
            selected.add(identity)
            row["candidate_rank"] = row["rank"]
            row["rank"] = rank
            row["surface_token"] = f"<MOST:FM:{rank:06d}>"
            row["locality_atom_count"] = shape.atom_count
            row["locality_glyph_count"] = shape.glyph_count
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    with removed_path.open("w", encoding="utf-8", newline="\n") as handle:
        for shape in removed:
            handle.write(
                json.dumps(
                    {
                        "candidate_rank": shape.row["rank"],
                        "fragment_identity": shape.row["fragment_identity"],
                        "selection_role": shape.row.get("selection_role"),
                        "atom_count": shape.atom_count,
                        "glyph_count": shape.glyph_count,
                        "reason": "exceeds_atom_or_glyph_locality_cap",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    coverage = {}
    for name, root in (
        ("phase1_train", phase1_census),
        ("phase2_train", phase2_census),
        ("chebi20_train", chebi20_census),
    ):
        coverage[name] = _coverage(
            root / "molecule_fragments.jsonl.gz", selected, _census_counts(root)
        )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate",
        "training_admission": False,
        "policy": {
            "name": policy.name,
            "max_atoms_inclusive": max_atoms,
            "max_smirk_glyphs_inclusive": max_glyphs,
            "removed_macros_use_lossless_smirk_phrase": True,
            "ranking_preserves_candidate_order_then_becomes_dense": True,
            "validation_or_test_used": False,
        },
        "runtime": {"rdkit_version": Chem.rdBase.rdkitVersion, "hidden_size": hidden_size},
        "counts": {
            "candidate_macros": len(shapes),
            "retained_macros": len(kept),
            "removed_macros": len(removed),
        },
        "weighted_analysis": analyze_policy(shapes, policy, hidden_size=hidden_size),
        "record_level_train_coverage": coverage,
        "inputs": {
            "candidate_registry": {
                "path": str(candidate_registry.resolve()),
                "bytes": candidate_registry.stat().st_size,
                "sha256": _sha256(candidate_registry),
            }
        },
        "artifacts": {},
    }
    manifest["artifacts"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in (registry_path, removed_path)
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-registry", required=True, type=Path)
    parser.add_argument("--phase1-census", required=True, type=Path)
    parser.add_argument("--phase2-census", required=True, type=Path)
    parser.add_argument("--chebi20-census", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-atoms", default=32, type=int)
    parser.add_argument("--max-glyphs", default=64, type=int)
    args = parser.parse_args(argv)
    manifest = freeze_registry(
        candidate_registry=args.candidate_registry,
        phase1_census=args.phase1_census,
        phase2_census=args.phase2_census,
        chebi20_census=args.chebi20_census,
        output_dir=args.output_dir,
        max_atoms=args.max_atoms,
        max_glyphs=args.max_glyphs,
    )
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
