"""Freeze a molecule-disjoint QM9 mechanism-probe subset.

The public 3D-MolT5 parquet repeats molecules across instruction paraphrases,
and its published validation shard overlaps the training shard by molecule.
This freezer therefore uses the training parquet as the source table, groups
by the exact molecular SMILES identity, and emits only source-row references.
E3FP tensors remain in the source parquet and are never duplicated here.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "most-t5-p2/qm9-3dmolt5-probe-subset/v1"
DEFAULT_SEED = 20260810
DEFAULT_SPLIT_COUNTS = {"train": 8000, "dev": 1000, "test": 1000}
PROPERTY_ORDER = {"homo": 0, "lumo": 1, "gap": 2}
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\.?$")


class QM9ProbeSubsetError(ValueError):
    """The source rows cannot define the frozen probe subset."""


def classify_property(instruction: str) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise QM9ProbeSubsetError("QM9 instruction must be non-empty text")
    text = instruction.lower()
    if "gap" in text or "separation" in text or "difference" in text:
        return "gap"
    if "lumo" in text or "lowest unoccupied" in text:
        return "lumo"
    if "homo" in text or "highest occupied" in text:
        return "homo"
    raise QM9ProbeSubsetError("unrecognized QM9 property instruction")


def parse_target(value: str) -> float:
    if not isinstance(value, str) or _FLOAT_RE.fullmatch(value.strip()) is None:
        raise QM9ProbeSubsetError("QM9 target is not one scalar")
    return float(value.strip().rstrip("."))


def state_sha256(selfies: str, molecule_fp: Sequence[Sequence[int]]) -> str:
    if not isinstance(selfies, str) or not selfies:
        raise QM9ProbeSubsetError("QM9 SELFIES must be non-empty")
    normalized: list[list[int]] = []
    for row in molecule_fp:
        if len(row) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in row):
            raise QM9ProbeSubsetError("QM9 molecule_fp must be integer [symbols,4]")
        normalized.append(list(row))
    payload = selfies + "\0" + json.dumps(normalized, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    split_counts: Mapping[str, int] = DEFAULT_SPLIT_COUNTS,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Freeze row references; duplicate instruction paraphrases collapse."""

    if tuple(split_counts) != ("train", "dev", "test") or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in split_counts.values()
    ):
        raise QM9ProbeSubsetError("split counts must be positive train/dev/test integers")
    materialized = list(rows)
    smiles_values = sorted({str(row["smiles"]) for row in materialized})
    required = sum(split_counts.values())
    if len(smiles_values) < required:
        raise QM9ProbeSubsetError("source has fewer molecular identities than requested")
    ordered = sorted(
        smiles_values,
        key=lambda smiles: (
            hashlib.sha256(f"{seed}\0{smiles}".encode()).digest(),
            smiles.encode("utf-8"),
        ),
    )[:required]
    split_by_smiles: dict[str, str] = {}
    cursor = 0
    for split, count in split_counts.items():
        for smiles in ordered[cursor : cursor + count]:
            split_by_smiles[smiles] = split
        cursor += count

    grouped: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for row_index, row in enumerate(materialized):
        smiles = str(row["smiles"])
        if smiles not in split_by_smiles:
            continue
        fp = row["molecule_fp"]
        if not isinstance(fp, Sequence):
            raise QM9ProbeSubsetError("QM9 molecule_fp must be a sequence")
        state = state_sha256(str(row["selfies"]), fp)
        prop = classify_property(str(row["instruction"]))
        grouped[(smiles, state, prop)].append((row_index, parse_target(str(row["output"]))))

    rejected_conflicts = 0
    membership: list[dict[str, object]] = []
    order_index = {smiles: index for index, smiles in enumerate(ordered)}
    for (smiles, state, prop), observations in grouped.items():
        targets = {target for _, target in observations}
        if len(targets) != 1:
            rejected_conflicts += 1
            continue
        membership.append({
            "split": split_by_smiles[smiles],
            "molecule_order_index": order_index[smiles],
            "source_row_index": min(row for row, _ in observations),
            "smiles": smiles,
            "state_sha256": state,
            "property": prop,
            "target_hartree": next(iter(targets)),
            "instruction_paraphrase_count": len(observations),
        })
    membership.sort(key=lambda row: (
        int(row["molecule_order_index"]),
        str(row["state_sha256"]),
        PROPERTY_ORDER[str(row["property"])],
    ))
    counts = {
        split: sum(row["split"] == split for row in membership)
        for split in split_counts
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "seed": seed,
        "source_policy": "training_parquet_only; published validation excluded for molecule overlap",
        "split_unit": "exact_smiles_molecular_identity",
        "requested_molecule_counts": dict(split_counts),
        "selected_molecule_count": required,
        "selected_state_property_rows": len(membership),
        "state_property_rows_by_split": counts,
        "conflicting_exact_state_property_groups_rejected": rejected_conflicts,
        "instruction_paraphrases_are_not_independent_samples": True,
        "target_unit": "hartree_as_published",
    }
    return membership, manifest


def freeze_parquet(
    *, source_parquet: Path, output_dir: Path, split_counts: Mapping[str, int], seed: int
) -> dict[str, object]:
    import pyarrow.parquet as pq

    if output_dir.exists():
        raise QM9ProbeSubsetError("QM9 probe output already exists")
    table = pq.read_table(
        source_parquet,
        columns=["instruction", "output", "molecule_fp", "selfies", "smiles"],
    )
    rows = (
        {name: table[name][index].as_py() for name in table.column_names}
        for index in range(table.num_rows)
    )
    membership, manifest = freeze_rows(rows, split_counts=split_counts, seed=seed)
    output_dir.mkdir(parents=True)
    with (output_dir / "membership.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in membership:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        **manifest,
        "source_parquet": str(source_parquet.resolve()),
        "source_row_count": table.num_rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-molecules", type=int, default=8000)
    parser.add_argument("--dev-molecules", type=int, default=1000)
    parser.add_argument("--test-molecules", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = freeze_parquet(
        source_parquet=args.source_parquet,
        output_dir=args.output_dir,
        split_counts={
            "train": args.train_molecules,
            "dev": args.dev_molecules,
            "test": args.test_molecules,
        },
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "rows": report["selected_state_property_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_SPLIT_COUNTS",
    "QM9ProbeSubsetError",
    "SCHEMA_VERSION",
    "classify_property",
    "freeze_parquet",
    "freeze_rows",
    "parse_target",
    "state_sha256",
]
