"""Append every Mol-Instructions train-target motif to a macro registry.

The three official tasks remain independent fine-tuning datasets.  Only their
canonical motif identity sets are unioned for tokenizer storage; no train row
is deduplicated and validation/test data are never read.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


SCHEMA_VERSION = "most-t5-next/fragsmiles-mol-instructions-extension/v1"
TASKS = ("reagent", "forward", "retro")


def _read(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def extend_registry(
    *,
    base_rows: Iterable[Mapping[str, object]],
    task_rows: Mapping[str, Iterable[Mapping[str, object]]],
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    if set(task_rows) != set(TASKS):
        raise ValueError("all three Mol-Instructions train task censuses are required")
    output = []
    identities: set[str] = set()
    for expected_rank, raw in enumerate(base_rows):
        row = dict(raw)
        identity = row.get("fragment_identity")
        if row.get("rank") != expected_rank or not isinstance(identity, str) or identity in identities:
            raise ValueError("base registry is not dense and unique")
        identities.add(identity)
        output.append(row)

    task_counts: dict[str, Counter[str]] = {}
    for task in TASKS:
        counts: Counter[str] = Counter()
        for expected_rank, row in enumerate(task_rows[task]):
            identity = row.get("fragment_identity")
            occurrences = row.get("occurrences")
            if (
                row.get("rank") != expected_rank
                or not isinstance(identity, str)
                or not identity
                or isinstance(occurrences, bool)
                or not isinstance(occurrences, int)
                or occurrences <= 0
                or identity in counts
            ):
                raise ValueError(f"invalid {task} train census")
            counts[identity] = occurrences
        task_counts[task] = counts

    new_identities = set().union(*(set(counts) for counts in task_counts.values())) - identities
    ordered_new = sorted(
        new_identities,
        key=lambda identity: (
            -sum(task_counts[task].get(identity, 0) for task in TASKS),
            identity.encode("utf-8"),
        ),
    )
    for identity in ordered_new:
        rank = len(output)
        output.append(
            {
                "rank": rank,
                "surface_token": f"<MOST:FM:{rank:06d}>",
                "fragment_identity": identity,
                "fragment_smiles": identity,
                "fragment_identity_sha256": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                "selection_role": "mol_instructions_train_target_extension",
                "mol_instructions_reagent_train_occurrences": task_counts["reagent"].get(identity, 0),
                "mol_instructions_forward_train_occurrences": task_counts["forward"].get(identity, 0),
                "mol_instructions_retro_train_occurrences": task_counts["retro"].get(identity, 0),
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate",
        "training_admission": False,
        "policy": {
            "all_train_target_motifs_admitted": True,
            "atom_or_glyph_locality_filter": False,
            "frequency_filter": False,
            "validation_or_test_used": False,
            "dataset_rows_deduplicated": False,
            "shared_identity_union_is_tokenizer_storage_only": True,
        },
        "counts": {
            "base_macros": len(identities),
            "new_union_macros": len(ordered_new),
            "final_macros": len(output),
            "task_unique_motifs": {task: len(task_counts[task]) for task in TASKS},
            "task_new_vs_base": {
                task: len(set(task_counts[task]) - identities) for task in TASKS
            },
        },
    }
    return tuple(output), report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-registry", required=True, type=Path)
    parser.add_argument("--reagent-census", required=True, type=Path)
    parser.add_argument("--forward-census", required=True, type=Path)
    parser.add_argument("--retro-census", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows, report = extend_registry(
        base_rows=_read(args.base_registry),
        task_rows={
            "reagent": _read(args.reagent_census),
            "forward": _read(args.forward_census),
            "retro": _read(args.retro_census),
        },
    )
    args.output_dir.mkdir(parents=True)
    registry = args.output_dir / "macro_registry.jsonl"
    registry.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    report["artifact"] = {
        "path": registry.name,
        "bytes": registry.stat().st_size,
        "sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "counts": report["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
