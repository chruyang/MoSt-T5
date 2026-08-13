#!/usr/bin/env python3
"""Build a train-only pure-motif census for registered downstream vocabulary.

This utility is intentionally separate from model training.  It reads only a
declared downstream *training* split, applies the frozen molecule-native motif
linearizer, removes molecule-local anchor text, and counts stereo-free pure
motifs.  Validation and test splits are not accepted by this CLI.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import datetime as dt
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
from typing import Sequence


SCHEMA_VERSION = "most-t5-next/registered-downstream-pure-motif-census/v1"
CENSUS_NAME = "pure_motif_census.jsonl"
REJECTS_NAME = "rejects.jsonl"
MANIFEST_NAME = "manifest.json"


class RegisteredDownstreamCensusError(RuntimeError):
    """The declared downstream training corpus cannot be projected exactly."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _project_smiles(task: tuple[int, str, str]) -> dict[str, object]:
    row_index, record_id, smiles = task
    try:
        from rdkit import Chem

        from most_t5_next.r1.adapter.mol_linearizer import linearize_mol
        from most_t5_next.r1.tokenizer.stereo_free_anchored_motif_surface_v1 import (
            canonicalize_legacy_fragment,
        )

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or molecule.GetNumAtoms() <= 0:
            raise ValueError("RDKit did not parse a nonempty molecule")
        result = linearize_mol(molecule)
        projected = tuple(
            canonicalize_legacy_fragment(fragment)
            for fragment in result.fragment_sequence
        )
        pure_motifs = tuple(pure for pure, _anchor_ids in projected)
        if not pure_motifs:
            raise ValueError("linearizer produced no motif")
        anchor_counts: Counter[int] = Counter(
            anchor_id
            for _pure, anchor_ids in projected
            for anchor_id in anchor_ids
        )
        if any(count != 2 for count in anchor_counts.values()):
            raise ValueError("legacy anchor IDs are not exact molecule-local pairs")
        return {
            "row_index": row_index,
            "record_id": record_id,
            "canonical_isomeric_smiles": Chem.MolToSmiles(
                molecule, canonical=True, isomericSmiles=True
            ),
            "pure_motifs": pure_motifs,
            "model_facing_anchor_count": len(anchor_counts),
            "error": None,
        }
    except Exception as exc:
        return {
            "row_index": row_index,
            "record_id": record_id,
            "canonical_isomeric_smiles": None,
            "pure_motifs": (),
            "model_facing_anchor_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def build(args: argparse.Namespace) -> dict[str, object]:
    try:
        import pandas as pd
        import rdkit
    except ImportError as exc:
        raise RegisteredDownstreamCensusError("pandas, pyarrow and RDKit are required") from exc

    if args.split != "train":
        raise RegisteredDownstreamCensusError(
            "only an explicitly registered train split may influence vocabulary"
        )
    source = Path(args.source_parquet).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists():
        raise RegisteredDownstreamCensusError("output and sibling staging must be absent")
    if args.workers <= 0:
        raise RegisteredDownstreamCensusError("workers must be positive")
    frame = pd.read_parquet(source, columns=[args.id_column, args.smiles_column])
    tasks = []
    for row_index, row in frame.iterrows():
        record_id = str(row[args.id_column])
        smiles = row[args.smiles_column]
        if not isinstance(smiles, str) or not smiles.strip():
            tasks.append((int(row_index), record_id, ""))
        else:
            tasks.append((int(row_index), record_id, smiles.strip()))
    if not tasks:
        raise RegisteredDownstreamCensusError("downstream training split is empty")

    staging.mkdir(parents=False)
    started = time.perf_counter()
    counts: Counter[str] = Counter()
    molecule_motif_counts = []
    molecule_anchor_counts = []
    reject_count = 0
    rejects_path = staging / REJECTS_NAME
    with rejects_path.open("x", encoding="utf-8", newline="\n") as rejects_handle:
        if args.workers == 1:
            results = map(_project_smiles, tasks)
            pool_context = None
        else:
            context = multiprocessing.get_context("spawn")
            pool_context = concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=context
            )
            results = pool_context.map(
                _project_smiles, tasks, chunksize=args.chunksize
            )
        try:
            for processed, result in enumerate(results, 1):
                error = result["error"]
                if error is not None:
                    reject_count += 1
                    rejects_handle.write(
                        _canonical_json(
                            {
                                "row_index": result["row_index"],
                                "record_id": result["record_id"],
                                "reason": error,
                            }
                        )
                        + "\n"
                    )
                else:
                    pure_motifs = result["pure_motifs"]
                    assert isinstance(pure_motifs, tuple)
                    counts.update(pure_motifs)
                    molecule_motif_counts.append(len(pure_motifs))
                    model_facing_anchor_count = result["model_facing_anchor_count"]
                    assert isinstance(model_facing_anchor_count, int)
                    molecule_anchor_counts.append(model_facing_anchor_count)
                if args.progress_every and processed % args.progress_every == 0:
                    print(f"downstream-motif-census {processed}/{len(tasks)}", flush=True)
        finally:
            if pool_context is not None:
                pool_context.shutdown(wait=True)

    census_path = staging / CENSUS_NAME
    with census_path.open("x", encoding="utf-8", newline="\n") as handle:
        for pure in sorted(counts, key=lambda value: value.encode("utf-8")):
            handle.write(
                _canonical_json(
                    {
                        "pure_motif": pure,
                        "pure_motif_sha256": hashlib.sha256(
                            pure.encode("utf-8")
                        ).hexdigest(),
                        "train_occurrences": counts[pure],
                    }
                )
                + "\n"
            )

    def artifact(path: Path) -> dict[str, object]:
        return {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }

    passed = reject_count == 0 and len(molecule_motif_counts) == len(tasks)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if passed else "failed",
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "dataset": args.dataset_id,
        "split": args.split,
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": _sha256_file(source),
            "id_column": args.id_column,
            "smiles_column": args.smiles_column,
        },
        "runtime": {
            "workers": args.workers,
            "chunksize": args.chunksize,
            "rdkit_version": rdkit.__version__,
            "wall_seconds": time.perf_counter() - started,
        },
        "counts": {
            "scheduled_records": len(tasks),
            "admitted_records": len(molecule_motif_counts),
            "rejected_records": reject_count,
            "unique_pure_motifs": len(counts),
            "pure_motif_occurrences": sum(counts.values()),
            "max_motifs_per_molecule": max(molecule_motif_counts, default=0),
            "max_model_facing_anchor_count_per_molecule": max(
                molecule_anchor_counts, default=0
            ),
            "max_model_facing_anchor_id": max(
                (count - 1 for count in molecule_anchor_counts if count > 0),
                default=-1,
            ),
        },
        "contracts": {
            "validation_or_test_used_for_vocabulary": False,
            "stereochemistry_removed_by_frozen_linearizer": True,
            "molecule_local_anchor_ids_removed_before_counting": True,
            "legacy_source_anchor_pairs_revalidated_before_counting": True,
            "task_aware_specialist_if_merged_into_pretraining_vocabulary": True,
            "training_admission": False,
        },
        "artifacts": {
            CENSUS_NAME: artifact(census_path),
            REJECTS_NAME: artifact(rejects_path),
        },
    }
    with (staging / MANIFEST_NAME).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    if not passed:
        raise RegisteredDownstreamCensusError(
            f"downstream projection rejected {reject_count} record(s); staging retained"
        )
    staging.rename(output)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--id-column", default="cid")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=2048)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        manifest = build(_parser().parse_args(argv))
    except Exception as exc:
        print(f"downstream motif census failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
