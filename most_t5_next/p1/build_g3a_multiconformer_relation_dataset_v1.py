#!/usr/bin/env python3
"""Build the frozen same-identity multiconformer relation dataset for G3a."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Mapping, Sequence

from most_t5_next.p1.run_g1_multiconformer_sensitivity_v1 import _generate_candidate


SCHEMA_VERSION = "most-t5-p1/g3a-multiconformer-relation-dataset/v1"
DEFAULT_SEED = 20260808


class G3ARelationDatasetError(ValueError):
    pass


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise G3ARelationDatasetError(
                    "{}:{} is not an object".format(path, line_number)
                )
            yield value


def _split_members(
    rows: Sequence[Mapping[str, Any]], *, train_fraction: float, seed: int
) -> Dict[str, str]:
    if not 0.0 < float(train_fraction) < 1.0:
        raise G3ARelationDatasetError("train_fraction must be in (0,1)")
    identities = [str(row["member_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise G3ARelationDatasetError("member identities are not unique")
    shuffled = list(identities)
    random.Random(int(seed)).shuffle(shuffled)
    train_count = int(round(len(shuffled) * float(train_fraction)))
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    train = set(shuffled[:train_count])
    return {identity: ("train" if identity in train else "dev") for identity in identities}


def _persisted_record(row: Mapping[str, Any], split: str) -> Dict[str, Any]:
    pairs = [
        {
            "left": int(pair["left"]),
            "right": int(pair["right"]),
            "aligned_rmsd_angstrom": float(pair["rmsd"]),
            "distance_matrix_rms_angstrom": float(pair["distance_matrix_rms"]),
        }
        for pair in row["rmsds"]
    ]
    return {
        "member_id": str(row["member_id"]),
        "selection_index": int(row["selection_index"]),
        "split": str(split),
        "e3fp_ids": row["e3fp_ids"],
        "atom_is_attachment": list(row["atom_is_attachment"]),
        "atom_to_motif": list(row["atom_to_motif"]),
        "motif_count": int(row["motif_count"]),
        "motif_edges": [list(edge) for edge in row["motif_edges"]],
        "atomic_numbers": list(row["atomic_numbers"]),
        "conformer_positions": row["conformer_positions"],
        "pairs": pairs,
    }


def run(args) -> Dict[str, Any]:
    source = Path(args.c0_molecules).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise G3ARelationDatasetError("output already exists: {}".format(output))
    candidate_limit = int(args.target_molecules) + int(args.replay_spares)
    candidates = []
    for row in _iter_jsonl(source):
        if row.get("status") == "pass":
            candidates.append(row)
        if len(candidates) >= candidate_limit:
            break
    if len(candidates) < int(args.target_molecules):
        raise G3ARelationDatasetError("C0 artifact has too few candidates")
    tasks = [
        {
            "candidate": candidate,
            "e3fp_source": str(Path(args.e3fp_source).expanduser().resolve()),
            "seed": int(args.seed),
            "requested_conformers": int(args.requested_conformers),
            "conformers_per_molecule": int(args.conformers_per_molecule),
            "prune_rms_threshold": float(args.prune_rms_threshold),
        }
        for candidate in candidates
    ]
    if int(args.workers) == 1:
        generated = [_generate_candidate(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            generated = list(pool.map(_generate_candidate, tasks, chunksize=1))
    accepted = [row for row in generated if row["status"] == "pass"]
    rejects = [row for row in generated if row["status"] != "pass"]
    if len(accepted) < int(args.target_molecules):
        raise G3ARelationDatasetError(
            "only {} accepted for target {}".format(
                len(accepted), int(args.target_molecules)
            )
        )
    accepted = accepted[: int(args.target_molecules)]
    splits = _split_members(
        accepted, train_fraction=float(args.train_fraction), seed=int(args.seed)
    )
    records = [
        _persisted_record(row, splits[str(row["member_id"])]) for row in accepted
    ]
    output.mkdir(parents=True)
    records_path = output / "records.jsonl"
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    with (output / "rejects.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rejects:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    pair_count = sum(len(record["pairs"]) for record in records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "pf1_train_identity_only_g3a_relation_screen",
        "source": str(source),
        "configuration": {
            "seed": int(args.seed),
            "target_molecules": int(args.target_molecules),
            "train_fraction": float(args.train_fraction),
            "conformers_per_molecule": int(args.conformers_per_molecule),
            "requested_conformers": int(args.requested_conformers),
            "prune_rms_threshold": float(args.prune_rms_threshold),
            "workers": int(args.workers),
            "geometry_target": "heavy_atom_pair_distance_matrix_rms_angstrom",
            "e3fp_semantics": "duplicate_pointer_inheritance_v1",
        },
        "result": {
            "molecules": len(records),
            "train_molecules": sum(record["split"] == "train" for record in records),
            "dev_molecules": sum(record["split"] == "dev" for record in records),
            "conformer_pairs": pair_count,
            "generation_rejects": len(rejects),
        },
        "artifacts": {"records": "records.jsonl", "rejects": "rejects.jsonl"},
        "not_a_t5_or_downstream_result": True,
    }
    with (output / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-molecules", required=True)
    parser.add_argument("--e3fp-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-molecules", type=int, default=1000)
    parser.add_argument("--replay-spares", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--conformers-per-molecule", type=int, default=4)
    parser.add_argument("--requested-conformers", type=int, default=8)
    parser.add_argument("--prune-rms-threshold", type=float, default=0.35)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv=None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
