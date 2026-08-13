#!/usr/bin/env python3
"""Measure frozen G1b motif-state sensitivity across same-2D conformers."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from most_t5_next.p1.audit_c0_multiconformer_e3fp_v1 import (
    _aligned_rmsd_same_atom_rows,
    _load_runtime,
    _one_conformer_mol,
    _rigid_copy,
)
from most_t5_next.p1.level_aware_motif_state_v1 import LevelAwareMotifStateEncoder
from most_t5_next.p1.motif_state_data_v1 import collate_motif_state_records


SCHEMA_VERSION = "most-t5-p1/g1-multiconformer-sensitivity/v1"


class G1MulticonformerError(ValueError):
    pass


def _attachment_roles(mol, groups: Sequence[Sequence[int]]) -> Tuple[bool, ...]:
    atom_to_group = {}
    for group_index, group in enumerate(groups):
        for atom_index in group:
            if int(atom_index) in atom_to_group:
                raise G1MulticonformerError("motif groups overlap")
            atom_to_group[int(atom_index)] = int(group_index)
    if set(atom_to_group) != set(range(mol.GetNumAtoms())):
        raise G1MulticonformerError("motif groups do not cover the atom domain")
    roles = [False] * mol.GetNumAtoms()
    for bond in mol.GetBonds():
        left = int(bond.GetBeginAtomIdx())
        right = int(bond.GetEndAtomIdx())
        if atom_to_group[left] != atom_to_group[right]:
            roles[left] = True
            roles[right] = True
    return tuple(roles)


def _motif_edges(mol, atom_to_group: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    """Return the undirected motif graph induced by cross-motif bonds."""

    edges = set()
    for bond in mol.GetBonds():
        left = int(atom_to_group[int(bond.GetBeginAtomIdx())])
        right = int(atom_to_group[int(bond.GetEndAtomIdx())])
        if left == right:
            continue
        edges.add((min(left, right), max(left, right)))
    return tuple(sorted(edges))


def _distance_matrix_rms(left_mol, right_mol) -> float:
    """RMS change of heavy-atom pair distances under frozen atom rows.

    Unlike aligned RMSD this target is directly invariant to rigid motion and
    does not require an atom permutation or an alignment algorithm.
    """

    if left_mol.GetNumAtoms() != right_mol.GetNumAtoms():
        raise G1MulticonformerError("distance matrices have different atom counts")
    atom_count = int(left_mol.GetNumAtoms())
    if atom_count < 2:
        return 0.0
    left = left_mol.GetConformer(0)
    right = right_mol.GetConformer(0)
    squared = 0.0
    pairs = 0
    for first in range(atom_count):
        for second in range(first + 1, atom_count):
            lp = left.GetAtomPosition(first)
            lq = left.GetAtomPosition(second)
            rp = right.GetAtomPosition(first)
            rq = right.GetAtomPosition(second)
            left_distance = math.sqrt(
                sum((float(lp[axis]) - float(lq[axis])) ** 2 for axis in range(3))
            )
            right_distance = math.sqrt(
                sum((float(rp[axis]) - float(rq[axis])) ** 2 for axis in range(3))
            )
            squared += (left_distance - right_distance) ** 2
            pairs += 1
    return math.sqrt(squared / float(pairs))


def _generate_candidate(task: Mapping[str, Any]) -> dict:
    candidate = dict(task["candidate"])
    runtime = _load_runtime(str(task["e3fp_source"]))
    np = runtime["np"]
    Chem = runtime["Chem"]
    AllChem = runtime["AllChem"]
    try:
        base = Chem.MolFromSmiles(candidate["smiles"], sanitize=True)
        if base is None:
            raise G1MulticonformerError("candidate SMILES cannot be parsed")
        groups = tuple(
            tuple(int(value) for value in group)
            for group in runtime["linearize_mol"](base).motif_atom_groups
        )
        roles = _attachment_roles(base, groups)
        atom_to_group = [-1] * base.GetNumAtoms()
        for group_index, group in enumerate(groups):
            for atom_index in group:
                atom_to_group[atom_index] = group_index
        motif_edges = _motif_edges(base, atom_to_group)
        with_h = Chem.AddHs(Chem.Mol(base), addCoords=False)
        parameters = AllChem.ETKDGv3() if hasattr(AllChem, "ETKDGv3") else AllChem.ETKDGv2()
        parameters.randomSeed = int(
            (int(task["seed"]) + 1_000_003 * int(candidate["selection_index"]))
            % 2_147_483_647
        )
        parameters.pruneRmsThresh = float(task["prune_rms_threshold"])
        parameters.numThreads = 1
        conformer_ids = list(
            AllChem.EmbedMultipleConfs(
                with_h,
                numConfs=int(task["requested_conformers"]),
                params=parameters,
            )
        )[: int(task["conformers_per_molecule"])]
        if len(conformer_ids) < 2:
            raise G1MulticonformerError("fewer than two conformers were reproduced")
        matrices = []
        geometries = []
        for local_index, conformer_id in enumerate(conformer_ids):
            single = _one_conformer_mol(Chem, with_h, int(conformer_id))
            tagged, source_count, _ = runtime["tag_source_atoms"](Chem, single)
            geometry, _ = runtime["project_hydrogens"](Chem, tagged, source_count)
            _, inherited, _, _, _ = runtime["generate"](
                np,
                runtime["e3fp_api"],
                geometry,
                int(candidate["selection_index"]) * 100 + local_index,
            )
            matrices.append(inherited.tolist())
            geometries.append(geometry)
        rigid = _rigid_copy(Chem, np, geometries[0])
        _, rigid_matrix, _, _, _ = runtime["generate"](
            np,
            runtime["e3fp_api"],
            rigid,
            int(candidate["selection_index"]) * 100 + 99,
        )
        rmsds = []
        for left in range(len(geometries)):
            for right in range(left + 1, len(geometries)):
                rmsds.append(
                    {
                        "left": left,
                        "right": right,
                        "rmsd": _aligned_rmsd_same_atom_rows(
                            Chem.Mol(geometries[left]), Chem.Mol(geometries[right])
                        ),
                        "distance_matrix_rms": _distance_matrix_rms(
                            Chem.Mol(geometries[left]), Chem.Mol(geometries[right])
                        ),
                    }
                )
        return {
            "status": "pass",
            "member_id": candidate["member_id"],
            "selection_index": int(candidate["selection_index"]),
            "e3fp_ids": matrices,
            "rigid_e3fp_ids": rigid_matrix.tolist(),
            "rigid_e3fp_exact": bool(
                np.array_equal(np.asarray(matrices[0]), rigid_matrix)
            ),
            "atom_is_attachment": roles,
            "atom_to_motif": tuple(atom_to_group),
            "motif_count": len(groups),
            "motif_edges": motif_edges,
            "atomic_numbers": tuple(int(atom.GetAtomicNum()) for atom in base.GetAtoms()),
            "conformer_positions": tuple(
                tuple(
                    tuple(
                        float(geometry.GetConformer(0).GetAtomPosition(atom_index)[axis])
                        for axis in range(3)
                    )
                    for atom_index in range(geometry.GetNumAtoms())
                )
                for geometry in geometries
            ),
            "rmsds": rmsds,
        }
    except Exception as exc:
        return {
            "status": "reject",
            "member_id": candidate.get("member_id"),
            "selection_index": candidate.get("selection_index"),
            "reason": type(exc).__name__,
            "detail": str(exc),
        }


def _record(candidate: Mapping[str, Any], matrix) -> dict:
    return {
        "member_id": candidate["member_id"],
        "selection_index": candidate["selection_index"],
        "e3fp_ids": matrix,
        "atom_valid": [True] * len(matrix),
        "atom_is_attachment": candidate["atom_is_attachment"],
        "atom_to_motif": candidate["atom_to_motif"],
        "motif_count": candidate["motif_count"],
    }


def _representation_distances(model, candidate, *, device, ablate_level3: bool) -> dict:
    matrices = []
    for matrix in candidate["e3fp_ids"] + [candidate["rigid_e3fp_ids"]]:
        copied = [list(row) for row in matrix]
        if ablate_level3:
            for row in copied:
                row[3] = -1
        matrices.append(copied)
    batch = collate_motif_state_records([_record(candidate, matrix) for matrix in matrices])
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = model(
            batch["e3fp_ids"].to(device),
            batch["atom_valid"].to(device),
            batch["atom_to_motif"].to(device),
            num_groups=int(batch["num_groups"]),
            atom_is_attachment=batch["atom_is_attachment"].to(device),
        )
    hidden = output.group_hidden[:, : int(candidate["motif_count"])].float().cpu()
    rigid_max_abs = float((hidden[0] - hidden[-1]).abs().max())
    pair_distances = []
    for row in candidate["rmsds"]:
        difference = hidden[int(row["left"])] - hidden[int(row["right"])]
        pair_distances.append(
            {
                "rmsd": float(row["rmsd"]),
                "representation_rms": float(torch.sqrt(torch.mean(difference.square()))),
            }
        )
    return {"rigid_max_abs": rigid_max_abs, "pairs": pair_distances}


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return 0.0 if denominator == 0.0 else numerator / denominator


def _summarize(rows: Sequence[Mapping[str, Any]], key: str) -> dict:
    rigid = [float(row[key]["rigid_max_abs"]) for row in rows]
    pairs = [pair for row in rows for pair in row[key]["pairs"]]
    distances = [float(pair["representation_rms"]) for pair in pairs]
    rmsds = [float(pair["rmsd"]) for pair in pairs]
    return {
        "molecules": len(rows),
        "conformer_pairs": len(pairs),
        "rigid_exact_count": sum(value == 0.0 for value in rigid),
        "rigid_max_abs_max": max(rigid),
        "changed_pair_fraction_at_1e_6": sum(value > 1e-6 for value in distances)
        / len(distances),
        "representation_rms_median": median(distances),
        "representation_rms_min": min(distances),
        "representation_rms_max": max(distances),
        "pearson_rmsd_vs_representation_rms": _pearson(rmsds, distances),
    }


def run(args) -> dict:
    candidates = []
    candidate_limit = int(args.target_molecules) + int(args.replay_spares)
    with Path(args.c0_molecules).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                candidates.append(json.loads(line))
            if len(candidates) >= candidate_limit:
                break
    if len(candidates) < int(args.target_molecules):
        raise G1MulticonformerError("C0 artifact has too few candidates")
    tasks = [
        {
            "candidate": candidate,
            "e3fp_source": str(Path(args.e3fp_source).expanduser().resolve()),
            "seed": 20260808,
            "requested_conformers": 8,
            "conformers_per_molecule": 4,
            "prune_rms_threshold": 0.35,
        }
        for candidate in candidates
    ]
    if int(args.workers) == 1:
        generated = [_generate_candidate(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            generated = list(pool.map(_generate_candidate, tasks, chunksize=1))
    rejects = [row for row in generated if row["status"] != "pass"]
    accepted = [row for row in generated if row["status"] == "pass"]
    if len(accepted) < int(args.target_molecules):
        raise G1MulticonformerError(
            "only {} reproduced candidates for target {}; rejects: {}".format(
                len(accepted), int(args.target_molecules), rejects[:3]
            )
        )
    accepted = accepted[: int(args.target_molecules)]

    with Path(args.g1_manifest).expanduser().resolve().open("r", encoding="utf-8") as handle:
        training_manifest = json.load(handle)
    configuration = training_manifest["configuration"]
    checkpoint = torch.load(Path(args.checkpoint).expanduser().resolve(), map_location="cpu")
    model = LevelAwareMotifStateEncoder(
        embedding_dim=int(configuration["embedding_dim"]),
        hidden_dim=int(configuration["hidden_dim"]),
        pooling=str(training_manifest["pooling"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if args.require_cuda and device.type != "cuda":
        raise G1MulticonformerError("CUDA was required but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise G1MulticonformerError("CUDA device was selected but is unavailable")
    model.to(device).eval()
    rows = []
    for candidate in accepted:
        rows.append(
            {
                "member_id": candidate["member_id"],
                "all_levels": _representation_distances(
                    model, candidate, device=device, ablate_level3=False
                ),
                "level3_ablated": _representation_distances(
                    model, candidate, device=device, ablate_level3=True
                ),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "frozen_g1b_same_2d_multiconformer_motif_representation",
        "device": str(device),
        "workers": int(args.workers),
        "c0_candidates_replayed": len(candidates),
        "reproduced_candidates": len(accepted),
        "replay_reject_count": len(rejects),
        "replay_rejects": [
            {
                "member_id": row.get("member_id"),
                "selection_index": row.get("selection_index"),
                "reason": row.get("reason"),
                "detail": row.get("detail"),
            }
            for row in rejects
        ],
        "rigid_e3fp_exact_count": sum(
            bool(candidate["rigid_e3fp_exact"]) for candidate in accepted
        ),
        "all_levels": _summarize(rows, "all_levels"),
        "level3_ablated": _summarize(rows, "level3_ablated"),
        "decision_boundary": (
            "all rigid copies must be exact and conformer pairs must change both the full "
            "and level3-ablated motif representations before a T5 bridge"
        ),
    }
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise G1MulticonformerError("output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-molecules", required=True)
    parser.add_argument("--e3fp-source", required=True)
    parser.add_argument("--g1-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-molecules", type=int, default=128)
    parser.add_argument("--replay-spares", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def main(argv=None):
    print(json.dumps(run(build_parser().parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
