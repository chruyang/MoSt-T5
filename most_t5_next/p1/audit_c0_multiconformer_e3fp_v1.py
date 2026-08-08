#!/usr/bin/env python3
"""C0 audit: can inherited E3FP distinguish conformers of one 2D molecule?

This is a chemistry/mechanism audit, not a training entry point.  It samples
identities from a published PF-1 train membership, generates several ETKDG
conformers for each identity, applies the production hydrogen projection and
explicit duplicate-shell inheritance, and reports sensitivity at shell, atom,
and motif granularity.

The audit deliberately does not alter the paired release, fit a model, or use
dev/test labels.  It answers whether the proposed G1 geometry target is
identifiable before GPU time is rented.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "most-t5-p1/c0-multiconformer-e3fp-audit/v1"
DEFAULT_SEED = 20260808


class C0AuditError(ValueError):
    pass


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise C0AuditError("{}:{} is not a JSON object".format(path, line_number))
            yield value


def _load_runtime(e3fp_source: str):
    root = str(Path(e3fp_source).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from most_t5_next.r1.adapter.mol_linearizer import linearize_mol
    from most_t5_next.r1.gates.pcqm_e3fp_preflight import (
        project_hydrogens,
        tag_source_atoms,
    )
    from most_t5_next.r1.semantic.e3fp_duplicate_inheritance_v1 import (
        generate_e3fp_projection_pair,
    )
    # Match the production import order.  On the pinned Windows Conda build,
    # importing E3FP before the projection/semantic modules causes RDKit's
    # conformer code and SciPy to initialize two OpenMP DLL copies.
    from e3fp.pipeline import fprints_from_mol_verbose
    from e3fp.fingerprint.fprinter import signed_to_unsigned_int

    api = {
        "fprints_from_mol_verbose": fprints_from_mol_verbose,
        "signed_to_unsigned_int": signed_to_unsigned_int,
    }
    return {
        "np": np,
        "Chem": Chem,
        "AllChem": AllChem,
        "linearize_mol": linearize_mol,
        "project_hydrogens": project_hydrogens,
        "tag_source_atoms": tag_source_atoms,
        "generate": generate_e3fp_projection_pair,
        "e3fp_api": api,
    }


def _select_candidates(
    release_root: Path,
    target_molecules: int,
    candidate_multiplier: float,
    seed: int,
    min_heavy_atoms: int,
    max_heavy_atoms: int,
    min_rotatable_bonds: int,
) -> List[Dict[str, Any]]:
    import lmdb
    from rdkit import Chem

    membership = list(_iter_jsonl(release_root / "train_membership.jsonl"))
    random.Random(int(seed)).shuffle(membership)
    wanted = max(target_molecules, int(math.ceil(target_molecules * candidate_multiplier)))
    selected: List[Dict[str, Any]] = []
    env = lmdb.open(
        str(release_root / "paired_records.lmdb"),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=64,
    )
    try:
        with env.begin(buffers=False) as txn:
            for shuffled_rank, member in enumerate(membership):
                storage_key = str(member["storage_key"])
                payload = txn.get(storage_key.encode("ascii"))
                if payload is None:
                    raise C0AuditError("paired LMDB lacks {}".format(storage_key))
                envelope = json.loads(payload)
                smiles = str(envelope["receipt"]["strict_isomeric_identity"])
                if "." in smiles:
                    continue
                mol = Chem.MolFromSmiles(smiles, sanitize=True)
                if mol is None:
                    continue
                heavy_atoms = int(mol.GetNumHeavyAtoms())
                rotatable = _acyclic_nonterminal_single_bonds(mol)
                if not (min_heavy_atoms <= heavy_atoms <= max_heavy_atoms):
                    continue
                if rotatable < min_rotatable_bonds:
                    continue
                selected.append(
                    {
                        "candidate_rank": len(selected),
                        "shuffled_membership_rank": shuffled_rank,
                        "selection_index": int(member["selection_index"]),
                        "member_id": str(member["member_id"]),
                        "storage_key": storage_key,
                        "smiles": smiles,
                        "heavy_atoms": heavy_atoms,
                        "rotatable_bonds": rotatable,
                    }
                )
                if len(selected) >= wanted:
                    break
    finally:
        env.close()
    if len(selected) < target_molecules:
        raise C0AuditError(
            "only {} eligible train identities for target {}".format(
                len(selected), target_molecules
            )
        )
    return selected


def _acyclic_nonterminal_single_bonds(mol) -> int:
    """Cheap flexibility proxy used only to stratify C0 train identities.

    Avoid compiled descriptor backends here: they are irrelevant to the E3FP
    calculation and conflict with the pinned Windows OpenMP runtime.  The audit
    records the proxy definition and does not treat it as a molecular label.
    """

    count = 0
    for bond in mol.GetBonds():
        if float(bond.GetBondTypeAsDouble()) != 1.0 or bond.IsInRing():
            continue
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetAtomicNum() <= 1 or end.GetAtomicNum() <= 1:
            continue
        if begin.GetDegree() <= 1 or end.GetDegree() <= 1:
            continue
        count += 1
    return count


def _one_conformer_mol(Chem, source_mol, conformer_id: int):
    result = Chem.Mol(source_mol)
    result.RemoveAllConformers()
    conformer = Chem.Conformer(source_mol.GetConformer(int(conformer_id)))
    conformer.SetId(0)
    result.AddConformer(conformer, assignId=True)
    return result


def _rigid_copy(Chem, np, geometry_mol):
    result = Chem.Mol(geometry_mol)
    conformer = result.GetConformer(0)
    for atom_index in range(result.GetNumAtoms()):
        point = conformer.GetAtomPosition(atom_index)
        conformer.SetAtomPosition(
            atom_index,
            (-float(point.y) + 7.25, float(point.x) - 3.5, float(point.z) + 1.75),
        )
    return result


def _aligned_rmsd_same_atom_rows(left_mol, right_mol) -> float:
    """Horn quaternion RMSD under the frozen atom-row correspondence.

    The scalar 4x4 power iteration avoids importing RDKit MolAlign or a BLAS
    eigensolver, both of which introduce an unrelated OpenMP runtime on the
    pinned Windows chemistry environment.
    """

    if left_mol.GetNumAtoms() != right_mol.GetNumAtoms():
        raise C0AuditError("RMSD molecules have different atom counts")
    count = int(left_mol.GetNumAtoms())
    if count == 0:
        return 0.0
    left_conf = left_mol.GetConformer(0)
    right_conf = right_mol.GetConformer(0)
    left = [left_conf.GetAtomPosition(index) for index in range(count)]
    right = [right_conf.GetAtomPosition(index) for index in range(count)]
    left_center = [sum(float(p[axis]) for p in left) / count for axis in range(3)]
    right_center = [sum(float(p[axis]) for p in right) / count for axis in range(3)]
    covariance = [[0.0] * 3 for _ in range(3)]
    squared_norm = 0.0
    for lp, rp in zip(left, right):
        x = [float(lp[axis]) - left_center[axis] for axis in range(3)]
        y = [float(rp[axis]) - right_center[axis] for axis in range(3)]
        squared_norm += sum(value * value for value in x)
        squared_norm += sum(value * value for value in y)
        for row in range(3):
            for column in range(3):
                covariance[row][column] += x[row] * y[column]
    sxx, sxy, sxz = covariance[0]
    syx, syy, syz = covariance[1]
    szx, szy, szz = covariance[2]
    horn = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ]
    shift = max(sum(abs(value) for value in row) for row in horn) + 1.0
    vector = [1.0, 0.0, 0.0, 0.0]
    for _ in range(80):
        updated = [
            sum(horn[row][column] * vector[column] for column in range(4))
            + shift * vector[row]
            for row in range(4)
        ]
        norm = math.sqrt(sum(value * value for value in updated))
        if norm == 0.0:
            break
        vector = [value / norm for value in updated]
    largest = sum(
        vector[row]
        * sum(horn[row][column] * vector[column] for column in range(4))
        for row in range(4)
    )
    return math.sqrt(max(0.0, (squared_norm - 2.0 * largest) / count))


def _pair_metrics(np, left, right, motif_groups, rmsd: float) -> Dict[str, Any]:
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 4:
        raise C0AuditError("paired E3FP matrices have incompatible shape")
    unequal = left != right
    by_level = []
    for level in range(4):
        union = (left[:, level] >= 0) | (right[:, level] >= 0)
        by_level.append(
            {
                "level": level,
                "rows": int(left.shape[0]),
                "changed_rows": int(unequal[:, level].sum()),
                "populated_union_rows": int(union.sum()),
                "changed_populated_union_rows": int((unequal[:, level] & union).sum()),
            }
        )
    atom_changed = unequal[:, 1:4].any(axis=1)
    motif_changed = [
        bool(any(bool(atom_changed[int(atom_index)]) for atom_index in group))
        for group in motif_groups
    ]
    return {
        "rmsd_angstrom": float(rmsd),
        "by_level": by_level,
        "atom_rows": int(left.shape[0]),
        "changed_atoms_l1_l3": int(atom_changed.sum()),
        "motifs": len(motif_groups),
        "changed_motifs_l1_l3": int(sum(motif_changed)),
        "exact_same_l1_l3": bool(np.array_equal(left[:, 1:4], right[:, 1:4])),
    }


def _worker(task: Mapping[str, Any]) -> Dict[str, Any]:
    candidate = dict(task["candidate"])
    runtime = _load_runtime(str(task["e3fp_source"]))
    np = runtime["np"]
    Chem = runtime["Chem"]
    AllChem = runtime["AllChem"]

    try:
        base = Chem.MolFromSmiles(candidate["smiles"], sanitize=True)
        if base is None:
            raise C0AuditError("canonical identity cannot be parsed")
        motif_groups = tuple(
            tuple(int(x) for x in group)
            for group in runtime["linearize_mol"](base).motif_atom_groups
        )
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
        )
        if len(conformer_ids) < 2:
            raise C0AuditError("ETKDG produced fewer than two distinct conformers")
        conformer_ids = conformer_ids[: int(task["conformers_per_molecule"])]

        energies: List[float] = []
        forcefield = "ETKDGv3_unoptimized"
        forcefield_mode = str(task["forcefield_mode"])
        if forcefield_mode == "auto" and AllChem.MMFFHasAllMoleculeParams(with_h):
            forcefield = "MMFF94s"
            optimized = AllChem.MMFFOptimizeMoleculeConfs(
                with_h, numThreads=1, maxIters=400, mmffVariant="MMFF94s"
            )
            energy_by_id = {int(cid): float(optimized[int(cid)][1]) for cid in conformer_ids}
            energies = [energy_by_id[int(cid)] for cid in conformer_ids]
        elif forcefield_mode == "auto" and AllChem.UFFHasAllMoleculeParams(with_h):
            forcefield = "UFF"
            optimized = AllChem.UFFOptimizeMoleculeConfs(with_h, numThreads=1, maxIters=400)
            energy_by_id = {int(cid): float(optimized[int(cid)][1]) for cid in conformer_ids}
            energies = [energy_by_id[int(cid)] for cid in conformer_ids]
        else:
            energies = [float("nan") for _ in conformer_ids]

        matrices = []
        geometries = []
        summaries = []
        for local_index, conformer_id in enumerate(conformer_ids):
            single = _one_conformer_mol(Chem, with_h, int(conformer_id))
            tagged, source_count, _ = runtime["tag_source_atoms"](Chem, single)
            geometry, _ = runtime["project_hydrogens"](Chem, tagged, source_count)
            raw, inherited, _, inheritance_summary, _ = runtime["generate"](
                np,
                runtime["e3fp_api"],
                geometry,
                int(candidate["selection_index"]) * 100 + local_index,
            )
            if inherited.shape != (base.GetNumAtoms(), 4):
                raise C0AuditError("projected E3FP rows do not match canonical atom domain")
            matrices.append(inherited)
            geometries.append(geometry)
            summaries.append(inheritance_summary)

        pair_rows: List[Dict[str, Any]] = []
        for left_index in range(len(matrices)):
            for right_index in range(left_index + 1, len(matrices)):
                probe = Chem.Mol(geometries[left_index])
                reference = Chem.Mol(geometries[right_index])
                rmsd = _aligned_rmsd_same_atom_rows(probe, reference)
                row = _pair_metrics(
                    np,
                    matrices[left_index],
                    matrices[right_index],
                    motif_groups,
                    rmsd,
                )
                row["left_conformer"] = left_index
                row["right_conformer"] = right_index
                pair_rows.append(row)

        rigid_exact = None
        if int(candidate["candidate_rank"]) < int(task["rigid_check_molecules"]):
            rigid = _rigid_copy(Chem, np, geometries[0])
            _, rigid_matrix, _, _, _ = runtime["generate"](
                np,
                runtime["e3fp_api"],
                rigid,
                int(candidate["selection_index"]) * 100 + 99,
            )
            rigid_exact = bool(np.array_equal(matrices[0], rigid_matrix))

        token_counts = []
        for level in range(4):
            counter: Counter = Counter()
            for matrix in matrices:
                counter.update(int(x) for x in matrix[:, level] if int(x) >= 0)
            token_counts.append(dict(counter))

        unique_l1_l3 = len({matrix[:, 1:4].tobytes() for matrix in matrices})
        return {
            "status": "pass",
            **candidate,
            "motif_count": len(motif_groups),
            "motif_sizes": [len(group) for group in motif_groups],
            "requested_conformers": int(task["conformers_per_molecule"]),
            "generated_conformers": len(matrices),
            "forcefield": forcefield,
            "energies": energies,
            "unique_e3fp_l1_l3_states": unique_l1_l3,
            "rigid_transform_exact": rigid_exact,
            "pairs": pair_rows,
            "token_counts_by_level": token_counts,
            "inheritance_summaries": summaries,
        }
    except Exception as exc:
        return {
            "status": "reject",
            **candidate,
            "reason": type(exc).__name__,
            "detail": str(exc),
        }


def _safe_ratio(numerator: int, denominator: int):
    return None if denominator == 0 else float(numerator) / float(denominator)


def _distribution_summary(counter: Counter) -> Dict[str, Any]:
    total = int(sum(counter.values()))
    if total == 0:
        return {"observations": 0, "unique_ids": 0, "entropy_nats": None, "top1_rate": None}
    probabilities = [float(count) / float(total) for count in counter.values()]
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
    return {
        "observations": total,
        "unique_ids": len(counter),
        "entropy_nats": entropy,
        "entropy_bits": entropy / math.log(2.0),
        "top1_rate": max(probabilities),
    }


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and float(values[order[stop]]) == float(values[order[start]]):
            stop += 1
        average = 0.5 * ((start + 1) + stop)
        for position in range(start, stop):
            ranks[order[position]] = average
        start = stop
    return ranks


def _spearman(values_a: Sequence[float], values_b: Sequence[float]):
    if len(values_a) != len(values_b) or len(values_a) < 3:
        return None
    ranks_a = _average_ranks(values_a)
    ranks_b = _average_ranks(values_b)
    mean_a = sum(ranks_a) / len(ranks_a)
    mean_b = sum(ranks_b) / len(ranks_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(ranks_a, ranks_b))
    denom_a = math.sqrt(sum((a - mean_a) ** 2 for a in ranks_a))
    denom_b = math.sqrt(sum((b - mean_b) ** 2 for b in ranks_b))
    if denom_a == 0.0 or denom_b == 0.0:
        return None
    return numerator / (denom_a * denom_b)


def summarize_passes(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    level_totals = [Counter() for _ in range(4)]
    token_counts = [Counter() for _ in range(4)]
    pair_rmsd: List[float] = []
    pair_change: List[float] = []
    atom_rows = changed_atoms = motifs = changed_motifs = 0
    exact_same_pairs = high_rmsd_same_pairs = 0
    motif_size_totals: Counter = Counter()
    motif_size_changed: Counter = Counter()
    rigid_checked = rigid_passed = 0

    for row in rows:
        for level, values in enumerate(row["token_counts_by_level"]):
            token_counts[level].update({int(k): int(v) for k, v in values.items()})
        if row["rigid_transform_exact"] is not None:
            rigid_checked += 1
            rigid_passed += int(bool(row["rigid_transform_exact"]))
        for size in row["motif_sizes"]:
            motif_size_totals[int(size)] += len(row["pairs"])
        for pair in row["pairs"]:
            pair_rmsd.append(float(pair["rmsd_angstrom"]))
            atom_rows += int(pair["atom_rows"])
            changed_atoms += int(pair["changed_atoms_l1_l3"])
            motifs += int(pair["motifs"])
            changed_motifs += int(pair["changed_motifs_l1_l3"])
            exact = bool(pair["exact_same_l1_l3"])
            exact_same_pairs += int(exact)
            high_rmsd_same_pairs += int(exact and float(pair["rmsd_angstrom"]) >= 1.0)
            pair_change.append(
                float(pair["changed_atoms_l1_l3"]) / float(pair["atom_rows"])
            )
            for level_row in pair["by_level"]:
                level = int(level_row["level"])
                level_totals[level].update(
                    {
                        "rows": int(level_row["rows"]),
                        "changed_rows": int(level_row["changed_rows"]),
                        "populated_union_rows": int(level_row["populated_union_rows"]),
                        "changed_populated_union_rows": int(
                            level_row["changed_populated_union_rows"]
                        ),
                    }
                )

    # Size-specific changed counts are reconstructed molecule by molecule so
    # the persisted per-pair record can remain compact.
    for row in rows:
        sizes = [int(x) for x in row["motif_sizes"]]
        # Per-motif flags are intentionally not persisted; use the aggregate
        # pair rate as a conservative allocation by size only in a separate
        # diagnostic, not as a scientific result.
        if len(set(sizes)) == 1:
            motif_size_changed[sizes[0]] += sum(
                int(pair["changed_motifs_l1_l3"]) for pair in row["pairs"]
            )

    correlation = None
    if len(pair_rmsd) >= 3 and len(set(pair_rmsd)) > 1 and len(set(pair_change)) > 1:
        correlation = _spearman(pair_rmsd, pair_change)

    by_level = []
    for level, counts in enumerate(level_totals):
        by_level.append(
            {
                "level": level,
                **dict(counts),
                "row_change_rate": _safe_ratio(counts["changed_rows"], counts["rows"]),
                "populated_union_change_rate": _safe_ratio(
                    counts["changed_populated_union_rows"],
                    counts["populated_union_rows"],
                ),
                "unigram_target": _distribution_summary(token_counts[level]),
            }
        )

    pair_count = len(pair_rmsd)
    return {
        "passed_molecules": len(rows),
        "generated_conformers": sum(int(row["generated_conformers"]) for row in rows),
        "pair_count": pair_count,
        "rigid_transform": {
            "checked_molecules": rigid_checked,
            "exact_passed": rigid_passed,
            "exact_rate": _safe_ratio(rigid_passed, rigid_checked),
        },
        "by_level": by_level,
        "atom_l1_l3_change_rate": _safe_ratio(changed_atoms, atom_rows),
        "motif_l1_l3_change_rate": _safe_ratio(changed_motifs, motifs),
        "exact_same_l1_l3_pairs": exact_same_pairs,
        "exact_same_l1_l3_pair_rate": _safe_ratio(exact_same_pairs, pair_count),
        "high_rmsd_ge_1A_exact_same_l1_l3_pairs": high_rmsd_same_pairs,
        "rmsd_angstrom": {
            "mean": None if not pair_rmsd else sum(pair_rmsd) / len(pair_rmsd),
            "min": None if not pair_rmsd else min(pair_rmsd),
            "max": None if not pair_rmsd else max(pair_rmsd),
        },
        "rmsd_vs_atom_change_spearman": correlation,
    }


def run(args) -> Dict[str, Any]:
    # Keep the known-good production import order on Windows Conda: NumPy's
    # runtime is loaded before RDKit during the membership chemistry filter.
    import numpy  # noqa: F401

    start = time.time()
    release_root = Path(args.paired_release).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise C0AuditError("output directory already exists: {}".format(output_dir))
    output_dir.mkdir(parents=True)

    candidates = _select_candidates(
        release_root=release_root,
        target_molecules=int(args.target_molecules),
        candidate_multiplier=float(args.candidate_multiplier),
        seed=int(args.seed),
        min_heavy_atoms=int(args.min_heavy_atoms),
        max_heavy_atoms=int(args.max_heavy_atoms),
        min_rotatable_bonds=int(args.min_rotatable_bonds),
    )
    tasks = [
        {
            "candidate": candidate,
            "e3fp_source": str(Path(args.e3fp_source).resolve()),
            "seed": int(args.seed),
            "requested_conformers": int(args.requested_conformers),
            "conformers_per_molecule": int(args.conformers_per_molecule),
            "prune_rms_threshold": float(args.prune_rms_threshold),
            "rigid_check_molecules": int(args.rigid_check_molecules),
            "forcefield_mode": str(args.forcefield_mode),
        }
        for candidate in candidates
    ]
    results: List[Dict[str, Any]] = []
    executor_kind = str(args.executor)
    if executor_kind == "auto":
        # Windows Conda distributions commonly load RDKit and NumPy from two
        # OpenMP runtimes in spawned children.  Independent molecules are
        # thread-safe, so use an ordered thread pool there without enabling
        # the unsafe KMP duplicate-runtime workaround.  Linux CPU instances
        # retain true process parallelism.
        executor_kind = "thread" if os.name == "nt" else "process"
    executor_class = ThreadPoolExecutor if executor_kind == "thread" else ProcessPoolExecutor
    if executor_kind == "thread":
        # Import RDKit/NumPy/SciPy-backed E3FP once before worker threads start;
        # concurrent first imports can race while loading the OpenMP DLLs.
        _load_runtime(str(Path(args.e3fp_source).resolve()))
    if int(args.workers) == 1:
        results = [_worker(task) for task in tasks]
    else:
        with executor_class(max_workers=int(args.workers)) as pool:
            for row in pool.map(_worker, tasks, chunksize=1):
                results.append(row)

    passes = [row for row in results if row["status"] == "pass"]
    passes.sort(key=lambda row: int(row["candidate_rank"]))
    accepted = passes[: int(args.target_molecules)]
    rejects = [row for row in results if row["status"] != "pass"]
    if len(accepted) < int(args.target_molecules):
        raise C0AuditError(
            "only {} successful molecules for target {}; inspect rejects".format(
                len(accepted), args.target_molecules
            )
        )

    per_molecule_path = output_dir / "molecules.jsonl"
    with per_molecule_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in accepted:
            persisted = dict(row)
            persisted.pop("token_counts_by_level", None)
            persisted.pop("inheritance_summaries", None)
            handle.write(json.dumps(persisted, sort_keys=True, ensure_ascii=False) + "\n")
    reject_path = output_dir / "rejects.jsonl"
    with reject_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rejects:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "train_identity_only_multiconformer_mechanism_audit",
        "not_a_training_or_performance_result": True,
        "configuration": {
            "seed": int(args.seed),
            "target_molecules": int(args.target_molecules),
            "candidate_count": len(candidates),
            "workers": int(args.workers),
            "executor": executor_kind,
            "requested_conformers": int(args.requested_conformers),
            "conformers_per_molecule": int(args.conformers_per_molecule),
            "prune_rms_threshold": float(args.prune_rms_threshold),
            "forcefield_mode": str(args.forcefield_mode),
            "min_heavy_atoms": int(args.min_heavy_atoms),
            "max_heavy_atoms": int(args.max_heavy_atoms),
            "min_rotatable_bonds": int(args.min_rotatable_bonds),
            "rotatable_selection_proxy": "nonring_nonterminal_heavy_single_bonds",
            "rigid_check_molecules": int(args.rigid_check_molecules),
            "e3fp_semantics": "duplicate_pointer_inheritance_v1",
            "e3fp_bits": 4096,
            "e3fp_levels": 4,
        },
        "selection": {
            "split": "train",
            "eligible_source": "published PF1 paired membership strict identities",
            "selected_candidates": len(candidates),
            "worker_rejects": len(rejects),
            "unused_successes": len(passes) - len(accepted),
            "accepted_molecules": len(accepted),
        },
        "result": summarize_passes(accepted),
        "wall_seconds": time.time() - start,
        "artifacts": {
            "molecules": per_molecule_path.name,
            "rejects": reject_path.name,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, sort_keys=True, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", required=True)
    parser.add_argument("--e3fp-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-molecules", type=int, default=1000)
    parser.add_argument("--candidate-multiplier", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--executor", choices=("auto", "process", "thread"), default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--requested-conformers", type=int, default=8)
    parser.add_argument("--conformers-per-molecule", type=int, default=4)
    parser.add_argument("--prune-rms-threshold", type=float, default=0.35)
    parser.add_argument(
        "--forcefield-mode",
        choices=("auto", "none"),
        default="auto",
        help="auto applies MMFF94s with UFF fallback; none audits the ETKDG ensemble directly",
    )
    parser.add_argument("--min-heavy-atoms", type=int, default=8)
    parser.add_argument("--max-heavy-atoms", type=int, default=25)
    parser.add_argument("--min-rotatable-bonds", type=int, default=2)
    parser.add_argument("--rigid-check-molecules", type=int, default=64)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.target_molecules <= 0 or args.workers <= 0:
        raise C0AuditError("target-molecules and workers must be positive")
    if args.requested_conformers < args.conformers_per_molecule:
        raise C0AuditError("requested-conformers must cover conformers-per-molecule")
    summary = run(args)
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
