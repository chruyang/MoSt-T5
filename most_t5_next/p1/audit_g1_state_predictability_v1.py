#!/usr/bin/env python3
"""Diagnose whether masked E3FP shell IDs are predictable from frozen context.

This is a train-to-dev lookup audit, not a learned model comparison.  It asks
whether each shell level has reusable conditional structure before changing the
G1 pooling architecture or bridging the state encoder into T5.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "most-t5-p1/g1-state-predictability-audit/v1"
NUM_CLASSES = 4096


class PredictabilityAuditError(ValueError):
    pass


@dataclass(frozen=True)
class CompactRecord:
    e3fp_ids: Tuple[Tuple[int, int, int, int], ...]
    atom_is_attachment: Tuple[bool, ...]
    atom_to_motif: Tuple[int, ...]


def _load_split(release_root: Path, split: str) -> List[CompactRecord]:
    try:
        import lmdb
    except ImportError as exc:
        raise PredictabilityAuditError("python-lmdb is required") from exc

    membership_path = release_root / "{}_membership.jsonl".format(split)
    lmdb_path = release_root / "paired_records.lmdb"
    if not membership_path.is_file() or not lmdb_path.is_dir():
        raise PredictabilityAuditError("paired release is incomplete")
    rows = []
    with membership_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    environment = lmdb.open(
        str(lmdb_path),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=8,
    )
    records: List[CompactRecord] = []
    try:
        with environment.begin(write=False) as transaction:
            for row in rows:
                payload = transaction.get(str(row["storage_key"]).encode("ascii"))
                if payload is None:
                    raise PredictabilityAuditError("membership key is absent from LMDB")
                envelope = json.loads(payload)
                domain = envelope["motif_training_document"]["atom_domain"]
                ids = tuple(tuple(int(value) for value in atom) for atom in domain["full_e3fp_ids"])
                roles = tuple(bool(value) for value in domain["atom_is_attachment"])
                groups = tuple(int(value) for value in domain["atom_to_logical_motif"])
                if not ids or not (len(ids) == len(roles) == len(groups)):
                    raise PredictabilityAuditError("atom-domain arrays disagree")
                records.append(CompactRecord(ids, roles, groups))
    finally:
        environment.close()
    return records


def _motif_members(record: CompactRecord) -> Mapping[int, Tuple[int, ...]]:
    members: Dict[int, List[int]] = defaultdict(list)
    for atom_index, motif_index in enumerate(record.atom_to_motif):
        members[motif_index].append(atom_index)
    return {key: tuple(value) for key, value in members.items()}


def _context(
    record: CompactRecord,
    atom_index: int,
    level: int,
    kind: str,
    motif_members: Mapping[int, Tuple[int, ...]],
) -> tuple:
    atom = record.e3fp_ids[atom_index]
    role = int(record.atom_is_attachment[atom_index])
    prefix = (role,) + tuple(atom[:level])
    if kind == "atom_prefix":
        return prefix
    if kind == "atom_other_shells":
        return (role,) + tuple(atom[index] for index in range(4) if index != level)
    motif_atoms = motif_members[record.atom_to_motif[atom_index]]
    if kind == "motif_scalar_context":
        attachment_count = sum(record.atom_is_attachment[index] for index in motif_atoms)
        return prefix + (len(motif_atoms), int(attachment_count))
    if kind == "motif_prefix_multiset":
        motif_prefixes = tuple(
            sorted(
                (int(record.atom_is_attachment[index]),)
                + tuple(record.e3fp_ids[index][:level])
                for index in motif_atoms
            )
        )
        return prefix + (motif_prefixes,)
    raise PredictabilityAuditError("unknown context kind: {}".format(kind))


def evaluate_context_predictor(
    train_records: Sequence[CompactRecord],
    dev_records: Sequence[CompactRecord],
    *,
    level: int,
    context_kind: str,
    num_classes: int = NUM_CLASSES,
) -> dict:
    if level not in (1, 2, 3):
        raise PredictabilityAuditError("level must be 1, 2, or 3")
    global_counts = [1] * int(num_classes)
    context_counts: Dict[tuple, Counter] = defaultdict(Counter)
    context_totals: Counter = Counter()
    train_targets = 0
    for record in train_records:
        members = _motif_members(record)
        for atom_index, atom in enumerate(record.e3fp_ids):
            target = int(atom[level])
            if target < 0:
                continue
            context = _context(record, atom_index, level, context_kind, members)
            global_counts[target] += 1
            context_counts[context][target] += 1
            context_totals[context] += 1
            train_targets += 1

    global_total = float(sum(global_counts))
    global_probabilities = [count / global_total for count in global_counts]
    global_mode = max(range(int(num_classes)), key=lambda value: global_counts[value])
    # Resolve each context mode once.  Recomputing max(Counter.values()) for
    # every dev atom is needlessly quadratic for broad low-level contexts.
    context_predictions = {}
    for context, counts in context_counts.items():
        highest = max(counts.values())
        context_predictions[context] = min(
            token for token, count in counts.items() if count == highest
        )
    static_nll_sum = 0.0
    conditional_nll_sum = 0.0
    static_correct = 0
    conditional_correct = 0
    dev_targets = 0
    seen_contexts = 0
    seen_support_sum = 0
    unique_dev_contexts = set()
    for record in dev_records:
        members = _motif_members(record)
        for atom_index, atom in enumerate(record.e3fp_ids):
            target = int(atom[level])
            if target < 0:
                continue
            context = _context(record, atom_index, level, context_kind, members)
            unique_dev_contexts.add(context)
            prior_probability = global_probabilities[target]
            static_nll_sum -= math.log(prior_probability)
            static_correct += int(target == global_mode)
            counts = context_counts.get(context)
            support = int(context_totals.get(context, 0))
            if counts is None:
                probability = prior_probability
                prediction = global_mode
            else:
                # One prior-equivalent observation avoids zero-probability
                # claims while keeping this diagnostic train-only.
                probability = (counts[target] + prior_probability) / (support + 1.0)
                prediction = context_predictions[context]
                seen_contexts += 1
                seen_support_sum += support
            conditional_nll_sum -= math.log(probability)
            conditional_correct += int(target == prediction)
            dev_targets += 1
    if not dev_targets:
        raise PredictabilityAuditError("dev split has no populated targets")
    static_nll = static_nll_sum / dev_targets
    conditional_nll = conditional_nll_sum / dev_targets
    return {
        "level": int(level),
        "context_kind": context_kind,
        "train_targets": train_targets,
        "dev_targets": dev_targets,
        "train_unique_contexts": len(context_counts),
        "dev_unique_contexts": len(unique_dev_contexts),
        "dev_seen_context_fraction": seen_contexts / dev_targets,
        "mean_train_support_for_seen_dev_context": (
            seen_support_sum / seen_contexts if seen_contexts else 0.0
        ),
        "laplace_unigram_nll": static_nll,
        "laplace_unigram_accuracy": static_correct / dev_targets,
        "conditional_nll": conditional_nll,
        "conditional_accuracy": conditional_correct / dev_targets,
        "nll_improvement_over_unigram": static_nll - conditional_nll,
        "uniform_nll": math.log(float(num_classes)),
    }


def run(args) -> dict:
    release_root = Path(args.paired_release).expanduser().resolve()
    train_records = _load_split(release_root, "train")
    dev_records = _load_split(release_root, "dev")
    context_kinds = (
        "atom_prefix",
        "atom_other_shells",
        "motif_scalar_context",
        "motif_prefix_multiset",
    )
    results = []
    for level in (1, 2, 3):
        for context_kind in context_kinds:
            results.append(
                evaluate_context_predictor(
                    train_records,
                    dev_records,
                    level=level,
                    context_kind=context_kind,
                )
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "train_to_dev_nonparametric_e3fp_shell_predictability",
        "paired_release": str(release_root),
        "train_members": len(train_records),
        "dev_members": len(dev_records),
        "target_levels": [1, 2, 3],
        "context_kinds": list(context_kinds),
        "posterior": "context_count_plus_one_laplace_unigram_prior_observation",
        "results": results,
        "interpretation_boundary": (
            "This audit measures reusable train-to-dev conditional structure. "
            "It is not a neural upper bound and does not compare downstream quality."
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise PredictabilityAuditError("output already exists: {}".format(output))
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
