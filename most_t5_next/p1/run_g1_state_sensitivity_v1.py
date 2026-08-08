#!/usr/bin/env python3
"""Evaluate aligned versus same-atom-count shuffled E3FP state context."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from most_t5_next.p1.level_aware_motif_state_v1 import (
    LevelAwareMotifStateEncoder,
    build_masked_e3fp_state_batch,
    masked_state_ce,
)
from most_t5_next.p1.motif_state_data_v1 import (
    PF1MotifStateDataset,
    collate_motif_state_records,
)


SCHEMA_VERSION = "most-t5-p1/g1-state-sensitivity/v1"


class G1SensitivityError(ValueError):
    pass


def same_size_derangement(records: Sequence[Mapping]) -> Tuple[Dict[int, int], Tuple[int, ...]]:
    """Return a deterministic cyclic donor map within each atom-count bucket."""

    buckets: Dict[int, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        buckets[len(record["e3fp_ids"])].append(index)
    mapping: Dict[int, int] = {}
    excluded: List[int] = []
    for atom_count in sorted(buckets):
        indices = buckets[atom_count]
        if len(indices) == 1:
            excluded.extend(indices)
            continue
        for position, index in enumerate(indices):
            mapping[index] = indices[(position + 1) % len(indices)]
    if any(index == donor for index, donor in mapping.items()):
        raise G1SensitivityError("derangement contains a self-pair")
    return mapping, tuple(excluded)


def _accumulate(totals, metrics) -> None:
    for level, row in metrics.items():
        if level not in totals:
            continue
        totals[level]["nll_sum"] += float(row["nll_sum"].detach().float().cpu())
        totals[level]["correct"] += int(row["correct"].detach().cpu())
        totals[level]["count"] += int(row["count"].detach().cpu())


@torch.no_grad()
def _evaluate(
    model,
    records,
    donor_map,
    *,
    device,
    batch_size: int,
    mask_probability: float,
    mask_seed: int,
    target_levels: Tuple[int, ...],
):
    aligned = {level: {"nll_sum": 0.0, "correct": 0, "count": 0} for level in target_levels}
    shuffled = {level: {"nll_sum": 0.0, "correct": 0, "count": 0} for level in target_levels}
    eligible_indices = tuple(sorted(donor_map))
    model.eval()
    for batch_index, start in enumerate(range(0, len(eligible_indices), int(batch_size))):
        indices = eligible_indices[start : start + int(batch_size)]
        recipients = [records[index] for index in indices]
        donors = [records[donor_map[index]] for index in indices]
        recipient_batch = collate_motif_state_records(recipients)
        donor_batch = collate_motif_state_records(donors)
        if recipient_batch["e3fp_ids"].shape != donor_batch["e3fp_ids"].shape:
            raise G1SensitivityError("same-size donor batch shape differs")
        masked = build_masked_e3fp_state_batch(
            recipient_batch["e3fp_ids"],
            recipient_batch["atom_valid"],
            mask_token_id=model.mask_token_id,
            probability=float(mask_probability),
            seed=int(mask_seed) + batch_index,
            target_levels=target_levels,
        )
        common = {
            "atom_valid": recipient_batch["atom_valid"].to(device, non_blocking=True),
            "atom_to_group": recipient_batch["atom_to_motif"].to(device, non_blocking=True),
            "num_groups": int(recipient_batch["num_groups"]),
            "atom_is_attachment": recipient_batch["atom_is_attachment"].to(
                device, non_blocking=True
            ),
        }
        targets = masked.target_ids.to(device, non_blocking=True)
        target_mask = masked.target_mask.to(device, non_blocking=True)
        shuffled_ids = donor_batch["e3fp_ids"].clone()
        shuffled_ids[masked.target_mask] = model.mask_token_id
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            aligned_output = model(
                masked.corrupted_ids.to(device, non_blocking=True), **common
            )
            shuffled_output = model(shuffled_ids.to(device, non_blocking=True), **common)
            _, aligned_metrics = masked_state_ce(
                aligned_output.logits, targets, target_mask
            )
            _, shuffled_metrics = masked_state_ce(
                shuffled_output.logits, targets, target_mask
            )
        _accumulate(aligned, aligned_metrics)
        _accumulate(shuffled, shuffled_metrics)
    report = {}
    for level in target_levels:
        count = aligned[level]["count"]
        if count <= 0 or count != shuffled[level]["count"]:
            raise G1SensitivityError("aligned/shuffled target counts disagree")
        aligned_nll = aligned[level]["nll_sum"] / count
        shuffled_nll = shuffled[level]["nll_sum"] / count
        report[str(level)] = {
            "count": count,
            "aligned_nll": aligned_nll,
            "shuffled_nll": shuffled_nll,
            "delta_nll_shuffled_minus_aligned": shuffled_nll - aligned_nll,
            "aligned_accuracy": aligned[level]["correct"] / count,
            "shuffled_accuracy": shuffled[level]["correct"] / count,
        }
    return report


def run(args) -> dict:
    manifest_path = Path(args.g1_manifest).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise G1SensitivityError("output already exists")
    with manifest_path.open("r", encoding="utf-8") as handle:
        training_manifest = json.load(handle)
    configuration = training_manifest["configuration"]
    target_levels = tuple(int(level) for level in configuration["masked_levels"])
    if target_levels != (1, 2):
        raise G1SensitivityError("G1c requires the frozen level 1+2 checkpoint")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if args.require_cuda and device.type != "cuda":
        raise G1SensitivityError("CUDA was required but is unavailable")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("completed_updates") != training_manifest["configuration"]["updates"]:
        raise G1SensitivityError("checkpoint update count differs from manifest")
    model = LevelAwareMotifStateEncoder(
        num_e3fp_embeddings=4096,
        embedding_dim=int(configuration["embedding_dim"]),
        hidden_dim=int(configuration["hidden_dim"]),
        pooling=str(training_manifest["pooling"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    dataset = PF1MotifStateDataset(args.paired_release, "dev")
    records = [dataset[index] for index in range(len(dataset))]
    dataset.close()
    donor_map, excluded = same_size_derangement(records)
    results = _evaluate(
        model,
        records,
        donor_map,
        device=device,
        batch_size=int(args.batch_size),
        mask_probability=float(configuration["mask_probability"]),
        mask_seed=int(configuration["dev_mask_seed"]),
        target_levels=target_levels,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "frozen_g1b_aligned_vs_same_atom_count_shuffled_state",
        "training_manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "paired_release": str(Path(args.paired_release).expanduser().resolve()),
        "device": str(device),
        "dev_members": len(records),
        "eligible_members": len(donor_map),
        "excluded_singleton_members": [records[index]["member_id"] for index in excluded],
        "same_atom_count": all(
            len(records[index]["e3fp_ids"]) == len(records[donor]["e3fp_ids"])
            for index, donor in donor_map.items()
        ),
        "no_self_pairs": all(index != donor for index, donor in donor_map.items()),
        "target_levels": list(target_levels),
        "results": results,
        "decision_boundary": (
            "positive shuffled-minus-aligned NLL at both levels is required before "
            "bridging the motif geometry carrier into T5"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", required=True)
    parser.add_argument("--g1-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def main(argv=None):
    print(json.dumps(run(build_parser().parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
