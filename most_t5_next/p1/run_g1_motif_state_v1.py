#!/usr/bin/env python3
"""Run the standalone Deep-Sets/gated E3FP masked-state G1 screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, Iterable, Mapping

import torch
from torch.utils.data import DataLoader

from most_t5_next.p1.level_aware_motif_state_v1 import (
    LevelAwareMotifStateEncoder,
    STATE_ENCODER_VERSION,
    build_masked_e3fp_state_batch,
    masked_state_ce,
)
from most_t5_next.p1.motif_state_data_v1 import (
    DATASET_VERSION,
    PF1MotifStateDataset,
    collate_motif_state_records,
)


RUN_SCHEMA = "most-t5-p1/g1-motif-state-screen/v1"


class G1RunnerError(ValueError):
    pass


def _loader(
    dataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    persistent_workers: bool = True,
):
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(workers),
        "collate_fn": collate_motif_state_records,
        "drop_last": False,
        "pin_memory": bool(pin_memory),
        "generator": torch.Generator().manual_seed(int(seed)),
    }
    if int(workers) > 0:
        kwargs.update(
            persistent_workers=bool(persistent_workers),
            prefetch_factor=2,
        )
    return DataLoader(**kwargs)


def _train_unigram_log_probs(loader: Iterable[Mapping[str, Any]], classes: int):
    counts = torch.ones((4, int(classes)), dtype=torch.float64)
    observations = torch.full((4,), int(classes), dtype=torch.long)
    for batch in loader:
        ids = batch["e3fp_ids"]
        valid = batch["atom_valid"].unsqueeze(-1) & (ids >= 0)
        for level in range(1, 4):
            values = ids[..., level][valid[..., level]]
            counts[level] += torch.bincount(values, minlength=int(classes)).to(torch.float64)
            observations[level] += values.numel()
    log_probs = torch.log(counts / counts.sum(dim=-1, keepdim=True))
    return log_probs, observations


@torch.no_grad()
def _evaluate(
    model,
    loader,
    *,
    device,
    mask_seed: int,
    mask_probability: float,
    unigram_log_probs,
    target_levels,
):
    model.eval()
    totals = {
        level: {"nll_sum": 0.0, "correct": 0, "count": 0, "unigram_nll_sum": 0.0}
        for level in range(1, 4)
    }
    for batch_index, batch in enumerate(loader):
        masked = build_masked_e3fp_state_batch(
            batch["e3fp_ids"],
            batch["atom_valid"],
            mask_token_id=model.mask_token_id,
            probability=float(mask_probability),
            seed=int(mask_seed) + int(batch_index),
            target_levels=tuple(int(level) for level in target_levels),
        )
        ids = masked.corrupted_ids.to(device, non_blocking=True)
        valid = batch["atom_valid"].to(device, non_blocking=True)
        groups = batch["atom_to_motif"].to(device, non_blocking=True)
        roles = batch["atom_is_attachment"].to(device, non_blocking=True)
        target_ids = masked.target_ids.to(device, non_blocking=True)
        target_mask = masked.target_mask.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                ids,
                valid,
                groups,
                num_groups=int(batch["num_groups"]),
                atom_is_attachment=roles,
            )
            _, metrics = masked_state_ce(output.logits, target_ids, target_mask)
        for level, row in metrics.items():
            if level == 0:
                continue
            local_targets = masked.target_ids[..., level][masked.target_mask[..., level]]
            totals[level]["nll_sum"] += float(row["nll_sum"].detach().float().cpu())
            totals[level]["correct"] += int(row["correct"].detach().cpu())
            totals[level]["count"] += int(row["count"].detach().cpu())
            totals[level]["unigram_nll_sum"] += float(
                (-unigram_log_probs[level, local_targets]).sum()
            )
    report = {}
    uniform_nll = math.log(float(model.num_e3fp_embeddings))
    selected_levels = {int(level) for level in target_levels}
    for level, row in totals.items():
        if level not in selected_levels:
            continue
        count = int(row["count"])
        unigram_nll = row["unigram_nll_sum"] / count
        model_nll = row["nll_sum"] / count
        best_static_prior_nll = min(unigram_nll, uniform_nll)
        report[str(level)] = {
            "count": count,
            "nll": model_nll,
            "accuracy": row["correct"] / count,
            "unigram_nll": unigram_nll,
            "uniform_nll": uniform_nll,
            "best_static_prior_nll": best_static_prior_nll,
            "improvement_over_unigram_nll": unigram_nll - model_nll,
            "improvement_over_best_static_prior_nll": best_static_prior_nll
            - model_nll,
        }
    return report


def run(args) -> Dict[str, Any]:
    total_start = time.time()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise G1RunnerError("output directory already exists: {}".format(output_dir))
    output_dir.mkdir(parents=True)
    if args.batch_size <= 0 or args.updates <= 0 or args.workers < 0:
        raise G1RunnerError("batch-size/updates must be positive and workers non-negative")
    target_levels = tuple(int(level) for level in args.target_levels)
    if (
        not target_levels
        or len(set(target_levels)) != len(target_levels)
        or any(level not in (1, 2, 3) for level in target_levels)
    ):
        raise G1RunnerError("target-levels must be unique values drawn from 1, 2, 3")

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if args.require_cuda and device.type != "cuda":
        raise G1RunnerError("CUDA was required but is unavailable")

    train_dataset = PF1MotifStateDataset(args.paired_release, "train")
    dev_dataset = PF1MotifStateDataset(args.paired_release, "dev")
    census_loader = _loader(
        train_dataset,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        shuffle=False,
        seed=int(args.seed),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    census_start = time.time()
    unigram_log_probs, train_observations = _train_unigram_log_probs(census_loader, 4096)
    census_seconds = time.time() - census_start
    train_loader = _loader(
        train_dataset,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        shuffle=True,
        seed=int(args.data_seed),
        pin_memory=device.type == "cuda",
    )
    dev_loader = _loader(
        dev_dataset,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        shuffle=False,
        seed=int(args.data_seed),
        pin_memory=device.type == "cuda",
    )

    model = LevelAwareMotifStateEncoder(
        num_e3fp_embeddings=4096,
        embedding_dim=int(args.embedding_dim),
        hidden_dim=int(args.hidden_dim),
        pooling=str(args.pooling),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=0.0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    eval_updates = {0, int(args.updates) // 2, int(args.updates)}
    evaluations = {
        "0": _evaluate(
            model,
            dev_loader,
            device=device,
            mask_seed=int(args.dev_mask_seed),
            mask_probability=float(args.mask_probability),
            unigram_log_probs=unigram_log_probs,
            target_levels=target_levels,
        )
    }
    iterator = iter(train_loader)
    epoch = 0
    batch_sizes = []
    losses = []
    start = time.time()
    model.train()
    for update in range(1, int(args.updates) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            iterator = iter(train_loader)
            batch = next(iterator)
        batch_sizes.append(len(batch["record_ids"]))
        masked = build_masked_e3fp_state_batch(
            batch["e3fp_ids"],
            batch["atom_valid"],
            mask_token_id=model.mask_token_id,
            probability=float(args.mask_probability),
            seed=int(args.train_mask_seed) + update,
            target_levels=target_levels,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                masked.corrupted_ids.to(device, non_blocking=True),
                batch["atom_valid"].to(device, non_blocking=True),
                batch["atom_to_motif"].to(device, non_blocking=True),
                num_groups=int(batch["num_groups"]),
                atom_is_attachment=batch["atom_is_attachment"].to(
                    device, non_blocking=True
                ),
            )
            loss, _ = masked_state_ce(
                output.logits,
                masked.target_ids.to(device, non_blocking=True),
                masked.target_mask.to(device, non_blocking=True),
            )
        if not bool(torch.isfinite(loss)):
            raise G1RunnerError("non-finite training loss at update {}".format(update))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().float().cpu()))
        if update in eval_updates:
            evaluations[str(update)] = _evaluate(
                model,
                dev_loader,
                device=device,
                mask_seed=int(args.dev_mask_seed),
                mask_probability=float(args.mask_probability),
                unigram_log_probs=unigram_log_probs,
                target_levels=target_levels,
            )
            model.train()

    torch.save(
        {
            "schema_version": RUN_SCHEMA,
            "pooling": str(args.pooling),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "completed_updates": int(args.updates),
        },
        output_dir / "final_state.pt",
    )
    elapsed = time.time() - start
    manifest = {
        "schema_version": RUN_SCHEMA,
        "status": "pass",
        "scope": "standalone_masked_e3fp_state_mechanism_screen",
        "not_t5_or_downstream_performance": True,
        "encoder_version": STATE_ENCODER_VERSION,
        "dataset_version": DATASET_VERSION,
        "paired_release": str(Path(args.paired_release).resolve()),
        "pooling": str(args.pooling),
        "configuration": {
            "seed": int(args.seed),
            "data_seed": int(args.data_seed),
            "train_mask_seed": int(args.train_mask_seed),
            "dev_mask_seed": int(args.dev_mask_seed),
            "updates": int(args.updates),
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "prefetch_factor": 2 if args.workers > 0 else None,
            "persistent_workers": bool(args.workers > 0),
            "drop_last": False,
            "mask_probability": float(args.mask_probability),
            "masked_levels": list(target_levels),
            "atom_role": "frozen_core_vs_attachment",
            "embedding_dim": int(args.embedding_dim),
            "hidden_dim": int(args.hidden_dim),
            "learning_rate": float(args.learning_rate),
            "device": str(device),
        },
        "data": {
            "train_members": len(train_dataset),
            "dev_members": len(dev_dataset),
            "train_level_observations_with_laplace_prior": {
                str(level): int(train_observations[level]) for level in range(1, 4)
            },
        },
        "train": {
            "loss_initial": losses[0],
            "loss_final": losses[-1],
            "loss_min": min(losses),
            "epochs_entered": epoch + 1,
            "members_per_update_min": min(batch_sizes),
            "members_per_update_max": max(batch_sizes),
            "members_per_update_mean": sum(batch_sizes) / len(batch_sizes),
            "wall_seconds": elapsed,
            "members_per_second": sum(batch_sizes) / elapsed,
            "peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
            "peak_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
            ),
        },
        "input_pipeline": {
            "full_train_unigram_census_seconds": census_seconds,
            "workers_per_process": int(args.workers),
            "multiworker_lmdb_decode": bool(args.workers > 0),
        },
        "total_wall_seconds": time.time() - total_start,
        "evaluations": evaluations,
        "decision_boundary": (
            "compare level-wise NLL with the better of train-unigram and uniform priors; "
            "compare gated and deep_sets under the same seeds before any T5 bridge"
        ),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")
    train_dataset.close()
    dev_dataset.close()
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pooling", choices=("deep_sets", "gated"), required=True)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--target-levels", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--data-seed", type=int, default=20260809)
    parser.add_argument("--train-mask-seed", type=int, default=20260810)
    parser.add_argument("--dev-mask-seed", type=int, default=20260811)
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def main(argv=None):
    manifest = run(build_parser().parse_args(argv))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
