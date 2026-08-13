"""Paired B2D/F3D V3 screen with same-identity motif-state matching.

This is a mechanism screen, not formal pretraining.  It keeps GraphPorts,
stock T5, and the E3FP/Morgan providers unchanged.  A disposable auxiliary
head asks each post-T5 motif carrier to retrieve its own atom-derived state
among cross-molecule motifs having the same exact identity.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

import torch
from torch import nn

from most_t5_next.p1.build_pf1_paired_release_v1 import (
    DONOR_ATOM_MAP_NAME,
    PF1PairedReleaseReader,
)
from most_t5_next.p1.pf1_optimization import (
    PF1LearningRateSchedule,
    PF1OptimizationProtocol,
    build_pf1_optimizer,
    clip_pf1_gradients,
)
from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)

from .build_pf10_morgan_overlay_v1 import MorganAtomStateProvider
from .factorized_model_init_v3 import (
    factorized_initialization_contract_v3,
    load_deterministic_factorized_model_v3,
)
from .factorized_view_collator_v2 import GraphPortsCanonicalAtomAddressProvider
from .motif_state_matching_v3 import MATCHING_ID, MotifStateMatchingHeadV3
from .run_pf10_3d_motif_v3_short_v1 import (
    DEV_RECORD_COUNT,
    NUM_E3FP_EMBEDDINGS,
    UNION_GEOMETRY_FUSION_SEED,
    _TrainCursor,
    _collate,
    _load_dev_prefix,
    _motif_records,
)


SCHEMA_VERSION = "most-t5-p2/pf10-3d-motif-v3-matching-screen/v1"
TRAIN_SEED = 20260815
DEV_SEED = 20260816
ADAPTER_SEED = 20260812
MATCHING_HEAD_SEED = 20260817
GEOMETRY_FRACTION = 0.15
MATCHING_LOSS_WEIGHT = 0.25
EVALUATION_UPDATES = (0, 500, 1000)
VIEW_CYCLE = ("m_plus_g", "g_only", "m_only", "m_plus_g")
PROTOCOL = PF1OptimizationProtocol(
    base_learning_rate=1.0e-4,
    warmup_updates=100,
    total_updates=1000,
    final_learning_rate=1.0e-5,
    warmup_start_factor=0.1,
    gradient_clip_norm=1.0,
    weight_decay=0.0,
    micro_batch_size=32,
    gradient_accumulation_steps=4,
    beta1=0.9,
    beta2=0.999,
    epsilon=1.0e-6,
)


class ThreeDMotifV3MatchingError(RuntimeError):
    """The paired matching screen failed its execution contract."""


def view_for_update(update: int) -> str:
    if isinstance(update, bool) or not isinstance(update, int) or update <= 0:
        raise ThreeDMotifV3MatchingError("update must be positive")
    return VIEW_CYCLE[(update - 1) % len(VIEW_CYCLE)]


def _autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _identities(rows: Sequence[Any]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(record.exact_identity_sha256) for record in _motif_records(rows))


def _eligible_anchor_count(identity_rows: Sequence[Sequence[str]]) -> int:
    occurrences: dict[str, list[int]] = {}
    for batch_index, identities in enumerate(identity_rows):
        for identity in identities:
            occurrences.setdefault(identity, []).append(batch_index)
    return sum(
        1
        for batch_index, identities in enumerate(identity_rows)
        for identity in identities
        if any(other != batch_index for other in occurrences[identity])
    )


def _forward(model: Any, batch: Any, *, memory_mode: str) -> Any:
    inputs = batch.model_inputs()
    inputs["state_memory_mode"] = memory_mode
    inputs["use_cache"] = False
    output = model(**inputs)
    loss = getattr(output, "loss", None)
    logits = getattr(getattr(output, "t5_output", None), "logits", None)
    if (
        not isinstance(loss, torch.Tensor)
        or loss.ndim != 0
        or not bool(torch.isfinite(loss))
        or not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
    ):
        raise ThreeDMotifV3MatchingError("V3 forward omitted finite CE/logits")
    return output


def _matching(head: MotifStateMatchingHeadV3, output: Any, batch: Any, identities):
    return head(
        encoder_hidden=output.encoder_last_hidden_state,
        motif_state=output.adapter_encoding.pre_t5_motif_context,
        motif_mask=batch.motif_mask,
        motif_to_carrier=batch.motif_to_carrier,
        exact_identity_sha256=identities,
    )


def _evaluate_view(
    model: nn.Module,
    head: MotifStateMatchingHeadV3,
    *,
    rows: Sequence[Any],
    view_id: str,
    tokenizer: Any,
    addresses: Any,
    provider: Any,
    device: torch.device,
    memory_mode: str,
) -> dict[str, object]:
    model.eval()
    head.eval()
    nll_sum = 0.0
    target_tokens = 0
    correct_tokens = 0
    matching_loss_sum = 0.0
    matching_anchors = 0
    matching_top1_credit = 0.0
    matching_positive_probability = 0.0
    matching_uniform_loss_sum = 0.0
    matching_groups = 0
    with torch.no_grad():
        for start in range(0, len(rows), PROTOCOL.micro_batch_size):
            selected = rows[start : start + PROTOCOL.micro_batch_size]
            batch = _collate(
                selected,
                view_id=view_id,
                tokenizer=tokenizer,
                addresses=addresses,
                provider=provider,
                seed=DEV_SEED,
                epoch=0,
                device=device,
            )
            labels = batch.labels
            if labels is None:
                raise ThreeDMotifV3MatchingError("dev batch lacks labels")
            count = int((labels != -100).sum().item())
            with _autocast(device):
                output = _forward(model, batch, memory_mode=memory_mode)
                matched = (
                    None
                    if view_id == "m_only"
                    else _matching(head, output, batch, _identities(selected))
                )
            logits = output.t5_output.logits
            mask = labels != -100
            correct_tokens += int((logits.argmax(-1)[mask] == labels[mask]).sum().item())
            nll_sum += float(output.loss.detach().float().cpu().item()) * count
            target_tokens += count
            if matched is not None and matched.eligible_anchors:
                matching_loss_sum += (
                    float(matched.loss.detach().float().cpu().item())
                    * matched.eligible_anchors
                )
                matching_anchors += matched.eligible_anchors
                matching_top1_credit += matched.tie_aware_top1_credit_sum
                matching_positive_probability += matched.positive_probability_sum
                matching_uniform_loss_sum += matched.uniform_loss_sum
                matching_groups += matched.eligible_identity_groups
    return {
        "view_id": view_id,
        "state_memory_mode": memory_mode,
        "members": len(rows),
        "target_tokens": target_tokens,
        "token_weighted_nll": nll_sum / target_tokens,
        "masked_token_accuracy": correct_tokens / target_tokens,
        "matching_eligible_anchors": matching_anchors,
        "matching_eligible_identity_groups_batch_sum": matching_groups,
        "matching_loss": (
            matching_loss_sum / matching_anchors if matching_anchors else None
        ),
        "matching_accuracy": (
            matching_top1_credit / matching_anchors if matching_anchors else None
        ),
        "matching_mean_positive_probability": (
            matching_positive_probability / matching_anchors
            if matching_anchors
            else None
        ),
        "matching_uniform_loss": (
            matching_uniform_loss_sum / matching_anchors
            if matching_anchors
            else None
        ),
        "matching_excess_loss_over_uniform": (
            (matching_loss_sum - matching_uniform_loss_sum) / matching_anchors
            if matching_anchors
            else None
        ),
    }


def evaluate(
    model: nn.Module,
    head: MotifStateMatchingHeadV3,
    *,
    rows: Sequence[Any],
    tokenizer: Any,
    addresses: Any,
    provider: Any,
    device: torch.device,
) -> dict[str, object]:
    reports = []
    for view_id in ("m_only", "m_plus_g", "g_only"):
        aligned = _evaluate_view(
            model,
            head,
            rows=rows,
            view_id=view_id,
            tokenizer=tokenizer,
            addresses=addresses,
            provider=provider,
            device=device,
            memory_mode="zero" if view_id == "m_only" else "aligned",
        )
        reports.append(aligned)
        if view_id != "m_only":
            zero = _evaluate_view(
                model,
                head,
                rows=rows,
                view_id=view_id,
                tokenizer=tokenizer,
                addresses=addresses,
                provider=provider,
                device=device,
                memory_mode="zero",
            )
            zero["zero_minus_aligned_delta_nll"] = (
                float(zero["token_weighted_nll"])
                - float(aligned["token_weighted_nll"])
            )
            if zero["matching_loss"] is not None and aligned["matching_loss"] is not None:
                zero["zero_minus_aligned_delta_matching_loss"] = (
                    float(zero["matching_loss"]) - float(aligned["matching_loss"])
                )
            reports.append(zero)
    return {"dev_records": len(rows), "views": reports}


def run_cell(
    *,
    cell: str,
    reader: PF1PairedReleaseReader,
    tokenizer: Any,
    addresses: Any,
    provider: Any,
    model: nn.Module,
    matching_head: MotifStateMatchingHeadV3,
    output_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    if cell not in {"B2D", "F3D"}:
        raise ThreeDMotifV3MatchingError("matching screen supports B2D/F3D only")
    output_dir.mkdir(parents=True, exist_ok=False)
    model.to(device)
    matching_head.to(device)
    trainable = nn.ModuleDict({"model": model, "matching_head": matching_head})
    optimizer = build_pf1_optimizer(trainable, PROTOCOL)
    scheduler = PF1LearningRateSchedule(optimizer, PROTOCOL)
    cursor = _TrainCursor(reader, PROTOCOL.micro_batch_size)
    dev_rows = _load_dev_prefix(reader)
    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    evaluations = [{
        "update": 0,
        **evaluate(
            model,
            matching_head,
            rows=dev_rows,
            tokenizer=tokenizer,
            addresses=addresses,
            provider=provider,
            device=device,
        ),
    }]
    view_updates = {"m_only": 0, "m_plus_g": 0, "g_only": 0}
    matching_anchor_count = 0
    matching_active_updates = 0
    preclip_norms: list[float] = []
    clipped_updates = 0
    members_seen = 0
    started = time.perf_counter()

    for update in range(1, PROTOCOL.total_updates + 1):
        view_id = view_for_update(update)
        model.train()
        matching_head.train()
        batches = []
        identities = []
        token_counts = []
        anchor_counts = []
        update_members = 0
        for _ in range(PROTOCOL.gradient_accumulation_steps):
            epoch, selected = cursor.next()
            update_members += len(selected)
            batch = _collate(
                selected,
                view_id=view_id,
                tokenizer=tokenizer,
                addresses=addresses,
                provider=provider,
                seed=TRAIN_SEED,
                epoch=epoch,
                device=device,
            )
            if batch.labels is None:
                raise ThreeDMotifV3MatchingError("train view lacks labels")
            identity_rows = _identities(selected)
            batches.append(batch)
            identities.append(identity_rows)
            token_counts.append(int((batch.labels != -100).sum().item()))
            anchor_counts.append(
                0 if view_id == "m_only" else _eligible_anchor_count(identity_rows)
            )
        total_tokens = sum(token_counts)
        total_anchors = sum(anchor_counts)
        optimizer.zero_grad(set_to_none=True)
        for batch, identity_rows, token_count, expected_anchors in zip(
            batches, identities, token_counts, anchor_counts
        ):
            with _autocast(device):
                output = _forward(
                    model,
                    batch,
                    memory_mode="zero" if view_id == "m_only" else "aligned",
                )
                loss = output.loss * (token_count / total_tokens)
                if view_id != "m_only":
                    matched = _matching(matching_head, output, batch, identity_rows)
                    if matched.eligible_anchors != expected_anchors:
                        raise ThreeDMotifV3MatchingError(
                            "matching coverage changed after model forward"
                        )
                    if total_anchors:
                        loss = loss + (
                            MATCHING_LOSS_WEIGHT
                            * matched.loss
                            * (matched.eligible_anchors / total_anchors)
                        )
            loss.backward()
        preclip = clip_pf1_gradients(trainable, PROTOCOL)
        if not math.isfinite(preclip):
            raise ThreeDMotifV3MatchingError("gradient norm is non-finite")
        preclip_norms.append(preclip)
        clipped_updates += int(preclip > PROTOCOL.gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        members_seen += update_members
        view_updates[view_id] += 1
        matching_anchor_count += total_anchors
        matching_active_updates += int(total_anchors > 0)

        if update in EVALUATION_UPDATES:
            evaluations.append({
                "update": update,
                **evaluate(
                    model,
                    matching_head,
                    rows=dev_rows,
                    tokenizer=tokenizer,
                    addresses=addresses,
                    provider=provider,
                    device=device,
                ),
            })

    wall_seconds = time.perf_counter() - started
    checkpoint = output_dir / "model_and_matching_state.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "cell": cell,
            "completed_updates": PROTOCOL.total_updates,
            "model_state_dict": model.state_dict(),
            "matching_head_state_dict": matching_head.state_dict(),
        },
        checkpoint,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "paired_geometry_mechanism_screen_not_formal_pretraining",
        "cell": cell,
        "state_kind": "e3fp" if provider is None else str(provider.state_kind),
        "protocol": asdict(PROTOCOL),
        "view_cycle": list(VIEW_CYCLE),
        "view_updates": view_updates,
        "geometry_fraction": GEOMETRY_FRACTION,
        "matching_contract": MATCHING_ID,
        "matching_loss_weight": MATCHING_LOSS_WEIGHT,
        "matching_eligible_anchors_train": matching_anchor_count,
        "matching_active_updates": matching_active_updates,
        "members_seen": members_seen,
        "evaluations": evaluations,
        "mean_preclip_gradient_norm": statistics.fmean(preclip_norms),
        "max_preclip_gradient_norm": max(preclip_norms),
        "clip_rate": clipped_updates / PROTOCOL.total_updates,
        "wall_seconds": wall_seconds,
        "members_per_second": members_seen / wall_seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "checkpoint": str(checkpoint),
        "scientific_boundary": (
            "same_identity_cross_record_matching_removes_the_direct_motif_name_shortcut; "
            "F3D_vs_B2D_is_the_primary_comparison; no_multi_conformer_claim"
        ),
    }
    (output_dir / "matching_screen_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _new_matching_head(hidden_size: int) -> MotifStateMatchingHeadV3:
    state = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(MATCHING_HEAD_SEED)
        return MotifStateMatchingHeadV3(
            hidden_size=hidden_size,
            projection_dim=128,
            temperature=0.1,
        )
    finally:
        torch.random.set_rng_state(state)


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ThreeDMotifV3MatchingError("one CUDA BF16 device is required")
    paired = Path(args.paired_release).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise ThreeDMotifV3MatchingError("output directory already exists")
    tokenizer = load_verified_canary_union_tokenizer(
        base_snapshot=Path(args.base_tokenizer_snapshot),
        output_dir=paired / "union_tokenizer",
    )
    reader = PF1PairedReleaseReader(paired)
    cache = reader.warm_decoded_record_cache(
        workers=int(args.cache_workers),
        max_pending=int(args.cache_workers) * 4,
    )
    addresses = GraphPortsCanonicalAtomAddressProvider(paired / DONOR_ATOM_MAP_NAME)
    provider = None
    if args.cell == "B2D":
        if args.morgan_overlay is None:
            raise ThreeDMotifV3MatchingError("B2D requires --morgan-overlay")
        provider = MorganAtomStateProvider(Path(args.morgan_overlay))
    elif args.morgan_overlay is not None:
        raise ThreeDMotifV3MatchingError("only B2D accepts --morgan-overlay")
    contract = factorized_initialization_contract_v3(
        adapter_seed=ADAPTER_SEED,
        num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
        state_level2_weight=0.25,
        state_embedding_dim=64,
        atom_memory_dim=128,
        max_identity_span_length=128,
        max_atoms_per_motif=128,
        geometry_fraction=GEOMETRY_FRACTION,
    )
    try:
        model = load_deterministic_factorized_model_v3(
            base_model_snapshot=Path(args.base_model_snapshot),
            base_tokenizer_snapshot=Path(args.base_tokenizer_snapshot),
            union_tokenizer_dir=paired / "union_tokenizer",
            union_init_dir=Path(args.union_init_dir),
            union_geometry_fusion_seed=UNION_GEOMETRY_FUSION_SEED,
            adapter_seed=ADAPTER_SEED,
            num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
            geometry_fraction=GEOMETRY_FRACTION,
        )
        hidden_size = int(model.get_input_embeddings().weight.shape[1])
        report = run_cell(
            cell=args.cell,
            reader=reader,
            tokenizer=tokenizer.runtime,
            addresses=addresses,
            provider=provider,
            model=model,
            matching_head=_new_matching_head(hidden_size),
            output_dir=output_dir,
            device=torch.device("cuda:0"),
        )
        report["initialization_contract"] = contract
        report["matching_head_seed"] = MATCHING_HEAD_SEED
        report["decoded_record_cache_warmup"] = cache
        (output_dir / "matching_screen_manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        if provider is not None:
            provider.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("B2D", "F3D"), required=True)
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--morgan-overlay", type=Path)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    report = run_cli(_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "cell": report["cell"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GEOMETRY_FRACTION",
    "MATCHING_LOSS_WEIGHT",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "ThreeDMotifV3MatchingError",
    "VIEW_CYCLE",
    "_eligible_anchor_count",
    "run_cell",
    "run_cli",
    "view_for_update",
]
