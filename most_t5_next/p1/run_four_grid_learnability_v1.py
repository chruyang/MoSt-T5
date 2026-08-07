#!/usr/bin/env python3
"""Run a fixed-minibatch A0/A1/M0/M1 learnability smoke on one GPU.

This runner starts every condition from an independent load of the same
published union initialization, reuses the paired-canary membership and
epoch-0 corruption, and performs a short AdamW overfit on that one frozen
mini-batch.  It answers only whether each plumbing path can take finite
optimization steps and reduce its own loss.  A and M losses are not directly
comparable because their masking units and target sequences differ.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime as dt
import gc
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

from most_t5_next.p1.build_union_init_checkpoint_v1 import (
    load_verified_four_grid_wrapper,
)
from most_t5_next.p1.run_four_grid_gpu_canary_v1 import (
    CONDITION_ORDER,
    CORRUPTION_EPOCH,
    CORRUPTION_SEED,
    FORWARD_SEED,
    MASK_PROBABILITY,
    _gradient_statistics,
    build_frozen_grid_batches,
    load_frozen_minibatch,
)
from most_t5_next.p1.training_adapter import (
    select_four_grid_forward_inputs,
    to_four_grid_batch_encoding,
)
from most_t5_next.r1.adapter.build_p1_paired_canary_v1 import TOKENIZER_DIRECTORY
from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)


REPORT_SCHEMA = "most-t5-p1/four-grid-learnability/v1"
REPORT_NAME = "learnability_manifest.json"
DEFAULT_BATCH_SIZE = 8
DEFAULT_STEPS = 20
DEFAULT_LEARNING_RATE = 5e-4


class FourGridLearnabilityError(RuntimeError):
    """The fixed-minibatch learnability smoke could not be completed."""


def _restart_forward_seed(torch_module: Any, device: Any) -> None:
    """Repeat one dropout realization at every curve point."""

    torch_module.manual_seed(FORWARD_SEED)
    if device.type == "cuda":
        torch_module.cuda.manual_seed_all(FORWARD_SEED)


def _loss_value(torch_module: Any, loss: Any, condition_id: str) -> float:
    if loss is None or loss.ndim != 0 or not bool(torch_module.isfinite(loss).item()):
        raise FourGridLearnabilityError(
            f"{condition_id} produced an absent or non-finite CE loss"
        )
    return float(loss.detach().float().cpu().item())


def _synchronize(torch_module: Any, device: Any) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)


def execute_four_grid_learnability(
    batches: Mapping[str, Any],
    *,
    base_model_snapshot: Path,
    base_tokenizer_snapshot: Path,
    union_tokenizer_dir: Path,
    union_init_dir: Path,
    geometry_fusion_seed: int,
    num_e3fp_embeddings: int,
    expected_vocab_size: int,
    device: Any,
    steps: int,
    learning_rate: float,
    use_bf16: bool,
    torch_module: Any,
    wrapper_loader: Callable[..., Any] = load_verified_four_grid_wrapper,
) -> list[dict[str, object]]:
    """Sequentially overfit each independently loaded grid cell."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise FourGridLearnabilityError("steps must be a positive integer")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
    ):
        raise FourGridLearnabilityError("learning_rate must be finite and positive")

    results: list[dict[str, object]] = []
    for condition_id in CONDITION_ORDER:
        if condition_id not in batches:
            raise FourGridLearnabilityError(f"missing frozen batch {condition_id}")

        model = None
        optimizer = None
        encoded = None
        forward_inputs = None
        outputs = None
        loss = None
        try:
            gc.collect()
            if device.type == "cuda":
                torch_module.cuda.empty_cache()
                torch_module.cuda.reset_peak_memory_stats(device)

            model = wrapper_loader(
                condition_id=condition_id,
                base_model_snapshot=base_model_snapshot,
                base_tokenizer_snapshot=base_tokenizer_snapshot,
                union_tokenizer_dir=union_tokenizer_dir,
                output_dir=union_init_dir,
                geometry_fusion_seed=geometry_fusion_seed,
                num_e3fp_embeddings=num_e3fp_embeddings,
            )
            if int(model.config.vocab_size) != expected_vocab_size:
                raise FourGridLearnabilityError(
                    "verified wrapper vocabulary differs from the paired tokenizer"
                )
            model.to(device)
            model.train()
            optimizer = torch_module.optim.AdamW(
                model.parameters(),
                lr=float(learning_rate),
                weight_decay=0.0,
            )

            batch = batches[condition_id]
            encoded = to_four_grid_batch_encoding(batch, device=device)
            forward_inputs = select_four_grid_forward_inputs(encoded)
            loss_curve: list[float] = []
            gradient_norm_curve: list[float] = []
            gradient_tensor_count_curve: list[int] = []
            step_time_seconds: list[float] = []

            for _step_index in range(steps):
                optimizer.zero_grad(set_to_none=True)
                _restart_forward_seed(torch_module, device)
                _synchronize(torch_module, device)
                started = time.perf_counter()
                autocast_context = (
                    torch_module.autocast(
                        device_type="cuda", dtype=torch_module.bfloat16
                    )
                    if use_bf16
                    else nullcontext()
                )
                with autocast_context:
                    outputs = model(
                        **forward_inputs,
                        use_cache=False,
                        return_dict=True,
                    )
                    loss = outputs.loss
                loss_curve.append(_loss_value(torch_module, loss, condition_id))
                loss.backward()
                try:
                    gradient_norm, gradient_tensor_count = _gradient_statistics(
                        torch_module, model
                    )
                except RuntimeError as exc:
                    raise FourGridLearnabilityError(str(exc)) from exc
                gradient_norm_curve.append(gradient_norm)
                gradient_tensor_count_curve.append(gradient_tensor_count)
                optimizer.step()
                _synchronize(torch_module, device)
                step_time_seconds.append(time.perf_counter() - started)

            # Curve index k denotes the loss after exactly k optimizer updates.
            optimizer.zero_grad(set_to_none=True)
            _restart_forward_seed(torch_module, device)
            with torch_module.no_grad():
                autocast_context = (
                    torch_module.autocast(
                        device_type="cuda", dtype=torch_module.bfloat16
                    )
                    if use_bf16
                    else nullcontext()
                )
                with autocast_context:
                    outputs = model(
                        **forward_inputs,
                        use_cache=False,
                        return_dict=True,
                    )
                    loss = outputs.loss
                loss_curve.append(_loss_value(torch_module, loss, condition_id))

            initial_loss = loss_curve[0]
            final_loss = loss_curve[-1]
            minimum_loss = min(loss_curve)
            minimum_step = loss_curve.index(minimum_loss)
            relative_drop = (initial_loss - final_loss) / initial_loss
            initial_to_final_decrease = final_loss < initial_loss
            minimum_below_initial = minimum_loss < initial_loss
            peak_bytes = (
                int(torch_module.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )
            results.append(
                {
                    "condition": condition_id,
                    "member_ids": list(batch.ce_batch.record_ids),
                    "input_shape": list(forward_inputs["input_ids"].shape),
                    "target_shape": list(forward_inputs["labels"].shape),
                    "geometry_enabled": batch.geometry is not None,
                    "optimizer_steps": steps,
                    "loss_curve_updates_applied": list(range(steps + 1)),
                    "loss_curve": loss_curve,
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "minimum_loss": minimum_loss,
                    "minimum_loss_step": minimum_step,
                    "relative_initial_to_final_loss_drop": relative_drop,
                    "initial_to_final_loss_decreased": initial_to_final_decrease,
                    "minimum_loss_below_initial": minimum_below_initial,
                    "learnability_smoke_pass": (
                        initial_to_final_decrease and minimum_below_initial
                    ),
                    "gradient_norm_curve": gradient_norm_curve,
                    "gradient_tensor_count_curve": gradient_tensor_count_curve,
                    "gradients_finite": True,
                    "step_time_seconds": step_time_seconds,
                    "mean_step_time_seconds": statistics.fmean(step_time_seconds),
                    "median_step_time_seconds": statistics.median(step_time_seconds),
                    "peak_gpu_memory_bytes": peak_bytes,
                }
            )
        finally:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if model is not None:
                model.zero_grad(set_to_none=True)
            del loss, outputs, forward_inputs, encoded, optimizer, model
            gc.collect()
            if device.type == "cuda":
                torch_module.cuda.empty_cache()
    return results


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the one-GPU fixed-minibatch learnability smoke."""

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import torch
        import transformers
    except ModuleNotFoundError as exc:  # pragma: no cover - integration boundary
        raise FourGridLearnabilityError("PyTorch and Transformers are required") from exc
    if not torch.cuda.is_available():
        raise FourGridLearnabilityError("learnability smoke requires one CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise FourGridLearnabilityError("learnability smoke requires BF16 support")
    if args.batch_size <= 0:
        raise FourGridLearnabilityError("batch_size must be positive")
    if args.steps <= 0:
        raise FourGridLearnabilityError("steps must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise FourGridLearnabilityError("learning_rate must be finite and positive")

    paired_release = Path(args.paired_release).expanduser().resolve()
    base_model_snapshot = Path(args.base_model_snapshot).expanduser().resolve()
    base_tokenizer_snapshot = Path(args.base_tokenizer_snapshot).expanduser().resolve()
    union_init_dir = Path(args.union_init_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FourGridLearnabilityError("output_dir must be a new path")

    records, paired_manifest = load_frozen_minibatch(
        paired_release=paired_release,
        batch_size=args.batch_size,
    )
    union_tokenizer_dir = paired_release / TOKENIZER_DIRECTORY
    tokenizer_build = load_verified_canary_union_tokenizer(
        base_snapshot=base_tokenizer_snapshot,
        output_dir=union_tokenizer_dir,
    )
    batches = build_frozen_grid_batches(
        records, tokenizer_runtime=tokenizer_build.runtime
    )
    device = torch.device("cuda", 0)
    condition_results = execute_four_grid_learnability(
        batches,
        base_model_snapshot=base_model_snapshot,
        base_tokenizer_snapshot=base_tokenizer_snapshot,
        union_tokenizer_dir=union_tokenizer_dir,
        union_init_dir=union_init_dir,
        geometry_fusion_seed=args.geometry_fusion_seed,
        num_e3fp_embeddings=args.num_e3fp_embeddings,
        expected_vocab_size=tokenizer_build.runtime.vocab_size,
        device=device,
        steps=args.steps,
        learning_rate=args.learning_rate,
        use_bf16=True,
        torch_module=torch,
    )

    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "created_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "scope": "fixed_single_minibatch_learnability_smoke_only",
        "interpretation": {
            "all_cells_learnability_smoke_pass": all(
                row["learnability_smoke_pass"] for row in condition_results
            ),
            "architecture_effect_ranking": False,
            "raw_A_M_losses_directly_comparable": False,
            "generalization_claim": False,
            "optimizer_step": True,
            "scheduler": False,
            "training_weights_saved": False,
        },
        "schedule": {
            "source": "first_rows_in_frozen_paired_membership_order",
            "batch_size": args.batch_size,
            "member_ids": [record.atom_record.record_id for record in records],
            "schedule_indices": [record.schedule_index for record in records],
            "corruption_seed": CORRUPTION_SEED,
            "epoch": CORRUPTION_EPOCH,
            "mask_probability": MASK_PROBABILITY,
            "same_frozen_minibatch_repeated_per_step": True,
            "forward_seed_restarted_per_curve_point": FORWARD_SEED,
            "sample_replacement": False,
            "sequence_truncation": False,
        },
        "parity": {
            "A0_A1_CE_batch_equal": True,
            "M0_M1_CE_batch_equal": True,
            "A1_M1_geometry_atom_rows_equal": True,
        },
        "initialization": {
            "one_published_union_init_shared_by_all_conditions": True,
            "independent_verified_load_per_condition": True,
            "tokenizer_vocab_size": tokenizer_build.runtime.vocab_size,
            "vocabulary_expansion_in_runner": False,
            "geometry_fusion_seed": args.geometry_fusion_seed,
            "num_e3fp_embeddings": args.num_e3fp_embeddings,
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "steps": args.steps,
            "scheduler": None,
        },
        "conditions": condition_results,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "visible_gpu_count": torch.cuda.device_count(),
            "precision": "bf16_autocast",
        },
        "inputs": {
            "paired_release_schema": paired_manifest.get("schema_version"),
            "paired_release": str(paired_release),
            "union_init": str(union_init_dir),
            "base_model_snapshot": str(base_model_snapshot),
        },
    }
    output_dir.mkdir(parents=True)
    with (output_dir / REPORT_NAME).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", required=True)
    parser.add_argument("--base-model-snapshot", required=True)
    parser.add_argument("--base-tokenizer-snapshot", required=True)
    parser.add_argument("--union-init-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--geometry-fusion-seed", type=int, required=True)
    parser.add_argument("--num-e3fp-embeddings", type=int, default=4096)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (FourGridLearnabilityError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_STEPS",
    "FourGridLearnabilityError",
    "REPORT_NAME",
    "REPORT_SCHEMA",
    "build_parser",
    "execute_four_grid_learnability",
    "run",
]
