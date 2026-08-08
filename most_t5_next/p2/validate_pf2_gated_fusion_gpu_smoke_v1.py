#!/usr/bin/env python3
"""Validate the real run3 F-Gate boundary before a 1,000-update screen.

The smoke uses one frozen motif minibatch and performs no optimizer update.  It
proves three implementation facts on the actual T5/union-tokenizer/runtime:

* M0 and M1 receive exactly the same CE tensors;
* a zero gate makes the M1 logits and loss bitwise equal to M0 in eval mode;
* the first M1 backward gives the scalar gate a finite nonzero gradient while
  the E3FP embedding table receives an exact zero gradient.

It is a runtime admission test, not a model-quality experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from most_t5_next.p1.build_pf1_paired_release_v1 import (
    PF1PairedReleaseReader,
    TOKENIZER_DIRECTORY,
)
from most_t5_next.p1.run_pf1_four_grid_v1 import collate_pf1_condition
from most_t5_next.p1.training_adapter import (
    select_four_grid_forward_inputs,
    to_four_grid_batch_encoding,
)
from most_t5_next.p2.gated_reference_geometry_fusion_v1 import (
    FUSION_ID,
    ZeroInitGatedE3FPCarrierFusion,
    load_verified_gated_four_grid_wrapper,
)
from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)


REPORT_SCHEMA = "most-t5-p2/zero-init-gated-fusion-gpu-smoke/v1"
SMOKE_BATCH_SIZE = 2
CORRUPTION_SEED = 0
CORRUPTION_EPOCH = 0
FORWARD_SEED = 20260807


class FGateSmokeError(RuntimeError):
    """The real-data zero-gate runtime boundary did not close."""


def _require_same_state(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> None:
    if tuple(left) != tuple(right):
        raise FGateSmokeError("M0/M1 wrapper state keys differ")
    for key in left:
        if not torch.equal(left[key], right[key]):
            raise FGateSmokeError("M0/M1 initialization differs at " + key)


def _require_zero_gate_gradient_boundary(model: Any) -> dict[str, float]:
    fusion = getattr(model, "geometry_fusion", None)
    if not isinstance(fusion, ZeroInitGatedE3FPCarrierFusion):
        raise FGateSmokeError("M1 uses the wrong geometry fusion")
    gate_grad = fusion.geometry_gate_logit.grad
    table_grad = fusion.shared_embedding.weight.grad
    if gate_grad is None or gate_grad.numel() != 1:
        raise FGateSmokeError("zero gate lacks its first-backward gradient")
    gate_value = float(gate_grad.detach().float().cpu().item())
    if not math.isfinite(gate_value) or gate_value == 0.0:
        raise FGateSmokeError("zero gate gradient is non-finite or zero")
    if table_grad is None or not bool(torch.isfinite(table_grad).all().item()):
        raise FGateSmokeError("E3FP table gradient is missing or non-finite")
    table_l1 = float(table_grad.detach().abs().sum().float().cpu().item())
    if table_l1 != 0.0:
        raise FGateSmokeError("E3FP table changed before the scalar gate opened")
    return {
        "geometry_gate_gradient": gate_value,
        "e3fp_table_gradient_l1": table_l1,
    }


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise FGateSmokeError("CUDA BF16 is required")
    paired_release = Path(args.paired_release).expanduser().resolve()
    base_model = Path(args.base_model_snapshot).expanduser().resolve()
    base_tokenizer = Path(args.base_tokenizer_snapshot).expanduser().resolve()
    union_init = Path(args.union_init_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FGateSmokeError("output must be a new path")

    union_dir = paired_release / TOKENIZER_DIRECTORY
    tokenizer = load_verified_canary_union_tokenizer(
        base_snapshot=base_tokenizer,
        output_dir=union_dir,
    )
    reader = PF1PairedReleaseReader(paired_release)
    rows = tuple(
        next(
            reader.iter_train_epoch(
                epoch=CORRUPTION_EPOCH,
                batch_size=SMOKE_BATCH_SIZE,
            )
        )
    )
    if len(rows) != SMOKE_BATCH_SIZE:
        raise FGateSmokeError("real smoke minibatch is incomplete")
    batches = {
        condition: collate_pf1_condition(
            rows,
            condition_id=condition,
            tokenizer_runtime=tokenizer.runtime,
            seed=CORRUPTION_SEED,
            epoch=CORRUPTION_EPOCH,
        )
        for condition in ("M0", "M1")
    }
    if batches["M0"].ce_batch != batches["M1"].ce_batch:
        raise FGateSmokeError("M0/M1 CE tensors differ before the model")

    wrappers = {
        condition: load_verified_gated_four_grid_wrapper(
            condition_id=condition,
            base_model_snapshot=base_model,
            base_tokenizer_snapshot=base_tokenizer,
            union_tokenizer_dir=union_dir,
            output_dir=union_init,
            geometry_fusion_seed=args.geometry_fusion_seed,
            num_e3fp_embeddings=args.num_e3fp_embeddings,
        )
        for condition in ("M0", "M1")
    }
    _require_same_state(wrappers["M0"].state_dict(), wrappers["M1"].state_dict())
    if any(
        float(model.geometry_fusion.geometry_gate_logit.detach().item()) != 0.0
        for model in wrappers.values()
    ):
        raise FGateSmokeError("F-Gate is not initialized at zero")

    device = torch.device("cuda", 0)
    for model in wrappers.values():
        model.to(device).eval()
    encoded = {
        condition: to_four_grid_batch_encoding(batch, device=device)
        for condition, batch in batches.items()
    }
    inputs = {
        condition: select_four_grid_forward_inputs(batch)
        for condition, batch in encoded.items()
    }
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        outputs = {}
        for condition in ("M0", "M1"):
            torch.manual_seed(FORWARD_SEED)
            torch.cuda.manual_seed_all(FORWARD_SEED)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs[condition] = wrappers[condition](
                    **inputs[condition],
                    use_cache=False,
                    return_dict=True,
                )
    if not torch.equal(outputs["M0"].logits, outputs["M1"].logits):
        max_diff = float(
            (outputs["M0"].logits - outputs["M1"].logits)
            .detach()
            .abs()
            .max()
            .float()
            .cpu()
            .item()
        )
        raise FGateSmokeError(
            f"zero-gate M0/M1 logits are not bitwise equal (max diff {max_diff})"
        )
    if not torch.equal(outputs["M0"].loss, outputs["M1"].loss):
        raise FGateSmokeError("zero-gate M0/M1 loss is not bitwise equal")
    loss_value = float(outputs["M1"].loss.detach().float().cpu().item())
    if not math.isfinite(loss_value):
        raise FGateSmokeError("zero-gate loss is non-finite")
    del outputs

    wrappers["M1"].zero_grad(set_to_none=True)
    torch.manual_seed(FORWARD_SEED)
    torch.cuda.manual_seed_all(FORWARD_SEED)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        train_output = wrappers["M1"](
            **inputs["M1"],
            use_cache=False,
            return_dict=True,
        )
    train_output.loss.backward()
    gradient_report = _require_zero_gate_gradient_boundary(wrappers["M1"])

    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "scope": "real_run3_runtime_admission_only_no_optimizer_step",
        "fusion_id": FUSION_ID,
        "members": len(rows),
        "record_ids": list(batches["M0"].ce_batch.record_ids),
        "m0_m1_ce_equal": True,
        "m0_m1_initial_state_equal": True,
        "m0_m1_zero_gate_logits_bitwise_equal": True,
        "m0_m1_zero_gate_loss_bitwise_equal": True,
        "loss": loss_value,
        **gradient_report,
        "optimizer_steps": 0,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "bf16": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", required=True)
    parser.add_argument("--base-model-snapshot", required=True)
    parser.add_argument("--base-tokenizer-snapshot", required=True)
    parser.add_argument("--union-init-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--geometry-fusion-seed", type=int, required=True)
    parser.add_argument("--num-e3fp-embeddings", type=int, default=4096)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_smoke(args)
    except (FGateSmokeError, RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FGateSmokeError",
    "REPORT_SCHEMA",
    "SMOKE_BATCH_SIZE",
    "build_parser",
    "main",
    "run_smoke",
]
