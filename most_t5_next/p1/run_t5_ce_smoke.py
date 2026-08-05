#!/usr/bin/env python3
"""Run one offline standard-T5 CE optimization step and save/reload probe.

This is a plumbing smoke test, not a P1 training launcher.  It accepts only an
explicit local T5 model snapshot and never asks Hugging Face to download one.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from most_t5_next.p1.runtime_bridge import PaddedCEBatch
from most_t5_next.p1.training_adapter import (
    MODEL_INPUT_KEYS,
    select_t5_forward_inputs,
    to_t5_batch_encoding,
)


REPORT_SCHEMA = "most-t5-p1/t5-ce-smoke-report/v2"
DEFAULT_FUNCTIONAL_RTOL = 1e-4
DEFAULT_FUNCTIONAL_ATOL = 5e-4


class SmokeError(RuntimeError):
    """The executable CE plumbing probe could not be completed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _functional_config_payload(model: Any) -> dict[str, object]:
    """Return config fields that can affect this model's forward semantics."""

    payload = dict(model.config.to_dict())
    # These fields identify the load location/revision, not model behavior.
    payload.pop("_name_or_path", None)
    payload.pop("_commit_hash", None)
    # ``save_pretrained`` persists the effective dtype even when the source
    # config omitted it.  Bind both sides to the tensors actually in memory.
    effective_dtype = str(next(model.parameters()).dtype)
    if effective_dtype.startswith("torch."):
        effective_dtype = effective_dtype[len("torch.") :]
    payload["torch_dtype"] = effective_dtype
    return payload


def _tuple_rows(value: Any, field: str) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(value, list) or not all(isinstance(row, list) for row in value):
        raise SmokeError(f"fixture {field} must be a JSON array of arrays")
    return tuple(tuple(row) for row in value)


def _load_fixture(path: Path) -> tuple[PaddedCEBatch, dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        batch = PaddedCEBatch(
            record_ids=tuple(payload["record_ids"]),
            input_ids=_tuple_rows(payload["input_ids"], "input_ids"),
            attention_mask=_tuple_rows(payload["attention_mask"], "attention_mask"),
            labels=_tuple_rows(payload["labels"], "labels"),
            input_lengths=tuple(payload["input_lengths"]),
            target_lengths=tuple(payload["target_lengths"]),
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SmokeError(f"invalid padded CE fixture: {exc}") from exc
    return batch, {
        "kind": "json_fixture",
        "path": str(path),
        "sha256": _sha256_file(path),
    }


def _build_builtin_batch(
    *, vocab_size: int, pad_token_id: int, eos_token_id: int
) -> PaddedCEBatch:
    candidates = [
        token_id
        for token_id in range(vocab_size)
        if token_id not in {pad_token_id, eos_token_id}
    ][:6]
    if len(candidates) < 6:
        raise SmokeError("model vocabulary is too small for the built-in CE batch")
    return PaddedCEBatch(
        record_ids=("builtin:0", "builtin:1"),
        input_ids=(
            (candidates[0], candidates[1], eos_token_id),
            (candidates[2], eos_token_id, pad_token_id),
        ),
        attention_mask=((True, True, True), (True, True, False)),
        labels=(
            (candidates[3], candidates[4], eos_token_id),
            (candidates[5], eos_token_id, -100),
        ),
        input_lengths=(3, 2),
        target_lengths=(3, 2),
    )


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SmokeError("--device cuda requested but CUDA is unavailable")
    return requested


def _compare_state_dicts(
    torch: Any,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> dict[str, object]:
    """Compare model parameters and persistent buffers without a forward pass.

    A checkpoint round trip is primarily a serialization question.  Comparing
    the state dictionaries directly keeps that question separate from small
    backend-dependent differences between two CUDA forward executions.
    """

    expected_keys = set(expected)
    observed_keys = set(observed)
    keys_match = expected_keys == observed_keys
    metadata_mismatches: list[str] = []
    stride_mismatches: list[str] = []
    nonexact_tensors: list[str] = []
    maximum_abs_difference = 0.0

    for key in sorted(expected_keys & observed_keys):
        left = expected[key]
        right = observed[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            metadata_mismatches.append(key)
            continue
        if tuple(left.stride()) != tuple(right.stride()):
            stride_mismatches.append(key)
        if bool(torch.equal(left, right)):
            continue
        nonexact_tensors.append(key)
        if left.numel() > 0:
            difference = float(
                (left.detach().double() - right.detach().double())
                .abs()
                .max()
                .item()
            )
            maximum_abs_difference = max(maximum_abs_difference, difference)

    tensors_exact = (
        keys_match and not metadata_mismatches and not nonexact_tensors
    )
    return {
        "keys_match": keys_match,
        "missing_keys": sorted(expected_keys - observed_keys),
        "unexpected_keys": sorted(observed_keys - expected_keys),
        "tensor_count": len(expected),
        "metadata_mismatch_count": len(metadata_mismatches),
        "metadata_mismatch_keys": metadata_mismatches,
        "stride_mismatch_count": len(stride_mismatches),
        "stride_mismatch_keys": stride_mismatches,
        "nonexact_tensor_count": len(nonexact_tensors),
        "nonexact_tensor_keys": nonexact_tensors,
        "tensors_exact": tensors_exact,
        "maximum_abs_difference": maximum_abs_difference,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        import torch
        import transformers
        from transformers import T5ForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - remote integration only
        raise SmokeError("PyTorch and Transformers are required for this smoke test") from exc

    snapshot = Path(args.model_snapshot).expanduser().resolve()
    if not snapshot.is_dir():
        raise SmokeError("--model-snapshot must be an existing local directory")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise SmokeError("--output-dir must be a new path")

    device = _resolve_device(torch, args.device)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = T5ForConditionalGeneration.from_pretrained(
        str(snapshot), local_files_only=True
    )
    model.to(device)
    pad_token_id = model.config.pad_token_id
    eos_token_id = model.config.eos_token_id
    if pad_token_id is None or eos_token_id is None:
        raise SmokeError("T5 config must define pad_token_id and eos_token_id")

    if args.batch_json:
        batch, batch_source = _load_fixture(Path(args.batch_json).expanduser().resolve())
    else:
        batch = _build_builtin_batch(
            vocab_size=int(model.config.vocab_size),
            pad_token_id=int(pad_token_id),
            eos_token_id=int(eos_token_id),
        )
        batch_source = {"kind": "deterministic_builtin", "seed": args.seed}

    encoded = to_t5_batch_encoding(batch, device=device)
    forward_inputs = select_t5_forward_inputs(encoded)
    if tuple(forward_inputs) != MODEL_INPUT_KEYS:
        raise SmokeError("unexpected T5 forward-input order")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**forward_inputs)
    loss = outputs.loss
    if loss is None or loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise SmokeError("forward loss is absent or non-finite")
    loss_value = float(loss.detach().cpu().item())
    loss.backward()

    gradient_squared_norm = 0.0
    gradient_tensor_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient_tensor_count += 1
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all().item()):
            raise SmokeError("backward produced a non-finite gradient")
        gradient_squared_norm += float(gradient.float().square().sum().item())
    gradient_norm = math.sqrt(gradient_squared_norm)
    if gradient_tensor_count == 0 or not math.isfinite(gradient_norm) or gradient_norm == 0.0:
        raise SmokeError("backward produced no finite nonzero gradient")

    optimizer.step()
    model.eval()
    with torch.no_grad():
        reference_logits = model(**forward_inputs).logits.detach().float().cpu()
        repeated_logits = model(**forward_inputs).logits.detach().float().cpu()
    if not bool(torch.isfinite(reference_logits).all().item()):
        raise SmokeError("post-step logits are non-finite")
    repeated_logits_finite = bool(torch.isfinite(repeated_logits).all().item())
    if not repeated_logits_finite:
        raise SmokeError("repeated post-step logits are non-finite")
    same_instance_repeat_max_abs_diff = float(
        (reference_logits - repeated_logits).abs().max().item()
    )

    output_dir.mkdir(parents=True)
    checkpoint_dir = output_dir / "checkpoint"
    optimizer = None
    model.to("cpu")
    cpu_forward_inputs = {
        key: value.detach().cpu() for key, value in forward_inputs.items()
    }
    with torch.no_grad():
        cpu_reference_logits = (
            model(**cpu_forward_inputs).logits.detach().float().cpu()
        )
    cpu_reference_logits_finite = bool(
        torch.isfinite(cpu_reference_logits).all().item()
    )
    if not cpu_reference_logits_finite:
        raise SmokeError("pre-save CPU logits are non-finite")
    model.save_pretrained(str(checkpoint_dir))

    reloaded = T5ForConditionalGeneration.from_pretrained(
        str(checkpoint_dir), local_files_only=True
    )
    original_config_sha256 = _canonical_json_sha256(
        _functional_config_payload(model)
    )
    reloaded_config_sha256 = _canonical_json_sha256(
        _functional_config_payload(reloaded)
    )
    config_exact = original_config_sha256 == reloaded_config_sha256
    state_dict_comparison = _compare_state_dicts(
        torch, model.state_dict(), reloaded.state_dict()
    )
    reloaded.eval()
    with torch.no_grad():
        cpu_reloaded_logits = (
            reloaded(**cpu_forward_inputs).logits.detach().float().cpu()
        )
    cpu_logits_finite = bool(torch.isfinite(cpu_reloaded_logits).all().item())
    cpu_logits_exact = bool(torch.equal(cpu_reference_logits, cpu_reloaded_logits))
    cpu_max_abs_diff = float(
        (cpu_reference_logits - cpu_reloaded_logits).abs().max().item()
    )
    cpu_logits_consistent = cpu_logits_finite and bool(
        torch.allclose(
            cpu_reference_logits,
            cpu_reloaded_logits,
            rtol=args.rtol,
            atol=args.atol,
        )
    )

    del model, outputs, loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reloaded.to(device)
    with torch.no_grad():
        runtime_reloaded_logits = (
            reloaded(**forward_inputs).logits.detach().float().cpu()
        )
    runtime_logits_finite = bool(
        torch.isfinite(runtime_reloaded_logits).all().item()
    )
    runtime_cross_reload_max_abs_diff = float(
        (reference_logits - runtime_reloaded_logits).abs().max().item()
    )
    runtime_cross_reload_logits_consistent = runtime_logits_finite and bool(
        torch.allclose(
            reference_logits,
            runtime_reloaded_logits,
            rtol=args.rtol,
            atol=args.atol,
        )
    )
    serialization_consistent = bool(
        config_exact
        and state_dict_comparison["tensors_exact"]
        and cpu_logits_consistent
    )
    smoke_pass = bool(serialization_consistent and runtime_logits_finite)

    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "pass": smoke_pass,
        "scope": "standard_t5_ce_plumbing_smoke_only",
        "model_snapshot": str(snapshot),
        "saved_checkpoint": str(checkpoint_dir),
        "batch_source": batch_source,
        "model_forward_keys": list(forward_inputs),
        "audit_metadata_forwarded": False,
        "batch": {
            "size": len(batch.record_ids),
            "input_shape": list(forward_inputs["input_ids"].shape),
            "label_shape": list(forward_inputs["labels"].shape),
            "tensor_dtype": str(forward_inputs["input_ids"].dtype),
        },
        "runtime": {
            "device": device,
            "device_name": (
                torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
            ),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "optimization": {
            "optimizer": "AdamW",
            "optimizer_steps": 1,
            "learning_rate": args.learning_rate,
            "loss": loss_value,
            "loss_finite": True,
            "gradient_tensor_count": gradient_tensor_count,
            "gradient_norm": gradient_norm,
            "gradients_finite": True,
        },
        "save_reload": {
            "pass_basis": (
                "exact_functional_config_and_state_dict_plus_cpu_functional_"
                "consistency_and_finite_runtime_reload_logits"
            ),
            "serialization_consistent": serialization_consistent,
            "config_exact": config_exact,
            "original_config_sha256": original_config_sha256,
            "reloaded_config_sha256": reloaded_config_sha256,
            "state_dict": state_dict_comparison,
            "cpu_logits_exact": cpu_logits_exact,
            "cpu_logits_finite": cpu_logits_finite,
            "cpu_logits_consistent": cpu_logits_consistent,
            "cpu_max_abs_diff": cpu_max_abs_diff,
            "runtime_device": device,
            "same_instance_repeat_logits_finite": repeated_logits_finite,
            "same_instance_repeat_max_abs_diff": same_instance_repeat_max_abs_diff,
            "cpu_reference_logits_finite": cpu_reference_logits_finite,
            "runtime_cross_reload_logits_finite": runtime_logits_finite,
            "runtime_cross_reload_logits_consistent": (
                runtime_cross_reload_logits_consistent
            ),
            "runtime_cross_reload_max_abs_diff": (
                runtime_cross_reload_max_abs_diff
            ),
            "runtime_cross_reload_is_diagnostic_only": True,
            "rtol": args.rtol,
            "atol": args.atol,
        },
    }
    report_path = output_dir / "smoke_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_FUNCTIONAL_RTOL,
        help="relative tolerance for functional logits checks; state tensors remain exact",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_FUNCTIONAL_ATOL,
        help="absolute tolerance for functional logits checks; state tensors remain exact",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except SmokeError as exc:
        print(json.dumps({"pass": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
