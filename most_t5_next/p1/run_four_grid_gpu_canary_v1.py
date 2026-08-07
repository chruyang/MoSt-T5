#!/usr/bin/env python3
"""Run the first real P1 A0/A1/M0/M1 GPU data-flow canary.

The paired canary release already fixes molecule membership, A/M records,
union-tokenizer semantics, and the admissible sequence lengths.  This runner
therefore performs only the next scientific plumbing check: take the first
mini-batch in frozen membership order, materialize the epoch-0 CE corruption,
and run one forward/backward pass for each condition from the same published
union initialization.

There is deliberately no optimizer and no checkpoint save.  The four losses
are runtime diagnostics only: A and M use different corruption units, so a
single raw CE value is not an architecture-effect comparison.
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
from typing import Any, Callable, Mapping, Sequence

from most_t5_next.p1.atom_production_bridge import (
    collate_production_atom_batch,
)
from most_t5_next.p1.build_union_init_checkpoint_v1 import (
    load_verified_four_grid_wrapper,
)
from most_t5_next.p1.experiment_grid import (
    P1ConditionBatch,
    validate_a1_m1_geometry_atom_parity,
)
from most_t5_next.p1.production_bridge import collate_production_batch
from most_t5_next.p1.training_adapter import (
    select_four_grid_forward_inputs,
    to_four_grid_batch_encoding,
)
from most_t5_next.r1.adapter.build_p1_paired_canary_v1 import (
    LMDB_DIRECTORY,
    MANIFEST_NAME as PAIRED_MANIFEST_NAME,
    MEMBERSHIP_NAME,
    TOKENIZER_DIRECTORY,
)
from most_t5_next.r1.adapter.paired_record_wire_v1 import (
    LoadedPairedTrainingRecord,
    decode_paired_training_record,
)
from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)


REPORT_SCHEMA = "most-t5-p1/four-grid-gpu-canary/v1"
REPORT_NAME = "gpu_canary_manifest.json"
CONDITION_ORDER = ("A0", "A1", "M0", "M1")
CORRUPTION_SEED = 0
CORRUPTION_EPOCH = 0
MASK_PROBABILITY = 0.15
FORWARD_SEED = 20260807
MAX_SEQUENCE_LENGTH = 512


class FourGridGPUCanaryError(RuntimeError):
    """The real four-grid forward/backward smoke could not be completed."""


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FourGridGPUCanaryError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise FourGridGPUCanaryError(f"{label} must be one JSON object")
    return value


def _read_membership(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise FourGridGPUCanaryError(
                        f"membership line {line_number} is not a JSON object"
                    )
                rows.append(value)
    except (OSError, ValueError) as exc:
        if isinstance(exc, FourGridGPUCanaryError):
            raise
        raise FourGridGPUCanaryError("paired membership is unreadable") from exc
    if not rows:
        raise FourGridGPUCanaryError("paired membership is empty")
    schedule = tuple(row.get("schedule_index") for row in rows)
    if schedule != tuple(range(len(rows))):
        raise FourGridGPUCanaryError(
            "paired membership must remain in frozen schedule order"
        )
    return tuple(rows)


def load_frozen_minibatch(
    *,
    paired_release: Path,
    batch_size: int,
) -> tuple[tuple[LoadedPairedTrainingRecord, ...], dict[str, object]]:
    """Load the first rows in frozen membership order from canonical LMDB."""

    paired_release = Path(paired_release).expanduser().resolve()
    if not paired_release.is_dir():
        raise FourGridGPUCanaryError("paired_release must be an existing directory")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise FourGridGPUCanaryError("batch_size must be a positive integer")

    manifest = _read_json(
        paired_release / PAIRED_MANIFEST_NAME, "paired canary manifest"
    )
    if (
        manifest.get("status") != "pass"
        or manifest.get("canary_execution_ready") is not True
        or manifest.get("training_admission") is not False
    ):
        raise FourGridGPUCanaryError("paired release is not a successful canary release")
    memberships = _read_membership(paired_release / MEMBERSHIP_NAME)
    if batch_size > len(memberships):
        raise FourGridGPUCanaryError("batch_size exceeds the frozen membership")
    selected = memberships[:batch_size]

    try:
        import lmdb
    except ModuleNotFoundError as exc:  # pragma: no cover - integration boundary
        raise FourGridGPUCanaryError("python-lmdb is required to read paired rows") from exc

    lmdb_path = paired_release / LMDB_DIRECTORY
    if not lmdb_path.is_dir():
        raise FourGridGPUCanaryError("paired LMDB directory is absent")
    environment = lmdb.open(
        str(lmdb_path),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=8,
    )
    records: list[LoadedPairedTrainingRecord] = []
    try:
        with environment.begin(write=False) as transaction:
            for membership in selected:
                storage_key = membership.get("storage_key")
                member_id = membership.get("member_id")
                if not isinstance(storage_key, str) or not isinstance(member_id, str):
                    raise FourGridGPUCanaryError(
                        "selected membership lacks member_id or storage_key"
                    )
                try:
                    payload = transaction.get(storage_key.encode("ascii"))
                except UnicodeEncodeError as exc:
                    raise FourGridGPUCanaryError(
                        "paired LMDB storage keys must be ASCII"
                    ) from exc
                if payload is None:
                    raise FourGridGPUCanaryError(
                        f"paired LMDB lacks selected storage key {storage_key}"
                    )
                record = decode_paired_training_record(bytes(payload))
                if (
                    record.schedule_index != membership.get("schedule_index")
                    or record.atom_record.record_id != member_id
                    or record.atom_record.storage_key != storage_key
                ):
                    raise FourGridGPUCanaryError(
                        "decoded paired row differs from its frozen membership"
                    )
                records.append(record)
    finally:
        environment.close()
    return tuple(records), manifest


def build_frozen_grid_batches(
    records: Sequence[LoadedPairedTrainingRecord],
    *,
    tokenizer_runtime: Any,
) -> dict[str, P1ConditionBatch]:
    """Collate the fixed epoch-0 batch and close the two comparison parities."""

    rows = tuple(records)
    if not rows:
        raise FourGridGPUCanaryError("the GPU canary requires a nonempty mini-batch")
    atoms = tuple(row.atom_record for row in rows)
    motifs = tuple(row.motif_record for row in rows)
    batches = {
        "A0": collate_production_atom_batch(
            atoms,
            condition_id="A0",
            tokenizer=tokenizer_runtime,
            seed=CORRUPTION_SEED,
            epoch=CORRUPTION_EPOCH,
            mask_probability=MASK_PROBABILITY,
        ),
        "A1": collate_production_atom_batch(
            atoms,
            condition_id="A1",
            tokenizer=tokenizer_runtime,
            seed=CORRUPTION_SEED,
            epoch=CORRUPTION_EPOCH,
            mask_probability=MASK_PROBABILITY,
        ),
        "M0": collate_production_batch(
            motifs,
            condition_id="M0",
            tokenizer=tokenizer_runtime,
            seed=CORRUPTION_SEED,
            epoch=CORRUPTION_EPOCH,
            mask_probability=MASK_PROBABILITY,
        ),
        "M1": collate_production_batch(
            motifs,
            condition_id="M1",
            tokenizer=tokenizer_runtime,
            seed=CORRUPTION_SEED,
            epoch=CORRUPTION_EPOCH,
            mask_probability=MASK_PROBABILITY,
        ),
    }
    if batches["A0"].ce_batch != batches["A1"].ce_batch:
        raise FourGridGPUCanaryError("A0/A1 CE mini-batches differ")
    if batches["M0"].ce_batch != batches["M1"].ce_batch:
        raise FourGridGPUCanaryError("M0/M1 CE mini-batches differ")
    try:
        validate_a1_m1_geometry_atom_parity(batches["A1"], batches["M1"])
    except ValueError as exc:
        raise FourGridGPUCanaryError("A1/M1 geometry atom rows differ") from exc
    for batch in batches.values():
        if (
            max(batch.ce_batch.input_lengths) > MAX_SEQUENCE_LENGTH
            or max(batch.ce_batch.target_lengths) > MAX_SEQUENCE_LENGTH
        ):
            raise FourGridGPUCanaryError(
                "selected mini-batch exceeds 512 tokens; this runner never truncates"
            )
    return batches


def _gradient_statistics(torch_module: Any, model: Any) -> tuple[float, int]:
    squared_norm = 0.0
    tensor_count = 0
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        tensor_count += 1
        detached = gradient.detach()
        if not bool(torch_module.isfinite(detached).all().item()):
            raise FourGridGPUCanaryError("backward produced a non-finite gradient")
        component = float(torch_module.linalg.vector_norm(detached.float()).item())
        squared_norm += component * component
    norm = math.sqrt(squared_norm)
    if tensor_count == 0 or not math.isfinite(norm) or norm <= 0.0:
        raise FourGridGPUCanaryError("backward produced no finite nonzero gradient")
    return norm, tensor_count


def execute_four_grid(
    batches: Mapping[str, P1ConditionBatch],
    *,
    base_model_snapshot: Path,
    base_tokenizer_snapshot: Path,
    union_tokenizer_dir: Path,
    union_init_dir: Path,
    geometry_fusion_seed: int,
    num_e3fp_embeddings: int,
    expected_vocab_size: int,
    device: Any,
    use_bf16: bool,
    torch_module: Any,
    wrapper_loader: Callable[..., Any] = load_verified_four_grid_wrapper,
) -> list[dict[str, object]]:
    """Sequentially forward/backward four independently loaded grid cells."""

    results: list[dict[str, object]] = []
    for condition_id in CONDITION_ORDER:
        if condition_id not in batches:
            raise FourGridGPUCanaryError(f"missing frozen batch {condition_id}")
        model = None
        outputs = None
        loss = None
        encoded = None
        forward_inputs = None
        try:
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
                raise FourGridGPUCanaryError(
                    "verified wrapper vocabulary differs from the paired tokenizer"
                )
            model.to(device)
            model.train()
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch_module.cuda.empty_cache()
                torch_module.cuda.reset_peak_memory_stats(device)
                torch_module.manual_seed(FORWARD_SEED)
                torch_module.cuda.manual_seed_all(FORWARD_SEED)
            else:  # CPU path exists only for the mock/unit test.
                torch_module.manual_seed(FORWARD_SEED)

            batch = batches[condition_id]
            encoded = to_four_grid_batch_encoding(batch, device=device)
            forward_inputs = select_four_grid_forward_inputs(encoded)
            if use_bf16:
                autocast_context = torch_module.autocast(
                    device_type="cuda", dtype=torch_module.bfloat16
                )
            else:
                autocast_context = nullcontext()
            with autocast_context:
                outputs = model(
                    **forward_inputs,
                    use_cache=False,
                    return_dict=True,
                )
                loss = outputs.loss
            if loss is None or loss.ndim != 0 or not bool(
                torch_module.isfinite(loss).item()
            ):
                raise FourGridGPUCanaryError(
                    f"{condition_id} produced an absent or non-finite CE loss"
                )
            loss_value = float(loss.detach().float().cpu().item())
            loss.backward()
            grad_norm, grad_tensors = _gradient_statistics(
                torch_module, model
            )
            peak_bytes = (
                int(torch_module.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )
            results.append(
                {
                    "condition": condition_id,
                    "member_ids": list(batch.ce_batch.record_ids),
                    "input_lengths": list(batch.ce_batch.input_lengths),
                    "target_lengths": list(batch.ce_batch.target_lengths),
                    "input_shape": list(forward_inputs["input_ids"].shape),
                    "target_shape": list(forward_inputs["labels"].shape),
                    "geometry_enabled": batch.geometry is not None,
                    "loss": loss_value,
                    "loss_finite": True,
                    "gradient_norm": grad_norm,
                    "gradients_finite": True,
                    "gradient_tensor_count": grad_tensors,
                    "peak_gpu_memory_bytes": peak_bytes,
                    "optimizer_steps": 0,
                }
            )
        finally:
            if model is not None:
                model.zero_grad(set_to_none=True)
            del forward_inputs, encoded, loss, outputs, model
            gc.collect()
            if device.type == "cuda":
                torch_module.cuda.empty_cache()
    return results


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute one real single-GPU canary and publish its compact manifest."""

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import torch
        import transformers
    except ModuleNotFoundError as exc:  # pragma: no cover - integration boundary
        raise FourGridGPUCanaryError("PyTorch and Transformers are required") from exc
    if not torch.cuda.is_available():
        raise FourGridGPUCanaryError("the real four-grid canary requires one CUDA GPU")
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise FourGridGPUCanaryError("BF16 was requested but is unsupported by this GPU")
    if args.batch_size <= 0:
        raise FourGridGPUCanaryError("batch_size must be positive")

    paired_release = Path(args.paired_release).expanduser().resolve()
    base_model_snapshot = Path(args.base_model_snapshot).expanduser().resolve()
    base_tokenizer_snapshot = Path(args.base_tokenizer_snapshot).expanduser().resolve()
    union_init_dir = Path(args.union_init_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FourGridGPUCanaryError("output_dir must be a new path")

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
    condition_results = execute_four_grid(
        batches,
        base_model_snapshot=base_model_snapshot,
        base_tokenizer_snapshot=base_tokenizer_snapshot,
        union_tokenizer_dir=union_tokenizer_dir,
        union_init_dir=union_init_dir,
        geometry_fusion_seed=args.geometry_fusion_seed,
        num_e3fp_embeddings=args.num_e3fp_embeddings,
        expected_vocab_size=tokenizer_build.runtime.vocab_size,
        device=device,
        use_bf16=use_bf16,
        torch_module=torch,
    )

    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "scope": "single_minibatch_runtime_and_dataflow_smoke_only",
        "interpretation": {
            "architecture_effect_ranking": False,
            "raw_A_M_CE_directly_comparable": False,
            "optimizer_step": False,
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
            "forward_seed_restarted_per_condition": FORWARD_SEED,
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
        "conditions": condition_results,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "visible_gpu_count": torch.cuda.device_count(),
            "precision": args.precision,
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--geometry-fusion-seed", type=int, required=True)
    parser.add_argument("--num-e3fp-embeddings", type=int, default=4096)
    parser.add_argument("--precision", choices=("bf16", "float32"), default="bf16")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except FourGridGPUCanaryError as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "CONDITION_ORDER",
    "FourGridGPUCanaryError",
    "REPORT_NAME",
    "REPORT_SCHEMA",
    "build_frozen_grid_batches",
    "build_parser",
    "execute_four_grid",
    "load_frozen_minibatch",
    "run",
]
