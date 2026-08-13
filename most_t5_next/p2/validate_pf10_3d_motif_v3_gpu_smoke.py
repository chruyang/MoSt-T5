"""One-card PF-10 mechanism smoke for 3D-MotifT5 V3.

The smoke performs exactly one optimizer update for each explicit input view
on the same first 128 frozen training records.  It writes no model checkpoint;
the result is a resource/gradient contract, not a training comparison.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Sequence

import torch

from most_t5_next.p1.build_pf1_paired_release_v1 import (
    DONOR_ATOM_MAP_NAME,
    PF1PairedReleaseReader,
)
from most_t5_next.p1.pf1_optimization import (
    FROZEN_PF1_PROTOCOL,
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
from .three_d_motif_training_views_v3 import (
    TRAINING_VIEW_ID,
    TRAINING_VIEWS,
    collate_3d_motif_training_view_v3,
)


REPORT_SCHEMA = "most-t5-p2/pf10-3d-motif-v3-gpu-smoke/v1"
RECORD_COUNT = 128
MICRO_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 8
SEED = 20260811
ADAPTER_SEED = 20260812
UNION_GEOMETRY_FUSION_SEED = 20260808
NUM_E3FP_EMBEDDINGS = 4096


class ThreeDMotifV3SmokeError(RuntimeError):
    """The fixed V3 GPU mechanism contract failed."""


def _autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _gradient_sum(parameter: torch.Tensor) -> float:
    gradient = parameter.grad
    if gradient is None:
        return 0.0
    return float(gradient.detach().abs().sum().float().cpu().item())


def _adapter_gradient_report(model) -> dict[str, float]:
    return {
        "atom_encoder": _gradient_sum(model.adapter.atom_encoder[0].weight),
        "carrier_projection": _gradient_sum(
            model.adapter.carrier_geometry_projection.weight
        ),
        "endpoint_projection": _gradient_sum(
            model.adapter.endpoint_geometry_projection.weight
        ),
    }


def _load_records(reader: PF1PairedReleaseReader):
    loaded = next(reader.iter_train_epoch(epoch=0, batch_size=RECORD_COUNT))
    if len(loaded) != RECORD_COUNT:
        raise ThreeDMotifV3SmokeError("paired release lacks 128 training records")
    records = tuple(pair.motif_record for pair in loaded)
    if len({record.record_id for record in records}) != RECORD_COUNT:
        raise ThreeDMotifV3SmokeError("smoke records are not unique")
    return records


def _run_view_update(
    *,
    view_id: str,
    records,
    tokenizer,
    addresses,
    state_provider,
    model,
    optimizer,
    device: torch.device,
) -> dict[str, object]:
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    target_total = 0
    weighted_loss = 0.0
    started = time.perf_counter()
    for micro_index, start in enumerate(
        range(0, RECORD_COUNT, MICRO_BATCH_SIZE)
    ):
        batch = collate_3d_motif_training_view_v3(
            records[start : start + MICRO_BATCH_SIZE],
            view_id=view_id,
            tokenizer=tokenizer,
            seed=SEED,
            epoch=0,
            atom_address_provider=addresses,
            atom_state_provider=state_provider,
            num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
            device=device,
        )
        target_count = int((batch.labels != -100).sum().item())
        if target_count <= 0:
            raise ThreeDMotifV3SmokeError("a V3 microbatch has no CE targets")
        # Each microbatch has the same record count.  Weighting by its actual
        # target count preserves the production token-weighted CE semantics.
        target_total += target_count
        with _autocast(device):
            inputs = batch.model_inputs()
            inputs["use_cache"] = False
            loss = model(**inputs).loss
        if not bool(torch.isfinite(loss)):
            raise ThreeDMotifV3SmokeError("V3 smoke produced non-finite CE")
        # Delay normalization until all target counts are known by using the
        # exact fixed 16-record microbatch target sum accumulated below.
        (loss * target_count).backward()
        weighted_loss += float(loss.detach().float().cpu().item()) * target_count
        del batch, inputs, loss
    if micro_index + 1 != GRADIENT_ACCUMULATION_STEPS:
        raise ThreeDMotifV3SmokeError("V3 smoke accumulation count drifted")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(target_total)

    gradients = _adapter_gradient_report(model)
    if view_id == "m_only":
        if any(value != 0.0 for value in gradients.values()):
            raise ThreeDMotifV3SmokeError(
                "M-only unexpectedly backpropagated through geometry"
            )
    elif any(value <= 0.0 for value in gradients.values()):
        raise ThreeDMotifV3SmokeError(
            "geometry-enabled view did not train every V3 geometry route"
        )
    preclip_norm = clip_pf1_gradients(model, FROZEN_PF1_PROTOCOL)
    if not (preclip_norm >= 0.0 and torch.isfinite(torch.tensor(preclip_norm))):
        raise ThreeDMotifV3SmokeError("V3 smoke gradient norm is non-finite")
    optimizer.step()
    torch.cuda.synchronize(device)
    return {
        "view_id": view_id,
        "records": RECORD_COUNT,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "target_tokens": target_total,
        "token_weighted_ce": weighted_loss / target_total,
        "preclip_gradient_norm": preclip_norm,
        "adapter_gradient_abs_sums": gradients,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ThreeDMotifV3SmokeError("one CUDA BF16 device is required")
    if torch.cuda.device_count() < 1:
        raise ThreeDMotifV3SmokeError("cuda:0 is unavailable")
    output = Path(args.output_report).expanduser().resolve()
    if output.exists():
        raise ThreeDMotifV3SmokeError("output report already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    paired = Path(args.paired_release).expanduser().resolve()
    tokenizer = load_verified_canary_union_tokenizer(
        base_snapshot=Path(args.base_tokenizer_snapshot),
        output_dir=paired / "union_tokenizer",
    )
    reader = PF1PairedReleaseReader(paired)
    records = _load_records(reader)
    addresses = GraphPortsCanonicalAtomAddressProvider(
        paired / DONOR_ATOM_MAP_NAME,
        required_record_ids=tuple(record.record_id for record in records),
    )
    provider = None
    if args.cell == "B2D":
        if args.morgan_overlay is None:
            raise ThreeDMotifV3SmokeError("B2D requires --morgan-overlay")
        provider = MorganAtomStateProvider(Path(args.morgan_overlay))
    elif args.morgan_overlay is not None:
        raise ThreeDMotifV3SmokeError("F3D does not accept --morgan-overlay")

    contract = factorized_initialization_contract_v3(
        adapter_seed=ADAPTER_SEED,
        num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
        state_level2_weight=0.25,
        state_embedding_dim=64,
        atom_memory_dim=128,
        max_identity_span_length=128,
        max_atoms_per_motif=128,
        geometry_fraction=0.5,
    )
    device = torch.device("cuda:0")
    try:
        model = load_deterministic_factorized_model_v3(
            base_model_snapshot=Path(args.base_model_snapshot),
            base_tokenizer_snapshot=Path(args.base_tokenizer_snapshot),
            union_tokenizer_dir=paired / "union_tokenizer",
            union_init_dir=Path(args.union_init_dir),
            union_geometry_fusion_seed=UNION_GEOMETRY_FUSION_SEED,
            adapter_seed=ADAPTER_SEED,
            num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
        ).to(device)
        model.train()
        optimizer = build_pf1_optimizer(model, FROZEN_PF1_PROTOCOL)
        view_reports = []
        for view_id in TRAINING_VIEWS:
            view_reports.append(
                _run_view_update(
                    view_id=view_id,
                    records=records,
                    tokenizer=tokenizer.runtime,
                    addresses=addresses,
                    state_provider=provider,
                    model=model,
                    optimizer=optimizer,
                    device=device,
                )
            )
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "pass",
            "scope": "mechanism_and_resource_smoke_only",
            "cell": args.cell,
            "state_kind": "e3fp" if provider is None else str(provider.state_kind),
            "training_view_contract": TRAINING_VIEW_ID,
            "initialization_contract": contract,
            "record_id_sha256": hashlib.sha256(
                "\n".join(record.record_id for record in records).encode()
            ).hexdigest(),
            "views": view_reports,
            "optimizer": "AdamWScale",
            "optimizer_updates": len(view_reports),
            "checkpoint_written": False,
            "scientific_comparison_authorized": False,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        }
        output.write_text(
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
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    report = run(_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "cell": report["cell"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
