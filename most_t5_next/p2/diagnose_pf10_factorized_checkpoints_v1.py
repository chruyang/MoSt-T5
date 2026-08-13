"""Read-only PF-10 F3D checkpoint sensitivity and retention diagnostic.

The diagnostic loads one already-published S or G checkpoint and evaluates the
same frozen dev cohort under aligned, zeroed, and matched-shuffle state memory.
It also replays the formal state-imputation dev view to measure whether the
S-stage capability survives grammar training.  No optimizer or training state
is created.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from most_t5_next.p1.build_pf1_paired_release_v1 import PF1PairedReleaseReader
from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)

from .build_pf10_matched_motif_overlay_v1 import MatchedMotifStateProvider
from .factorized_model_init_v1 import load_deterministic_factorized_model
from .factorized_motif_t5_v1 import FactorizedMotifT5V1
from .run_pf10_factorized_grammar_v1 import (
    CHECKPOINT_SCHEMA as G_CHECKPOINT_SCHEMA,
    _cell_protocol,
    evaluate_grammar_stage,
)
from .run_pf10_factorized_smoke_cli_v1 import (
    ADAPTER_SEED,
    NUM_E3FP_EMBEDDINGS,
    UNION_GEOMETRY_FUSION_SEED,
)
from .run_pf10_factorized_state_v1 import (
    CHECKPOINT_SCHEMA as S_CHECKPOINT_SCHEMA,
    S_PROTOCOL,
    _EligibleReader,
    _load_eligible_indices,
    evaluate_state_stage,
)


SCHEMA_VERSION = "most-t5-p2/pf10-factorized-checkpoint-diagnostic/v1"


class PF10CheckpointDiagnosticError(RuntimeError):
    """The requested checkpoint or evaluation contract is inconsistent."""


def _validate_checkpoint_payload(
    payload: Mapping[str, Any], *, stage: str, update: int
) -> None:
    if stage == "S":
        schema = S_CHECKPOINT_SCHEMA
        protocol = asdict(S_PROTOCOL)
    elif stage == "G":
        schema = G_CHECKPOINT_SCHEMA
        protocol = asdict(_cell_protocol("F3D"))
    else:
        raise PF10CheckpointDiagnosticError("stage must be S or G")
    if not (
        payload.get("schema_version") == schema
        and payload.get("cell") == "F3D"
        and payload.get("stage") == stage
        and payload.get("state_kind") == "e3fp"
        and payload.get("completed_updates") == update
        and payload.get("protocol") == protocol
        and isinstance(payload.get("model_state_dict"), Mapping)
    ):
        raise PF10CheckpointDiagnosticError(
            "checkpoint differs from the requested formal F3D stage/update"
        )


def load_diagnostic_checkpoint(
    model: FactorizedMotifT5V1,
    *,
    checkpoint_dir: Path,
    stage: str,
    update: int,
) -> None:
    path = Path(checkpoint_dir) / "training_state.pt"
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise PF10CheckpointDiagnosticError("checkpoint payload is not a mapping")
    _validate_checkpoint_payload(payload, stage=stage, update=update)
    model.load_state_dict(payload["model_state_dict"], strict=True)


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    paired_release = Path(args.paired_release).expanduser().resolve()
    verified_tokenizer = load_verified_canary_union_tokenizer(
        base_snapshot=Path(args.base_tokenizer_snapshot),
        output_dir=paired_release / "union_tokenizer",
    )
    reader = PF1PairedReleaseReader(paired_release)
    cache = reader.warm_decoded_record_cache(
        workers=args.cache_workers,
        max_pending=args.cache_max_pending,
    )
    support = Path(args.support_census).expanduser().resolve()
    eligible = _EligibleReader(
        reader,
        train_indices=_load_eligible_indices(
            support / "train_state_eligible_membership.jsonl", expected_split="train"
        ),
        dev_indices=_load_eligible_indices(
            support / "dev_state_eligible_membership.jsonl", expected_split="dev"
        ),
    )
    model = load_deterministic_factorized_model(
        base_model_snapshot=Path(args.base_model_snapshot),
        base_tokenizer_snapshot=Path(args.base_tokenizer_snapshot),
        union_tokenizer_dir=paired_release / "union_tokenizer",
        union_init_dir=Path(args.union_init_dir),
        union_geometry_fusion_seed=UNION_GEOMETRY_FUSION_SEED,
        adapter_seed=ADAPTER_SEED,
        num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
    )
    load_diagnostic_checkpoint(
        model,
        checkpoint_dir=Path(args.checkpoint_dir),
        stage=args.stage,
        update=args.update,
    )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise PF10CheckpointDiagnosticError("one CUDA BF16 device is required")
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    shuffle = MatchedMotifStateProvider(Path(args.shuffle_overlay))
    shuffle_coverage = dict(shuffle.manifest["coverage"])
    try:
        aligned = evaluate_grammar_stage(
            model,
            reader=reader,
            tokenizer=verified_tokenizer.runtime,
            provider=None,
            raw_t5=False,
            device=device,
            use_bf16=True,
        )
        zero = evaluate_grammar_stage(
            model,
            reader=reader,
            tokenizer=verified_tokenizer.runtime,
            provider=None,
            raw_t5=False,
            device=device,
            use_bf16=True,
            state_memory_mode="zero",
        )
        shuffled = evaluate_grammar_stage(
            model,
            reader=reader,
            tokenizer=verified_tokenizer.runtime,
            provider=shuffle,
            raw_t5=False,
            device=device,
            use_bf16=True,
        )
        state = evaluate_state_stage(
            model,
            reader=eligible,
            tokenizer=verified_tokenizer.runtime,
            provider=None,
            device=device,
            use_bf16=True,
        )
    finally:
        shuffle.close()
    aligned_nll = float(aligned["token_weighted_nll"])
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "cell": "F3D",
        "stage": args.stage,
        "update": args.update,
        "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser().resolve()),
        "grammar": {
            "aligned": aligned,
            "zero": zero,
            "matched_shuffle": shuffled,
            "zero_minus_aligned_delta_nll": float(zero["token_weighted_nll"])
            - aligned_nll,
            "shuffle_minus_aligned_delta_nll": float(
                shuffled["token_weighted_nll"]
            )
            - aligned_nll,
        },
        "state_imputation": state,
        "data_contract": {
            "paired_release": str(paired_release),
            "dev_members": reader.dev_member_count,
            "state_eligible_dev_members": eligible.dev_member_count,
            "shuffle_overlay": str(Path(args.shuffle_overlay).expanduser().resolve()),
            "shuffle_coverage": shuffle_coverage,
            "tokenizer_contract_sha256": (
                verified_tokenizer.runtime.tokenizer_contract_sha256
            ),
            "tokenizer_snapshot_sha256": (
                verified_tokenizer.runtime.tokenizer_snapshot_sha256
            ),
        },
        "decoded_cache_warmup": cache,
        "decoded_cache_final": reader.decoded_record_cache_stats(),
    }
    output = Path(args.output_report).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise PF10CheckpointDiagnosticError("output report already exists")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("S", "G"), required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--support-census", type=Path, required=True)
    parser.add_argument("--shuffle-overlay", type=Path, required=True)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=12)
    parser.add_argument("--cache-max-pending", type=int, default=48)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_diagnostic(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PF10CheckpointDiagnosticError",
    "SCHEMA_VERSION",
    "_validate_checkpoint_payload",
    "load_diagnostic_checkpoint",
    "main",
    "run_diagnostic",
]
