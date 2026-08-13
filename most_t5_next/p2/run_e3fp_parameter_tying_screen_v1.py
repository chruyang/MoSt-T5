#!/usr/bin/env python3
"""Run one PF-10 E3FP parameter-tying cell from the tensor cache.

The six-cell matrix is three atom-table sharing schemes crossed with the
coordinate-blind B2D control and F3D.  Training is carrier-only so endpoint
placement is not factorially mixed into this atom-encoder decision.  Final
evaluation still reports carrier-only, endpoint-only, both, zero, and an
optional matched-state perturbation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence

from most_t5_next.p1.pf1_optimization import G_CODEC_PF1_PROTOCOL

from .e3fp_atom_embedding_v1 import (
    LEVEL_SPECIFIC_FIXED4,
    L0_STATE_FIXED4,
    REFERENCE_SHARED_FIXED4,
)
from .factorized_model_init_v10 import (
    factorized_initialization_contract_v10,
    load_deterministic_factorized_model_v10,
)
from .run_anchored_v4_shell_screen_v1 import run as run_shared_screen


SCHEMA_VERSION = "most-t5-p2/e3fp-parameter-tying-screen/v1"
MANIFEST_FILENAME = "e3fp_parameter_tying_manifest.json"
CANDIDATES = (
    REFERENCE_SHARED_FIXED4,
    L0_STATE_FIXED4,
    LEVEL_SPECIFIC_FIXED4,
)
PF10_PROTOCOL = replace(
    G_CODEC_PF1_PROTOCOL,
    warmup_updates=1000,
    total_updates=10000,
    micro_batch_size=32,
    gradient_accumulation_steps=4,
)
EVALUATION_UPDATES = (0, 2500, 5000, 7500, 10000)


class E3FPParameterTyingScreenError(RuntimeError):
    pass


EXPECTED_PF10_COUNTS = {
    "records": 336006,
    "train_records": 302406,
    "dev_records": 33600,
}

ARCHITECTURE_CONTRACT_ID = "most-t5/anchored-v10-e3fp-parameter-tying/20260813"


def validate_source_control_boundary(expected_commit: str) -> dict[str, object]:
    """Bind an experiment to one clean Git checkout before CUDA allocation."""

    repo_root = Path(__file__).resolve().parents[2]
    expected = str(expected_commit).strip().lower()
    if len(expected) != 40 or any(ch not in "0123456789abcdef" for ch in expected):
        raise E3FPParameterTyingScreenError("code commit must be a full 40-character Git SHA")
    try:
        actual = subprocess.run(
            ("git", "-C", str(repo_root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        dirty = subprocess.run(
            (
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise E3FPParameterTyingScreenError("training source is not a readable Git checkout") from exc
    if actual != expected:
        raise E3FPParameterTyingScreenError("training checkout does not match --code-commit")
    if dirty:
        raise E3FPParameterTyingScreenError("training checkout is not clean")
    return {
        "git_commit": actual,
        "tracked_worktree_clean": True,
        "architecture_contract_id": ARCHITECTURE_CONTRACT_ID,
    }


def validate_pf10_cache_boundary(cache_root: Path) -> dict[str, object]:
    """Reject the historical 1% cache before allocating a CUDA model."""

    path = Path(cache_root).expanduser().resolve() / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E3FPParameterTyingScreenError(
            "PF-10 cache manifest is unreadable"
        ) from exc
    counts = manifest.get("counts")
    source = manifest.get("source")
    if manifest.get("status") != "pass" or not isinstance(counts, dict):
        raise E3FPParameterTyingScreenError("PF-10 cache is not passed")
    if any(int(counts.get(key, -1)) != value for key, value in EXPECTED_PF10_COUNTS.items()):
        raise E3FPParameterTyingScreenError(
            "cache is not the frozen 336,006-member PF-10 population"
        )
    derived = source.get("derived_representation") if isinstance(source, dict) else None
    if not isinstance(derived, dict) or derived.get("schema_version") != (
        "most-t5-p2/anchored-training-tensor-cache-build/v1"
    ):
        raise E3FPParameterTyingScreenError(
            "cache does not expose the anchored PF-10 training surface"
        )
    return {
        "records": EXPECTED_PF10_COUNTS["records"],
        "train_records": EXPECTED_PF10_COUNTS["train_records"],
        "dev_records": EXPECTED_PF10_COUNTS["dev_records"],
        "surface": "anchored_interim_not_final_fragsmiles",
    }


def cell_contract(cell: str, candidate: str) -> dict[str, str]:
    if candidate not in CANDIDATES:
        raise E3FPParameterTyingScreenError("unknown parameter-tying candidate")
    if cell == "B2D":
        state_kind = "coordinate_blind_morgan"
    elif cell == "F3D":
        state_kind = "e3fp"
    else:
        raise E3FPParameterTyingScreenError("cell must be B2D or F3D")
    return {
        "cell": cell,
        "state_kind": state_kind,
        "view_id": "m_plus_g",
        "memory_mode": "aligned",
        "e3fp_parameter_tying": candidate,
        "factor_under_test": "e3fp_embedding_table_parameter_tying_only",
    }


def _model_loader(candidate: str):
    def load(**kwargs: object):
        if kwargs.pop("shell_fusion_mode", None) != "not_applicable":
            raise E3FPParameterTyingScreenError("legacy shell mode crossed the V10 boundary")
        kwargs["parameter_tying"] = candidate
        kwargs["atom_memory_dim"] = 768
        return load_deterministic_factorized_model_v10(**kwargs)

    return load


def _contract_builder(candidate: str):
    def build(**kwargs: object) -> dict[str, object]:
        if kwargs.pop("shell_fusion_mode", None) != "not_applicable":
            raise E3FPParameterTyingScreenError("legacy shell mode crossed the V10 boundary")
        kwargs["parameter_tying"] = candidate
        kwargs["atom_memory_dim"] = 768
        return factorized_initialization_contract_v10(**kwargs)

    return build


def run(args: argparse.Namespace) -> dict[str, object]:
    candidate = str(args.parameter_tying)
    contract = cell_contract(str(args.cell), candidate)
    source_control = validate_source_control_boundary(str(args.code_commit))
    cache_boundary = validate_pf10_cache_boundary(Path(args.cache_root))
    bridge = argparse.Namespace(**vars(args))
    bridge.shell_fusion_mode = "not_applicable"

    def build_cell_contract(cell: str, shell: str) -> dict[str, str]:
        if cell != contract["cell"] or shell != "not_applicable":
            raise E3FPParameterTyingScreenError("cell contract bridge drifted")
        return dict(contract)

    report = run_shared_screen(
        bridge,
        model_loader=_model_loader(candidate),
        initialization_contract_builder=_contract_builder(candidate),
        contract_builder=build_cell_contract,
        schema_version=SCHEMA_VERSION,
        scope="pf10_e3fp_parameter_tying_decision_not_formal_pretraining",
        manifest_filename=MANIFEST_FILENAME,
        training_component_mode="carrier_only",
        optimization_protocol=PF10_PROTOCOL,
        evaluation_updates=EVALUATION_UPDATES,
    )
    report["decision_contract"] = {
        "three_parameter_tying_arms": list(CANDIDATES),
        "paired_b2d_f3d_required": True,
        "update_zero_exact_equivalence_required": True,
        "e3fp_embedding_width": 768,
        "carrier_only_during_training": True,
        "endpoint_injection_selection_deferred": True,
        "identity_ce_alone_selects_winner": False,
        "three_dimensional_probe_required_after_training": True,
        "cache_boundary": cache_boundary,
        "final_phase_i_three_part_cache_inferred": False,
        "final_fragsmiles_contract_inferred": False,
    }
    report["source_control"] = source_control
    output = Path(args.output_dir).expanduser().resolve()
    (output / MANIFEST_FILENAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("B2D", "F3D"), required=True)
    parser.add_argument("--parameter-tying", choices=CANDIDATES, required=True)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--anchored-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--semantic-plan-sha256", required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=5)
    parser.add_argument("--matched-overlay", type=Path)
    parser.add_argument("--save-final-checkpoint", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    report = run(_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "cell": report["cell"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATES",
    "ARCHITECTURE_CONTRACT_ID",
    "EXPECTED_PF10_COUNTS",
    "PF10_PROTOCOL",
    "cell_contract",
    "run",
    "validate_source_control_boundary",
    "validate_pf10_cache_boundary",
]
