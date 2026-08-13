#!/usr/bin/env python3
"""Run one PF1-scale atom-encoder reduction screen cell.

This is the final one-percent architecture screen.  It reuses the established
anchored carrier/endpoint training loop and changes only the atom-state
reducer:

* ``reference_fixed_four_mean``: the 3D-MolT5-style fixed four-slot mean;
* ``l0_high_minimal_phi``: L0 separated from the masked L1--L3 mean;
* ``l0_high_level_aware_phi``: the same minimal phi plus shell-level context.

Attachment role and explicit presence features are absent from every cell.
Each candidate must be run once with B2D and once with F3D before selection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from .factorized_model_init_v5 import (
    factorized_initialization_contract_v5,
    load_deterministic_factorized_model_v5,
)
from .factorized_model_init_v8 import (
    factorized_initialization_contract_v8,
    load_deterministic_factorized_model_v8,
)
from .factorized_model_init_v9 import (
    factorized_initialization_contract_v9,
    load_deterministic_factorized_model_v9,
)
from .run_anchored_v4_shell_screen_v1 import (
    AnchoredV4ShellScreenError,
    run as run_shared_screen,
)


SCHEMA_VERSION = "most-t5-p2/anchored-atom-encoder-screen/v1"
MANIFEST_FILENAME = "atom_encoder_screen_manifest.json"
CANDIDATES = (
    "reference_fixed_four_mean",
    "l0_high_minimal_phi",
    "l0_high_level_aware_phi",
)

_MODEL_LOADERS: Mapping[str, Callable[..., torch.nn.Module]] = {
    "reference_fixed_four_mean": load_deterministic_factorized_model_v5,
    "l0_high_minimal_phi": load_deterministic_factorized_model_v8,
    "l0_high_level_aware_phi": load_deterministic_factorized_model_v9,
}
_CONTRACT_BUILDERS: Mapping[str, Callable[..., dict[str, object]]] = {
    "reference_fixed_four_mean": factorized_initialization_contract_v5,
    "l0_high_minimal_phi": factorized_initialization_contract_v8,
    "l0_high_level_aware_phi": factorized_initialization_contract_v9,
}


class AnchoredAtomEncoderScreenError(AnchoredV4ShellScreenError):
    """The final paired one-percent atom-encoder screen is invalid."""


def cell_contract(cell: str, candidate: str) -> dict[str, str]:
    if candidate not in CANDIDATES:
        raise AnchoredAtomEncoderScreenError("atom-encoder candidate is not frozen")
    if cell == "B2D":
        state_kind = "coordinate_blind_morgan"
    elif cell == "F3D":
        state_kind = "e3fp"
    else:
        raise AnchoredAtomEncoderScreenError("cell must be B2D or F3D")
    return {
        "cell": cell,
        "state_kind": state_kind,
        "view_id": "m_plus_g",
        "memory_mode": "aligned",
        "atom_encoder_candidate": candidate,
        "factor_under_test": (
            "level_embedding"
            if candidate == "l0_high_level_aware_phi"
            else "reference_or_l0_separation"
        ),
    }


def _without_legacy_shell_argument(function: Callable[..., object]) -> Callable[..., object]:
    def wrapped(**kwargs: object) -> object:
        shell = kwargs.pop("shell_fusion_mode", None)
        if shell != "not_applicable":
            raise AnchoredAtomEncoderScreenError(
                "generic reducer screen received a legacy V4 shell mode"
            )
        return function(**kwargs)

    return wrapped


def run(args: argparse.Namespace) -> dict[str, object]:
    candidate = str(args.atom_encoder_candidate)
    contract = cell_contract(str(args.cell), candidate)
    bridge_args = argparse.Namespace(**vars(args))
    bridge_args.shell_fusion_mode = "not_applicable"

    def build_contract(_cell: str, _shell: str) -> dict[str, str]:
        if _cell != contract["cell"] or _shell != "not_applicable":
            raise AnchoredAtomEncoderScreenError("screen contract bridge drifted")
        return dict(contract)

    report = run_shared_screen(
        bridge_args,
        model_loader=_without_legacy_shell_argument(_MODEL_LOADERS[candidate]),
        initialization_contract_builder=_without_legacy_shell_argument(
            _CONTRACT_BUILDERS[candidate]
        ),
        contract_builder=build_contract,
        schema_version=SCHEMA_VERSION,
        scope="pf1_final_atom_encoder_selection_not_formal_pretraining",
        manifest_filename=MANIFEST_FILENAME,
    )
    report["selection_boundary"] = {
        "one_percent_is_elimination_only": True,
        "ten_percent_is_required_for_architecture_freeze": True,
        "b2d_and_f3d_pair_required": True,
        "level_embedding_is_single_isolated_factor": (
            candidate == "l0_high_level_aware_phi"
        ),
        "attachment_role_feature": False,
        "explicit_presence_feature": False,
        "historical_v4_is_external_complexity_upper_bound": True,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("B2D", "F3D"), required=True)
    parser.add_argument(
        "--atom-encoder-candidate", choices=CANDIDATES, required=True
    )
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--anchored-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--semantic-plan-sha256", required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
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
