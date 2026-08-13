"""Merge three completed PF-10 grammar-cell reports into one causal screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from .run_pf10_factorized_grammar_v1 import SCHEMA_VERSION as CELL_SCHEMA


SCHEMA_VERSION = "most-t5-p2/pf10-factorized-grammar-merge/v1"


class PF10GrammarMergeError(RuntimeError):
    """Cell reports do not represent one comparable PF-10 matrix."""


def _load(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not (
        value.get("schema_version") == CELL_SCHEMA and value.get("status") == "pass"
    ):
        raise PF10GrammarMergeError("one grammar cell is absent or not passed")
    return value


def _final(report: Mapping[str, object]) -> Mapping[str, object]:
    rows = report.get("evaluations")
    if not isinstance(rows, list) or not rows:
        raise PF10GrammarMergeError("grammar cell lacks evaluations")
    final = rows[-1]
    if not isinstance(final, dict) or final.get("update") != 10000:
        raise PF10GrammarMergeError("grammar cell lacks its frozen final evaluation")
    return final


def merge_grammar_matrix(
    *, b0_manifest: Path, b2d_manifest: Path, f3d_manifest: Path, output: Path
) -> dict[str, object]:
    reports = {
        "B0": _load(b0_manifest),
        "B2D": _load(b2d_manifest),
        "F3D": _load(f3d_manifest),
    }
    for cell, report in reports.items():
        if report.get("cell") != cell:
            raise PF10GrammarMergeError("grammar cell label differs from its input")
    protocols = {cell: report.get("protocol") for cell, report in reports.items()}
    if any(not isinstance(value, dict) for value in protocols.values()):
        raise PF10GrammarMergeError("grammar cells lack optimization protocols")
    common_protocol = dict(protocols["B0"])  # type: ignore[arg-type]
    common_protocol.pop("micro_batch_size", None)
    common_protocol.pop("gradient_accumulation_steps", None)
    common_data_contract = reports["B0"].get("data_contract")
    if not isinstance(common_data_contract, dict):
        raise PF10GrammarMergeError("grammar cells lack their common data contract")
    update_axes = []
    for report in reports.values():
        protocol = dict(report["protocol"])  # type: ignore[arg-type]
        effective_batch = int(protocol["micro_batch_size"]) * int(
            protocol["gradient_accumulation_steps"]
        )
        protocol.pop("micro_batch_size")
        protocol.pop("gradient_accumulation_steps")
        if protocol != common_protocol or effective_batch != 128:
            raise PF10GrammarMergeError(
                "grammar cells differ beyond an effective-batch-preserving microbatch split"
            )
        if report.get("data_contract") != common_data_contract:
            raise PF10GrammarMergeError("grammar cells use different data or tokenizer contracts")
        update_axes.append(tuple(row["update"] for row in report["evaluations"]))  # type: ignore[index]
    if len(set(update_axes)) != 1:
        raise PF10GrammarMergeError("grammar cells use different evaluation schedules")

    final = {cell: _final(report) for cell, report in reports.items()}
    nll = {cell: float(row["token_weighted_nll"]) for cell, row in final.items()}
    accuracy = {cell: float(row["masked_token_accuracy"]) for cell, row in final.items()}
    diagnostics = reports["F3D"].get("f3d_state_diagnostics")
    if not isinstance(diagnostics, dict):
        raise PF10GrammarMergeError("F3D lacks zero and matched-shuffle diagnostics")
    zero_delta = float(diagnostics["zero_minus_aligned_delta_nll"])
    shuffle_delta = float(diagnostics["shuffle_minus_aligned_delta_nll"])
    directional = {
        "f3d_nll_lower_than_b2d": nll["F3D"] < nll["B2D"],
        "f3d_nll_lower_than_b0": nll["F3D"] < nll["B0"],
        "zero_worsens_f3d_nll": zero_delta > 0.0,
        "matched_shuffle_worsens_f3d_nll": shuffle_delta > 0.0,
    }
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "pf10_one_seed_causal_screen",
        "data_contract": common_data_contract,
        "cell_batch_geometry": {
            cell: {
                "micro_batch_size": report["protocol"]["micro_batch_size"],  # type: ignore[index]
                "gradient_accumulation_steps": report["protocol"]["gradient_accumulation_steps"],  # type: ignore[index]
                "effective_members_per_update": 128,
            }
            for cell, report in reports.items()
        },
        "final_token_weighted_nll": nll,
        "final_masked_token_accuracy": accuracy,
        "nll_differences": {
            "f3d_minus_b2d": nll["F3D"] - nll["B2D"],
            "f3d_minus_b0": nll["F3D"] - nll["B0"],
            "f3d_zero_minus_aligned": zero_delta,
            "f3d_matched_shuffle_minus_aligned": shuffle_delta,
        },
        "directional_gates": directional,
        "all_directional_gates_pass": all(directional.values()),
        "interpretation_boundary": (
            "One PF-10 seed is an architecture screen, not final pretraining or "
            "a statistical significance claim."
        ),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-manifest", type=Path, required=True)
    parser.add_argument("--b2d-manifest", type=Path, required=True)
    parser.add_argument("--f3d-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    merge_grammar_matrix(
        b0_manifest=args.b0_manifest,
        b2d_manifest=args.b2d_manifest,
        f3d_manifest=args.f3d_manifest,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PF10GrammarMergeError", "SCHEMA_VERSION", "merge_grammar_matrix"]
