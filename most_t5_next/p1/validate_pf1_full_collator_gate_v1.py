#!/usr/bin/env python3
"""Stream every frozen PF-1 row through the four production collators.

The gate is deliberately read-only with respect to the paired release.  It
uses one reader pass per split and keeps only the current micro-batch.  There
is no sampling, replacement, truncation, or alternate collation path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from most_t5_next.p1.atom_production_bridge import collate_production_atom_record
from most_t5_next.p1.build_pf1_paired_release_v1 import (
    MANIFEST_NAME as PAIRED_MANIFEST_NAME,
    TOKENIZER_DIRECTORY,
    PF1PairedReleaseReader,
)
from most_t5_next.p1.experiment_grid import (
    P1_CONDITION_SPECS,
    P1ConditionBatch,
    validate_a1_m1_geometry_atom_parity,
)
from most_t5_next.p1.production_bridge import collate_production_motif_record
from most_t5_next.p1.runtime_bridge import pad_ce_first_batch
from most_t5_next.p1.training_adapter import to_four_grid_batch_encoding
from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)


REPORT_SCHEMA = "most-t5-p1/pf1-full-collator-gate/v1"
DEFAULT_BATCH_SIZE = 8
CONDITION_ORDER = tuple(P1_CONDITION_SPECS)
TRAIN_CORRUPTION_SEED = 0
DEV_CORRUPTION_SEED = 1
DEV_CORRUPTION_EPOCH = 0
MASK_PROBABILITY = 0.15
TRAIN_GATE_EPOCH = 0


class PF1FullCollatorGateError(RuntimeError):
    """The complete PF-1 release cannot enter four-grid training."""


def _load_frozen_runner_collator() -> Callable[..., P1ConditionBatch]:
    """Lazily bind the exact runner collator and reject protocol drift."""

    # The runner imports the PyTorch optimizer, so this stays lazy for the
    # pure-Python structural tests.
    from most_t5_next.p1 import run_pf1_four_grid_v1 as runner

    observed = (
        tuple(runner.CONDITION_ORDER),
        runner.TRAIN_CORRUPTION_SEED,
        runner.DEV_CORRUPTION_SEED,
        runner.DEV_CORRUPTION_EPOCH,
        runner.MASK_PROBABILITY,
    )
    expected = (
        CONDITION_ORDER,
        TRAIN_CORRUPTION_SEED,
        DEV_CORRUPTION_SEED,
        DEV_CORRUPTION_EPOCH,
        MASK_PROBABILITY,
    )
    if observed != expected:
        raise PF1FullCollatorGateError(
            "collator gate constants differ from the frozen PF-1 runner"
        )
    return runner.collate_pf1_condition


@dataclass
class Coverage:
    members: int = 0
    members_selected: int = 0
    eligible_units: int = 0
    selected_units: int = 0
    eligible_identity_tokens: int = 0
    selected_identity_tokens: int = 0
    eligible_atoms: int = 0
    selected_atoms: int = 0
    max_selected_units: int = 0
    max_sentinels: int = 0

    def add(
        self,
        *,
        identity_lengths: Sequence[int],
        selected: Sequence[int],
        atom_count: int,
        selected_atom_count: int,
    ) -> None:
        self.members += 1
        self.members_selected += int(bool(selected))
        self.eligible_units += len(identity_lengths)
        self.selected_units += len(selected)
        self.eligible_identity_tokens += sum(identity_lengths)
        self.selected_identity_tokens += sum(
            identity_lengths[index] for index in selected
        )
        self.eligible_atoms += atom_count
        self.selected_atoms += selected_atom_count
        self.max_selected_units = max(self.max_selected_units, len(selected))
        # One sentinel per selected span plus the terminal sentinel before EOS.
        self.max_sentinels = max(self.max_sentinels, len(selected) + 1)

    def merged(self, other: "Coverage") -> "Coverage":
        result = Coverage()
        for name in (
            "members",
            "members_selected",
            "eligible_units",
            "selected_units",
            "eligible_identity_tokens",
            "selected_identity_tokens",
            "eligible_atoms",
            "selected_atoms",
        ):
            setattr(result, name, getattr(self, name) + getattr(other, name))
        result.max_selected_units = max(self.max_selected_units, other.max_selected_units)
        result.max_sentinels = max(self.max_sentinels, other.max_sentinels)
        return result

    def report(self) -> dict[str, object]:
        if min(
            self.members,
            self.eligible_units,
            self.eligible_identity_tokens,
            self.eligible_atoms,
        ) <= 0:
            raise PF1FullCollatorGateError("mask coverage domain is empty")
        return {
            "member_count": self.members,
            "members_with_selected_mask": self.members_selected,
            "eligible_mask_units": self.eligible_units,
            "selected_mask_units": self.selected_units,
            "eligible_identity_tokens": self.eligible_identity_tokens,
            "selected_identity_tokens": self.selected_identity_tokens,
            "eligible_atoms": self.eligible_atoms,
            "selected_atoms": self.selected_atoms,
            "mask_unit_coverage": self.selected_units / self.eligible_units,
            "identity_token_coverage": (
                self.selected_identity_tokens / self.eligible_identity_tokens
            ),
            "atom_coverage": self.selected_atoms / self.eligible_atoms,
            "member_coverage": self.members_selected / self.members,
            "max_selected_mask_units_per_member": self.max_selected_units,
            "max_sentinel_tokens_per_target": self.max_sentinels,
        }


@dataclass
class ConditionAudit:
    condition_id: str
    batch_count: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "dev": 0}
    )
    member_count: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "dev": 0}
    )
    max_batch: int = 0
    max_input: int = 0
    max_target: int = 0
    max_atoms: int = 0
    tensor_keys: tuple[str, ...] | None = None
    tensor_dtypes: dict[str, str] = field(default_factory=dict)
    tensor_ranks: dict[str, int] = field(default_factory=dict)
    e3fp_level_width: int | None = None

    def observe(
        self,
        *,
        split: str,
        batch: P1ConditionBatch,
        encoded: Mapping[str, Any],
        torch_module: Any,
    ) -> None:
        geometry = self.condition_id in {"A1", "M1"}
        keys = ("input_ids", "attention_mask", "labels") + (
            ("e3fp_ids", "e3fp_atom_mask", "e3fp_atom_to_token")
            if geometry
            else ()
        )
        if tuple(encoded) != keys:
            raise PF1FullCollatorGateError(
                self.condition_id + " model-input allowlist changed"
            )
        shapes = {key: tuple(int(value) for value in encoded[key].shape) for key in keys}
        size = len(batch.ce_batch.record_ids)
        if not (
            len(shapes["input_ids"]) == len(shapes["labels"]) == 2
            and shapes["attention_mask"] == shapes["input_ids"]
            and shapes["input_ids"][0] == shapes["labels"][0] == size
        ):
            raise PF1FullCollatorGateError(self.condition_id + " CE shape changed")
        if geometry:
            atom_shape = shapes["e3fp_ids"]
            if not (
                len(atom_shape) == 3
                and shapes["e3fp_atom_mask"] == atom_shape[:2]
                and shapes["e3fp_atom_to_token"] == atom_shape[:2]
                and atom_shape[0] == size
            ):
                raise PF1FullCollatorGateError(
                    self.condition_id + " geometry shape changed"
                )
            if self.e3fp_level_width not in {None, atom_shape[2]}:
                raise PF1FullCollatorGateError("E3FP level width changed")
            self.e3fp_level_width = atom_shape[2]

        expected_dtype = {
            "input_ids": torch_module.long,
            "attention_mask": torch_module.long,
            "labels": torch_module.long,
            "e3fp_ids": torch_module.long,
            "e3fp_atom_mask": torch_module.bool,
            "e3fp_atom_to_token": torch_module.long,
        }
        dtypes = {}
        for key in keys:
            label = str(encoded[key].dtype)
            dtypes[key] = label[6:] if label.startswith("torch.") else label
        ranks = {key: len(shapes[key]) for key in keys}
        if any(encoded[key].dtype != expected_dtype[key] for key in keys):
            raise PF1FullCollatorGateError(self.condition_id + " dtype changed")
        if self.tensor_keys not in {None, keys} or (
            self.tensor_dtypes and self.tensor_dtypes != dtypes
        ) or (self.tensor_ranks and self.tensor_ranks != ranks):
            raise PF1FullCollatorGateError(
                self.condition_id + " tensor contract changed between batches"
            )
        self.tensor_keys, self.tensor_dtypes, self.tensor_ranks = keys, dtypes, ranks
        self.batch_count[split] += 1
        self.member_count[split] += size
        self.max_batch = max(self.max_batch, size)
        self.max_input = max(self.max_input, max(batch.ce_batch.input_lengths))
        self.max_target = max(self.max_target, max(batch.ce_batch.target_lengths))
        if batch.geometry is not None:
            self.max_atoms = max(self.max_atoms, max(batch.geometry.atom_lengths))

    def report(self, coverage: Mapping[str, object]) -> dict[str, object]:
        total_coverage = coverage["total"]
        assert isinstance(total_coverage, Mapping)
        return {
            "batch_count": {
                **self.batch_count,
                "total": sum(self.batch_count.values()),
            },
            "member_count": {
                **self.member_count,
                "total": sum(self.member_count.values()),
            },
            "selected_mask_coverage": coverage,
            "maxima": {
                "batch_members": self.max_batch,
                "input_tokens": self.max_input,
                "target_tokens": self.max_target,
                "atom_rows": self.max_atoms,
                "selected_mask_units_per_member": total_coverage[
                    "max_selected_mask_units_per_member"
                ],
                "sentinel_tokens_per_target": total_coverage[
                    "max_sentinel_tokens_per_target"
                ],
            },
            "tensor_contract": {
                "verified_batch_count": sum(self.batch_count.values()),
                "model_input_keys": list(self.tensor_keys or ()),
                "dtypes": self.tensor_dtypes,
                "ranks": self.tensor_ranks,
                "e3fp_level_width": self.e3fp_level_width,
            },
        }


def _audit_examples(
    rows: Sequence[Any], *, tokenizer: Any, seed: int, epoch: int
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    atom = tuple(
        collate_production_atom_record(
            row.atom_record,
            tokenizer=tokenizer,
            seed=seed,
            epoch=epoch,
            mask_probability=MASK_PROBABILITY,
        )
        for row in rows
    )
    motif = tuple(
        collate_production_motif_record(
            row.motif_record,
            tokenizer=tokenizer,
            seed=seed,
            epoch=epoch,
            mask_probability=MASK_PROBABILITY,
        )
        for row in rows
    )
    return atom, motif


def _update_coverage(
    atom_coverage: Coverage,
    motif_coverage: Coverage,
    rows: Sequence[Any],
    atom_examples: Sequence[Any],
    motif_examples: Sequence[Any],
) -> None:
    for row, atom_example, motif_example in zip(
        rows, atom_examples, motif_examples
    ):
        atom_selected = tuple(
            index
            for index, selected in enumerate(atom_example.identity_recovery_mask)
            if selected
        )
        atom_lengths = tuple(
            span.stop - span.start for span in row.atom_record.atom_identity_spans
        )
        atom_coverage.add(
            identity_lengths=atom_lengths,
            selected=atom_selected,
            atom_count=len(atom_lengths),
            selected_atom_count=len(atom_selected),
        )

        motif_selected = tuple(
            index
            for index, selected in enumerate(motif_example.identity_recovery_mask)
            if selected
        )
        motif_lengths = tuple(
            span.stop - span.start for span in row.motif_record.identity_spans
        )
        selected_set = frozenset(motif_selected)
        motif_coverage.add(
            identity_lengths=motif_lengths,
            selected=motif_selected,
            atom_count=len(row.motif_record.atom_to_logical_motif),
            selected_atom_count=sum(
                motif_id in selected_set
                for motif_id in row.motif_record.atom_to_logical_motif
            ),
        )


def validate_full_collator_gate(
    *,
    reader: Any,
    tokenizer_runtime: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
    torch_module: Any | None = None,
    condition_collator: Callable[..., P1ConditionBatch] | None = None,
    tensorizer: Callable[..., Mapping[str, Any]] = to_four_grid_batch_encoding,
) -> dict[str, object]:
    """Replay all train/dev members once and return a compact pass report."""

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise PF1FullCollatorGateError("batch_size must be positive")
    if torch_module is None:
        try:
            import torch as torch_module
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime boundary
            raise PF1FullCollatorGateError("PyTorch is required") from exc
    collate = condition_collator or _load_frozen_runner_collator()
    expected = {
        "train": int(reader.train_member_count),
        "dev": int(reader.dev_member_count),
    }
    if min(expected.values()) <= 0:
        raise PF1FullCollatorGateError("train and dev must be nonempty")

    audits = {condition: ConditionAudit(condition) for condition in CONDITION_ORDER}
    coverage = {
        family: {split: Coverage() for split in ("train", "dev")}
        for family in ("atom", "motif")
    }
    split_report: dict[str, object] = {}

    for split in ("train", "dev"):
        seed = TRAIN_CORRUPTION_SEED if split == "train" else DEV_CORRUPTION_SEED
        epoch = TRAIN_GATE_EPOCH if split == "train" else DEV_CORRUPTION_EPOCH
        iterator = (
            reader.iter_train_epoch(epoch=epoch, batch_size=batch_size)
            if split == "train"
            else reader.iter_dev(batch_size=batch_size)
        )
        members = batches_seen = 0
        first_index = last_index = None
        for raw_rows in iterator:
            rows = tuple(raw_rows)
            if not rows or len(rows) > batch_size:
                raise PF1FullCollatorGateError(split + " yielded an invalid batch")
            indices = tuple(row.schedule_index for row in rows)
            ordered = ((last_index,) if last_index is not None else ()) + indices
            if any(
                right <= left
                for left, right in zip(ordered, ordered[1:])
            ):
                raise PF1FullCollatorGateError(split + " frozen order changed")
            if first_index is None:
                first_index = indices[0]
            last_index = indices[-1]
            if members + len(rows) > expected[split]:
                raise PF1FullCollatorGateError(split + " exceeded frozen membership")
            for row in rows:
                atom, motif = row.atom_record, row.motif_record
                if not (
                    atom.record_id == motif.record_id
                    and atom.geometry_record_content_sha256
                    == motif.geometry_record_content_sha256
                    and atom.model_to_source_atom_index
                    == motif.model_to_source_atom_index
                    and atom.full_e3fp_ids == motif.full_e3fp_ids
                ):
                    raise PF1FullCollatorGateError(
                        split + " raw paired geometry/source mapping differs"
                    )

            try:
                grid = {
                    condition: collate(
                        rows,
                        condition_id=condition,
                        tokenizer_runtime=tokenizer_runtime,
                        seed=seed,
                        epoch=epoch,
                    )
                    for condition in CONDITION_ORDER
                }
                atom_examples, motif_examples = _audit_examples(
                    rows, tokenizer=tokenizer_runtime, seed=seed, epoch=epoch
                )
            except Exception as exc:
                raise PF1FullCollatorGateError(
                    f"{split} batch {batches_seen} collator reject: {exc}"
                ) from exc
            if grid["A0"].ce_batch != grid["A1"].ce_batch:
                raise PF1FullCollatorGateError("A0/A1 CE differs")
            if grid["M0"].ce_batch != grid["M1"].ce_batch:
                raise PF1FullCollatorGateError("M0/M1 CE differs")
            try:
                validate_a1_m1_geometry_atom_parity(grid["A1"], grid["M1"])
            except Exception as exc:
                raise PF1FullCollatorGateError("A1/M1 geometry parity differs") from exc
            if (
                pad_ce_first_batch(
                    atom_examples, pad_token_id=tokenizer_runtime.pad_token_id
                )
                != grid["A0"].ce_batch
                or pad_ce_first_batch(
                    motif_examples, pad_token_id=tokenizer_runtime.pad_token_id
                )
                != grid["M0"].ce_batch
            ):
                raise PF1FullCollatorGateError("mask audit and four-grid CE differ")
            _update_coverage(
                coverage["atom"][split],
                coverage["motif"][split],
                rows,
                atom_examples,
                motif_examples,
            )
            for condition, batch in grid.items():
                try:
                    encoded = tensorizer(
                        batch,
                        torch_module=torch_module,
                        batch_encoding_cls=dict,
                    )
                except Exception as exc:
                    raise PF1FullCollatorGateError(
                        condition + " tensor conversion rejected"
                    ) from exc
                audits[condition].observe(
                    split=split,
                    batch=batch,
                    encoded=encoded,
                    torch_module=torch_module,
                )
            members += len(rows)
            batches_seen += 1

        if members != expected[split] or batches_seen != math.ceil(
            expected[split] / batch_size
        ):
            raise PF1FullCollatorGateError(
                f"{split} reader ended at {members:,}/{expected[split]:,} members"
            )
        split_report[split] = {
            "member_count": members,
            "batch_count": batches_seen,
            "first_schedule_index": first_index,
            "last_schedule_index": last_index,
            "strictly_increasing_frozen_order": True,
        }

    family_reports = {}
    for family in ("atom", "motif"):
        total = coverage[family]["train"].merged(coverage[family]["dev"])
        family_reports[family] = {
            "train": coverage[family]["train"].report(),
            "dev": coverage[family]["dev"].report(),
            "total": total.report(),
        }
    conditions = {
        condition: audit.report(
            family_reports["atom" if condition.startswith("A") else "motif"]
        )
        for condition, audit in audits.items()
    }
    total_members = expected["train"] + expected["dev"]
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "created_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "scope": "complete_pf1_release_cpu_collator_and_tensor_boundary_only",
        "interpretation": {
            "model_forward_or_optimizer_step": False,
            "architecture_effect_claim": False,
            "raw_atom_motif_ce_directly_comparable": False,
        },
        "schedule": {
            "batch_size": batch_size,
            "train_seed": TRAIN_CORRUPTION_SEED,
            "train_epoch": TRAIN_GATE_EPOCH,
            "dev_seed": DEV_CORRUPTION_SEED,
            "dev_epoch": DEV_CORRUPTION_EPOCH,
            "reader_passes_per_split": 1,
            "record_sampling": False,
            "sample_replacement": False,
            "sequence_truncation": False,
            "maximum_decoded_records_resident": batch_size,
            "full_release_record_residency": False,
        },
        "members": {**expected, "total": total_members, "rejected": 0},
        "splits": split_report,
        "parity": {
            "A0_A1_CE_equal_all_batches": True,
            "M0_M1_CE_equal_all_batches": True,
            "A1_M1_geometry_rows_E3FP_and_source_mapping_equal_all_batches": True,
        },
        "sentinel_contract": {
            "available_sentinel_tokens": len(tokenizer_runtime.sentinel_token_ids),
            "maximum_sentinel_token_id": max(tokenizer_runtime.sentinel_token_ids),
            "selected_plus_terminal_capacity_verified_all_members": True,
        },
        "conditions": conditions,
        "rejects": {"member_count": 0, "batch_count": 0},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", required=True)
    parser.add_argument("--base-tokenizer-snapshot", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def run(args: argparse.Namespace, *, torch_module: Any | None = None) -> dict[str, object]:
    release = Path(args.paired_release).expanduser().resolve()
    output = Path(args.output_report).expanduser().resolve()
    if output.exists():
        raise PF1FullCollatorGateError("output_report must be a new file")
    if not (release / PAIRED_MANIFEST_NAME).is_file():
        raise PF1FullCollatorGateError("paired release manifest is absent")
    tokenizer = load_verified_canary_union_tokenizer(
        base_snapshot=Path(args.base_tokenizer_snapshot).expanduser().resolve(),
        output_dir=release / TOKENIZER_DIRECTORY,
    )
    reader = PF1PairedReleaseReader(release)
    report = validate_full_collator_gate(
        reader=reader,
        tokenizer_runtime=tokenizer.runtime,
        batch_size=args.batch_size,
        torch_module=torch_module,
    )
    report["release"] = {
        "paired_release_root": str(release),
        "paired_release_schema": reader.manifest.get("schema_version"),
        "union_tokenizer_contract_sha256": tokenizer.runtime.tokenizer_contract_sha256,
        "union_tokenizer_snapshot_sha256": tokenizer.runtime.tokenizer_snapshot_sha256,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "PF1FullCollatorGateError",
    "REPORT_SCHEMA",
    "build_parser",
    "run",
    "validate_full_collator_gate",
]
