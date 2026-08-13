#!/usr/bin/env python3
"""Fine-tune an anchored 3D-MotifT5 encoder on frozen QM9 properties.

This is a representation probe, not a numeric text-generation benchmark.  A
shared regression head consumes the mean-pooled T5 encoder state, partial
HOMO/LUMO/gap labels are masked, and metrics are reported in the original
Hartree unit.  All cells share the same model/head initialization and schedule;
only the atom-state provider and shell-fusion mode differ.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from most_t5_next.p1.build_anchored_candidate_tokenizer_v1 import (
    load_verified_anchored_candidate_tokenizer,
)
from most_t5_next.p2.build_qm9_anchored_probe_cache_v1 import (
    PROPERTY_NAMES,
    record_from_document,
)
from most_t5_next.p2.factorized_model_init_v4 import (
    load_deterministic_factorized_model_v4,
)
from most_t5_next.p2.factorized_model_init_v5 import (
    load_deterministic_factorized_model_v5,
)
from most_t5_next.p2.factorized_model_init_v6 import (
    load_deterministic_factorized_model_v6,
)
from most_t5_next.p2.factorized_model_init_v7 import (
    load_deterministic_factorized_model_v7,
)
from most_t5_next.p2.factorized_model_init_v8 import (
    load_deterministic_factorized_model_v8,
)
from most_t5_next.p2.factorized_model_init_v9 import (
    load_deterministic_factorized_model_v9,
)
from most_t5_next.p2.motif_geometry_adapter_v4 import SHELL_FUSION_MODES


SCHEMA_VERSION = "most-t5-p2/qm9-anchored-property-probe/v1"
TRAIN_SEED = 20260810
ADAPTER_SEED = 20260809
GEOMETRY_FUSION_SEED = 20260808
CELLS = ("B0", "B2D", "F3D")
REFERENCE_SHELL_MODES = (
    "reference_fixed_four_mean",
    "adaptive_l0_high",
    "linear_l0_high",
    "minimal_phi_l0_high",
    "level_aware_phi_l0_high",
)
ALL_SHELL_MODES = SHELL_FUSION_MODES + REFERENCE_SHELL_MODES


class QM9AnchoredProbeError(RuntimeError):
    """The property probe contract or one training step is invalid."""


@dataclass(frozen=True)
class ProbeBatch:
    inputs: Mapping[str, Tensor]
    targets: Tensor
    target_mask: Tensor
    record_ids: tuple[str, ...]

    def pin_memory(self) -> "ProbeBatch":
        return ProbeBatch(
            inputs={key: value.pin_memory() for key, value in self.inputs.items()},
            targets=self.targets.pin_memory(),
            target_mask=self.target_mask.pin_memory(),
            record_ids=self.record_ids,
        )

    def to(self, device: torch.device) -> "ProbeBatch":
        return ProbeBatch(
            inputs={
                key: value.to(device=device, non_blocking=True)
                for key, value in self.inputs.items()
            },
            targets=self.targets.to(device=device, non_blocking=True),
            target_mask=self.target_mask.to(device=device, non_blocking=True),
            record_ids=self.record_ids,
        )


class QM9ProbeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_root: Path,
        *,
        split: str,
        target_overlay_dir: Path | None = None,
        property_names: Sequence[str] = PROPERTY_NAMES,
    ) -> None:
        if split not in {"train", "dev", "test"}:
            raise QM9AnchoredProbeError("split must be train, dev or test")
        property_names = tuple(str(name) for name in property_names)
        if not property_names or len(set(property_names)) != len(property_names):
            raise QM9AnchoredProbeError("property names must be unique and non-empty")
        overlay_by_record_id: dict[str, Mapping[str, object]] | None = None
        if target_overlay_dir is not None:
            overlay_by_record_id = {}
            overlay_path = Path(target_overlay_dir) / "targets.jsonl"
            if not overlay_path.is_file():
                raise QM9AnchoredProbeError("target overlay lacks targets.jsonl")
            with overlay_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    overlay = json.loads(line)
                    record_id = overlay.get("record_id")
                    targets = overlay.get("targets")
                    if (
                        not isinstance(record_id, str)
                        or record_id in overlay_by_record_id
                        or not isinstance(targets, dict)
                        or any(name not in targets for name in property_names)
                    ):
                        raise QM9AnchoredProbeError("target overlay row is invalid")
                    overlay_by_record_id[record_id] = overlay
        self.rows = []
        path = Path(cache_root) / "records.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split") != split:
                    continue
                if overlay_by_record_id is not None:
                    record = row.get("record")
                    if not isinstance(record, dict):
                        raise QM9AnchoredProbeError("probe row lacks record document")
                    overlay = overlay_by_record_id.get(str(record.get("record_id")))
                    if overlay is None:
                        continue
                    if (
                        overlay.get("split") != split
                        or overlay.get("storage_key") != record.get("storage_key")
                    ):
                        raise QM9AnchoredProbeError("target overlay binding differs")
                    row = dict(row)
                    targets = overlay["targets"]
                    row["targets_hartree"] = [float(targets[name]) for name in property_names]
                    row["target_mask"] = [True] * len(property_names)
                self.rows.append(row)
        if not self.rows:
            raise QM9AnchoredProbeError(f"QM9 {split} split is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        return self.rows[index]


def training_target_statistics(
    dataset: QM9ProbeDataset,
    *,
    property_names: Sequence[str] = PROPERTY_NAMES,
) -> tuple[Tensor, Tensor]:
    values = [[] for _ in property_names]
    for row in dataset.rows:
        for index, (value, present) in enumerate(
            zip(row["targets_hartree"], row["target_mask"])
        ):
            if present:
                values[index].append(float(value))
    if any(len(row) < 2 for row in values):
        raise QM9AnchoredProbeError("one QM9 property has insufficient training labels")
    means = torch.tensor([statistics.fmean(row) for row in values], dtype=torch.float32)
    stds = torch.tensor([statistics.stdev(row) for row in values], dtype=torch.float32)
    if not bool(torch.isfinite(stds).all()) or bool((stds <= 0).any()):
        raise QM9AnchoredProbeError("training target standard deviation is invalid")
    return means, stds


class ProbeCollator:
    def __init__(
        self,
        *,
        pad_token_id: int,
        cell: str,
        property_names: Sequence[str] = PROPERTY_NAMES,
    ) -> None:
        if cell not in CELLS:
            raise QM9AnchoredProbeError("unknown probe cell")
        self.pad_token_id = int(pad_token_id)
        self.cell = cell
        self.property_names = tuple(property_names)
        if not self.property_names:
            raise QM9AnchoredProbeError("property names must be non-empty")

    def __call__(self, rows: Sequence[Mapping[str, object]]) -> ProbeBatch:
        if not rows:
            raise QM9AnchoredProbeError("probe batch is empty")
        records = [record_from_document(row["record"]) for row in rows]
        token_width = max(len(record.input_ids) for record in records)
        atom_width = max(len(record.full_e3fp_ids) for record in records)
        motif_width = max(len(record.identity_spans) for record in records)
        batch = len(records)
        input_ids = torch.full((batch, token_width), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros((batch, token_width), dtype=torch.bool)
        states = torch.full((batch, atom_width, 4), -1, dtype=torch.long)
        atom_mask = torch.zeros((batch, atom_width), dtype=torch.bool)
        atom_to_motif = torch.full((batch, atom_width), -1, dtype=torch.long)
        attachment = torch.zeros((batch, atom_width), dtype=torch.bool)
        motif_mask = torch.zeros((batch, motif_width), dtype=torch.bool)
        carriers = torch.full((batch, motif_width), -1, dtype=torch.long)
        spans = torch.full((batch, motif_width, 2), -1, dtype=torch.long)
        endpoints = torch.full((batch, token_width), -1, dtype=torch.long)
        targets = torch.zeros((batch, len(self.property_names)), dtype=torch.float32)
        target_mask = torch.zeros_like(targets, dtype=torch.bool)
        record_ids = []
        for index, (row, record) in enumerate(zip(rows, records)):
            token_count = len(record.input_ids)
            atom_count = len(record.full_e3fp_ids)
            motif_count = len(record.identity_spans)
            source_states = (
                row["morgan_state_ids"]
                if self.cell == "B2D"
                else record.full_e3fp_ids
            )
            if len(source_states) != atom_count:
                raise QM9AnchoredProbeError("atom-state rows differ from record atoms")
            input_ids[index, :token_count] = torch.tensor(record.input_ids)
            attention[index, :token_count] = True
            states[index, :atom_count] = torch.tensor(source_states, dtype=torch.long)
            atom_mask[index, :atom_count] = torch.tensor(record.atom_valid_mask)
            atom_to_motif[index, :atom_count] = torch.tensor(record.atom_to_logical_motif)
            attachment[index, :atom_count] = torch.tensor(record.atom_is_attachment)
            motif_mask[index, :motif_count] = True
            carriers[index, :motif_count] = torch.tensor(record.logical_to_carrier)
            spans[index, :motif_count] = torch.tensor(
                [(span.start, span.stop) for span in record.identity_spans]
            )
            endpoints[index, :token_count] = torch.tensor(
                record.connection_token_to_atom
            )
            for prop, (value, present) in enumerate(
                zip(row["targets_hartree"], row["target_mask"])
            ):
                if present:
                    targets[index, prop] = float(value)
                    target_mask[index, prop] = True
            record_ids.append(record.record_id)
        return ProbeBatch(
            inputs={
                "input_ids": input_ids,
                "attention_mask": attention,
                "e3fp_input_ids": states,
                "atom_mask": atom_mask,
                "atom_to_motif": atom_to_motif,
                "motif_mask": motif_mask,
                "motif_to_carrier": carriers,
                "identity_span_bounds": spans,
                "endpoint_token_to_atom": endpoints,
                "atom_is_attachment": attachment,
            },
            targets=targets,
            target_mask=target_mask,
            record_ids=tuple(record_ids),
        )


class AnchoredQM9Regressor(nn.Module):
    def __init__(self, backbone: nn.Module, *, hidden_size: int, output_size: int = len(PROPERTY_NAMES)) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size),
        )

    def forward(
        self,
        inputs: Mapping[str, Tensor],
        *,
        state_memory_mode: str,
        geometry_component_mode: str,
    ) -> Tensor:
        input_ids = inputs["input_ids"]
        attention = inputs["attention_mask"]
        embeddings = self.backbone.t5.get_input_embeddings()(input_ids)
        encoded = self.backbone.adapter.encode(
            embeddings,
            attention_mask=attention,
            e3fp_input_ids=inputs["e3fp_input_ids"],
            atom_mask=inputs["atom_mask"],
            atom_to_motif=inputs["atom_to_motif"],
            motif_mask=inputs["motif_mask"],
            motif_to_carrier=inputs["motif_to_carrier"],
            identity_span_bounds=inputs["identity_span_bounds"],
            endpoint_token_to_atom=inputs["endpoint_token_to_atom"],
            atom_is_attachment=inputs["atom_is_attachment"],
            state_memory_mode=state_memory_mode,
            geometry_component_mode=geometry_component_mode,
        )
        output = self.backbone.t5.encoder(
            inputs_embeds=encoded.fused_embeddings,
            attention_mask=attention,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        weight = attention.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1)
        return self.head(pooled)


def _evaluate(
    model: AnchoredQM9Regressor,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    means: Tensor,
    stds: Tensor,
    memory_mode: str,
    component_mode: str,
    property_names: Sequence[str] = PROPERTY_NAMES,
    target_units: Mapping[str, str] | None = None,
) -> dict[str, object]:
    model.eval()
    property_names = tuple(property_names)
    absolute = torch.zeros(len(property_names), dtype=torch.float64)
    squared = torch.zeros_like(absolute)
    standardized_absolute = torch.zeros_like(absolute)
    counts = torch.zeros(len(property_names), dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                standardized = model(
                    batch.inputs,
                    state_memory_mode=memory_mode,
                    geometry_component_mode=component_mode,
                )
            predictions = standardized.float().cpu() * stds + means
            errors = predictions - batch.targets.cpu()
            standardized_errors = errors / stds
            mask = batch.target_mask.cpu()
            absolute += (errors.abs() * mask).sum(dim=0).double()
            squared += (errors.square() * mask).sum(dim=0).double()
            standardized_absolute += (standardized_errors.abs() * mask).sum(dim=0).double()
            counts += mask.sum(dim=0)
    if bool((counts == 0).any()):
        raise QM9AnchoredProbeError("evaluation split lacks one property")
    result = {
        "state_memory_mode": memory_mode,
        "geometry_component_mode": component_mode,
        "metrics": {
            name: {
                "count": int(counts[index]),
                "unit": None if target_units is None else target_units.get(name),
                "mae": float(absolute[index] / counts[index]),
                "rmse": float(torch.sqrt(squared[index] / counts[index])),
                "standardized_mae": float(standardized_absolute[index] / counts[index]),
            }
            for index, name in enumerate(property_names)
        },
        "macro_average_standardized_mae": float(
            statistics.fmean(
                float(standardized_absolute[index] / counts[index])
                for index in range(len(property_names))
            )
        ),
    }
    if target_units is None or len({target_units.get(name) for name in property_names}) == 1:
        macro_average = float(
            statistics.fmean(
                float(absolute[index] / counts[index])
                for index in range(len(property_names))
            )
        )
        result["macro_average_mae"] = macro_average
        if target_units is None or {target_units.get(name) for name in property_names} == {"hartree"}:
            result["metrics_hartree"] = result["metrics"]
            result["macro_average_mae_hartree"] = macro_average
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise QM9AnchoredProbeError("one CUDA BF16 device is required")
    if args.cell not in CELLS or args.shell_fusion_mode not in ALL_SHELL_MODES:
        raise QM9AnchoredProbeError("cell or shell-fusion mode is invalid")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise QM9AnchoredProbeError("output directory must be new")
    output_dir.mkdir(parents=True)
    property_names = tuple(args.property_names)
    if not property_names or len(set(property_names)) != len(property_names):
        raise QM9AnchoredProbeError("property names must be unique and non-empty")
    target_overlay_dir = (
        None if args.target_overlay_dir is None else Path(args.target_overlay_dir).resolve()
    )
    if target_overlay_dir is None:
        if property_names != tuple(PROPERTY_NAMES):
            raise QM9AnchoredProbeError(
                "the built-in probe cache only exposes HOMO/LUMO/gap"
            )
        target_units = {name: "hartree" for name in property_names}
        target_source = "3dmolt5_qm9_instruction_targets"
    else:
        manifest_path = target_overlay_dir / "manifest.json"
        if not manifest_path.is_file():
            raise QM9AnchoredProbeError("target overlay lacks manifest.json")
        overlay_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_units = overlay_manifest.get("target_units")
        if not isinstance(manifest_units, dict) or any(
            name not in manifest_units for name in property_names
        ):
            raise QM9AnchoredProbeError("target overlay manifest lacks requested units")
        target_units = {name: str(manifest_units[name]) for name in property_names}
        target_source = str(overlay_manifest.get("schema_version"))
    random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed_all(args.train_seed)
    tokenizer = load_verified_anchored_candidate_tokenizer(
        base_snapshot=args.base_tokenizer_snapshot,
        output_dir=args.anchored_tokenizer_dir,
        semantic_plan_sha256=args.semantic_plan_sha256,
    )
    dataset_kwargs = {
        "target_overlay_dir": target_overlay_dir,
        "property_names": property_names,
    }
    train_data = QM9ProbeDataset(args.cache_root, split="train", **dataset_kwargs)
    dev_data = QM9ProbeDataset(args.cache_root, split="dev", **dataset_kwargs)
    test_data = QM9ProbeDataset(args.cache_root, split="test", **dataset_kwargs)
    means, stds = training_target_statistics(train_data, property_names=property_names)
    collator = ProbeCollator(
        pad_token_id=tokenizer.runtime.pad_token_id,
        cell=args.cell,
        property_names=property_names,
    )
    generator = torch.Generator().manual_seed(args.train_seed)
    loader_kwargs = dict(
        batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = torch.utils.data.DataLoader(
        train_data, shuffle=True, generator=generator, drop_last=False, **loader_kwargs
    )
    dev_loader = torch.utils.data.DataLoader(dev_data, shuffle=False, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **loader_kwargs)
    model_kwargs = {
        "base_model_snapshot": args.base_model_snapshot,
        "base_tokenizer_snapshot": args.base_tokenizer_snapshot,
        "anchored_tokenizer_dir": args.anchored_tokenizer_dir,
        "semantic_plan_sha256": args.semantic_plan_sha256,
        "union_init_dir": args.union_init_dir,
        "union_geometry_fusion_seed": GEOMETRY_FUSION_SEED,
        "adapter_seed": args.adapter_seed,
        "num_e3fp_embeddings": 4096,
        "state_level2_weight": 0.25,
        "state_embedding_dim": 64,
        "atom_memory_dim": 128,
        "max_identity_span_length": 128,
        "max_atoms_per_motif": 128,
        "geometry_fraction": 0.5,
    }
    if args.shell_fusion_mode == "reference_fixed_four_mean":
        backbone = load_deterministic_factorized_model_v5(**model_kwargs)
        atom_encoder_variant = "v5_reference_fixed_four_mean"
    elif args.shell_fusion_mode == "adaptive_l0_high":
        backbone = load_deterministic_factorized_model_v6(
            **model_kwargs,
            shell_reducer_mode="adaptive_l0_high",
        )
        atom_encoder_variant = "v6_adaptive_l0_high"
    elif args.shell_fusion_mode == "linear_l0_high":
        backbone = load_deterministic_factorized_model_v7(**model_kwargs)
        atom_encoder_variant = "v7_linear_l0_high"
    elif args.shell_fusion_mode == "minimal_phi_l0_high":
        backbone = load_deterministic_factorized_model_v8(**model_kwargs)
        atom_encoder_variant = "v8_minimal_phi_l0_high"
    elif args.shell_fusion_mode == "level_aware_phi_l0_high":
        backbone = load_deterministic_factorized_model_v9(**model_kwargs)
        atom_encoder_variant = "v9_level_aware_phi_l0_high"
    else:
        backbone = load_deterministic_factorized_model_v4(
            **model_kwargs,
            shell_fusion_mode=args.shell_fusion_mode,
        )
        atom_encoder_variant = "v4_level_explicit_research_baseline"
    hidden_size = int(backbone.get_input_embeddings().weight.shape[1])
    model = AnchoredQM9Regressor(
        backbone,
        hidden_size=hidden_size,
        output_size=len(property_names),
    )
    device = torch.device("cuda", 0)
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
        fused=True,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = min(args.warmup_updates, max(1, total_updates - 1))
    def lr_scale(step: int) -> float:
        if step < warmup_updates:
            return (step + 1) / warmup_updates
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    memory_mode = "zero" if args.cell == "B0" else "aligned"
    means_device = means.to(device)
    stds_device = stds.to(device)
    completed_updates = 0
    clip_count = 0
    losses = []
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        micro_count = 0
        for batch_index, batch in enumerate(train_loader):
            batch = batch.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(
                    batch.inputs,
                    state_memory_mode=memory_mode,
                    geometry_component_mode="both",
                )
                normalized_target = (batch.targets - means_device) / stds_device
                squared = (prediction - normalized_target).square()
                loss = squared[batch.target_mask].mean()
                scaled_loss = loss / args.gradient_accumulation_steps
            if not bool(torch.isfinite(loss).item()):
                raise QM9AnchoredProbeError("training loss is non-finite")
            scaled_loss.backward()
            losses.append(float(loss.detach().float().item()))
            micro_count += 1
            is_last = batch_index + 1 == len(train_loader)
            if micro_count == args.gradient_accumulation_steps or is_last:
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if not bool(torch.isfinite(norm).item()):
                    raise QM9AnchoredProbeError("gradient norm is non-finite")
                clip_count += int(float(norm) > 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                completed_updates += 1
                micro_count = 0
        if completed_updates != (epoch + 1) * updates_per_epoch:
            raise QM9AnchoredProbeError("optimizer update accounting drifted")

    component_conditions = (
        (("zero", "zero"),)
        if args.cell == "B0"
        else (
            ("aligned", "both"),
            ("aligned", "carrier_only"),
            ("aligned", "endpoint_only"),
            ("zero", "zero"),
        )
    )
    evaluations = {
        split: [
            _evaluate(
                model,
                loader,
                device=device,
                means=means,
                stds=stds,
                memory_mode=condition_memory,
                component_mode=component,
                property_names=property_names,
                target_units=target_units,
            )
            for condition_memory, component in component_conditions
        ]
        for split, loader in (("dev", dev_loader), ("test", test_loader))
    }
    checkpoint = output_dir / "final_model_state.pt"
    final_l0_weight = (
        float(backbone.adapter.l0_weight().detach().cpu())
        if hasattr(backbone.adapter, "l0_weight")
        else None
    )
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "cell": args.cell,
            "shell_fusion_mode": args.shell_fusion_mode,
            "atom_encoder_variant": atom_encoder_variant,
            "final_l0_weight": final_l0_weight,
            "completed_updates": completed_updates,
            "train_seed": args.train_seed,
            "adapter_seed": args.adapter_seed,
            "model_state_dict": {
                key: (
                    value.detach().cpu()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in model.state_dict().items()
            },
            "target_mean_hartree": means,
            "target_std_hartree": stds,
        },
        checkpoint,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "qm9_molecule_disjoint_3d_sensitive_representation_probe",
        "cell": args.cell,
        "shell_fusion_mode": args.shell_fusion_mode,
        "atom_encoder_variant": atom_encoder_variant,
        "initial_l0_weight": (
            0.25 if args.shell_fusion_mode in REFERENCE_SHELL_MODES else None
        ),
        "final_l0_weight": final_l0_weight,
        "state_memory_mode": memory_mode,
        "data": {
            "cache_root": str(Path(args.cache_root).resolve()),
            "records": {
                "train": len(train_data), "dev": len(dev_data), "test": len(test_data)
            },
            "property_names": list(property_names),
            "target_units": target_units,
            "target_source": target_source,
            "target_overlay_dir": (
                None if target_overlay_dir is None else str(target_overlay_dir)
            ),
            "target_mean": means.tolist(),
            "target_std": stds.tolist(),
            "partial_targets_masked": True,
        },
        "optimization": {
            "epochs": args.epochs,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "nominal_effective_batch": args.micro_batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_updates": warmup_updates,
            "total_updates": total_updates,
            "completed_updates": completed_updates,
            "schedule": "linear_warmup_then_cosine",
            "full_encoder_finetuning": True,
            "train_seed": args.train_seed,
            "adapter_seed": args.adapter_seed,
        },
        "training": {
            "wall_seconds": time.perf_counter() - started,
            "mean_normalized_mse": statistics.fmean(losses),
            "clip_rate": clip_count / completed_updates,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "evaluations": evaluations,
        "checkpoint": {"file": checkpoint.name, "bytes": checkpoint.stat().st_size},
        "scientific_boundary": {
            "numeric_text_generation_evaluated": False,
            "molecule_disjoint_split": True,
            "b2d_excludes_chirality": True,
            "b2d_is_parameter_matched_coordinate_blind_control": True,
            "selected_properties_are_not_a_complete_3d_benchmark": True,
            "geometry_claim_requires_f3d_better_than_b2d_and_zero": True,
            "endpoint_claim_requires_endpoint_only_and_carrier_only_ablation": True,
        },
    }
    (output_dir / "probe_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "COMPLETE").write_text("pass\n", encoding="ascii")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--shell-fusion-mode", choices=ALL_SHELL_MODES, required=True)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--anchored-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--semantic-plan-sha256", required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--target-overlay-dir", type=Path)
    parser.add_argument(
        "--property-names",
        nargs="+",
        default=list(PROPERTY_NAMES),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--warmup-updates", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--adapter-seed", type=int, default=ADAPTER_SEED)
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
    "AnchoredQM9Regressor",
    "ProbeBatch",
    "ProbeCollator",
    "QM9AnchoredProbeError",
    "QM9ProbeDataset",
    "training_target_statistics",
]
