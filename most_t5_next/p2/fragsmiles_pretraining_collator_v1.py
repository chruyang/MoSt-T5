"""Online molecular denoising views for the final fragSMILES T5 path.

The immutable mmap cache stores complete records.  This module performs the
epoch-conditioned operation that must remain online:

* a motif consists of its fragment phrase and every explicit endpoint span on
  that fragment side; one uniform-heavy-atom owner draw masks them together;
* non-fragment molecular syntax uses the reference T5 random-span policy;
* selected motif carrier/endpoint geometry is disabled;
* every token address is remapped after sentinel replacement before dynamic
  padding.

The two candidate sets are disjoint and each uses the same 15 percent density,
so their union preserves the nominal density over all eligible surface tokens
without making a multi-token fallback fragment easier to split than a macro.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Sequence

import numpy as np

from most_t5_next.data.motif_corruption import (
    MotifUnit,
    build_motif_units,
    geometry_visibility,
    select_motif_units,
)

from .fragsmiles_training_tensor_cache_v1 import (
    CachedFragSmilesRecord,
    CachedFragSmilesSample,
    ROLE_TO_ID,
    collate_cached_fragsmiles,
)
from .semantic_span_corruption_v1 import (
    SemanticUnit,
    apply_t5_semantic_span_corruption,
)


SCHEMA_VERSION = "most-t5-p2/fragsmiles-pretraining-collator/v1"
MOLECULAR_DENOISING_VIEWS = frozenset({"P1-SYN", "P2-M", "P2-MG"})
GEOMETRY_VIEWS = frozenset({"P1-SYN", "P2-MG"})
PROTECTED_ROLE_IDS = frozenset(
    {ROLE_TO_ID["control"], ROLE_TO_ID["molecule_boundary"]}
)


class FragSmilesPretrainingCollatorError(ValueError):
    pass


@dataclass(frozen=True)
class CorruptedFragSmilesRecord:
    record: CachedFragSmilesRecord
    labels: tuple[int, ...]
    selected_fragment_ids: tuple[int, ...]
    selected_motif_spans: tuple[tuple[int, int], ...]
    selected_syntax_spans: tuple[tuple[int, int], ...]
    fragment_geometry_mask: tuple[bool, ...]
    endpoint_geometry_mask: tuple[bool, ...]
    view: str
    seed: int


def _view_seed(*, global_seed: int, epoch: int, ordinal: int, view: str) -> int:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (global_seed, epoch, ordinal)
    ):
        raise FragSmilesPretrainingCollatorError(
            "global seed, epoch and ordinal must be nonnegative integers"
        )
    # P2-M and P2-MG are a paired intervention: identity corruption must be
    # byte-identical and only geometry visibility may differ.
    corruption_family = "P2-M-PAIRED" if view in {"P2-M", "P2-MG"} else view
    payload = f"{global_seed}:{epoch}:{ordinal}:{corruption_family}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _random_segmentation(
    num_items: int, num_segments: int, *, generator: random.Random
) -> tuple[int, ...]:
    if not 1 <= num_segments <= num_items:
        raise FragSmilesPretrainingCollatorError("random segmentation is invalid")
    if num_segments == 1:
        return (num_items,)
    cuts = sorted(generator.sample(range(1, num_items), num_segments - 1))
    edges = (0, *cuts, num_items)
    return tuple(edges[index + 1] - edges[index] for index in range(num_segments))


def standard_t5_noise_mask(
    length: int,
    *,
    noise_density: float,
    mean_noise_span_length: float,
    seed: int,
) -> tuple[bool, ...]:
    """Reference T5 alternating random segmentation over one logical axis."""

    if (
        isinstance(length, bool)
        or not isinstance(length, int)
        or length <= 0
        or not 0.0 < noise_density < 1.0
        or mean_noise_span_length <= 0.0
    ):
        raise FragSmilesPretrainingCollatorError("T5 corruption settings are invalid")
    if length == 1:
        return (True,)
    noise_tokens = min(max(int(round(length * noise_density)), 1), length - 1)
    nonnoise_tokens = length - noise_tokens
    noise_spans = max(int(round(noise_tokens / mean_noise_span_length)), 1)
    noise_spans = min(noise_spans, noise_tokens, nonnoise_tokens)
    generator = random.Random(seed)
    noise_lengths = _random_segmentation(
        noise_tokens, noise_spans, generator=generator
    )
    nonnoise_lengths = _random_segmentation(
        nonnoise_tokens, noise_spans, generator=generator
    )
    mask: list[bool] = []
    for clean, noise in zip(nonnoise_lengths, noise_lengths):
        mask.extend([False] * clean)
        mask.extend([True] * noise)
    if len(mask) != length:
        raise FragSmilesPretrainingCollatorError("T5 mask length does not close")
    return tuple(mask)


def _syntax_units(
    record: CachedFragSmilesRecord,
    *,
    noise_density: float,
    mean_noise_span_length: float,
    seed: int,
) -> tuple[SemanticUnit, ...]:
    eligible_positions = [
        index
        for index, (role, owner) in enumerate(
            zip(record.token_roles.tolist(), record.token_to_fragment.tolist())
        )
        if (
            int(owner) < 0
            and int(role) not in PROTECTED_ROLE_IDS
            and int(role) != ROLE_TO_ID["connector_endpoint"]
        )
    ]
    if not eligible_positions:
        return ()
    noise = standard_t5_noise_mask(
        len(eligible_positions),
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
        seed=seed,
    )
    selected_positions = [
        position for position, selected in zip(eligible_positions, noise) if selected
    ]
    units: list[SemanticUnit] = []
    start = stop = -1
    logical_id = len(record.fragment_spans)
    for position in selected_positions:
        if start < 0:
            start, stop = position, position + 1
        elif position == stop:
            stop += 1
        else:
            units.append(
                SemanticUnit(start, stop, logical_id, semantic_type="syntax")
            )
            logical_id += 1
            start, stop = position, position + 1
    if start >= 0:
        units.append(SemanticUnit(start, stop, logical_id, semantic_type="syntax"))
    return tuple(units)


def _explicit_endpoint_spans(
    record: CachedFragSmilesRecord,
) -> dict[int, tuple[tuple[int, int], ...]]:
    """Recover each explicit connector lexeme from its terminator address."""

    endpoint_role = ROLE_TO_ID["connector_endpoint"]
    explicit_carriers: dict[int, list[int]] = {}
    for endpoint in record.endpoints:
        if not bool(endpoint[5]):
            continue
        owner = int(endpoint[2])
        carrier = int(endpoint[4])
        if (
            not 0 <= owner < len(record.fragment_spans)
            or not 0 <= carrier < len(record.input_ids)
            or int(record.token_roles[carrier]) != endpoint_role
            or int(record.token_to_fragment[carrier]) != owner
        ):
            raise FragSmilesPretrainingCollatorError(
                "explicit endpoint ownership metadata is inconsistent"
            )
        explicit_carriers.setdefault(owner, []).append(carrier)

    spans: dict[int, list[tuple[int, int]]] = {}
    index = 0
    while index < len(record.input_ids):
        if int(record.token_roles[index]) != endpoint_role:
            index += 1
            continue
        owner = int(record.token_to_fragment[index])
        if not 0 <= owner < len(record.fragment_spans):
            raise FragSmilesPretrainingCollatorError(
                "connector endpoint token has no motif owner"
            )
        stop = index + 1
        while (
            stop < len(record.input_ids)
            and int(record.token_roles[stop]) == endpoint_role
            and int(record.token_to_fragment[stop]) == owner
        ):
            stop += 1
        carriers = sorted(
            carrier
            for carrier in explicit_carriers.get(owner, ())
            if index <= carrier < stop
        )
        if not carriers or carriers[-1] != stop - 1:
            raise FragSmilesPretrainingCollatorError(
                "connector endpoint span lacks its explicit terminator"
            )
        start = index
        for carrier in carriers:
            spans.setdefault(owner, []).append((start, carrier + 1))
            start = carrier + 1
        if start != stop:
            raise FragSmilesPretrainingCollatorError(
                "connector endpoint span does not close"
            )
        index = stop

    observed = sorted(stop - 1 for rows in spans.values() for _start, stop in rows)
    expected = sorted(carrier for rows in explicit_carriers.values() for carrier in rows)
    if observed != expected:
        raise FragSmilesPretrainingCollatorError(
            "explicit endpoint rows and surface spans disagree"
        )
    return {owner: tuple(rows) for owner, rows in spans.items()}


def _motif_units(record: CachedFragSmilesRecord) -> tuple[MotifUnit, ...]:
    return build_motif_units(
        tuple((int(start), int(stop)) for start, stop in record.fragment_spans),
        _explicit_endpoint_spans(record),
        tuple(int(owner) for owner in record.atom_to_fragment),
        sequence_length=len(record.input_ids),
    )


def _selected_motif_units(
    record: CachedFragSmilesRecord, *, noise_density: float, seed: int
) -> tuple[MotifUnit, ...]:
    units = _motif_units(record)
    return select_motif_units(units, noise_density=noise_density, seed=seed)


def _sentinel_metadata(
    record: CachedFragSmilesRecord,
    unit: SemanticUnit,
    motif_span_metadata: dict[tuple[int, int], tuple[int, int]],
) -> tuple[int, int]:
    motif_metadata = motif_span_metadata.get((unit.start, unit.stop))
    if motif_metadata is not None:
        return motif_metadata
    roles = record.token_roles[unit.start : unit.stop]
    role = (
        ROLE_TO_ID["connector_endpoint"]
        if bool(np.any(roles == ROLE_TO_ID["connector_endpoint"]))
        else ROLE_TO_ID["syntax_glyph"]
    )
    return role, -1


def corrupt_cached_fragsmiles_record(
    sample: CachedFragSmilesSample,
    *,
    view: str,
    sentinel_token_ids: Sequence[int],
    eos_token_id: int,
    global_seed: int,
    noise_density: float = 0.15,
    mean_noise_span_length: float = 3.0,
    encoder_cap: int = 512,
    target_cap: int = 114,
) -> CorruptedFragSmilesRecord:
    if view not in MOLECULAR_DENOISING_VIEWS:
        raise FragSmilesPretrainingCollatorError("unknown molecular denoising view")
    record = sample.record
    seed = _view_seed(
        global_seed=global_seed,
        epoch=sample.epoch,
        ordinal=record.ordinal,
        view=view,
    )
    motif_units = _selected_motif_units(
        record, noise_density=noise_density, seed=seed
    )
    syntax_units = _syntax_units(
        record,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
        seed=seed ^ 0x9E3779B97F4A7C15,
    )
    motif_span_metadata: dict[tuple[int, int], tuple[int, int]] = {}
    motif_spans: list[SemanticUnit] = []
    fragment_role = ROLE_TO_ID["fragment_phrase"]
    endpoint_role = ROLE_TO_ID["connector_endpoint"]
    for motif in motif_units:
        fragment_span = tuple(int(value) for value in record.fragment_spans[motif.fragment_id])
        for start, stop in motif.spans:
            role = fragment_role if (start, stop) == fragment_span else endpoint_role
            motif_span_metadata[(start, stop)] = (role, motif.fragment_id)
            motif_spans.append(
                SemanticUnit(start, stop, 0, semantic_type="motif")
            )
    ordered_units = sorted((*motif_spans, *syntax_units), key=lambda unit: unit.start)
    selected = tuple(
        SemanticUnit(
            unit.start,
            unit.stop,
            logical_id,
            heavy_atom_count=unit.heavy_atom_count,
            semantic_type=unit.semantic_type,
        )
        for logical_id, unit in enumerate(ordered_units)
    )
    if not selected:
        raise FragSmilesPretrainingCollatorError(
            "record has no eligible molecular corruption unit"
        )
    corruption = apply_t5_semantic_span_corruption(
        record.input_ids.tolist(),
        selected,
        sentinel_token_ids=sentinel_token_ids,
        eos_token_id=eos_token_id,
    )
    if len(corruption.input_ids) > encoder_cap:
        raise FragSmilesPretrainingCollatorError(
            f"record {record.ordinal} {view} encoder length exceeds {encoder_cap}"
        )
    if len(corruption.labels) > target_cap:
        raise FragSmilesPretrainingCollatorError(
            f"record {record.ordinal} {view} target length exceeds {target_cap}"
        )

    old_to_new = np.full((len(record.input_ids),), -1, dtype=np.int64)
    new_roles: list[int] = []
    new_owners: list[int] = []
    cursor = 0
    for unit in selected:
        for old_index in range(cursor, unit.start):
            old_to_new[old_index] = len(new_roles)
            new_roles.append(int(record.token_roles[old_index]))
            new_owners.append(int(record.token_to_fragment[old_index]))
        sentinel_index = len(new_roles)
        old_to_new[unit.start : unit.stop] = sentinel_index
        role, owner = _sentinel_metadata(record, unit, motif_span_metadata)
        new_roles.append(role)
        new_owners.append(owner)
        cursor = unit.stop
    for old_index in range(cursor, len(record.input_ids)):
        old_to_new[old_index] = len(new_roles)
        new_roles.append(int(record.token_roles[old_index]))
        new_owners.append(int(record.token_to_fragment[old_index]))
    if len(new_roles) != len(corruption.input_ids) or bool((old_to_new < 0).any()):
        raise FragSmilesPretrainingCollatorError("token remapping does not close")

    selected_fragments = tuple(
        motif.fragment_id for motif in motif_units
    )
    selected_fragment_set = set(selected_fragments)
    selected_syntax = tuple((unit.start, unit.stop) for unit in syntax_units)
    remapped_spans: list[tuple[int, int]] = []
    remapped_carriers: list[int] = []
    for fragment_id, (start, stop) in enumerate(record.fragment_spans):
        if fragment_id in selected_fragment_set:
            sentinel = int(old_to_new[int(start)])
            remapped_spans.append((sentinel, sentinel + 1))
            remapped_carriers.append(sentinel)
        else:
            remapped_spans.append(
                (int(old_to_new[int(start)]), int(old_to_new[int(stop) - 1]) + 1)
            )
            remapped_carriers.append(int(old_to_new[int(record.fragment_carriers[fragment_id])]))

    remapped_endpoints = np.asarray(record.endpoints, dtype=np.int64).copy()
    if remapped_endpoints.size:
        remapped_endpoints[:, 4] = old_to_new[remapped_endpoints[:, 4]]
    remapped_atom_carriers = old_to_new[np.asarray(record.atom_carriers, dtype=np.int64)]
    remapped_molecule_carrier = (
        -1
        if record.molecule_carrier < 0
        else int(old_to_new[int(record.molecule_carrier)])
    )

    fragment_geometry, endpoint_geometry = geometry_visibility(
        len(record.fragment_spans),
        tuple(int(endpoint[2]) for endpoint in record.endpoints),
        selected_fragments,
        enabled=view in GEOMETRY_VIEWS,
    )

    transformed = CachedFragSmilesRecord(
        cache_index=record.cache_index,
        ordinal=record.ordinal,
        source_segment=record.source_segment,
        mode=record.mode,
        component_count=record.component_count,
        molecule_carrier=remapped_molecule_carrier,
        input_ids=np.asarray(corruption.input_ids, dtype=np.int64),
        token_roles=np.asarray(new_roles, dtype=np.int64),
        token_to_fragment=np.asarray(new_owners, dtype=np.int64),
        fragment_spans=np.asarray(remapped_spans, dtype=np.int64).reshape((-1, 2)),
        fragment_carriers=np.asarray(remapped_carriers, dtype=np.int64),
        fragment_components=record.fragment_components,
        fragment_representations=record.fragment_representations,
        atom_to_fragment=record.atom_to_fragment,
        atom_local_index=record.atom_local_index,
        atom_components=record.atom_components,
        atom_carriers=np.asarray(remapped_atom_carriers, dtype=np.int64),
        atom_is_attachment=record.atom_is_attachment,
        e3fp=record.e3fp,
        endpoints=remapped_endpoints.reshape((-1, 6)),
    )
    return CorruptedFragSmilesRecord(
        record=transformed,
        labels=corruption.labels,
        selected_fragment_ids=selected_fragments,
        selected_motif_spans=tuple(
            sorted((unit.start, unit.stop) for unit in motif_spans)
        ),
        selected_syntax_spans=selected_syntax,
        fragment_geometry_mask=fragment_geometry,
        endpoint_geometry_mask=endpoint_geometry,
        view=view,
        seed=seed,
    )


def collate_molecular_denoising_samples(
    samples: Sequence[CachedFragSmilesSample],
    *,
    view: str,
    pad_token_id: int,
    sentinel_token_ids: Sequence[int],
    eos_token_id: int,
    global_seed: int,
    encoder_cap: int = 512,
    target_cap: int = 114,
) -> dict[str, object]:
    if not samples or len({sample.epoch for sample in samples}) != 1:
        raise FragSmilesPretrainingCollatorError(
            "one molecular batch must contain one nonempty epoch"
        )
    rows = tuple(
        corrupt_cached_fragsmiles_record(
            sample,
            view=view,
            sentinel_token_ids=sentinel_token_ids,
            eos_token_id=eos_token_id,
            global_seed=global_seed,
            encoder_cap=encoder_cap,
            target_cap=target_cap,
        )
        for sample in samples
    )
    batch = collate_cached_fragsmiles(
        tuple(row.record for row in rows), pad_token_id=pad_token_id
    )
    import torch

    max_target = max(len(row.labels) for row in rows)
    labels = torch.full((len(rows), max_target), -100, dtype=torch.long)
    fragment_geometry_mask = torch.zeros_like(batch["fragment_mask"])
    endpoint_geometry_mask = torch.zeros_like(batch["endpoint_mask"])
    for index, row in enumerate(rows):
        labels[index, : len(row.labels)] = torch.tensor(row.labels, dtype=torch.long)
        fragment_geometry_mask[index, : len(row.fragment_geometry_mask)] = torch.tensor(
            row.fragment_geometry_mask, dtype=torch.bool
        )
        endpoint_geometry_mask[index, : len(row.endpoint_geometry_mask)] = torch.tensor(
            row.endpoint_geometry_mask, dtype=torch.bool
        )
    batch.update(
        {
            "labels": labels,
            "fragment_geometry_mask": fragment_geometry_mask,
            "endpoint_geometry_mask": endpoint_geometry_mask,
            "view": view,
            "epoch": samples[0].epoch,
            "corruption_rows": rows,
        }
    )
    return batch


__all__ = [
    "CorruptedFragSmilesRecord",
    "FragSmilesPretrainingCollatorError",
    "GEOMETRY_VIEWS",
    "MOLECULAR_DENOISING_VIEWS",
    "SCHEMA_VERSION",
    "collate_molecular_denoising_samples",
    "corrupt_cached_fragsmiles_record",
    "standard_t5_noise_mask",
]
