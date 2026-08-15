"""T5-compatible semantic span selection for molecular identity units.

The model-facing operation remains ordinary T5 sentinel reconstruction.  This
module changes only which complete source spans are eligible:

* atom-bearing molecular identity units are sampled by first drawing one
  retained heavy atom uniformly and selecting its owning logical unit;
* natural text and non-atom molecular syntax bypass this module and retain
  standard T5 random-span corruption.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence


MASK_POLICY_ID = "most-t5-p2/semantic-span-corruption/v1"
C4_TEXT_POLICY = "standard_t5_random_span_corruption"
MOLECULE_IDENTITY_POLICY = "uniform_heavy_atom_then_owner_unit"
MOLECULE_SYNTAX_POLICY = "standard_t5_random_span_corruption"


class SemanticSpanCorruptionError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class SemanticUnit:
    start: int
    stop: int
    logical_id: int
    heavy_atom_count: int = 0
    semantic_type: str = ""

    @property
    def token_count(self) -> int:
        return self.stop - self.start

    def validate(self, *, sequence_length: int) -> None:
        if (
            isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or isinstance(self.stop, bool)
            or not isinstance(self.stop, int)
            or not 0 <= self.start < self.stop <= sequence_length
            or isinstance(self.logical_id, bool)
            or not isinstance(self.logical_id, int)
            or self.logical_id < 0
            or isinstance(self.heavy_atom_count, bool)
            or not isinstance(self.heavy_atom_count, int)
            or self.heavy_atom_count < 0
        ):
            raise SemanticSpanCorruptionError("semantic unit is invalid")


@dataclass(frozen=True)
class T5Corruption:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    selected_units: tuple[SemanticUnit, ...]


def _validate_units(
    units: Sequence[SemanticUnit], *, sequence_length: int
) -> tuple[SemanticUnit, ...]:
    ordered = tuple(sorted(units, key=lambda unit: (unit.start, unit.stop)))
    seen_ids: set[int] = set()
    previous_stop = -1
    for unit in ordered:
        unit.validate(sequence_length=sequence_length)
        if unit.logical_id in seen_ids:
            raise SemanticSpanCorruptionError("logical unit IDs must be unique")
        if unit.start < previous_stop:
            raise SemanticSpanCorruptionError("semantic units cannot overlap")
        seen_ids.add(unit.logical_id)
        previous_stop = unit.stop
    return ordered


def _greedy_nearest_budget(
    ordered_candidates: Sequence[SemanticUnit], *, target_tokens: int
) -> tuple[SemanticUnit, ...]:
    if isinstance(target_tokens, bool) or not isinstance(target_tokens, int) or target_tokens <= 0:
        raise SemanticSpanCorruptionError("target_tokens must be positive")
    selected: list[SemanticUnit] = []
    realized = 0
    for candidate in ordered_candidates:
        proposed = realized + candidate.token_count
        if not selected or abs(target_tokens - proposed) < abs(target_tokens - realized):
            selected.append(candidate)
            realized = proposed
        if realized >= target_tokens:
            break
    return tuple(sorted(selected, key=lambda unit: unit.start))


def select_heavy_atom_anchored_units(
    units: Sequence[SemanticUnit],
    *,
    sequence_length: int,
    target_tokens: int,
    seed: int,
) -> tuple[SemanticUnit, ...]:
    """Draw units without replacement through a uniform heavy-atom ticket pool."""

    ordered = _validate_units(units, sequence_length=sequence_length)
    eligible = [unit for unit in ordered if unit.heavy_atom_count > 0]
    if not eligible:
        raise SemanticSpanCorruptionError("molecular identity has no retained heavy atom")
    generator = random.Random(seed)
    remaining = list(eligible)
    candidates: list[SemanticUnit] = []
    while remaining:
        ticket_count = sum(unit.heavy_atom_count for unit in remaining)
        ticket = generator.randrange(ticket_count)
        cursor = 0
        selected_index = -1
        for index, unit in enumerate(remaining):
            cursor += unit.heavy_atom_count
            if ticket < cursor:
                selected_index = index
                break
        if selected_index < 0:  # pragma: no cover - arithmetic invariant
            raise SemanticSpanCorruptionError("heavy-atom ticket draw failed")
        candidates.append(remaining.pop(selected_index))
    return _greedy_nearest_budget(candidates, target_tokens=target_tokens)


def apply_t5_semantic_span_corruption(
    input_ids: Sequence[int],
    selected_units: Sequence[SemanticUnit],
    *,
    sentinel_token_ids: Sequence[int],
    eos_token_id: int,
) -> T5Corruption:
    """Replace complete spans with sentinels and construct standard T5 labels."""

    source = tuple(int(value) for value in input_ids)
    selected = _validate_units(selected_units, sequence_length=len(source))
    if not selected:
        raise SemanticSpanCorruptionError("at least one semantic span is required")
    if len(sentinel_token_ids) < len(selected):
        raise SemanticSpanCorruptionError(
            "selected spans require one sentinel each"
        )
    sentinels = tuple(int(value) for value in sentinel_token_ids)
    if len(set(sentinels)) != len(sentinels) or eos_token_id in sentinels:
        raise SemanticSpanCorruptionError("sentinel/EOS domain is invalid")

    corrupted: list[int] = []
    labels: list[int] = []
    cursor = 0
    for index, unit in enumerate(selected):
        corrupted.extend(source[cursor : unit.start])
        corrupted.append(sentinels[index])
        labels.extend((sentinels[index], *source[unit.start : unit.stop]))
        cursor = unit.stop
    corrupted.extend(source[cursor:])
    labels.append(int(eos_token_id))
    return T5Corruption(
        input_ids=tuple(corrupted),
        labels=tuple(labels),
        selected_units=selected,
    )


__all__ = [
    "C4_TEXT_POLICY",
    "MASK_POLICY_ID",
    "MOLECULE_IDENTITY_POLICY",
    "MOLECULE_SYNTAX_POLICY",
    "SemanticSpanCorruptionError",
    "SemanticUnit",
    "T5Corruption",
    "apply_t5_semantic_span_corruption",
    "select_heavy_atom_anchored_units",
]
