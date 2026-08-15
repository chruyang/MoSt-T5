"""Heavy-atom-weighted sampling of compound fragSMILES motif units."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
from typing import Mapping, Sequence, Tuple


Span = Tuple[int, int]


class MotifCorruptionError(ValueError):
    pass


@dataclass(frozen=True)
class MotifUnit:
    """A fragment span and every explicit endpoint span owned on its side."""

    fragment_id: int
    spans: tuple[Span, ...]
    heavy_atom_count: int

    @property
    def token_count(self) -> int:
        return sum(stop - start for start, stop in self.spans)

    @property
    def start(self) -> int:
        return self.spans[0][0]


def build_motif_units(
    fragment_spans: Sequence[Span],
    explicit_endpoint_spans: Mapping[int, Sequence[Span]],
    atom_to_fragment: Sequence[int],
    *,
    sequence_length: int,
) -> tuple[MotifUnit, ...]:
    """Build one compound unit per fragment without merging its surface spans."""

    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length <= 0
    ):
        raise MotifCorruptionError("sequence_length must be a positive integer")
    fragment_count = len(fragment_spans)
    if fragment_count == 0:
        return ()
    unknown_endpoint_owners = set(explicit_endpoint_spans).difference(
        range(fragment_count)
    )
    if unknown_endpoint_owners:
        raise MotifCorruptionError("endpoint span has no valid fragment owner")
    counts = Counter(int(owner) for owner in atom_to_fragment)
    observed: list[tuple[int, int, int]] = []
    units: list[MotifUnit] = []
    for fragment_id, fragment_span in enumerate(fragment_spans):
        spans = tuple(
            sorted(
                (
                    (int(fragment_span[0]), int(fragment_span[1])),
                    *(
                        (int(start), int(stop))
                        for start, stop in explicit_endpoint_spans.get(fragment_id, ())
                    ),
                )
            )
        )
        for start, stop in spans:
            if not 0 <= start < stop <= sequence_length:
                raise MotifCorruptionError("motif span lies outside the token sequence")
            observed.append((start, stop, fragment_id))
        atom_count = counts.get(fragment_id, 0)
        units.append(MotifUnit(fragment_id, spans, atom_count))

    ordered = sorted(observed)
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise MotifCorruptionError("compound motif spans cannot overlap")
    unknown_owners = set(counts).difference(range(fragment_count))
    if unknown_owners:
        raise MotifCorruptionError("heavy atom has no valid fragment owner")
    if any(unit.heavy_atom_count == 0 for unit in units):
        raise MotifCorruptionError("motif fragment owns no retained heavy atom")
    return tuple(units)


def select_motif_units(
    units: Sequence[MotifUnit],
    *,
    noise_density: float,
    seed: int,
) -> tuple[MotifUnit, ...]:
    """Sample owners by heavy atoms, then stop nearest the token budget."""

    if not units:
        return ()
    if not 0.0 < noise_density < 1.0:
        raise MotifCorruptionError("noise_density must lie between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise MotifCorruptionError("seed must be a nonnegative integer")
    remaining = [unit for unit in units if unit.heavy_atom_count > 0]
    if not remaining:
        raise MotifCorruptionError("no motif owns a retained heavy atom")
    generator = random.Random(seed)
    candidates: list[MotifUnit] = []
    while remaining:
        ticket = generator.randrange(sum(unit.heavy_atom_count for unit in remaining))
        cursor = 0
        for index, unit in enumerate(remaining):
            cursor += unit.heavy_atom_count
            if ticket < cursor:
                candidates.append(remaining.pop(index))
                break

    target_tokens = max(1, int(round(sum(unit.token_count for unit in units) * noise_density)))
    selected: list[MotifUnit] = []
    realized = 0
    for candidate in candidates:
        proposed = realized + candidate.token_count
        if not selected or abs(target_tokens - proposed) < abs(target_tokens - realized):
            selected.append(candidate)
            realized = proposed
        if realized >= target_tokens:
            break
    return tuple(sorted(selected, key=lambda unit: unit.start))


def geometry_visibility(
    fragment_count: int,
    endpoint_to_fragment: Sequence[int],
    selected_fragment_ids: Sequence[int],
    *,
    enabled: bool,
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Close carrier and owned endpoint geometry for every selected motif."""

    selected = set(int(value) for value in selected_fragment_ids)
    if selected.difference(range(fragment_count)):
        raise MotifCorruptionError("selected fragment lies outside the fragment axis")
    fragments = tuple(enabled and index not in selected for index in range(fragment_count))
    endpoints: list[bool] = []
    for owner in endpoint_to_fragment:
        owner = int(owner)
        if not 0 <= owner < fragment_count:
            raise MotifCorruptionError("endpoint has no valid fragment owner")
        endpoints.append(enabled and fragments[owner])
    return fragments, tuple(endpoints)


__all__ = [
    "MotifCorruptionError",
    "MotifUnit",
    "build_motif_units",
    "geometry_visibility",
    "select_motif_units",
]
