"""Shared configuration and batch boundary for the P1 four-condition grid.

The grid changes only two scientific factors: identity granularity and whether
E3FP is exposed.  Geometry is carried beside the ordinary T5 CE batch so A0
and M0 remain exact standard-T5 forwards.  A1 and M1 use the same explicit
atom-to-token carrier interface; only the cardinality of that mapping differs.

This module deliberately contains no encoder, fusion layer, teacher, C1-G or
C1-R implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .runtime_bridge import PaddedCEBatch


GRID_SPEC_VERSION = "most-t5-next/p1-four-condition-grid/v1"
BASE_T5_INPUT_KEYS = ("input_ids", "attention_mask", "labels")
GEOMETRY_INPUT_KEYS = ("e3fp_ids", "e3fp_atom_mask", "e3fp_atom_to_token")

ATOM_SELFIES_IDENTITY = "atom_selfies_aligned_identity"
HYBRID_MOTIF_IDENTITY = "hybrid_motif_identity_and_connection"
NO_GEOMETRY = "none"
ATOM_ALIGNED_E3FP = "atom_aligned_e3fp"
MOTIF_MEAN_E3FP = "atom_to_logical_motif_invariant_mean"


class FourGridContractError(ValueError):
    """A condition or its model-facing batch does not match the frozen grid."""


@dataclass(frozen=True)
class P1ConditionSpec:
    """One preregistered cell in the A0/A1/M0/M1 comparison."""

    condition_id: str
    identity_representation: str
    record_family: str
    corruption_unit: str
    geometry_condition: str
    atom_to_carrier_cardinality: str

    @property
    def uses_geometry(self) -> bool:
        return self.geometry_condition != NO_GEOMETRY

    @property
    def uses_logical_motifs(self) -> bool:
        return self.record_family == "logical_motif"

    @property
    def t5_input_keys(self) -> tuple[str, ...]:
        return BASE_T5_INPUT_KEYS

    @property
    def geometry_input_keys(self) -> tuple[str, ...]:
        return GEOMETRY_INPUT_KEYS if self.uses_geometry else ()

    @property
    def wrapper_input_keys(self) -> tuple[str, ...]:
        return self.t5_input_keys + self.geometry_input_keys


_CONDITIONS = {
    "A0": P1ConditionSpec(
        condition_id="A0",
        identity_representation=ATOM_SELFIES_IDENTITY,
        record_family="atom_selfies",
        corruption_unit="atom_identity_span",
        geometry_condition=NO_GEOMETRY,
        atom_to_carrier_cardinality="none",
    ),
    "A1": P1ConditionSpec(
        condition_id="A1",
        identity_representation=ATOM_SELFIES_IDENTITY,
        record_family="atom_selfies",
        corruption_unit="atom_identity_span",
        geometry_condition=ATOM_ALIGNED_E3FP,
        atom_to_carrier_cardinality="one_atom_per_carrier",
    ),
    "M0": P1ConditionSpec(
        condition_id="M0",
        identity_representation=HYBRID_MOTIF_IDENTITY,
        record_family="logical_motif",
        corruption_unit="complete_logical_motif_identity_span",
        geometry_condition=NO_GEOMETRY,
        atom_to_carrier_cardinality="none",
    ),
    "M1": P1ConditionSpec(
        condition_id="M1",
        identity_representation=HYBRID_MOTIF_IDENTITY,
        record_family="logical_motif",
        corruption_unit="complete_logical_motif_identity_span",
        geometry_condition=MOTIF_MEAN_E3FP,
        atom_to_carrier_cardinality="many_atoms_per_motif_carrier",
    ),
}
P1_CONDITION_SPECS: Mapping[str, P1ConditionSpec] = MappingProxyType(_CONDITIONS)


def get_p1_condition_spec(condition_id: str) -> P1ConditionSpec:
    """Return one exact grid cell; aliases are intentionally not accepted."""

    try:
        return P1_CONDITION_SPECS[condition_id]
    except (KeyError, TypeError) as exc:
        raise FourGridContractError(
            "condition_id must be exactly one of A0, A1, M0, M1"
        ) from exc


@dataclass(frozen=True)
class GeometryBatchSidecar:
    """Padded atom E3FP rows and their explicit carrier-token mapping.

    The admitted narrow-P1 schema requires geometry for every real model atom,
    so ``e3fp_atom_mask`` is exactly ``True`` for the first ``atom_length``
    positions and ``False`` only for batch padding.  Partial-geometry atoms
    require a later record schema and are not silently represented here.

    ``e3fp_atom_to_token`` is computed after CE corruption.  A1 will bind one
    model atom to one atom/SELFIES carrier; M1 binds all atoms in a logical
    motif to the same motif carrier.  The future model can therefore embed
    E3FP per atom and use one masked scatter-mean implementation for both.

    ``model_to_source_atom_index`` is audit-only provenance for that same row
    axis; it is not a T5 model input.  It is copied from the post-hydrogen-
    projection geometry record so A1/M1 parity never infers a heavy-atom
    subset from SELFIES or E3FP shape.
    """

    record_ids: tuple[str, ...]
    e3fp_ids: tuple[tuple[tuple[int, ...], ...], ...]
    e3fp_atom_mask: tuple[tuple[bool, ...], ...]
    e3fp_atom_to_token: tuple[tuple[int, ...], ...]
    model_to_source_atom_index: tuple[tuple[int, ...], ...]
    atom_lengths: tuple[int, ...]
    e3fp_level_count: int
    token_width: int
    e3fp_atom_is_attachment: tuple[tuple[bool, ...], ...] | None = None

    def __post_init__(self) -> None:
        batch_size = len(self.record_ids)
        parallel = (
            len(self.e3fp_ids),
            len(self.e3fp_atom_mask),
            len(self.e3fp_atom_to_token),
            len(self.model_to_source_atom_index),
            len(self.atom_lengths),
        )
        if batch_size == 0 or any(length != batch_size for length in parallel):
            raise FourGridContractError("geometry sidecar batch dimensions disagree")
        if self.e3fp_level_count <= 0 or self.token_width <= 0:
            raise FourGridContractError("geometry dimensions must be positive")
        if self.e3fp_atom_is_attachment is not None and len(
            self.e3fp_atom_is_attachment
        ) != batch_size:
            raise FourGridContractError("attachment-role batch dimension disagrees")

        atom_widths = {len(row) for row in self.e3fp_ids}
        if len(atom_widths) != 1 or 0 in atom_widths:
            raise FourGridContractError("e3fp_ids must be a nonempty rectangular atom batch")
        atom_width = next(iter(atom_widths))
        for batch_index in range(batch_size):
            rows = self.e3fp_ids[batch_index]
            mask = self.e3fp_atom_mask[batch_index]
            carriers = self.e3fp_atom_to_token[batch_index]
            source_indices = self.model_to_source_atom_index[batch_index]
            attachment_roles = (
                None
                if self.e3fp_atom_is_attachment is None
                else self.e3fp_atom_is_attachment[batch_index]
            )
            atom_length = self.atom_lengths[batch_index]
            if (
                len(mask) != atom_width
                or len(carriers) != atom_width
                or len(source_indices) != atom_width
            ):
                raise FourGridContractError("geometry atom-domain arrays disagree")
            if attachment_roles is not None and len(attachment_roles) != atom_width:
                raise FourGridContractError("attachment roles must match the atom width")
            if not 0 < atom_length <= atom_width:
                raise FourGridContractError("atom_lengths disagree with padded geometry")
            expected_mask = (True,) * atom_length + (False,) * (atom_width - atom_length)
            if mask != expected_mask:
                raise FourGridContractError(
                    "narrow P1 requires all real atoms valid and padding false"
                )
            active_source_indices = source_indices[:atom_length]
            if (
                any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in active_source_indices
                )
                or active_source_indices != tuple(sorted(set(active_source_indices)))
            ):
                raise FourGridContractError(
                    "active model_to_source_atom_index must be strictly increasing"
                )
            if source_indices[atom_length:] != (-1,) * (atom_width - atom_length):
                raise FourGridContractError(
                    "padded model_to_source_atom_index values must be -1"
                )
            if attachment_roles is not None:
                if any(not isinstance(value, bool) for value in attachment_roles):
                    raise FourGridContractError("attachment roles must be Boolean")
                if any(attachment_roles[atom_length:]):
                    raise FourGridContractError("padded attachment roles must be false")
            for levels, active, carrier in zip(rows, mask, carriers):
                if len(levels) != self.e3fp_level_count:
                    raise FourGridContractError("E3FP level width changed within a batch")
                if active:
                    if levels[0] < 0 or any(value < -1 or value > 4095 for value in levels):
                        raise FourGridContractError("active E3FP rows violate the narrow P1 domain")
                    if not 0 <= carrier < self.token_width:
                        raise FourGridContractError("active atom carrier is outside the token batch")
                elif levels != (-1,) * self.e3fp_level_count or carrier != -1:
                    raise FourGridContractError("padded atoms must use all--1 geometry and carrier rows")

    def model_inputs(self) -> dict[str, tuple[object, ...]]:
        """Return only geometry-side names reserved by the four-grid wrapper."""

        result = {
            "e3fp_ids": self.e3fp_ids,
            "e3fp_atom_mask": self.e3fp_atom_mask,
            "e3fp_atom_to_token": self.e3fp_atom_to_token,
        }
        if self.e3fp_atom_is_attachment is not None:
            result["e3fp_atom_is_attachment"] = self.e3fp_atom_is_attachment
        return result


@dataclass(frozen=True)
class P1ConditionBatch:
    """One condition-tagged CE batch with an optional geometry sidecar."""

    condition_id: str
    ce_batch: PaddedCEBatch
    geometry: GeometryBatchSidecar | None = None

    def __post_init__(self) -> None:
        spec = get_p1_condition_spec(self.condition_id)
        if spec.uses_geometry != (self.geometry is not None):
            raise FourGridContractError(
                "geometry sidecar presence must match the selected grid condition"
            )
        if self.geometry is None:
            return
        if self.geometry.record_ids != self.ce_batch.record_ids:
            raise FourGridContractError("CE and geometry record order differs")
        if self.geometry.token_width != len(self.ce_batch.input_ids[0]):
            raise FourGridContractError("CE and geometry token widths differ")
        for batch_index, input_length in enumerate(self.ce_batch.input_lengths):
            for atom_index in range(self.geometry.atom_lengths[batch_index]):
                if self.geometry.e3fp_atom_to_token[batch_index][atom_index] >= input_length:
                    raise FourGridContractError("atom carrier points into CE padding")

    @property
    def spec(self) -> P1ConditionSpec:
        return get_p1_condition_spec(self.condition_id)

    def t5_inputs(self) -> dict[str, tuple[tuple[object, ...], ...]]:
        """Return the unchanged standard-T5 CE input triplet."""

        return self.ce_batch.model_inputs()

    def geometry_inputs(self) -> dict[str, tuple[object, ...]]:
        """Return model-wrapper side inputs without leaking them to bare T5."""

        return {} if self.geometry is None else self.geometry.model_inputs()


def validate_a1_m1_geometry_atom_parity(
    a1_batch: P1ConditionBatch,
    m1_batch: P1ConditionBatch,
) -> None:
    """Require A1 and M1 to consume the same post-projection atom rows.

    PCQM production projects explicit hydrogen before E3FP and rejects any
    residual hydrogen.  Consequently the common comparison domain is the
    ordered model-atom row axis plus its preserved pre-projection source atom
    indices.  Carrier mappings intentionally differ and are not compared.
    """

    if not isinstance(a1_batch, P1ConditionBatch) or a1_batch.condition_id != "A1":
        raise FourGridContractError("left parity input must be an A1 condition batch")
    if not isinstance(m1_batch, P1ConditionBatch) or m1_batch.condition_id != "M1":
        raise FourGridContractError("right parity input must be an M1 condition batch")
    if a1_batch.geometry is None or m1_batch.geometry is None:
        raise FourGridContractError("A1/M1 parity requires both geometry sidecars")

    a1 = a1_batch.geometry
    m1 = m1_batch.geometry
    comparisons = (
        ("record order", a1.record_ids, m1.record_ids),
        ("atom lengths", a1.atom_lengths, m1.atom_lengths),
        ("E3FP level count", a1.e3fp_level_count, m1.e3fp_level_count),
        ("atom mask", a1.e3fp_atom_mask, m1.e3fp_atom_mask),
        (
            "model-to-source atom mapping",
            a1.model_to_source_atom_index,
            m1.model_to_source_atom_index,
        ),
        ("E3FP atom rows", a1.e3fp_ids, m1.e3fp_ids),
    )
    for label, left, right in comparisons:
        if left != right:
            raise FourGridContractError("A1/M1 geometry parity differs in " + label)
