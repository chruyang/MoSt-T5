"""Coordinate-blind atom-state control matched to the E3FP slot interface.

The B2D control uses RDKit Morgan/ECFP with radius 3 and a 4096-bit folded
domain.  ``AdditionalOutput.bitInfoMap`` recovers one folded categorical ID
for every decoded atom and radius 0..3; the persisted SELFIES carrier order
maps those decoded atoms back to the frozen geometry model-row axis.

This module creates only a PF-10 overlay.  It does not modify the production
E3FP release or infer geometry from coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


MORGAN_STATE_ID = "most-t5-p2/coordinate-blind-morgan-atom-state/r3-fp4096-v1"


class MorganAtomStateError(ValueError):
    """The decoded 2D graph cannot be mapped to the frozen atom row axis."""


@dataclass(frozen=True)
class MorganAtomState:
    state_ids: tuple[tuple[int, int, int, int], ...]
    radius: int = 3
    fp_size: int = 4096
    include_chirality: bool = True
    use_bond_types: bool = True
    include_redundant_environments: bool = True


def derive_morgan_atom_state(
    *,
    Chem,
    rdFingerprintGenerator,
    selfies_decoder: Callable[[str], str],
    selfies: str,
    atom_to_carrier: Sequence[int],
    radius: int = 3,
    fp_size: int = 4096,
) -> MorganAtomState:
    """Return a coordinate-blind ``[model_atom, radius]`` folded-bit matrix."""

    if radius != 3 or fp_size != 4096:
        raise MorganAtomStateError("B2D-v1 is frozen to radius=3 and fp_size=4096")
    carriers = tuple(atom_to_carrier)
    if not carriers or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in carriers
    ):
        raise MorganAtomStateError("atom_to_carrier must contain nonnegative integers")
    if len(set(carriers)) != len(carriers):
        raise MorganAtomStateError("atom carriers must be unique")
    if not isinstance(selfies, str) or not selfies:
        raise MorganAtomStateError("SELFIES surface must be nonempty")
    try:
        smiles = selfies_decoder(selfies)
    except Exception as exc:
        raise MorganAtomStateError("SELFIES decoding failed") from exc
    if not isinstance(smiles, str) or not smiles:
        raise MorganAtomStateError("SELFIES decoder returned no SMILES")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise MorganAtomStateError("decoded SMILES is not an RDKit molecule")
    if molecule.GetNumAtoms() != len(carriers):
        raise MorganAtomStateError(
            "decoded atom count differs from the frozen model atom count"
        )

    # Atom-producing SELFIES tokens occur in decoded atom order.  The persisted
    # carrier positions therefore provide the explicit decoded->model mapping.
    model_atom_by_decoded_index = tuple(
        sorted(range(len(carriers)), key=lambda atom_id: carriers[atom_id])
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size,
        includeChirality=True,
        useBondTypes=True,
        includeRedundantEnvironments=True,
    )
    additional = rdFingerprintGenerator.AdditionalOutput()
    additional.AllocateBitInfoMap()
    generator.GetFingerprint(molecule, additionalOutput=additional)
    bit_info = additional.GetBitInfoMap()
    decoded_rows = [[-1] * (radius + 1) for _ in range(molecule.GetNumAtoms())]
    for bit_id, occurrences in bit_info.items():
        if not 0 <= int(bit_id) < fp_size:
            raise MorganAtomStateError("Morgan bit lies outside the folded domain")
        for atom_index, environment_radius in occurrences:
            if not 0 <= int(atom_index) < molecule.GetNumAtoms():
                raise MorganAtomStateError("Morgan atom occurrence is out of range")
            if not 0 <= int(environment_radius) <= radius:
                continue
            previous = decoded_rows[int(atom_index)][int(environment_radius)]
            if previous not in (-1, int(bit_id)):
                raise MorganAtomStateError(
                    "one atom/radius maps to multiple folded Morgan bits"
                )
            decoded_rows[int(atom_index)][int(environment_radius)] = int(bit_id)
    if any(row[0] < 0 for row in decoded_rows):
        raise MorganAtomStateError(
            "Morgan generator did not emit every radius-zero atom environment"
        )

    model_rows = [[-1] * (radius + 1) for _ in carriers]
    for decoded_index, model_atom_id in enumerate(model_atom_by_decoded_index):
        model_rows[model_atom_id] = decoded_rows[decoded_index]
    return MorganAtomState(
        state_ids=tuple(tuple(row) for row in model_rows),
        radius=radius,
        fp_size=fp_size,
    )


__all__ = [
    "MORGAN_STATE_ID",
    "MorganAtomState",
    "MorganAtomStateError",
    "derive_morgan_atom_state",
]
