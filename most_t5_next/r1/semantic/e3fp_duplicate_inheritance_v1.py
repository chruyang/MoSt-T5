"""Explicit duplicate-shell inheritance for atom-aligned E3FP matrices.

The frozen PCQM production-v2 release projects every shell's *raw*
``shell.identifier`` into an ``[atom, level]`` matrix.  Vendored E3FP also
retains a more precise provenance edge for a shell whose atom-set substructure
has already been accepted: ``shell.is_duplicate`` is true and
``shell.duplicate`` points to that accepted shell.

This module derives the legacy raw matrix and the explicit-inheritance matrix
from the same completed :class:`Fingerprinter`.  It deliberately does not
mutate shells and does not reproduce 3D-MolT5's folded-bit/candidate-search
heuristic.  The frozen raw producer and its independent replay remain
unchanged; a later adapter may store the paired result as an additive artifact.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable


FP_BITS = 4096
FP_LEVEL = 3
SEMANTICS_ID = "duplicate_pointer_inheritance_v1"

# Kept explicit instead of importing the frozen preflight implementation.  A
# paired run must use the same E3FP invocation while remaining a new semantic
# layer rather than changing the source-hashed production-v2 helper.
E3FP_INVOCATION = {
    "bits": FP_BITS,
    "level": FP_LEVEL,
    "rdkit_invariants": True,
    "all_iters": True,
    "exclude_floating": False,
}


class E3FPInheritanceError(ValueError):
    """A shell graph cannot satisfy explicit duplicate inheritance."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _shell_level(shell, radius_multiplier: float) -> int:
    try:
        radius = float(shell.radius)
        multiplier = float(radius_multiplier)
    except Exception as exc:
        raise E3FPInheritanceError(
            "E3FP_SHELL_RADIUS_INVALID", "shell radius or radius multiplier is not numeric"
        ) from exc
    if (
        not math.isfinite(radius)
        or radius < 0.0
        or not math.isfinite(multiplier)
        or multiplier <= 0.0
    ):
        raise E3FPInheritanceError(
            "E3FP_SHELL_RADIUS_INVALID", "shell radius or radius multiplier is invalid"
        )
    if radius == 0.0:
        return 0
    level = int(round(radius / multiplier))
    tolerance = max(1e-10, abs(multiplier) * 1e-9)
    if level < 0 or not math.isclose(
        radius, level * multiplier, rel_tol=0.0, abs_tol=tolerance
    ):
        raise E3FPInheritanceError(
            "E3FP_SHELL_RADIUS_UNMAPPABLE", "shell radius does not map to an E3FP level"
        )
    return level


def _fold_identifier(identifier, signed_to_unsigned_int) -> int:
    if identifier is None:
        raise E3FPInheritanceError(
            "E3FP_SHELL_IDENTIFIER_MISSING", "shell identifier is missing"
        )
    try:
        folded = int(signed_to_unsigned_int(int(identifier)) % FP_BITS)
    except Exception as exc:
        raise E3FPInheritanceError(
            "E3FP_IDENTIFIER_FOLD_FAILED", "shell identifier cannot be folded"
        ) from exc
    if folded < 0 or folded >= FP_BITS:
        raise E3FPInheritanceError(
            "E3FP_IDENTIFIER_OUT_OF_RANGE", "folded shell identifier is outside the vocabulary"
        )
    return folded


def _final_index_set(final_fingerprint_indices: Iterable[int]) -> set[int]:
    try:
        result = {int(value) for value in final_fingerprint_indices}
    except Exception as exc:
        raise E3FPInheritanceError(
            "E3FP_FINAL_FINGERPRINT_INVALID", "final fingerprint indices are not iterable integers"
        ) from exc
    if not result or min(result) < 0 or max(result) >= FP_BITS:
        raise E3FPInheritanceError(
            "E3FP_FINAL_FINGERPRINT_INVALID", "final fingerprint indices violate the folded range"
        )
    return result


def build_shell_projection_pair(
    np,
    fingerprinter,
    signed_to_unsigned_int,
    model_atom_count: int,
    final_fingerprint_indices: Iterable[int],
):
    """Build raw and duplicate-inherited ``int32[A, 4]`` matrices.

    A slot is addressed only by ``shell.center_atom`` and the level recovered
    from ``shell.radius``.  For a duplicate shell, the inherited identifier is
    taken directly from ``shell.duplicate.identifier``.  Folded-bit membership
    is a postcondition, never a mechanism for selecting the inherited shell.

    Returns
    -------
    tuple
        ``(raw, inherited, duplicate_mask, summary)``.  The Boolean mask is an
        audit aid; a training record only needs ``inherited``.
    """

    if (
        isinstance(model_atom_count, bool)
        or not isinstance(model_atom_count, int)
        or model_atom_count <= 0
    ):
        raise E3FPInheritanceError(
            "E3FP_MODEL_ATOM_COUNT_INVALID", "model_atom_count must be a positive integer"
        )
    if not hasattr(fingerprinter, "all_shells") or not hasattr(
        fingerprinter, "radius_multiplier"
    ):
        raise E3FPInheritanceError(
            "E3FP_FINGERPRINTER_INVALID", "fingerprinter lacks shells or radius multiplier"
        )

    final_indices = _final_index_set(final_fingerprint_indices)
    raw = np.full((model_atom_count, FP_LEVEL + 1), -1, dtype=np.int32)
    inherited = np.full((model_atom_count, FP_LEVEL + 1), -1, dtype=np.int32)
    duplicate_mask = np.zeros((model_atom_count, FP_LEVEL + 1), dtype=np.bool_)
    slots_seen = set()
    shells_seen = 0
    duplicate_slots = 0
    changed_identifier_slots = 0
    changed_token_slots = 0
    duplicate_atoms = set()

    for shell in fingerprinter.all_shells:
        shells_seen += 1
        try:
            center_atom = int(shell.center_atom)
        except Exception as exc:
            raise E3FPInheritanceError(
                "E3FP_SHELL_CENTER_INVALID", "shell center is not an integer"
            ) from exc
        if center_atom < 0 or center_atom >= model_atom_count:
            raise E3FPInheritanceError(
                "E3FP_SHELL_CENTER_OUT_OF_RANGE", "shell center is outside the model atom domain"
            )
        level = _shell_level(shell, fingerprinter.radius_multiplier)
        if level > FP_LEVEL:
            raise E3FPInheritanceError(
                "E3FP_SHELL_LEVEL_ABOVE_REQUESTED", "shell level exceeds the requested E3FP level"
            )
        slot = (center_atom, level)
        if slot in slots_seen:
            raise E3FPInheritanceError(
                "E3FP_DUPLICATE_CENTER_RADIUS_SLOT", "two shells address the same atom/level slot"
            )
        slots_seen.add(slot)

        raw_identifier = getattr(shell, "identifier", None)
        raw_bit = _fold_identifier(raw_identifier, signed_to_unsigned_int)
        is_duplicate = bool(getattr(shell, "is_duplicate", False))
        inherited_identifier = raw_identifier

        if is_duplicate:
            duplicate = getattr(shell, "duplicate", None)
            if duplicate is None or duplicate is shell:
                raise E3FPInheritanceError(
                    "E3FP_DUPLICATE_POINTER_MISSING", "duplicate shell lacks a prior accepted shell"
                )
            if bool(getattr(duplicate, "is_duplicate", False)):
                raise E3FPInheritanceError(
                    "E3FP_DUPLICATE_POINTER_NOT_ACCEPTED",
                    "duplicate pointer must identify a non-duplicate accepted shell",
                )
            inherited_identifier = getattr(duplicate, "identifier", None)
            if inherited_identifier is None:
                raise E3FPInheritanceError(
                    "E3FP_DUPLICATE_IDENTIFIER_MISSING",
                    "duplicate pointer has no accepted-shell identifier",
                )
            try:
                same_substructure = bool(shell.substruct == duplicate.substruct)
            except Exception as exc:
                raise E3FPInheritanceError(
                    "E3FP_DUPLICATE_SUBSTRUCTURE_INVALID",
                    "duplicate substructure equality cannot be evaluated",
                ) from exc
            if not same_substructure:
                raise E3FPInheritanceError(
                    "E3FP_DUPLICATE_SUBSTRUCTURE_MISMATCH",
                    "duplicate pointer refers to a different atom-set substructure",
                )
            duplicate_slots += 1
            duplicate_atoms.add(center_atom)
            duplicate_mask[center_atom, level] = True
            if int(inherited_identifier) != int(raw_identifier):
                changed_identifier_slots += 1

        inherited_bit = _fold_identifier(inherited_identifier, signed_to_unsigned_int)
        if inherited_bit not in final_indices:
            raise E3FPInheritanceError(
                "E3FP_INHERITED_BIT_NOT_IN_FINAL_FINGERPRINT",
                "inherited folded identifier is absent from the final fingerprint",
            )
        if raw_bit != inherited_bit:
            changed_token_slots += 1
        raw[center_atom, level] = raw_bit
        inherited[center_atom, level] = inherited_bit

    if shells_seen == 0:
        raise E3FPInheritanceError("E3FP_NO_SHELLS", "fingerprinter emitted no shells")
    raw_padding = raw == -1
    inherited_padding = inherited == -1
    if not bool(np.array_equal(raw_padding, inherited_padding)):
        raise E3FPInheritanceError(
            "E3FP_PADDING_MASK_MISMATCH", "raw and inherited padding masks differ"
        )
    if bool(np.any(raw[:, 0] == -1)) or bool(np.any(inherited[:, 0] == -1)):
        raise E3FPInheritanceError(
            "E3FP_LEVEL0_MISSING", "every model atom requires a level-0 identifier"
        )
    if not bool(np.array_equal(raw[:, 0], inherited[:, 0])):
        raise E3FPInheritanceError(
            "E3FP_LEVEL0_INHERITANCE_MISMATCH", "level-0 identifiers must not use inheritance"
        )
    if bool(np.any(np.all(raw_padding, axis=1))) or bool(
        np.any(np.all(inherited_padding, axis=1))
    ):
        raise E3FPInheritanceError(
            "E3FP_ALL_PADDING_MODEL_ROW", "a model atom has no E3FP shell identifier"
        )
    if (
        int(raw.min()) < -1
        or int(raw.max()) >= FP_BITS
        or int(inherited.min()) < -1
        or int(inherited.max()) >= FP_BITS
    ):
        raise E3FPInheritanceError(
            "E3FP_VALUE_RANGE_INVALID", "raw or inherited matrix violates the folded range"
        )

    raw = np.ascontiguousarray(raw, dtype=np.int32)
    inherited = np.ascontiguousarray(inherited, dtype=np.int32)
    duplicate_mask = np.ascontiguousarray(duplicate_mask, dtype=np.bool_)
    summary = {
        "semantics_id": SEMANTICS_ID,
        "shells_seen": int(shells_seen),
        "slots_populated": int(len(slots_seen)),
        "duplicate_slots": int(duplicate_slots),
        "duplicate_atoms": int(len(duplicate_atoms)),
        "changed_identifier_slots": int(changed_identifier_slots),
        "changed_token_slots": int(changed_token_slots),
    }
    return raw, inherited, duplicate_mask, summary


def generate_e3fp_projection_pair(np, e3fp_api, geometry_mol, ordinal: int):
    """Run E3FP once and derive both shell projections from that run."""

    if geometry_mol is None or not hasattr(geometry_mol, "GetNumAtoms"):
        raise E3FPInheritanceError(
            "E3FP_GEOMETRY_MOL_INVALID", "geometry_mol is not a molecule-like object"
        )
    try:
        model_atom_count = int(geometry_mol.GetNumAtoms())
    except Exception as exc:
        raise E3FPInheritanceError(
            "E3FP_GEOMETRY_MOL_INVALID", "geometry_mol atom count is unavailable"
        ) from exc
    if model_atom_count <= 0:
        raise E3FPInheritanceError(
            "E3FP_GEOMETRY_MOL_INVALID", "geometry_mol has no model atoms"
        )
    try:
        geometry_mol.SetProp("_Name", "r1_pcqm_e3fp_inheritance_{:09d}".format(int(ordinal)))
    except Exception as exc:
        raise E3FPInheritanceError(
            "E3FP_GEOMETRY_MOL_INVALID", "geometry_mol cannot receive the required E3FP name"
        ) from exc

    generator = e3fp_api.get("fprints_from_mol_verbose") if isinstance(e3fp_api, dict) else None
    signed_to_unsigned_int = (
        e3fp_api.get("signed_to_unsigned_int") if isinstance(e3fp_api, dict) else None
    )
    if not callable(generator) or not callable(signed_to_unsigned_int):
        raise E3FPInheritanceError(
            "E3FP_API_INVALID", "E3FP API lacks the verbose generator or folding function"
        )

    root_logger = logging.getLogger()
    previous_root_level = root_logger.level
    try:
        if previous_root_level < logging.WARNING:
            root_logger.setLevel(logging.WARNING)
        # This is intentionally the only E3FP generation call in the function.
        fprints, fingerprinter = generator(
            geometry_mol, fprint_params=dict(E3FP_INVOCATION)
        )
    except Exception as exc:
        raise E3FPInheritanceError(
            "E3FP_GENERATION_FAILED", "E3FP generation failed"
        ) from exc
    finally:
        root_logger.setLevel(previous_root_level)

    try:
        fingerprint_count = len(fprints)
    except Exception as exc:
        raise E3FPInheritanceError(
            "E3FP_FINGERPRINT_RESULT_INVALID", "E3FP result is not a fingerprint sequence"
        ) from exc
    if fingerprint_count != 1:
        raise E3FPInheritanceError(
            "E3FP_FINGERPRINT_RESULT_INVALID",
            "a paired geometry record requires exactly one conformer fingerprint",
        )
    final_fingerprint = fprints[0]
    if not hasattr(final_fingerprint, "indices"):
        raise E3FPInheritanceError(
            "E3FP_FINGERPRINT_RESULT_INVALID", "final fingerprint has no folded indices"
        )

    resolved = {
        "bits": int(fingerprinter.bits),
        "level": int(fingerprinter.level),
        "radius_multiplier": float(fingerprinter.radius_multiplier),
        "stereo": bool(fingerprinter.stereo),
        "include_disconnected": bool(fingerprinter.include_disconnected),
        "rdkit_invariants": bool(fingerprinter.rdkit_invariants),
        "exclude_floating": bool(fingerprinter.exclude_floating),
        "remove_duplicate_substructs": bool(fingerprinter.remove_duplicate_substructs),
        "fingerprint_type": getattr(
            fingerprinter.fp_type, "__name__", str(fingerprinter.fp_type)
        ),
        "all_iters": True,
    }
    if (
        resolved["bits"] != FP_BITS
        or resolved["level"] != FP_LEVEL
        or resolved["rdkit_invariants"] is not True
        or resolved["exclude_floating"] is not False
        or resolved["remove_duplicate_substructs"] is not True
    ):
        raise E3FPInheritanceError(
            "E3FP_RESOLVED_CONFIG_MISMATCH",
            "resolved E3FP configuration cannot provide the frozen inheritance semantic",
        )

    raw, inherited, duplicate_mask, summary = build_shell_projection_pair(
        np,
        fingerprinter,
        signed_to_unsigned_int,
        model_atom_count,
        final_fingerprint.indices,
    )
    return raw, inherited, duplicate_mask, summary, resolved
