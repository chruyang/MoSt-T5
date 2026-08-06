#!/usr/bin/env python3
"""One RDKit identity universe shared by downstream split builders.

The project-wide molecule identity contract is intentionally narrower than a
generic ``RemoveHs`` policy.  It copies the parsed molecule and changes only
``RemoveHsParameters.removeDefiningBondStereo`` before sanitization and stereo
assignment.  Isotopic, mapped, query, and otherwise chemically meaningful
hydrogens retain RDKit's defaults.

This module owns the executable implementation so a builder cannot claim the
PCQM identity contract while silently serializing a different molecule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem


class IdentityNormalizationError(ValueError):
    """Raised when a molecule cannot enter the frozen identity universe."""


@dataclass(frozen=True)
class CanonicalIdentityForms:
    """Post-projection molecule plus its strict and connectivity identities."""

    molecule: Any
    strict_isomeric_smiles: str
    connectivity_smiles: str


def normalize_molecule(molecule: Any) -> Any:
    """Return a copied molecule under the frozen minimal explicit-H projection."""

    if molecule is None:
        raise IdentityNormalizationError("molecule must not be None")
    parameters = Chem.RemoveHsParameters()
    if not hasattr(parameters, "removeDefiningBondStereo"):
        raise RuntimeError(
            "installed RDKit lacks RemoveHsParameters.removeDefiningBondStereo"
        )
    parameters.removeDefiningBondStereo = True
    try:
        normalized = Chem.RemoveHs(Chem.Mol(molecule), parameters, sanitize=True)
        Chem.SanitizeMol(normalized)
        Chem.AssignStereochemistry(normalized, cleanIt=True, force=True)
    except Exception as exc:
        raise IdentityNormalizationError(
            "RDKit failed during the frozen explicit-H projection"
        ) from exc
    return normalized


def canonical_forms_from_molecule(molecule: Any) -> CanonicalIdentityForms:
    """Canonicalize an already parsed molecule under the shared contract."""

    normalized = normalize_molecule(molecule)
    return CanonicalIdentityForms(
        molecule=normalized,
        strict_isomeric_smiles=Chem.MolToSmiles(
            normalized,
            canonical=True,
            isomericSmiles=True,
            kekuleSmiles=False,
        ),
        connectivity_smiles=Chem.MolToSmiles(
            normalized,
            canonical=True,
            isomericSmiles=False,
            kekuleSmiles=False,
        ),
    )


def canonical_forms_from_smiles(smiles: str) -> CanonicalIdentityForms:
    """Parse a non-empty SMILES and canonicalize it under the shared contract."""

    if not isinstance(smiles, str) or not smiles.strip():
        raise IdentityNormalizationError("smiles must be a non-empty string")
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        raise IdentityNormalizationError("RDKit could not parse smiles")
    return canonical_forms_from_molecule(molecule)
