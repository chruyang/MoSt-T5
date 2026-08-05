"""R1 sidecar adapters.

Nothing in this package mutates or imports the historical CAMT5 implementation.
"""

from .mol_linearizer import (
    CrossMotifBond,
    LinearizationMetadata,
    LinearizationResult,
    linearize_mol,
)

__all__ = [
    "CrossMotifBond",
    "LinearizationMetadata",
    "LinearizationResult",
    "linearize_mol",
]
