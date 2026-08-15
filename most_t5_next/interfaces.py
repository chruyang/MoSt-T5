"""Shared task and model-interface types for MoSt-T5."""

from typing import Literal


GeometryMode = Literal["full", "carrier", "endpoint", "none"]

PHASE_TASKS = {
    1: ("M", "MG"),
    2: ("SYN", "TXT", "CAP", "T2M"),
}

GEOMETRY_INPUT_NAMES = frozenset(
    {
        "e3fp_ids",
        "atom_mask",
        "atom_to_fragment",
        "fragment_mask",
        "fragment_to_carrier",
        "identity_span_bounds",
        "endpoint_mask",
        "endpoint_to_atom",
        "endpoint_to_token",
        "endpoint_to_fragment",
        "endpoint_is_explicit",
        "token_is_connector_endpoint",
        "atom_is_attachment",
        "fragment_geometry_mask",
        "endpoint_geometry_mask",
    }
)

OPTIONAL_GEOMETRY_INPUT_NAMES = frozenset(
    {"fragment_geometry_mask", "endpoint_geometry_mask"}
)
REQUIRED_GEOMETRY_INPUT_NAMES = GEOMETRY_INPUT_NAMES - OPTIONAL_GEOMETRY_INPUT_NAMES

__all__ = [
    "GEOMETRY_INPUT_NAMES",
    "GeometryMode",
    "OPTIONAL_GEOMETRY_INPUT_NAMES",
    "PHASE_TASKS",
    "REQUIRED_GEOMETRY_INPUT_NAMES",
]
