"""Isolated P1 candidate interfaces.

The modules in this package are deliberately independent of the historical
``dataset/``, ``model/`` and ``tokenization/`` trees.  The first candidate is
only a synthetic, pre-training-inadmissible contract for a hybrid motif codec,
a three-domain :class:`BoundRecord`, and CE-first whole-identity corruption.
"""

from .hybrid_codec import (
    CodecContractError,
    ConnectionEndpoint,
    CrossMotifConnection,
    HybridMotifCodec,
    LogicalMoleculeSchema,
    LogicalMotif,
    LogicalMotifIdentity,
    SurfaceEncoding,
)
from .bound_record import (
    BoundRecord,
    BoundRecordInvariantError,
    Span,
    build_bound_record,
    build_synthetic_token_table,
)
from .ce_collator import (
    CEFirstExample,
    CollatorContractError,
    MaskedIdentityTarget,
    SyntheticCEFirstCollator,
)
from .runtime_bridge import (
    CEModelExample,
    CE_FIRST_PROFILE,
    LABEL_PAD_ID,
    P1ArtifactBindings,
    P1MemberRef,
    PaddedCEBatch,
    RuntimeBridgeError,
    materialize_training_record,
    pad_ce_first_batch,
)
from .experiment_grid import (
    ATOM_ALIGNED_E3FP,
    ATOM_SELFIES_IDENTITY,
    BASE_T5_INPUT_KEYS,
    GEOMETRY_INPUT_KEYS,
    GRID_SPEC_VERSION,
    HYBRID_MOTIF_IDENTITY,
    MOTIF_MEAN_E3FP,
    NO_GEOMETRY,
    FourGridContractError,
    GeometryBatchSidecar,
    P1ConditionBatch,
    P1ConditionSpec,
    P1_CONDITION_SPECS,
    get_p1_condition_spec,
    validate_a1_m1_geometry_atom_parity,
)
from .production_bridge import (
    ProductionBridgeError,
    ProductionCEExample,
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
    collate_production_batch,
    collate_production_motif_record,
    collate_production_training_record,
    load_production_motif_record,
)
from .atom_production_bridge import (
    ALLOWED_UNCORRUPTED_TOKEN_ROLES,
    ATOM_IDENTITY_ROLE,
    ATOM_SELFIES_RECORD_SCHEMA,
    IDENTITY_SENTINEL_ROLE as ATOM_IDENTITY_SENTINEL_ROLE,
    AtomProductionBridgeError,
    ProductionAtomCEExample,
    ProductionAtomSelfiesRecord,
    collate_production_atom_batch,
    collate_production_atom_record,
)
from .training_adapter import (
    FOUR_GRID_MODEL_INPUT_KEYS,
    GEOMETRY_MODEL_INPUT_KEYS,
    MODEL_INPUT_KEYS,
    TrainingAdapterError,
    select_four_grid_forward_inputs,
    select_t5_forward_inputs,
    to_four_grid_batch_encoding,
    to_t5_batch_encoding,
)

# Keep pure-Python data preparation usable on hosts without PyTorch.  Training
# hosts receive these exports through the ordinary package namespace.
try:
    from .shared_geometry_fusion import (
        GeometryFusionError,
        GeometryTensorSidecar,
        SharedE3FPCarrierFusion,
    )
    from .four_grid_t5_wrapper import (
        FourGridT5Wrapper,
        FourGridT5WrapperError,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _GEOMETRY_FUSION_EXPORTS: tuple[str, ...] = ()
else:
    _GEOMETRY_FUSION_EXPORTS = (
        "GeometryFusionError",
        "GeometryTensorSidecar",
        "SharedE3FPCarrierFusion",
        "FourGridT5Wrapper",
        "FourGridT5WrapperError",
    )

__all__ = [
    "BoundRecord",
    "BoundRecordInvariantError",
    "CEFirstExample",
    "CEModelExample",
    "CollatorContractError",
    "CodecContractError",
    "FourGridContractError",
    "GeometryBatchSidecar",
    "GRID_SPEC_VERSION",
    "BASE_T5_INPUT_KEYS",
    "GEOMETRY_INPUT_KEYS",
    "ATOM_SELFIES_IDENTITY",
    "ATOM_IDENTITY_ROLE",
    "ATOM_IDENTITY_SENTINEL_ROLE",
    "ATOM_SELFIES_RECORD_SCHEMA",
    "ALLOWED_UNCORRUPTED_TOKEN_ROLES",
    "HYBRID_MOTIF_IDENTITY",
    "NO_GEOMETRY",
    "ATOM_ALIGNED_E3FP",
    "MOTIF_MEAN_E3FP",
    "CE_FIRST_PROFILE",
    "ConnectionEndpoint",
    "CrossMotifConnection",
    "HybridMotifCodec",
    "LogicalMoleculeSchema",
    "LogicalMotif",
    "LogicalMotifIdentity",
    "LABEL_PAD_ID",
    "MaskedIdentityTarget",
    "FOUR_GRID_MODEL_INPUT_KEYS",
    "GEOMETRY_MODEL_INPUT_KEYS",
    "MODEL_INPUT_KEYS",
    "P1ArtifactBindings",
    "P1ConditionBatch",
    "P1ConditionSpec",
    "P1_CONDITION_SPECS",
    "P1MemberRef",
    "PaddedCEBatch",
    "RuntimeBridgeError",
    "ProductionBridgeError",
    "AtomProductionBridgeError",
    "ProductionAtomCEExample",
    "ProductionAtomSelfiesRecord",
    "ProductionCEExample",
    "ProductionMotifRecord",
    "ProductionTokenizerRuntime",
    "Span",
    "SurfaceEncoding",
    "SyntheticCEFirstCollator",
    "TrainingAdapterError",
    "build_bound_record",
    "build_synthetic_token_table",
    "collate_production_batch",
    "collate_production_atom_batch",
    "collate_production_atom_record",
    "collate_production_motif_record",
    "collate_production_training_record",
    "get_p1_condition_spec",
    "validate_a1_m1_geometry_atom_parity",
    "materialize_training_record",
    "load_production_motif_record",
    "pad_ce_first_batch",
    "select_four_grid_forward_inputs",
    "select_t5_forward_inputs",
    "to_four_grid_batch_encoding",
    "to_t5_batch_encoding",
] + list(_GEOMETRY_FUSION_EXPORTS)
