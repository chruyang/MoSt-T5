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
    CE_FIRST_PROFILE,
    LABEL_PAD_ID,
    P1ArtifactBindings,
    P1MemberRef,
    PaddedCEBatch,
    RuntimeBridgeError,
    materialize_training_record,
    pad_ce_first_batch,
)
from .training_adapter import (
    MODEL_INPUT_KEYS,
    TrainingAdapterError,
    select_t5_forward_inputs,
    to_t5_batch_encoding,
)

__all__ = [
    "BoundRecord",
    "BoundRecordInvariantError",
    "CEFirstExample",
    "CollatorContractError",
    "CodecContractError",
    "CE_FIRST_PROFILE",
    "ConnectionEndpoint",
    "CrossMotifConnection",
    "HybridMotifCodec",
    "LogicalMoleculeSchema",
    "LogicalMotif",
    "LogicalMotifIdentity",
    "LABEL_PAD_ID",
    "MaskedIdentityTarget",
    "MODEL_INPUT_KEYS",
    "P1ArtifactBindings",
    "P1MemberRef",
    "PaddedCEBatch",
    "RuntimeBridgeError",
    "Span",
    "SurfaceEncoding",
    "SyntheticCEFirstCollator",
    "TrainingAdapterError",
    "build_bound_record",
    "build_synthetic_token_table",
    "materialize_training_record",
    "pad_ce_first_batch",
    "select_t5_forward_inputs",
    "to_t5_batch_encoding",
]
