"""Frozen G1b Deep Sets geometry carrier used by the G2 bridge screen.

The proven G1b encoder is reused without updating its parameters.  Its
permutation-invariant motif representation is projected to the T5 width and
added only at the already-declared motif carrier tokens.  The projection is
the sole trainable geometry bridge; both G2 cells instantiate the same module
and initialization, while the topology-only cell does not execute it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from most_t5_next.p1.build_union_init_checkpoint_v1 import (
    load_verified_union_init_checkpoint,
)
from most_t5_next.p1.experiment_grid import GeometryBatchSidecar
from most_t5_next.p1.four_grid_t5_wrapper import FourGridT5Wrapper
from most_t5_next.p1.level_aware_motif_state_v1 import (
    LevelAwareMotifStateEncoder,
)
from most_t5_next.p1.shared_geometry_fusion import GeometryTensorSidecar


FUSION_ID = "frozen-g1b-deep-sets-motif-carrier-linear-v1"
G1B_SCHEMA = "most-t5-p1/g1-motif-state-screen/v1"


class G1DeepSetsFusionError(ValueError):
    """The G1b checkpoint or geometry carrier input is incompatible."""


def load_g1b_encoder(checkpoint_path: Path) -> LevelAwareMotifStateEncoder:
    """Load the accepted 64->128 Deep Sets encoder and freeze it."""

    path = Path(checkpoint_path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise G1DeepSetsFusionError("G1b checkpoint is absent or empty")
    try:
        payload = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as exc:
        raise G1DeepSetsFusionError("G1b checkpoint cannot be loaded") from exc
    if not isinstance(payload, Mapping):
        raise G1DeepSetsFusionError("G1b checkpoint must contain one mapping")
    if (
        payload.get("schema_version") != G1B_SCHEMA
        or payload.get("pooling") != "deep_sets"
        or payload.get("completed_updates") != 500
    ):
        raise G1DeepSetsFusionError("G1b checkpoint contract differs")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise G1DeepSetsFusionError("G1b checkpoint lacks its model state")
    encoder = LevelAwareMotifStateEncoder(
        num_e3fp_embeddings=4096,
        embedding_dim=64,
        hidden_dim=128,
        pooling="deep_sets",
    )
    encoder.load_state_dict(dict(state), strict=True)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


class FrozenG1DeepSetsCarrierFusion(nn.Module):
    """Project a frozen G1b motif state to its T5 sentinel carrier."""

    def __init__(
        self,
        *,
        num_e3fp_embeddings: int,
        hidden_size: int,
        g1_checkpoint: Path,
    ) -> None:
        super().__init__()
        if int(num_e3fp_embeddings) != 4096:
            raise G1DeepSetsFusionError("G1b is fixed to the 4096-state E3FP domain")
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise G1DeepSetsFusionError("hidden_size must be a positive integer")
        self.num_e3fp_embeddings = 4096
        self.hidden_size = hidden_size
        self.g1_encoder = load_g1b_encoder(g1_checkpoint)
        self.bridge_norm = nn.LayerNorm(128)
        self.bridge_projection = nn.Linear(128, hidden_size, bias=False)

    def train(self, mode: bool = True) -> "FrozenG1DeepSetsCarrierFusion":
        super().train(mode)
        # The state encoder is a fixed scientific input, not a jointly tuned
        # auxiliary network.  Only norm/projection follow the requested mode.
        self.g1_encoder.eval()
        return self

    @staticmethod
    def _tensorize(
        geometry: GeometryBatchSidecar | GeometryTensorSidecar,
        *,
        device: torch.device,
    ) -> GeometryTensorSidecar:
        if isinstance(geometry, GeometryBatchSidecar):
            return GeometryTensorSidecar.from_contract(geometry, device=device)
        if isinstance(geometry, GeometryTensorSidecar):
            return geometry
        raise G1DeepSetsFusionError("unknown geometry sidecar type")

    def forward(
        self,
        input_embeddings: Tensor,
        geometry: GeometryBatchSidecar | GeometryTensorSidecar,
        *,
        attention_mask: Tensor,
    ) -> Tensor:
        if input_embeddings.ndim != 3:
            raise G1DeepSetsFusionError("input embeddings must be [B,T,H]")
        batch_size, token_width, hidden_size = input_embeddings.shape
        if hidden_size != self.hidden_size:
            raise G1DeepSetsFusionError("T5 hidden width differs from the bridge")
        tensor = self._tensorize(geometry, device=input_embeddings.device)
        roles = tensor.e3fp_atom_is_attachment
        if roles is None:
            raise G1DeepSetsFusionError(
                "G2 requires the persisted core-versus-attachment atom roles"
            )
        if (
            tensor.e3fp_ids.shape[:2] != tensor.e3fp_atom_mask.shape
            or tensor.e3fp_atom_mask.shape != tensor.e3fp_atom_to_token.shape
            or roles.shape != tensor.e3fp_atom_mask.shape
        ):
            raise G1DeepSetsFusionError("G2 geometry atom tensors disagree")
        if tensor.e3fp_ids.shape[0] != batch_size or tensor.e3fp_ids.shape[2] != 4:
            raise G1DeepSetsFusionError("G2 requires four E3FP levels for every batch row")
        if attention_mask.shape != (batch_size, token_width):
            raise G1DeepSetsFusionError("attention mask shape differs from T5 input")

        atom_valid = tensor.e3fp_atom_mask
        carriers = tensor.e3fp_atom_to_token.to(torch.long)
        if bool((atom_valid & ((carriers < 0) | (carriers >= token_width))).any()):
            raise G1DeepSetsFusionError("active motif carrier is outside the token domain")
        safe_groups = carriers.clamp_min(0)
        with torch.set_grad_enabled(False):
            state = self.g1_encoder(
                tensor.e3fp_ids,
                atom_valid,
                safe_groups,
                num_groups=token_width,
                atom_is_attachment=roles,
            ).group_hidden
        projected = self.bridge_projection(self.bridge_norm(state.to(input_embeddings.dtype)))

        carrier_counts = input_embeddings.new_zeros((batch_size, token_width))
        carrier_counts.scatter_add_(
            1,
            safe_groups,
            atom_valid.to(input_embeddings.dtype),
        )
        carrier_mask = (carrier_counts > 0) & attention_mask.to(torch.bool)
        return input_embeddings + projected * carrier_mask.unsqueeze(-1)


def load_verified_g1_bridge_wrapper(
    *,
    condition_id: str,
    base_model_snapshot: Path,
    base_tokenizer_snapshot: Path,
    union_tokenizer_dir: Path,
    output_dir: Path,
    geometry_fusion_seed: int,
    num_e3fp_embeddings: int,
    g1_checkpoint: Path,
) -> Any:
    """Load one matched T5 cell with the same deterministic G2 bridge."""

    verified = load_verified_union_init_checkpoint(
        base_model_snapshot=base_model_snapshot,
        base_tokenizer_snapshot=base_tokenizer_snapshot,
        union_tokenizer_dir=union_tokenizer_dir,
        output_dir=output_dir,
        geometry_fusion_seed=geometry_fusion_seed,
        num_e3fp_embeddings=num_e3fp_embeddings,
    )
    rng_state = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(geometry_fusion_seed)

        def factory(*, num_e3fp_embeddings: int, hidden_size: int) -> nn.Module:
            return FrozenG1DeepSetsCarrierFusion(
                num_e3fp_embeddings=num_e3fp_embeddings,
                hidden_size=hidden_size,
                g1_checkpoint=g1_checkpoint,
            )

        wrapper = FourGridT5Wrapper(
            t5_model=verified.model,
            condition_id=condition_id,
            num_e3fp_embeddings=num_e3fp_embeddings,
            geometry_fusion_factory=factory,
        )
    finally:
        torch.random.set_rng_state(rng_state)
    if {parameter.device.type for parameter in wrapper.parameters()} != {"cpu"}:
        raise G1DeepSetsFusionError("verified G2 wrapper must start on CPU")
    return wrapper


__all__ = [
    "FUSION_ID",
    "FrozenG1DeepSetsCarrierFusion",
    "G1DeepSetsFusionError",
    "load_g1b_encoder",
    "load_verified_g1_bridge_wrapper",
]
