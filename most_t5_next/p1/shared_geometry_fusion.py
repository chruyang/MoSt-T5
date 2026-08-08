"""One shared E3FP-to-carrier addition path for the P1 A1/M1 cells.

The scientific comparison changes only ``e3fp_atom_to_token``: A1 assigns
one atom to each atom/SELFIES carrier, whereas M1 assigns every atom in a
logical motif to the motif carrier.  Both cells therefore use the same four
level-specific embedding tables, level sum, and carrier mean implemented here.

This module intentionally contains no condition-specific parameters, teacher,
regression loss, gate, concatenation, or geometry-free branch.  A0 and M0 use
the ordinary T5 input embeddings without calling this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor, nn

from .experiment_grid import GeometryBatchSidecar


class GeometryFusionError(ValueError):
    """Geometry tensors cannot be applied to the supplied T5 embeddings."""


@dataclass(frozen=True)
class GeometryTensorSidecar:
    """Tensor form of the model-facing geometry inputs.

    Shapes are ``[B, A, L]`` for E3FP IDs and ``[B, A]`` for both the atom
    mask and atom-to-token mapping.  The optional Boolean attachment role is
    retained for a later motif-state encoder and ignored by the original P1
    fusion.  Real atoms are selected by the Boolean
    mask.  Every padded atom must contain only ``-1`` in both ID and carrier
    fields.
    """

    e3fp_ids: Tensor
    e3fp_atom_mask: Tensor
    e3fp_atom_to_token: Tensor
    e3fp_atom_is_attachment: Tensor | None = None

    @classmethod
    def from_contract(
        cls,
        sidecar: GeometryBatchSidecar,
        *,
        device: torch.device,
    ) -> "GeometryTensorSidecar":
        """Materialize a validated tuple contract on the model device."""

        if not isinstance(sidecar, GeometryBatchSidecar):
            raise GeometryFusionError("sidecar must be a GeometryBatchSidecar")
        return cls(
            e3fp_ids=torch.as_tensor(
                sidecar.e3fp_ids,
                dtype=torch.long,
                device=device,
            ),
            e3fp_atom_mask=torch.as_tensor(
                sidecar.e3fp_atom_mask,
                dtype=torch.bool,
                device=device,
            ),
            e3fp_atom_to_token=torch.as_tensor(
                sidecar.e3fp_atom_to_token,
                dtype=torch.long,
                device=device,
            ),
            e3fp_atom_is_attachment=(
                None
                if sidecar.e3fp_atom_is_attachment is None
                else torch.as_tensor(
                    sidecar.e3fp_atom_is_attachment,
                    dtype=torch.bool,
                    device=device,
                )
            ),
        )


GeometryInput = Union[GeometryBatchSidecar, GeometryTensorSidecar]


class SharedE3FPCarrierFusion(nn.Module):
    """Add invariantly aggregated E3FP states to T5 carrier embeddings.

    The chain is fixed as:

    ``level-specific E3FP embeddings -> sum over four shell levels -> atom state``
    ``-> mean over atoms that``
    ``share a carrier token -> addition to inputs_embeds``.

    ``num_e3fp_embeddings`` counts valid E3FP IDs per shell level.  Each of
    the four tables has ``num_e3fp_embeddings + 1`` rows because row zero is
    reserved for the shifted ``-1`` padding value.  The geometry parameter
    count is therefore exactly
    ``4 * (num_e3fp_embeddings + 1) * hidden_size`` in every A0/A1/M0/M1
    wrapper; only A1/M1 execute the module.

    ``attention_mask`` is required so a carrier can never point into T5 input
    padding.  A :class:`GeometryBatchSidecar` is converted to tensors on the
    embedding device; a pre-tensorized sidecar must already be colocated.
    """

    def __init__(self, *, num_e3fp_embeddings: int, hidden_size: int) -> None:
        super().__init__()
        if isinstance(num_e3fp_embeddings, bool) or not isinstance(
            num_e3fp_embeddings, int
        ):
            raise GeometryFusionError("num_e3fp_embeddings must be an integer")
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
            raise GeometryFusionError("hidden_size must be an integer")
        if num_e3fp_embeddings <= 0 or hidden_size <= 0:
            raise GeometryFusionError("embedding dimensions must be positive")

        self.num_e3fp_embeddings = num_e3fp_embeddings
        self.hidden_size = hidden_size
        self.num_e3fp_levels = 4
        self.level_embeddings = nn.ModuleList(
            nn.Embedding(
                num_e3fp_embeddings + 1,
                hidden_size,
                padding_idx=0,
            )
            for _ in range(self.num_e3fp_levels)
        )

    def _tensorize(
        self,
        geometry: GeometryInput,
        *,
        device: torch.device,
    ) -> GeometryTensorSidecar:
        if isinstance(geometry, GeometryBatchSidecar):
            return GeometryTensorSidecar.from_contract(geometry, device=device)
        if isinstance(geometry, GeometryTensorSidecar):
            return geometry
        raise GeometryFusionError(
            "geometry must be GeometryBatchSidecar or GeometryTensorSidecar"
        )

    def _validate(
        self,
        input_embeddings: Tensor,
        attention_mask: Tensor,
        geometry: GeometryTensorSidecar,
        *,
        contract: GeometryBatchSidecar | None,
    ) -> tuple[int, int, int, Tensor]:
        if not isinstance(input_embeddings, Tensor) or input_embeddings.ndim != 3:
            raise GeometryFusionError("input_embeddings must have shape [B, T, H]")
        if not input_embeddings.is_floating_point():
            raise GeometryFusionError("input_embeddings must be floating point")

        batch_size, token_width, hidden_size = input_embeddings.shape
        if batch_size <= 0 or token_width <= 0:
            raise GeometryFusionError("input_embeddings must be nonempty")
        if hidden_size != self.hidden_size:
            raise GeometryFusionError("input embedding width differs from hidden_size")
        reference_weight = self.level_embeddings[0].weight
        if input_embeddings.device != reference_weight.device:
            raise GeometryFusionError("module and input_embeddings must share a device")
        if input_embeddings.dtype != reference_weight.dtype:
            raise GeometryFusionError("module and input_embeddings must share a dtype")

        if not isinstance(attention_mask, Tensor) or attention_mask.shape != (
            batch_size,
            token_width,
        ):
            raise GeometryFusionError("attention_mask must have shape [B, T]")
        if attention_mask.device != input_embeddings.device:
            raise GeometryFusionError("attention_mask must share the model device")
        if attention_mask.is_floating_point() or attention_mask.is_complex():
            raise GeometryFusionError("attention_mask must be Boolean or integer")
        if bool(((attention_mask != 0) & (attention_mask != 1)).any().item()):
            raise GeometryFusionError("attention_mask must be binary")
        token_mask = attention_mask.to(dtype=torch.bool)
        if bool((~token_mask.any(dim=1)).any().item()):
            raise GeometryFusionError("every record must expose at least one token")

        e3fp_ids = geometry.e3fp_ids
        atom_mask = geometry.e3fp_atom_mask
        atom_to_token = geometry.e3fp_atom_to_token
        if not all(isinstance(value, Tensor) for value in (e3fp_ids, atom_mask, atom_to_token)):
            raise GeometryFusionError("all geometry fields must be tensors")
        if e3fp_ids.ndim != 3:
            raise GeometryFusionError("e3fp_ids must have shape [B, A, L]")
        if e3fp_ids.shape[0] != batch_size or e3fp_ids.shape[1] <= 0 or e3fp_ids.shape[2] <= 0:
            raise GeometryFusionError("e3fp_ids batch and atom/level widths are invalid")
        atom_width, level_count = e3fp_ids.shape[1], e3fp_ids.shape[2]
        if level_count != self.num_e3fp_levels:
            raise GeometryFusionError(
                "e3fp_ids must expose exactly four ordered E3FP shell levels"
            )
        expected_atom_shape = (batch_size, atom_width)
        if atom_mask.shape != expected_atom_shape:
            raise GeometryFusionError("e3fp_atom_mask must have shape [B, A]")
        if atom_to_token.shape != expected_atom_shape:
            raise GeometryFusionError("e3fp_atom_to_token must have shape [B, A]")
        if e3fp_ids.device != input_embeddings.device:
            raise GeometryFusionError("e3fp_ids must share the model device")
        if atom_mask.device != input_embeddings.device:
            raise GeometryFusionError("e3fp_atom_mask must share the model device")
        if atom_to_token.device != input_embeddings.device:
            raise GeometryFusionError("e3fp_atom_to_token must share the model device")
        if e3fp_ids.dtype == torch.bool or e3fp_ids.is_floating_point() or e3fp_ids.is_complex():
            raise GeometryFusionError("e3fp_ids must use an integer dtype")
        if atom_mask.dtype != torch.bool:
            raise GeometryFusionError("e3fp_atom_mask must use torch.bool")
        if (
            atom_to_token.dtype == torch.bool
            or atom_to_token.is_floating_point()
            or atom_to_token.is_complex()
        ):
            raise GeometryFusionError("e3fp_atom_to_token must use an integer dtype")

        if contract is not None:
            if contract.token_width != token_width:
                raise GeometryFusionError("geometry contract token_width differs from T5")
            if contract.e3fp_level_count != level_count:
                raise GeometryFusionError("geometry contract level count differs from tensors")

        if bool((~atom_mask.any(dim=1)).any().item()):
            raise GeometryFusionError("every record must contain at least one active atom")
        invalid_level = (e3fp_ids < -1) | (e3fp_ids >= self.num_e3fp_embeddings)
        if bool(invalid_level.any().item()):
            raise GeometryFusionError("E3FP IDs must be -1 or in the embedding range")
        valid_level = e3fp_ids >= 0
        if bool((atom_mask & ~valid_level.any(dim=2)).any().item()):
            raise GeometryFusionError("every active atom needs at least one E3FP level")
        if bool(((~atom_mask).unsqueeze(2) & (e3fp_ids != -1)).any().item()):
            raise GeometryFusionError("padded atoms must contain only -1 E3FP IDs")
        if bool((atom_mask & ((atom_to_token < 0) | (atom_to_token >= token_width))).any().item()):
            raise GeometryFusionError("active atom carrier is outside the token domain")
        if bool(((~atom_mask) & (atom_to_token != -1)).any().item()):
            raise GeometryFusionError("padded atom carriers must be -1")

        safe_carriers = atom_to_token.clamp_min(0).to(dtype=torch.long)
        carrier_is_visible = token_mask.gather(1, safe_carriers)
        if bool((atom_mask & ~carrier_is_visible).any().item()):
            raise GeometryFusionError("active atom carrier points into T5 padding")
        return batch_size, token_width, atom_width, valid_level

    def forward(
        self,
        input_embeddings: Tensor,
        geometry: GeometryInput,
        *,
        attention_mask: Tensor,
    ) -> Tensor:
        """Return T5 input embeddings with geometry added at carrier tokens."""

        contract = geometry if isinstance(geometry, GeometryBatchSidecar) else None
        tensor_geometry = self._tensorize(geometry, device=input_embeddings.device)
        batch_size, token_width, _atom_width, valid_level = self._validate(
            input_embeddings,
            attention_mask,
            tensor_geometry,
            contract=contract,
        )

        # Preserve shell identity: level i always uses table i.  Shifting by
        # one maps the external -1 sentinel to each table's padding row zero,
        # matching the established GSMATEmbeddings convention in
        # model/modeling.py.  Shell states are summed, not averaged.
        shifted_ids = (tensor_geometry.e3fp_ids + 1).to(dtype=torch.long)
        atom_states = input_embeddings.new_zeros(
            (
                shifted_ids.shape[0],
                shifted_ids.shape[1],
                self.hidden_size,
            )
        )
        for level_index, embedding in enumerate(self.level_embeddings):
            level_state = embedding(shifted_ids[:, :, level_index])
            level_state = level_state * valid_level[:, :, level_index].unsqueeze(2)
            atom_states = atom_states + level_state

        atom_mask = tensor_geometry.e3fp_atom_mask
        batch_offsets = (
            torch.arange(batch_size, device=input_embeddings.device).unsqueeze(1)
            * token_width
        )
        flat_carriers = (
            tensor_geometry.e3fp_atom_to_token.to(dtype=torch.long) + batch_offsets
        )[atom_mask]
        flat_atom_states = atom_states[atom_mask]

        flat_carrier_states = input_embeddings.new_zeros(
            (batch_size * token_width, self.hidden_size)
        )
        flat_carrier_states = flat_carrier_states.index_add(
            0,
            flat_carriers,
            flat_atom_states,
        )
        carrier_counts = input_embeddings.new_zeros((batch_size * token_width, 1))
        carrier_counts = carrier_counts.index_add(
            0,
            flat_carriers,
            input_embeddings.new_ones((flat_carriers.numel(), 1)),
        )
        flat_carrier_states = flat_carrier_states / carrier_counts.clamp_min(1)
        carrier_states = flat_carrier_states.view(
            batch_size,
            token_width,
            self.hidden_size,
        )
        return input_embeddings + carrier_states
