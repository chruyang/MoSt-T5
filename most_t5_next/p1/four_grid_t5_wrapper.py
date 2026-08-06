"""One Trainer-facing T5 wrapper for every P1 A0/A1/M0/M1 cell.

The wrapper deliberately fixes the scientific difference outside T5 itself:

* every cell owns the same base-T5 module and the same
  :class:`SharedE3FPCarrierFusion` parameter schema;
* A0/M0 call the ordinary ``input_ids`` + ``labels`` T5 CE forward and reject
  geometry tensors;
* A1/M1 obtain the base input embeddings, apply the one shared additive E3FP
  carrier path, and call that same T5 with ``inputs_embeds`` and ``labels``.

No auxiliary loss, teacher, gate, concatenation, or condition-specific
geometry module is implemented here.  The fixed ``condition_id`` is ordinary
wrapper metadata rather than a model parameter, so all four cells have
identical state-dict keys when constructed from identical T5 backbones.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .experiment_grid import P1ConditionSpec, get_p1_condition_spec
from .shared_geometry_fusion import (
    GeometryTensorSidecar,
    SharedE3FPCarrierFusion,
)


class FourGridT5WrapperError(ValueError):
    """The configured grid cell or its Trainer batch violates the contract."""


class FourGridT5Wrapper(nn.Module):
    """Run one fixed A0/A1/M0/M1 condition through a common T5 architecture.

    Parameters
    ----------
    t5_model:
        A T5-compatible :class:`torch.nn.Module`.  It must expose
        ``get_input_embeddings()``, ``get_output_embeddings()`` and a config
        whose ``d_model``/``vocab_size`` agree with those modules.
    condition_id:
        Exactly one of ``A0``, ``A1``, ``M0`` or ``M1``.  This is fixed for
        the lifetime of the wrapper.  If a collator also supplies a
        ``condition_id`` field, it is checked and removed before the T5 call.
    num_e3fp_embeddings:
        Number of valid E3FP IDs at each of the four ordered shell levels.
        Four level-specific tables (plus one padding row per table) are
        constructed for every cell, including A0/M0, to preserve the complete
        parameter and checkpoint schema.

    The explicit forward signature lets Hugging Face Trainer retain the three
    geometry columns when ``remove_unused_columns=True``.  Returned objects
    are forwarded unchanged from the wrapped T5 model.
    """

    def __init__(
        self,
        t5_model: nn.Module,
        *,
        condition_id: str,
        num_e3fp_embeddings: int,
    ) -> None:
        super().__init__()
        if not isinstance(t5_model, nn.Module):
            raise FourGridT5WrapperError("t5_model must be a torch.nn.Module")
        if isinstance(num_e3fp_embeddings, bool) or not isinstance(
            num_e3fp_embeddings, int
        ):
            raise FourGridT5WrapperError(
                "num_e3fp_embeddings must be an integer"
            )
        if num_e3fp_embeddings <= 0:
            raise FourGridT5WrapperError(
                "num_e3fp_embeddings must be positive"
            )

        try:
            spec = get_p1_condition_spec(condition_id)
        except ValueError as exc:
            raise FourGridT5WrapperError(str(exc)) from exc

        input_embedding = self._get_embedding_module(
            t5_model,
            getter_name="get_input_embeddings",
            label="input embedding",
        )
        output_embedding = self._get_embedding_module(
            t5_model,
            getter_name="get_output_embeddings",
            label="LM head",
        )
        input_weight = self._get_embedding_weight(input_embedding, "input embedding")
        output_weight = self._get_embedding_weight(output_embedding, "LM head")

        vocab_size, hidden_size = input_weight.shape
        if output_weight.shape != (vocab_size, hidden_size):
            raise FourGridT5WrapperError(
                "T5 input embedding and LM head must share [vocab, hidden] shape"
            )

        config = getattr(t5_model, "config", None)
        if config is None:
            raise FourGridT5WrapperError("t5_model must expose config")
        config_hidden = getattr(config, "d_model", hidden_size)
        config_vocab = getattr(config, "vocab_size", vocab_size)
        if config_hidden != hidden_size:
            raise FourGridT5WrapperError(
                "config.d_model differs from the T5 input embedding width"
            )
        if config_vocab != vocab_size:
            raise FourGridT5WrapperError(
                "config.vocab_size differs from the frozen union vocabulary"
            )

        self.t5 = t5_model
        self.condition_id = spec.condition_id
        self.condition_spec: P1ConditionSpec = spec
        # This module name and shape are intentionally present in every cell.
        self.geometry_fusion = SharedE3FPCarrierFusion(
            num_e3fp_embeddings=num_e3fp_embeddings,
            hidden_size=hidden_size,
        )

    @staticmethod
    def _get_embedding_module(
        model: nn.Module,
        *,
        getter_name: str,
        label: str,
    ) -> nn.Module:
        getter = getattr(model, getter_name, None)
        if not callable(getter):
            raise FourGridT5WrapperError(
                f"t5_model must expose {getter_name}()"
            )
        module = getter()
        if not isinstance(module, nn.Module):
            raise FourGridT5WrapperError(f"T5 {label} must be a torch.nn.Module")
        return module

    @staticmethod
    def _get_embedding_weight(module: nn.Module, label: str) -> Tensor:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, Tensor) or weight.ndim != 2:
            raise FourGridT5WrapperError(f"T5 {label} must expose a rank-2 weight")
        if weight.shape[0] <= 0 or weight.shape[1] <= 0:
            raise FourGridT5WrapperError(f"T5 {label} dimensions must be positive")
        return weight

    @property
    def config(self) -> Any:
        """Expose the untouched base-T5 config expected by Trainer."""

        return self.t5.config

    @property
    def uses_geometry(self) -> bool:
        return self.condition_spec.uses_geometry

    def get_input_embeddings(self) -> nn.Module:
        """Delegate the standard model-introspection API without resizing it."""

        return self.t5.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        """Delegate access to the common LM head."""

        return self.t5.get_output_embeddings()

    def _validate_batch_condition(self, condition_id: object | None) -> None:
        if condition_id is None:
            return
        if isinstance(condition_id, str):
            observed = (condition_id,)
        elif isinstance(condition_id, Sequence) and not isinstance(
            condition_id, (bytes, bytearray)
        ):
            observed = tuple(condition_id)
            if not observed:
                raise FourGridT5WrapperError("batch condition_id cannot be empty")
        else:
            raise FourGridT5WrapperError(
                "batch condition_id must be a string or a nonempty string sequence"
            )
        if any(value != self.condition_id for value in observed):
            raise FourGridT5WrapperError(
                f"batch condition_id must match fixed wrapper condition {self.condition_id}"
            )

    def _validate_geometry_presence(
        self,
        e3fp_ids: Tensor | None,
        e3fp_atom_mask: Tensor | None,
        e3fp_atom_to_token: Tensor | None,
    ) -> bool:
        fields = (e3fp_ids, e3fp_atom_mask, e3fp_atom_to_token)
        present = tuple(value is not None for value in fields)
        if any(present) and not all(present):
            raise FourGridT5WrapperError(
                "e3fp_ids, e3fp_atom_mask and e3fp_atom_to_token are all-or-none"
            )
        has_geometry = all(present)
        if self.uses_geometry and not has_geometry:
            raise FourGridT5WrapperError(
                f"condition {self.condition_id} requires all geometry inputs"
            )
        if not self.uses_geometry and has_geometry:
            raise FourGridT5WrapperError(
                f"condition {self.condition_id} rejects geometry inputs"
            )
        return has_geometry

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        e3fp_ids: Tensor | None = None,
        e3fp_atom_mask: Tensor | None = None,
        e3fp_atom_to_token: Tensor | None = None,
        condition_id: object | None = None,
        **t5_kwargs: Any,
    ) -> Any:
        """Forward a Trainer batch and return the base T5 output unchanged."""

        self._validate_batch_condition(condition_id)
        has_geometry = self._validate_geometry_presence(
            e3fp_ids,
            e3fp_atom_mask,
            e3fp_atom_to_token,
        )
        if input_ids is None:
            raise FourGridT5WrapperError("input_ids are required by the frozen grid")
        if "inputs_embeds" in t5_kwargs:
            raise FourGridT5WrapperError(
                "external inputs_embeds are outside the frozen four-grid interface"
            )

        if not has_geometry:
            # This is intentionally the unmodified standard T5 CE call.
            return self.t5(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **t5_kwargs,
            )

        if attention_mask is None:
            raise FourGridT5WrapperError(
                "geometry-enabled cells require attention_mask"
            )
        geometry = GeometryTensorSidecar(
            e3fp_ids=e3fp_ids,
            e3fp_atom_mask=e3fp_atom_mask,
            e3fp_atom_to_token=e3fp_atom_to_token,
        )
        input_embeddings = self.get_input_embeddings()(input_ids)
        fused_embeddings = self.geometry_fusion(
            input_embeddings,
            geometry,
            attention_mask=attention_mask,
        )
        return self.t5(
            inputs_embeds=fused_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            **t5_kwargs,
        )


__all__ = [
    "FourGridT5Wrapper",
    "FourGridT5WrapperError",
]
