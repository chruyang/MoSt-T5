"""MoSt-T5: a stock T5 backbone with optional fragSMILES geometry injection."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from most_t5_next.interfaces import (
    GEOMETRY_INPUT_NAMES,
    GeometryMode,
    REQUIRED_GEOMETRY_INPUT_NAMES,
)

from .geometry import GeometryAdapter, GeometryEncoding


class MoStT5Error(ValueError):
    pass


class MoStT5(nn.Module):
    """Run text tasks through T5 unchanged and molecular tasks through one adapter."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        fp_bits: int = 4096,
        atom_embedding_dim: int = 768,
        geometry_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        embedding = backbone.get_input_embeddings()
        weight = getattr(embedding, "weight", None)
        if not isinstance(weight, Tensor) or weight.ndim != 2:
            raise MoStT5Error("backbone must expose a rank-two input embedding table")
        configured_width = getattr(getattr(backbone, "config", None), "d_model", None)
        if configured_width is not None and int(configured_width) != int(weight.shape[1]):
            raise MoStT5Error("backbone config and embedding width disagree")
        self.geometry = GeometryAdapter(
            int(weight.shape[1]),
            fp_bits=fp_bits,
            atom_embedding_dim=atom_embedding_dim,
            geometry_fraction=geometry_fraction,
        )

    @property
    def config(self) -> Any:
        return self.backbone.config

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    @staticmethod
    def _take_geometry_inputs(kwargs: dict[str, Any]) -> dict[str, Tensor]:
        return {
            name: kwargs.pop(name)
            for name in tuple(kwargs)
            if name in GEOMETRY_INPUT_NAMES
        }

    @staticmethod
    def _resolve_geometry_mode(
        geometry_mode: GeometryMode | None,
        geometry_inputs: dict[str, Tensor],
    ) -> GeometryMode | None:
        """Select geometry from the payload while retaining explicit ablations."""

        if not geometry_inputs:
            if geometry_mode not in {None, "none"}:
                raise MoStT5Error("geometry mode requires molecular structure inputs")
            return None
        missing = sorted(REQUIRED_GEOMETRY_INPUT_NAMES - geometry_inputs.keys())
        if missing:
            raise MoStT5Error(
                "partial molecular input is missing: " + ", ".join(missing)
            )
        return "full" if geometry_mode is None else geometry_mode

    def encode_molecule(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        geometry_mode: GeometryMode = "full",
        **geometry_inputs: Tensor,
    ) -> GeometryEncoding:
        token_embeddings = self.get_input_embeddings()(input_ids)
        return self.geometry(
            token_embeddings,
            attention_mask=attention_mask,
            geometry_mode=geometry_mode,
            **geometry_inputs,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        labels: Tensor | None = None,
        geometry_mode: GeometryMode | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("return_dict", None)
        geometry_inputs = self._take_geometry_inputs(kwargs)
        resolved_mode = self._resolve_geometry_mode(geometry_mode, geometry_inputs)
        if resolved_mode is None:
            return self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
                return_dict=True,
                **kwargs,
            )
        encoding = self.encode_molecule(
            input_ids,
            attention_mask,
            geometry_mode=resolved_mode,
            **geometry_inputs,
        )
        return self.backbone(
            inputs_embeds=encoding.fused_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )

    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        geometry_mode: GeometryMode | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Generate text from plain T5 tokens or geometry-fused molecular tokens."""

        geometry_inputs = self._take_geometry_inputs(kwargs)
        resolved_mode = self._resolve_geometry_mode(geometry_mode, geometry_inputs)
        if resolved_mode is None:
            return self.backbone.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        encoding = self.encode_molecule(
            input_ids,
            attention_mask,
            geometry_mode=resolved_mode,
            **geometry_inputs,
        )
        return self.backbone.generate(
            inputs_embeds=encoding.fused_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
