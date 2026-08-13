"""Reference-controlled E3FP atom embeddings for 3D-MotifT5.

The reference arm reproduces the atom-state part of 3D-MolT5: one shared
``(fp_bits + 1) x d_model`` table, external ``-1`` shifted to padding row zero,
and a fixed mean over four shell slots.  Two project arms change exactly the
parameter-tying pattern: one table for L0 plus one shared table for L1--L3, or
one table per ordered shell level.  Every table can be initialized from the
same reference weight, making all three encoders numerically identical before
optimization.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


E3FP_ATOM_EMBEDDING_ID = "most-t5-next/e3fp-atom-embedding/v1"
REFERENCE_SHARED_FIXED4 = "reference_shared_fixed4"
L0_STATE_FIXED4 = "l0_state_fixed4"
LEVEL_SPECIFIC_FIXED4 = "level_specific_fixed4"


class E3FPAtomEmbeddingError(ValueError):
    pass


class E3FPAtomEmbeddingV1(nn.Module):
    """Encode ``[B,A,4]`` folded E3FP IDs into one ``[B,A,D]`` atom state."""

    def __init__(
        self,
        *,
        fp_bits: int = 4096,
        embedding_dim: int = 768,
        variant: Literal[
            "reference_shared_fixed4", "l0_state_fixed4", "level_specific_fixed4"
        ] = REFERENCE_SHARED_FIXED4,
    ) -> None:
        super().__init__()
        if isinstance(fp_bits, bool) or not isinstance(fp_bits, int) or fp_bits <= 1:
            raise E3FPAtomEmbeddingError("fp_bits must be an integer greater than one")
        if (
            isinstance(embedding_dim, bool)
            or not isinstance(embedding_dim, int)
            or embedding_dim <= 0
        ):
            raise E3FPAtomEmbeddingError("embedding_dim must be positive")
        if variant not in {
            REFERENCE_SHARED_FIXED4,
            L0_STATE_FIXED4,
            LEVEL_SPECIFIC_FIXED4,
        }:
            raise E3FPAtomEmbeddingError("unknown E3FP embedding variant")
        self.fp_bits = fp_bits
        self.embedding_dim = embedding_dim
        self.variant = variant
        # Match 3D-MolT5 exactly: external -1 is shifted to row 0 and real ID
        # k is stored in row k+1.  There is no learned state-mask row here.
        self.padding_row = 0
        self.real_row_offset = 1
        if variant == REFERENCE_SHARED_FIXED4:
            self.shared_embedding = nn.Embedding(
                fp_bits + 1, embedding_dim, padding_idx=self.padding_row
            )
        elif variant == L0_STATE_FIXED4:
            self.l0_embedding = nn.Embedding(
                fp_bits + 1, embedding_dim, padding_idx=self.padding_row
            )
            self.state_embedding = nn.Embedding(
                fp_bits + 1, embedding_dim, padding_idx=self.padding_row
            )
        else:
            self.level_embeddings = nn.ModuleList(
                nn.Embedding(fp_bits + 1, embedding_dim, padding_idx=self.padding_row)
                for _ in range(4)
            )
        self._zero_padding_rows()

    @torch.no_grad()
    def _zero_padding_rows(self) -> None:
        if self.variant == REFERENCE_SHARED_FIXED4:
            self.shared_embedding.weight[self.padding_row].zero_()
        elif self.variant == L0_STATE_FIXED4:
            self.l0_embedding.weight[self.padding_row].zero_()
            self.state_embedding.weight[self.padding_row].zero_()
        else:
            for table in self.level_embeddings:
                table.weight[self.padding_row].zero_()

    @torch.no_grad()
    def initialize_tied_tables_from_shared(self, shared_weight: Tensor) -> None:
        """Copy one reference table into every table of a candidate arm."""

        if self.variant == REFERENCE_SHARED_FIXED4:
            raise E3FPAtomEmbeddingError(
                "shared-table initialization requires a multi-table candidate"
            )
        expected = (self.fp_bits + 1, self.embedding_dim)
        if shared_weight.shape != expected or not shared_weight.is_floating_point():
            raise E3FPAtomEmbeddingError(
                f"shared_weight must be floating point with shape {expected}"
            )
        tables = (
            (self.l0_embedding, self.state_embedding)
            if self.variant == L0_STATE_FIXED4
            else tuple(self.level_embeddings)
        )
        for table in tables:
            table.weight.copy_(shared_weight.to(table.weight))
        self._zero_padding_rows()

    @torch.no_grad()
    def initialize_level_tables_from_shared(self, shared_weight: Tensor) -> None:
        """Backward-compatible alias for the four-table arm only."""

        if self.variant != LEVEL_SPECIFIC_FIXED4:
            raise E3FPAtomEmbeddingError(
                "level-table initialization requires the level-specific variant"
            )
        self.initialize_tied_tables_from_shared(shared_weight)

    def forward(self, e3fp_ids: Tensor, atom_mask: Tensor) -> Tensor:
        if e3fp_ids.ndim != 3 or e3fp_ids.shape[-1] != 4:
            raise E3FPAtomEmbeddingError("e3fp_ids must be [B,A,4]")
        if e3fp_ids.dtype == torch.bool or e3fp_ids.is_floating_point():
            raise E3FPAtomEmbeddingError("e3fp_ids must use an integer dtype")
        if atom_mask.shape != e3fp_ids.shape[:2] or atom_mask.dtype != torch.bool:
            raise E3FPAtomEmbeddingError("atom_mask must be bool [B,A]")
        if atom_mask.device != e3fp_ids.device:
            raise E3FPAtomEmbeddingError("E3FP IDs and atom mask must share a device")
        if bool(((e3fp_ids < -1) | (e3fp_ids >= self.fp_bits)).any()):
            raise E3FPAtomEmbeddingError("E3FP ID is outside -1 or the folded domain")
        if bool(((~atom_mask).unsqueeze(-1) & (e3fp_ids != -1)).any()):
            raise E3FPAtomEmbeddingError("padded atoms must contain four -1 slots")

        shifted = (e3fp_ids + self.real_row_offset).to(torch.long)
        if self.variant == REFERENCE_SHARED_FIXED4:
            shell_hidden = self.shared_embedding(shifted)
        elif self.variant == L0_STATE_FIXED4:
            shell_hidden = torch.cat(
                [
                    self.l0_embedding(shifted[..., :1]),
                    self.state_embedding(shifted[..., 1:]),
                ],
                dim=2,
            )
        else:
            shell_hidden = torch.stack(
                [
                    table(shifted[..., level])
                    for level, table in enumerate(self.level_embeddings)
                ],
                dim=2,
            )
        # Keep the reference denominator fixed at four.  A missing high shell
        # contributes the exact zero padding vector instead of changing scale.
        atom_hidden = shell_hidden.mean(dim=2)
        return atom_hidden * atom_mask.unsqueeze(-1).to(atom_hidden.dtype)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


__all__ = [
    "E3FP_ATOM_EMBEDDING_ID",
    "E3FPAtomEmbeddingError",
    "E3FPAtomEmbeddingV1",
    "L0_STATE_FIXED4",
    "LEVEL_SPECIFIC_FIXED4",
    "REFERENCE_SHARED_FIXED4",
]
