"""E3FP shell embeddings used by MoSt-T5."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class E3FPEmbeddingError(ValueError):
    pass


class E3FPShellEmbedding(nn.Module):
    """Embed L0 identity and shared L1--L3 state with a fixed four-slot mean."""

    def __init__(self, fp_bits: int = 4096, embedding_dim: int = 768) -> None:
        super().__init__()
        if fp_bits <= 1 or embedding_dim <= 0:
            raise E3FPEmbeddingError("fp_bits and embedding_dim must be positive")
        self.fp_bits = int(fp_bits)
        self.embedding_dim = int(embedding_dim)
        self.identity = nn.Embedding(fp_bits + 1, embedding_dim, padding_idx=0)
        self.state = nn.Embedding(fp_bits + 1, embedding_dim, padding_idx=0)
        self.reset_padding()

    @torch.no_grad()
    def reset_padding(self) -> None:
        self.identity.weight[0].zero_()
        self.state.weight[0].zero_()

    def forward(self, e3fp_ids: Tensor, atom_mask: Tensor) -> Tensor:
        if e3fp_ids.ndim != 3 or e3fp_ids.shape[-1] != 4:
            raise E3FPEmbeddingError("e3fp_ids must have shape [batch, atoms, 4]")
        if atom_mask.shape != e3fp_ids.shape[:2] or atom_mask.dtype != torch.bool:
            raise E3FPEmbeddingError("atom_mask must be boolean [batch, atoms]")
        if e3fp_ids.is_floating_point() or e3fp_ids.dtype == torch.bool:
            raise E3FPEmbeddingError("e3fp_ids must use an integer dtype")
        if bool(((e3fp_ids < -1) | (e3fp_ids >= self.fp_bits)).any()):
            raise E3FPEmbeddingError("E3FP ID lies outside the folded domain")
        if bool(((~atom_mask).unsqueeze(-1) & e3fp_ids.ne(-1)).any()):
            raise E3FPEmbeddingError("padded atoms must contain four -1 values")

        rows = (e3fp_ids + 1).to(torch.long)
        shells = torch.cat(
            (self.identity(rows[..., :1]), self.state(rows[..., 1:])), dim=2
        )
        return shells.mean(dim=2) * atom_mask.unsqueeze(-1).to(shells.dtype)
