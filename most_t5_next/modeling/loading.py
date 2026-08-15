"""Load the frozen vocabulary-expanded T5 checkpoint and geometry adapter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from transformers import T5ForConditionalGeneration

from .model import MoStT5


FROZEN_VOCAB_SIZE = 53_368


def _set_dropout(model: torch.nn.Module, dropout_rate: float) -> None:
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must lie in [0, 1)")
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "dropout_rate"):
            config.dropout_rate = float(dropout_rate)
        if isinstance(module, torch.nn.Dropout):
            module.p = float(dropout_rate)


def load_pretrained_model(
    checkpoint: str | Path,
    *,
    adapter_seed: int = 42,
    expected_vocab_size: int = FROZEN_VOCAB_SIZE,
    fp_bits: int = 4096,
    atom_embedding_dim: int = 768,
    geometry_fraction: float = 0.5,
    dropout_rate: float = 0.0,
) -> MoStT5:
    """Load initialized vocabulary rows before constructing the 3D adapter."""

    with torch.random.fork_rng(devices=[]):
        backbone = T5ForConditionalGeneration.from_pretrained(
            str(checkpoint), local_files_only=True
        )
        if backbone.get_input_embeddings().num_embeddings != expected_vocab_size:
            raise ValueError("checkpoint vocabulary does not match the frozen tokenizer")
        torch.manual_seed(adapter_seed)
        model = MoStT5(
            backbone,
            fp_bits=fp_bits,
            atom_embedding_dim=atom_embedding_dim,
            geometry_fraction=geometry_fraction,
        )
    _set_dropout(model, dropout_rate)
    return model


def load_model_from_config(
    checkpoint: str | Path,
    config: Mapping[str, Any],
) -> MoStT5:
    """Construct the model from the validated public configuration."""

    model_config = config["model"]
    return load_pretrained_model(
        checkpoint,
        adapter_seed=int(config["seed"]),
        expected_vocab_size=int(model_config["vocabulary_size"]),
        fp_bits=int(model_config["e3fp_bits"]),
        atom_embedding_dim=int(model_config["e3fp_embedding_dim"]),
        geometry_fraction=float(model_config["geometry_fraction"]),
        dropout_rate=float(model_config["dropout_rate"]),
    )


__all__ = ["FROZEN_VOCAB_SIZE", "load_model_from_config", "load_pretrained_model"]
