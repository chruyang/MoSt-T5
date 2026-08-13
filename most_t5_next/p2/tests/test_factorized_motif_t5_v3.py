from __future__ import annotations

from dataclasses import dataclass
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from most_t5_next.p2.factorized_motif_t5_v3 import (
    FACTORISATION_ID,
    FactorizedMotifT5V3,
)


@dataclass
class _Config:
    vocab_size: int
    d_model: int


@dataclass
class _EncoderOutput:
    last_hidden_state: object


@dataclass
class _T5Output:
    loss: object
    logits: object
    encoder_last_hidden_state: object


class _Encoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, *, inputs_embeds, attention_mask, return_dict=True):
        hidden = self.projection(inputs_embeds)
        return _EncoderOutput(
            hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        )


class _TinyT5(nn.Module):
    def __init__(self, vocab_size: int = 19, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = _Config(vocab_size, hidden_size)
        self.shared = nn.Embedding(vocab_size, hidden_size)
        self.encoder = _Encoder(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.shared

    def forward(self, *, inputs_embeds, attention_mask, labels, return_dict=True, **_kwargs):
        hidden = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        logits = self.lm_head(hidden)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return _T5Output(loss, logits, hidden)


class FactorizedMotifT5V3Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(401)
        self.model = FactorizedMotifT5V3(
            _TinyT5(),
            num_e3fp_embeddings=16,
            state_embedding_dim=4,
            atom_memory_dim=8,
            max_identity_span_length=8,
            max_atoms_per_motif=4,
        )
        self.common = {
            "input_ids": torch.tensor([[2, 3, 4, 5, 6, 7]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "e3fp_mask_token_id": 17,
            "e3fp_input_ids": torch.tensor(
                [[[7, 8, 9, 10], [8, 9, 10, 11], [4, 5, 6, 7]]]
            ),
            "atom_mask": torch.tensor([[True, True, True]]),
            "atom_to_motif": torch.tensor([[0, 0, 1]]),
            "atom_local_positions": torch.tensor([[0, 1, 0]]),
            "motif_mask": torch.tensor([[True, True]]),
            "motif_to_carrier": torch.tensor([[0, 3]]),
            "identity_span_bounds": torch.tensor([[[0, 2], [3, 5]]]),
            "endpoint_token_to_atom": torch.tensor([[-1, -1, 1, -1, -1, 2]]),
            "atom_is_attachment": torch.tensor([[False, True, True]]),
        }

    def test_grammar_path_and_exact_zero_geometry_path(self) -> None:
        labels = torch.tensor([[3, 4, 5, 6, 7, 8]])
        aligned = self.model(
            **self.common,
            labels=labels,
            objective_mode="cross_view",
        )
        zero = self.model(
            **self.common,
            labels=labels,
            objective_mode="cross_view",
            state_memory_mode="zero",
        )
        self.assertIsNotNone(aligned.grammar_loss)
        self.assertIsNotNone(zero.grammar_loss)
        self.assertFalse(
            torch.equal(
                aligned.adapter_encoding.fused_embeddings,
                zero.adapter_encoding.fused_embeddings,
            )
        )
        baseline = self.model.get_input_embeddings()(self.common["input_ids"])
        self.assertTrue(
            torch.equal(zero.adapter_encoding.fused_embeddings, baseline)
        )

    def test_factorisation_and_parameter_topology_are_explicit(self) -> None:
        self.assertEqual(self.model.factorisation_id, FACTORISATION_ID)
        self.assertIn("carrier-endpoint", FACTORISATION_ID)
        self.assertFalse(
            any("gate" in name for name, _ in self.model.adapter.named_parameters())
        )


if __name__ == "__main__":
    unittest.main()
