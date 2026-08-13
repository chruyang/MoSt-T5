from __future__ import annotations

from dataclasses import dataclass
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from most_t5_next.p2.factorized_motif_t5_v2 import (
    FACTORISATION_ID,
    FactorizedMotifT5V2,
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
        hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        return _EncoderOutput(last_hidden_state=hidden)


class _TinyT5(nn.Module):
    def __init__(self, vocab_size: int = 19, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = _Config(vocab_size=vocab_size, d_model=hidden_size)
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


class FactorizedMotifT5V2Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(211)
        self.model = FactorizedMotifT5V2(
            _TinyT5(),
            num_e3fp_embeddings=16,
            state_embedding_dim=4,
            atom_memory_dim=8,
            max_identity_span_length=8,
            max_atoms_per_motif=4,
        )
        self.common = {
            "input_ids": torch.tensor([[2, 3, 4, 5, 6]]),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "e3fp_mask_token_id": 17,
            "e3fp_input_ids": torch.tensor(
                [[[7, 17, 17, 17], [8, 9, 17, 17], [4, 5, 6, 7]]]
            ),
            "atom_mask": torch.tensor([[True, True, True]]),
            "atom_to_motif": torch.tensor([[0, 0, 1]]),
            "atom_local_positions": torch.tensor([[0, 1, 0]]),
            "motif_mask": torch.tensor([[True, True]]),
            "motif_to_carrier": torch.tensor([[0, 3]]),
            "identity_span_bounds": torch.tensor([[[0, 2], [3, 5]]]),
            "atom_is_attachment": torch.tensor([[False, True, False]]),
        }
        targets = torch.tensor(
            [[[7, 8, 9, 10], [8, 9, 10, 11], [4, 5, 6, 7]]]
        )
        target_mask = torch.zeros_like(targets, dtype=torch.bool)
        target_mask[0, 0, 1:3] = True
        target_mask[0, 1, 2] = True
        corruption = torch.zeros_like(target_mask)
        corruption[0, 0, 1:4] = True
        corruption[0, 1, 2:4] = True
        self.state = {
            "state_target_ids": targets,
            "state_target_mask": target_mask,
            "state_corruption_mask": corruption,
        }

    def test_v2_state_and_grammar_paths_share_the_v1_batch_contract(self) -> None:
        state = self.model(**self.common, **self.state, objective_mode="state")
        self.assertEqual(tuple(state.state_logits.shape), (1, 3, 2, 16))
        self.assertIsNotNone(state.state_loss)
        state.loss.backward()
        self.assertGreater(
            float(self.model.adapter.geometry_output.gate_logits.grad.abs().sum()),
            0.0,
        )

        self.model.zero_grad(set_to_none=True)
        labels = torch.tensor([[3, 4, 5, 6, 7]])
        grammar = self.model(
            **self.common,
            labels=labels,
            objective_mode="grammar",
        )
        self.assertIsNotNone(grammar.grammar_loss)
        self.assertIsNone(grammar.state_loss)

    def test_factorisation_identity_is_new_and_explicit(self) -> None:
        self.assertEqual(self.model.factorisation_id, FACTORISATION_ID)
        self.assertIn("v2-carrier-only", FACTORISATION_ID)


if __name__ == "__main__":
    unittest.main()
