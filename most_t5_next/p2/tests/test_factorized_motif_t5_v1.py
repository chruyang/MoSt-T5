from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from torch.nn import functional as F

    from most_t5_next.p2.factorized_motif_t5_v1 import (
        FactorizedMotifT5Error,
        FactorizedMotifT5V1,
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


if TORCH_AVAILABLE:

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

        def forward(
            self,
            *,
            inputs_embeds,
            attention_mask,
            labels,
            return_dict=True,
            **_kwargs,
        ):
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
            return _T5Output(
                loss=loss,
                logits=logits,
                encoder_last_hidden_state=hidden,
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class FactorizedMotifT5Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model = FactorizedMotifT5V1(
            _TinyT5(),
            num_e3fp_embeddings=16,
            state_embedding_dim=4,
            atom_memory_dim=8,
            max_identity_span_length=8,
        )
        self.common = {
            "input_ids": torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "e3fp_mask_token_id": 17,
            "e3fp_input_ids": torch.tensor(
                [[[7, 17, 17, 17], [8, 9, 17, 17], [4, 5, 6, 7]]],
                dtype=torch.long,
            ),
            "atom_mask": torch.tensor([[True, True, True]]),
            "atom_to_motif": torch.tensor([[0, 0, 1]], dtype=torch.long),
            "motif_mask": torch.tensor([[True, True]]),
            "motif_to_carrier": torch.tensor([[0, 3]], dtype=torch.long),
            "identity_span_bounds": torch.tensor([[[0, 2], [3, 5]]]),
            "atom_is_attachment": torch.tensor([[False, True, False]]),
        }
        target_ids = torch.tensor(
            [[[7, 8, 9, 10], [8, 9, 10, 11], [4, 5, 6, 7]]],
            dtype=torch.long,
        )
        target_mask = torch.zeros_like(target_ids, dtype=torch.bool)
        target_mask[0, 0, 1] = True
        target_mask[0, 0, 2] = True
        target_mask[0, 1, 2] = True
        corruption = torch.zeros_like(target_mask)
        corruption[0, 0, 1:4] = True
        corruption[0, 1, 2:4] = True
        self.state = {
            "state_target_ids": target_ids,
            "state_target_mask": target_mask,
            "state_corruption_mask": corruption,
        }

    def test_state_objective_uses_post_t5_context_and_backpropagates(self):
        output = self.model(
            **self.common,
            **self.state,
            objective_mode="state",
        )
        self.assertIsNone(output.grammar_loss)
        self.assertIsNotNone(output.state_loss)
        self.assertEqual(tuple(output.state_logits.shape), (1, 3, 2, 16))
        self.assertEqual(set(output.state_level_losses), {1, 2})
        output.loss.backward()
        self.assertGreater(
            float(self.model.t5.encoder.projection.weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(self.model.adapter.state_embedding.weight.grad.abs().sum()),
            0.0,
        )

    def test_grammar_is_one_plain_decoder_ce_without_state_loss(self):
        labels = torch.tensor([[3, 4, 5, 6, 7]], dtype=torch.long)
        output = self.model(
            **self.common,
            objective_mode="grammar",
            labels=labels,
        )
        self.assertIsNotNone(output.grammar_loss)
        self.assertIsNone(output.state_loss)
        self.assertIsNone(output.state_logits)
        self.assertTrue(torch.equal(output.loss, output.t5_output.loss))

    def test_cross_view_is_explicit_not_an_implicit_second_loss(self):
        labels = torch.tensor([[3, 4, 5, 6, 7]], dtype=torch.long)
        output = self.model(
            **self.common,
            objective_mode="cross_view",
            labels=labels,
        )
        self.assertEqual(output.objective_mode, "cross_view")
        self.assertIsNone(output.state_loss)

    def test_scored_slot_must_be_corrupted(self):
        broken = dict(self.state)
        broken["state_corruption_mask"] = torch.zeros_like(
            self.state["state_corruption_mask"]
        )
        with self.assertRaisesRegex(FactorizedMotifT5Error, "must be hidden"):
            self.model(
                **self.common,
                **broken,
                objective_mode="state",
            )

    def test_corruption_mask_exactly_matches_mask_tokens(self):
        broken_common = dict(self.common)
        broken_ids = self.common["e3fp_input_ids"].clone()
        broken_ids[0, 2, 1] = 17
        broken_common["e3fp_input_ids"] = broken_ids
        with self.assertRaisesRegex(FactorizedMotifT5Error, "exactly name"):
            self.model(
                **broken_common,
                **self.state,
                objective_mode="state",
            )

    def test_nested_shell_state_cannot_leak_after_an_earlier_mask(self):
        broken_common = dict(self.common)
        target_ids = self.state["state_target_ids"].clone()
        broken_ids = target_ids.clone()
        broken_ids[0, 0, 1] = 17
        broken_common["e3fp_input_ids"] = broken_ids
        target_mask = torch.zeros_like(target_ids, dtype=torch.bool)
        target_mask[0, 0, 1] = True
        corruption = torch.zeros_like(target_mask)
        corruption[0, 0, 1] = True
        with self.assertRaisesRegex(FactorizedMotifT5Error, "suffix-closed"):
            self.model(
                **broken_common,
                state_target_ids=target_ids,
                state_target_mask=target_mask,
                state_corruption_mask=corruption,
                objective_mode="state",
            )

    def test_state_and_grammar_targets_cannot_be_silently_mixed(self):
        labels = torch.tensor([[3, 4, 5, 6, 7]], dtype=torch.long)
        with self.assertRaisesRegex(FactorizedMotifT5Error, "silently add"):
            self.model(
                **self.common,
                **self.state,
                objective_mode="grammar",
                labels=labels,
            )


if __name__ == "__main__":
    unittest.main()
