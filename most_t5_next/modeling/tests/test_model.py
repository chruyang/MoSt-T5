from __future__ import annotations

from dataclasses import dataclass
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from most_t5_next.modeling import GeometryAdapter, MoStT5
from most_t5_next.modeling.geometry import GeometryInputError
from most_t5_next.modeling.model import MoStT5Error


@dataclass
class _Output:
    loss: torch.Tensor | None
    logits: torch.Tensor
    encoder_last_hidden_state: torch.Tensor


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Embedding(32, 8)
        self.lm_head = nn.Linear(8, 32, bias=False)
        self.config = type("Config", (), {"d_model": 8})()

    def get_input_embeddings(self) -> nn.Module:
        return self.shared

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        attention_mask,
        labels=None,
        **_kwargs,
    ) -> _Output:
        hidden = self.shared(input_ids) if inputs_embeds is None else inputs_embeds
        hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.flatten(0, 1), labels.flatten(), ignore_index=-100
            )
        return _Output(loss, logits, hidden)

    def generate(self, *, input_ids=None, inputs_embeds=None, attention_mask, **_kwargs):
        hidden = self.shared(input_ids) if inputs_embeds is None else inputs_embeds
        scores = self.lm_head(hidden * attention_mask.unsqueeze(-1))
        return scores.argmax(dim=-1)


def _geometry_inputs() -> dict[str, torch.Tensor]:
    return {
        "e3fp_ids": torch.tensor([[[0, 1, 2, -1], [3, 4, 5, 6]]]),
        "atom_mask": torch.tensor([[True, True]]),
        "atom_to_fragment": torch.tensor([[0, 0]]),
        "fragment_mask": torch.tensor([[True]]),
        "fragment_to_carrier": torch.tensor([[0]]),
        "identity_span_bounds": torch.tensor([[[0, 2]]]),
        "endpoint_mask": torch.tensor([[True]]),
        "endpoint_to_atom": torch.tensor([[1]]),
        "endpoint_to_token": torch.tensor([[2]]),
        "endpoint_to_fragment": torch.tensor([[0]]),
        "endpoint_is_explicit": torch.tensor([[True]]),
        "token_is_connector_endpoint": torch.tensor(
            [[False, False, True, False]]
        ),
        "atom_is_attachment": torch.tensor([[False, True]]),
    }


def _mixed_geometry_inputs() -> dict[str, torch.Tensor]:
    """One molecular row followed by one text-only padded structural row."""

    return {
        "e3fp_ids": torch.tensor(
            [
                [[0, 1, 2, -1], [3, 4, 5, 6]],
                [[-1, -1, -1, -1], [-1, -1, -1, -1]],
            ]
        ),
        "atom_mask": torch.tensor([[True, True], [False, False]]),
        "atom_to_fragment": torch.tensor([[0, 0], [-1, -1]]),
        "fragment_mask": torch.tensor([[True], [False]]),
        "fragment_to_carrier": torch.tensor([[0], [-1]]),
        "identity_span_bounds": torch.tensor([[[0, 2]], [[-1, -1]]]),
        "endpoint_mask": torch.tensor([[True], [False]]),
        "endpoint_to_atom": torch.tensor([[1], [-1]]),
        "endpoint_to_token": torch.tensor([[2], [-1]]),
        "endpoint_to_fragment": torch.tensor([[0], [-1]]),
        "endpoint_is_explicit": torch.tensor([[True], [False]]),
        "token_is_connector_endpoint": torch.tensor(
            [[False, False, True, False], [False, False, False, False]]
        ),
        "atom_is_attachment": torch.tensor([[False, True], [False, False]]),
    }


def _mixed_fallback_geometry_inputs() -> dict[str, torch.Tensor]:
    """One fragmented row followed by one whole-molecule fallback row."""

    inputs = _mixed_geometry_inputs()
    inputs["e3fp_ids"][1] = torch.tensor(
        [[7, 8, 9, -1], [10, 11, 12, 13]]
    )
    inputs["atom_mask"][1] = True
    return inputs


class GeometryAdapterTest(unittest.TestCase):
    def test_missing_geometry_and_explicit_none_are_exact_identity_paths(self) -> None:
        torch.manual_seed(7)
        adapter = GeometryAdapter(8, fp_bits=16, atom_embedding_dim=8)
        tokens = torch.randn(1, 4, 8)
        common = _geometry_inputs()
        missing = dict(common)
        missing["e3fp_ids"] = torch.full_like(common["e3fp_ids"], -1)
        output = adapter(
            tokens,
            attention_mask=torch.ones((1, 4), dtype=torch.bool),
            **missing,
        )
        torch.testing.assert_close(output.fused_embeddings, tokens, rtol=0, atol=0)
        disabled = adapter(
            tokens,
            attention_mask=torch.ones((1, 4), dtype=torch.bool),
            geometry_mode="none",
            **common,
        )
        torch.testing.assert_close(disabled.fused_embeddings, tokens, rtol=0, atol=0)

    def test_carrier_and_endpoint_share_one_normalized_token_update(self) -> None:
        torch.manual_seed(8)
        adapter = GeometryAdapter(8, fp_bits=16, atom_embedding_dim=8)
        tokens = torch.randn(1, 4, 8, requires_grad=True)
        common = _geometry_inputs()
        common["endpoint_is_explicit"] = torch.tensor([[False]])
        common["endpoint_to_token"] = torch.tensor([[0]])
        common["token_is_connector_endpoint"] = torch.zeros((1, 4), dtype=torch.bool)
        output = adapter(
            tokens,
            attention_mask=torch.ones((1, 4), dtype=torch.bool),
            **common,
        )
        changed = output.fused_embeddings.ne(tokens).any(dim=-1)
        self.assertEqual(changed.tolist(), [[True, False, False, False]])
        probe = torch.randn_like(output.fused_embeddings)
        (output.fused_embeddings * probe).sum().backward()
        self.assertGreater(float(adapter.carrier_projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(adapter.endpoint_projection.weight.grad.abs().sum()), 0)


class MoStT5Test(unittest.TestCase):
    def test_one_wrapper_supports_plain_and_molecular_tasks(self) -> None:
        torch.manual_seed(9)
        model = MoStT5(
            _Backbone(), fp_bits=16, atom_embedding_dim=8, geometry_fraction=0.5
        )
        self.assertEqual(model.geometry.identity_position_capacity, 512)
        self.assertEqual(model.geometry.identity_position_score.num_embeddings, 512)
        input_ids = torch.tensor([[2, 3, 4, 5]])
        attention_mask = torch.ones((1, 4), dtype=torch.long)
        labels = torch.tensor([[3, 4, 5, 6]])
        plain = model(input_ids, attention_mask, labels=labels)
        molecular_off = model(
            input_ids,
            attention_mask,
            labels=labels,
            geometry_mode="none",
            **_geometry_inputs(),
        )
        torch.testing.assert_close(plain.logits, molecular_off.logits, rtol=0, atol=0)
        molecular_on = model(
            input_ids,
            attention_mask,
            labels=labels,
            geometry_mode="full",
            **_geometry_inputs(),
        )
        self.assertFalse(torch.equal(plain.logits, molecular_on.logits))
        self.assertEqual(molecular_on.loss.ndim, 0)

    def test_generation_uses_the_same_plain_and_molecular_routes(self) -> None:
        torch.manual_seed(10)
        model = MoStT5(_Backbone(), fp_bits=16, atom_embedding_dim=8)
        input_ids = torch.tensor([[2, 3, 4, 5]])
        attention_mask = torch.ones((1, 4), dtype=torch.long)
        plain = model.generate(input_ids, attention_mask)
        molecular_off = model.generate(
            input_ids,
            attention_mask,
            geometry_mode="none",
            **_geometry_inputs(),
        )
        torch.testing.assert_close(plain, molecular_off, rtol=0, atol=0)
        molecular_on = model.generate(
            input_ids,
            attention_mask,
            geometry_mode="full",
            **_geometry_inputs(),
        )
        self.assertEqual(molecular_on.shape, plain.shape)

    def test_payload_selects_geometry_without_a_task_name(self) -> None:
        torch.manual_seed(11)
        model = MoStT5(_Backbone(), fp_bits=16, atom_embedding_dim=8)
        input_ids = torch.tensor([[2, 3, 4, 5]])
        attention_mask = torch.ones((1, 4), dtype=torch.long)
        labels = torch.tensor([[3, 4, 5, 6]])
        plain = model(input_ids, attention_mask, labels=labels)
        automatic = model(
            input_ids,
            attention_mask,
            labels=labels,
            **_geometry_inputs(),
        )
        self.assertFalse(torch.equal(plain.logits, automatic.logits))

    def test_mixed_batch_does_not_inject_geometry_into_text_only_rows(self) -> None:
        torch.manual_seed(12)
        model = MoStT5(_Backbone(), fp_bits=16, atom_embedding_dim=8)
        input_ids = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
        attention_mask = torch.ones((2, 4), dtype=torch.long)
        labels = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
        lexical = model(input_ids, attention_mask, labels=labels)
        mixed = model(
            input_ids,
            attention_mask,
            labels=labels,
            **_mixed_geometry_inputs(),
        )
        self.assertFalse(torch.equal(lexical.logits[0], mixed.logits[0]))
        torch.testing.assert_close(lexical.logits[1], mixed.logits[1], rtol=0, atol=0)

    def test_mixed_batch_accepts_unowned_whole_molecule_fallback_atoms(self) -> None:
        torch.manual_seed(14)
        model = MoStT5(_Backbone(), fp_bits=16, atom_embedding_dim=8)
        input_ids = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
        attention_mask = torch.ones((2, 4), dtype=torch.long)
        labels = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
        lexical = model(input_ids, attention_mask, labels=labels)
        mixed = model(
            input_ids,
            attention_mask,
            labels=labels,
            **_mixed_fallback_geometry_inputs(),
        )
        self.assertFalse(torch.equal(lexical.logits[0], mixed.logits[0]))
        torch.testing.assert_close(lexical.logits[1], mixed.logits[1], rtol=0, atol=0)

    def test_whole_molecule_fallback_rejects_fragment_ownership(self) -> None:
        adapter = GeometryAdapter(8, fp_bits=16, atom_embedding_dim=8)
        inputs = _mixed_fallback_geometry_inputs()
        inputs["atom_to_fragment"][1, 0] = 0
        with self.assertRaisesRegex(
            GeometryInputError, "whole-molecule fallback atoms must be unowned"
        ):
            adapter(
                torch.randn(2, 4, 8),
                attention_mask=torch.ones((2, 4), dtype=torch.bool),
                **inputs,
            )

    def test_all_minus_one_payload_has_zero_geometry_gradient(self) -> None:
        torch.manual_seed(13)
        model = MoStT5(_Backbone(), fp_bits=16, atom_embedding_dim=8)
        inputs = _geometry_inputs()
        inputs["e3fp_ids"] = torch.full_like(inputs["e3fp_ids"], -1)
        output = model(
            torch.tensor([[2, 3, 4, 5]]),
            torch.ones((1, 4), dtype=torch.long),
            labels=torch.tensor([[3, 4, 5, 6]]),
            **inputs,
        )
        output.loss.backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.geometry.parameters()
            if parameter.grad is not None
        )
        self.assertEqual(gradient, 0.0)

    def test_partial_geometry_payload_fails_closed(self) -> None:
        model = MoStT5(_Backbone(), fp_bits=16, atom_embedding_dim=8)
        with self.assertRaisesRegex(MoStT5Error, "partial molecular input"):
            model(
                torch.tensor([[2, 3]]),
                torch.ones((1, 2), dtype=torch.long),
                e3fp_ids=torch.full((1, 1, 4), -1, dtype=torch.long),
            )


if __name__ == "__main__":
    unittest.main()
