from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from torch.nn import functional as F

    from most_t5_next.p1.four_grid_t5_wrapper import (
        FourGridT5Wrapper,
        FourGridT5WrapperError,
    )
    from most_t5_next.p1.shared_geometry_fusion import SharedE3FPCarrierFusion


@dataclass
class DummyConfig:
    vocab_size: int
    d_model: int
    model_type: str = "t5"


@dataclass
class DummyT5Output:
    loss: object
    logits: object
    call_mode: str
    forwarded_kwargs: dict[str, object]


if TORCH_AVAILABLE:

    class DummyT5ForConditionalGeneration(nn.Module):
        """Small deterministic T5-shaped module; no Transformers dependency."""

        def __init__(self, *, vocab_size: int = 13, hidden_size: int = 5) -> None:
            super().__init__()
            self.config = DummyConfig(vocab_size=vocab_size, d_model=hidden_size)
            self.shared = nn.Embedding(vocab_size, hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            self.lm_head.weight = self.shared.weight

        def get_input_embeddings(self):
            return self.shared

        def get_output_embeddings(self):
            return self.lm_head

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            inputs_embeds=None,
            **kwargs,
        ):
            if (input_ids is None) == (inputs_embeds is None):
                raise AssertionError("dummy T5 needs exactly one input representation")
            if inputs_embeds is None:
                hidden = self.shared(input_ids)
                call_mode = "input_ids"
            else:
                hidden = inputs_embeds
                call_mode = "inputs_embeds"
            logits = self.lm_head(hidden)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
            return DummyT5Output(
                loss=loss,
                logits=logits,
                call_mode=call_mode,
                forwarded_kwargs=dict(kwargs),
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for T5 wrapper tests")
class FourGridT5WrapperTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        base = DummyT5ForConditionalGeneration()
        self.models = {
            condition_id: FourGridT5Wrapper(
                copy.deepcopy(base),
                condition_id=condition_id,
                num_e3fp_embeddings=8,
            )
            for condition_id in ("A0", "A1", "M0", "M1")
        }
        reference_state = self.models["A0"].state_dict()
        for model in self.models.values():
            model.load_state_dict(reference_state, strict=True)

        self.input_ids = torch.tensor([[2, 3, 4, 1]], dtype=torch.long)
        self.attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        self.labels = torch.tensor([[5, 6, 7, 1]], dtype=torch.long)
        self.geometry = {
            "e3fp_ids": torch.tensor(
                [[[1, 2, -1, -1], [3, -1, -1, -1]]],
                dtype=torch.long,
            ),
            "e3fp_atom_mask": torch.tensor([[True, True]], dtype=torch.bool),
            "e3fp_atom_to_token": torch.tensor([[1, 2]], dtype=torch.long),
        }

    def standard_inputs(self):
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
        }

    def test_all_cells_have_identical_state_schema_and_parameter_count(self):
        reference = self.models["A0"]
        reference_keys = tuple(reference.state_dict())
        reference_shapes = {
            name: tuple(value.shape)
            for name, value in reference.state_dict().items()
        }
        reference_count = sum(parameter.numel() for parameter in reference.parameters())

        for level_index in range(4):
            self.assertIn(
                f"geometry_fusion.level_embeddings.{level_index}.weight",
                reference_keys,
            )
        for model in self.models.values():
            self.assertEqual(tuple(model.state_dict()), reference_keys)
            self.assertEqual(
                {name: tuple(value.shape) for name, value in model.state_dict().items()},
                reference_shapes,
            )
            self.assertEqual(
                sum(parameter.numel() for parameter in model.parameters()),
                reference_count,
            )

    def test_a0_and_m0_are_the_same_standard_t5_ce_forward(self):
        a0 = self.models["A0"](**self.standard_inputs(), condition_id="A0")
        m0 = self.models["M0"](**self.standard_inputs(), condition_id=("M0",))

        self.assertIsInstance(a0, DummyT5Output)
        self.assertEqual(a0.call_mode, "input_ids")
        self.assertEqual(m0.call_mode, "input_ids")
        self.assertTrue(torch.equal(a0.logits, m0.logits))
        self.assertTrue(torch.equal(a0.loss, m0.loss))

        direct = self.models["A0"].t5(**self.standard_inputs())
        self.assertTrue(torch.equal(a0.logits, direct.logits))
        self.assertTrue(torch.equal(a0.loss, direct.loss))

    def test_a1_and_m1_use_the_same_shared_fusion_path(self):
        a1_model = self.models["A1"]
        m1_model = self.models["M1"]
        self.assertIsInstance(a1_model.geometry_fusion, SharedE3FPCarrierFusion)
        self.assertIsInstance(m1_model.geometry_fusion, SharedE3FPCarrierFusion)
        for level_index in range(4):
            self.assertTrue(
                torch.equal(
                    a1_model.geometry_fusion.level_embeddings[level_index].weight,
                    m1_model.geometry_fusion.level_embeddings[level_index].weight,
                )
            )

        a1 = a1_model(**self.standard_inputs(), **self.geometry)
        m1 = m1_model(**self.standard_inputs(), **self.geometry)
        self.assertEqual(a1.call_mode, "inputs_embeds")
        self.assertEqual(m1.call_mode, "inputs_embeds")
        self.assertTrue(torch.equal(a1.logits, m1.logits))
        self.assertTrue(torch.equal(a1.loss, m1.loss))

    def test_geometry_presence_and_fixed_condition_fail_closed(self):
        with self.assertRaisesRegex(FourGridT5WrapperError, "rejects geometry"):
            self.models["A0"](**self.standard_inputs(), **self.geometry)
        with self.assertRaisesRegex(FourGridT5WrapperError, "requires all geometry"):
            self.models["A1"](**self.standard_inputs())
        with self.assertRaisesRegex(FourGridT5WrapperError, "all-or-none"):
            self.models["A1"](
                **self.standard_inputs(),
                e3fp_ids=self.geometry["e3fp_ids"],
            )
        with self.assertRaisesRegex(FourGridT5WrapperError, "fixed wrapper"):
            self.models["A1"](
                **self.standard_inputs(),
                **self.geometry,
                condition_id=("A1", "M1"),
            )

    def test_ce_backward_reaches_t5_and_only_active_geometry_path(self):
        a1 = self.models["A1"]
        output = a1(**self.standard_inputs(), **self.geometry)
        output.loss.backward()
        self.assertIsNotNone(a1.t5.shared.weight.grad)
        self.assertGreater(float(a1.t5.shared.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(a1.geometry_fusion.level_embeddings[0].weight.grad)
        self.assertGreater(
            sum(
                float(embedding.weight.grad.abs().sum())
                for embedding in a1.geometry_fusion.level_embeddings
            ),
            0.0,
        )

        a0 = self.models["A0"]
        a0(**self.standard_inputs()).loss.backward()
        self.assertIsNotNone(a0.t5.shared.weight.grad)
        self.assertTrue(
            all(
                embedding.weight.grad is None
                for embedding in a0.geometry_fusion.level_embeddings
            )
        )

    def test_t5_kwargs_and_output_object_are_forwarded_unchanged(self):
        output = self.models["M1"](
            **self.standard_inputs(),
            **self.geometry,
            output_hidden_states=True,
        )
        self.assertIsInstance(output, DummyT5Output)
        self.assertEqual(output.forwarded_kwargs, {"output_hidden_states": True})

    def test_invalid_constructor_contracts_are_rejected(self):
        with self.assertRaisesRegex(FourGridT5WrapperError, "A0, A1, M0, M1"):
            FourGridT5Wrapper(
                DummyT5ForConditionalGeneration(),
                condition_id="C1",
                num_e3fp_embeddings=8,
            )

        invalid_vocab = DummyT5ForConditionalGeneration()
        invalid_vocab.config.vocab_size += 1
        with self.assertRaisesRegex(FourGridT5WrapperError, "union vocabulary"):
            FourGridT5Wrapper(
                invalid_vocab,
                condition_id="A0",
                num_e3fp_embeddings=8,
            )


if __name__ == "__main__":
    unittest.main()
