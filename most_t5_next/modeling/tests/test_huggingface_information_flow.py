from __future__ import annotations

import unittest

import torch
from transformers import T5Config, T5ForConditionalGeneration

from most_t5_next.modeling import MoStT5
from most_t5_next.training import forward_task


def _molecular_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[2, 3, 4, 5]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "labels": torch.tensor([[6, 7, 1]]),
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
        "token_is_connector_endpoint": torch.tensor([[False, False, True, False]]),
        "atom_is_attachment": torch.tensor([[False, True]]),
    }


def _text_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[2, 3, 4, 5]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "labels": torch.tensor([[6, 7, 1]]),
    }


def _tiny_model() -> MoStT5:
    config = T5Config(
        vocab_size=64,
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    return MoStT5(
        T5ForConditionalGeneration(config), fp_bits=16, atom_embedding_dim=32
    )


def _geometry_gradient(model: MoStT5) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.geometry.parameters()
        if parameter.grad is not None
    )


class HuggingFaceInformationFlowTest(unittest.TestCase):
    def test_six_tasks_reach_t5_and_only_molecular_3d_tasks_reach_adapter(self) -> None:
        torch.manual_seed(11)
        model = _tiny_model()
        molecular_tasks = {"M", "MG", "SYN", "CAP"}
        geometry_tasks = {"MG", "SYN", "CAP"}
        for task in ("M", "MG", "SYN", "TXT", "CAP", "T2M"):
            model.zero_grad(set_to_none=True)
            batch = _molecular_batch() if task in molecular_tasks else _text_batch()
            output = forward_task(model, task, batch)
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertGreater(
                float(model.backbone.shared.weight.grad.detach().abs().sum()), 0
            )
            if task in geometry_tasks:
                self.assertGreater(_geometry_gradient(model), 0, task)
            else:
                self.assertEqual(_geometry_gradient(model), 0, task)

    def test_geometry_changes_only_carrier_and_explicit_endpoint_positions(self) -> None:
        torch.manual_seed(12)
        model = _tiny_model()
        batch = _molecular_batch()
        lexical = model.get_input_embeddings()(batch["input_ids"])
        encoding = model.encode_molecule(
            batch["input_ids"],
            batch["attention_mask"],
            geometry_mode="full",
            **{
                key: value
                for key, value in batch.items()
                if key not in {"input_ids", "attention_mask", "labels"}
            },
        )
        changed = encoding.fused_embeddings.ne(lexical).any(dim=-1)
        self.assertEqual(changed.tolist(), [[True, False, True, False]])

    def test_plain_and_molecular_generation_use_the_stock_t5_decoder(self) -> None:
        torch.manual_seed(13)
        model = _tiny_model()
        text = _text_batch()
        plain = model.generate(
            text["input_ids"], text["attention_mask"], max_new_tokens=2
        )
        molecule = _molecular_batch()
        geometry = {
            key: value
            for key, value in molecule.items()
            if key not in {"input_ids", "attention_mask", "labels"}
        }
        molecular = model.generate(
            molecule["input_ids"],
            molecule["attention_mask"],
            geometry_mode="full",
            max_new_tokens=2,
            **geometry,
        )
        self.assertEqual(plain.shape, molecular.shape)
        self.assertEqual(plain.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
