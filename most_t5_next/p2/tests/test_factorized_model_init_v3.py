from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from most_t5_next.p2.factorized_model_init_v3 import (
    FactorizedModelInitV3Error,
    factorized_initialization_contract_v3,
    load_deterministic_factorized_model_v3,
)


class _TinyT5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=20, d_model=8)
        self.shared = nn.Embedding(20, 8)
        self.encoder = nn.Identity()

    def get_input_embeddings(self):
        return self.shared


class FactorizedModelInitV3Test(unittest.TestCase):
    def kwargs(self):
        return dict(
            adapter_seed=17,
            num_e3fp_embeddings=16,
            state_level2_weight=0.25,
            state_embedding_dim=4,
            atom_memory_dim=6,
            max_identity_span_length=8,
            max_atoms_per_motif=4,
            geometry_fraction=0.5,
        )

    def test_contract_freezes_the_thin_v3_boundary(self) -> None:
        contract = factorized_initialization_contract_v3(**self.kwargs())
        self.assertEqual(contract["geometry_shells"], [1, 2])
        self.assertEqual(contract["text_tokens_added"], 0)
        self.assertFalse(contract["trainable_collapse_gate"])
        self.assertIn("carrier_and_endpoint", contract["geometry_injection"])

    def test_loader_is_deterministic_and_restores_external_rng(self) -> None:
        def loader(**_kwargs):
            return SimpleNamespace(model=_TinyT5())

        torch.manual_seed(91)
        before = torch.random.get_rng_state().clone()
        first = load_deterministic_factorized_model_v3(
            base_model_snapshot="base",
            base_tokenizer_snapshot="tokenizer",
            union_tokenizer_dir="union",
            union_init_dir="init",
            union_geometry_fusion_seed=3,
            union_loader=loader,
            **self.kwargs(),
        )
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))
        second = load_deterministic_factorized_model_v3(
            base_model_snapshot="base",
            base_tokenizer_snapshot="tokenizer",
            union_tokenizer_dir="union",
            union_init_dir="init",
            union_geometry_fusion_seed=3,
            union_loader=loader,
            **self.kwargs(),
        )
        for left, right in zip(first.adapter.parameters(), second.adapter.parameters()):
            self.assertTrue(torch.equal(left, right))
            self.assertNotEqual(left.data_ptr(), right.data_ptr())

    def test_invalid_geometry_fraction_is_rejected(self) -> None:
        values = self.kwargs()
        values["geometry_fraction"] = 0.0
        with self.assertRaisesRegex(FactorizedModelInitV3Error, "geometry_fraction"):
            factorized_initialization_contract_v3(**values)


if __name__ == "__main__":
    unittest.main()
