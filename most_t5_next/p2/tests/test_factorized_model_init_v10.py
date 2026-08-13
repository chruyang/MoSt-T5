from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from most_t5_next.p2.e3fp_atom_embedding_v1 import (
    LEVEL_SPECIFIC_FIXED4,
    L0_STATE_FIXED4,
    REFERENCE_SHARED_FIXED4,
)
from most_t5_next.p2.factorized_model_init_v10 import (
    factorized_initialization_contract_v10,
    load_deterministic_factorized_model_v10,
)


class _TinyT5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=20, d_model=8)
        self.shared = nn.Embedding(20, 8)
        self.encoder = nn.Identity()

    def get_input_embeddings(self):
        return self.shared


class FactorizedModelInitV10Test(unittest.TestCase):
    def _kwargs(self, variant: str) -> dict[str, object]:
        return {
            "semantic_plan_sha256": "a" * 64,
            "adapter_seed": 17,
            "num_e3fp_embeddings": 16,
            "state_level2_weight": 0.25,
            "state_embedding_dim": 4,
            "atom_memory_dim": 8,
            "max_identity_span_length": 8,
            "max_atoms_per_motif": 4,
            "geometry_fraction": 0.5,
            "parameter_tying": variant,
        }

    def _load(self, variant: str):
        def loader(**_kwargs):
            return SimpleNamespace(model=_TinyT5())

        return load_deterministic_factorized_model_v10(
            base_model_snapshot="base",
            base_tokenizer_snapshot="tokenizer",
            anchored_tokenizer_dir="anchored",
            union_init_dir="init",
            union_geometry_fusion_seed=3,
            union_loader=loader,
            **self._kwargs(variant),
        )

    def test_contract_records_exact_parameter_inventory(self) -> None:
        expected = {
            REFERENCE_SHARED_FIXED4: (1, 17 * 8),
            L0_STATE_FIXED4: (2, 2 * 17 * 8),
            LEVEL_SPECIFIC_FIXED4: (4, 4 * 17 * 8),
        }
        for variant, (tables, parameters) in expected.items():
            contract = factorized_initialization_contract_v10(**self._kwargs(variant))
            self.assertEqual(contract["e3fp_table_count"], tables)
            self.assertEqual(contract["e3fp_parameter_count"], parameters)
            self.assertTrue(contract["update_zero_reference_equivalence_required"])

    def test_all_full_models_are_update_zero_equivalent(self) -> None:
        torch.manual_seed(29)
        before = torch.random.get_rng_state().clone()
        models = {
            variant: self._load(variant)
            for variant in (
                REFERENCE_SHARED_FIXED4,
                L0_STATE_FIXED4,
                LEVEL_SPECIFIC_FIXED4,
            )
        }
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        ids = torch.tensor([[[0, 1, 2, -1], [3, 4, 5, 6]]])
        mask = torch.tensor([[True, True]])
        role = torch.tensor([[False, True]])
        reference = models[REFERENCE_SHARED_FIXED4].adapter.encode_atom_memory(ids, mask, role)
        for variant, model in models.items():
            torch.testing.assert_close(
                model.adapter.encode_atom_memory(ids, mask, role),
                reference,
                rtol=0,
                atol=0,
                msg=variant,
            )
        common_reference = {
            key: value
            for key, value in models[REFERENCE_SHARED_FIXED4].adapter.state_dict().items()
            if not key.startswith("e3fp_atom_embedding") and key != "_extra_state"
        }
        for variant in (L0_STATE_FIXED4, LEVEL_SPECIFIC_FIXED4):
            candidate = models[variant].adapter.state_dict()
            for key, value in common_reference.items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, candidate[key]), f"{variant}:{key}")


if __name__ == "__main__":
    unittest.main()
