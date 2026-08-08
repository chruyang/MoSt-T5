from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from most_t5_next.p2.factorized_model_init_v1 import (
    FactorizedModelInitError,
    factorized_initialization_contract,
    load_deterministic_factorized_model,
)


class _TinyT5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Embedding(19, 12)
        self.config = SimpleNamespace(d_model=12)

    def get_input_embeddings(self) -> nn.Module:
        return self.shared


class _Verified:
    def __init__(self) -> None:
        self.model = _TinyT5()


def _loader(**_kwargs: object) -> _Verified:
    # A real loader independently reloads raw T5 on every call.
    torch.manual_seed(991)
    return _Verified()


class FactorizedModelInitTests(unittest.TestCase):
    def _load(self, *, seed: int = 41):
        return load_deterministic_factorized_model(
            base_model_snapshot=Path("base-model"),
            base_tokenizer_snapshot=Path("base-tokenizer"),
            union_tokenizer_dir=Path("union-tokenizer"),
            union_init_dir=Path("union-init"),
            union_geometry_fusion_seed=17,
            adapter_seed=seed,
            num_e3fp_embeddings=23,
            state_embedding_dim=8,
            atom_memory_dim=10,
            max_identity_span_length=16,
            union_loader=_loader,
        )

    def test_contract_is_explicit_and_data_independent(self) -> None:
        contract = factorized_initialization_contract(
            adapter_seed=41,
            num_e3fp_embeddings=4096,
            state_level2_weight=0.25,
            state_embedding_dim=64,
            atom_memory_dim=128,
            max_identity_span_length=64,
        )
        self.assertFalse(contract["data_dependent_initialization"])
        self.assertEqual(contract["paired_cells"], ["B2D", "F3D"])
        with self.assertRaises(FactorizedModelInitError):
            factorized_initialization_contract(
                adapter_seed=-1,
                num_e3fp_embeddings=4096,
                state_level2_weight=0.25,
                state_embedding_dim=64,
                atom_memory_dim=128,
                max_identity_span_length=64,
            )

    def test_repeated_loads_are_bitwise_equal_but_independent(self) -> None:
        first = self._load()
        second = self._load()
        first_state = first.state_dict()
        second_state = second.state_dict()
        self.assertEqual(first_state.keys(), second_state.keys())
        for key, value in first_state.items():
            self.assertTrue(torch.equal(value, second_state[key]), key)
            self.assertNotEqual(value.data_ptr(), second_state[key].data_ptr(), key)

    def test_adapter_seed_changes_adapter_but_not_t5(self) -> None:
        first = self._load(seed=41)
        second = self._load(seed=42)
        self.assertTrue(
            torch.equal(first.t5.shared.weight, second.t5.shared.weight)
        )
        self.assertFalse(
            torch.equal(
                first.adapter.state_embedding.weight,
                second.adapter.state_embedding.weight,
            )
        )

    def test_loader_restores_caller_rng(self) -> None:
        torch.manual_seed(1234)
        before = torch.random.get_rng_state().clone()
        self._load()
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
