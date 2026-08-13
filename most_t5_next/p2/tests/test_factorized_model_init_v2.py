from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from most_t5_next.p2.factorized_model_init_v2 import (
    FactorizedModelInitV2Error,
    factorized_initialization_contract_v2,
    load_deterministic_factorized_model_v2,
)
from most_t5_next.p2.factorized_motif_t5_v2 import FactorizedMotifT5V2


class _TinyT5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Embedding(19, 12)
        self.config = SimpleNamespace(d_model=12)

    def get_input_embeddings(self) -> nn.Module:
        return self.shared


def _loader(**_kwargs):
    torch.manual_seed(991)
    return SimpleNamespace(model=_TinyT5())


class FactorizedModelInitV2Tests(unittest.TestCase):
    def _load(self, seed: int = 41):
        return load_deterministic_factorized_model_v2(
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
            max_atoms_per_motif=12,
            initial_geometry_gate=0.1,
            union_loader=_loader,
        )

    def test_contract_names_the_mechanism_and_forbids_the_bypass(self) -> None:
        contract = factorized_initialization_contract_v2(
            adapter_seed=41,
            num_e3fp_embeddings=4096,
            state_level2_weight=0.25,
            state_embedding_dim=64,
            atom_memory_dim=128,
            max_identity_span_length=128,
            max_atoms_per_motif=128,
            initial_geometry_gate=0.1,
        )
        self.assertFalse(contract["state_decoder_reads_atom_memory_directly"])
        self.assertEqual(
            contract["geometry_injection"],
            "normalized_per_channel_sigmoid_gate",
        )
        self.assertEqual(
            contract["target_atom_address"],
            "graphports_canonical_local_atom_id",
        )
        self.assertIn("v2-carrier-only", contract["factorisation_id"])
        with self.assertRaises(FactorizedModelInitV2Error):
            factorized_initialization_contract_v2(
                adapter_seed=41,
                num_e3fp_embeddings=4096,
                state_level2_weight=0.25,
                state_embedding_dim=64,
                atom_memory_dim=128,
                max_identity_span_length=128,
                max_atoms_per_motif=128,
                initial_geometry_gate=0.0,
            )

    def test_repeated_loads_are_equal_and_storage_independent(self) -> None:
        first = self._load()
        second = self._load()
        self.assertIsInstance(first, FactorizedMotifT5V2)
        for key, value in first.state_dict().items():
            other = second.state_dict()[key]
            self.assertTrue(torch.equal(value, other), key)
            self.assertNotEqual(value.data_ptr(), other.data_ptr(), key)

    def test_seed_changes_adapter_not_union_init_t5(self) -> None:
        first = self._load(41)
        second = self._load(42)
        self.assertTrue(torch.equal(first.t5.shared.weight, second.t5.shared.weight))
        self.assertFalse(
            torch.equal(
                first.adapter.target_atom_position_embedding.weight,
                second.adapter.target_atom_position_embedding.weight,
            )
        )

    def test_loader_restores_caller_rng(self) -> None:
        torch.manual_seed(1234)
        before = torch.random.get_rng_state().clone()
        self._load()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
