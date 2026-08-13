from __future__ import annotations

import hashlib
import unittest

import torch

from most_t5_next.p1.bound_record import Span
from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
    _connection_token_to_atom_from_document,
)
from most_t5_next.p2.factorized_view_collator_v3 import (
    collate_factorized_motif_view_v3,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _AddressProvider:
    def get(self, _record):
        return (0, 1, 0)


class FactorizedViewCollatorV3Test(unittest.TestCase):
    def setUp(self) -> None:
        contract = _digest("contract")
        snapshot = _digest("snapshot")
        self.tokenizer = ProductionTokenizerRuntime(
            tokenizer_contract_sha256=contract,
            tokenizer_snapshot_sha256=snapshot,
            vocab_size=128,
            pad_token_id=0,
            eos_token_id=1,
            sentinel_token_ids=tuple(range(127, 117, -1)),
        )
        self.record = ProductionMotifRecord(
            record_artifact_sha256=_digest("record"),
            record_id="molecule-1",
            storage_key="fixture/1",
            release_id="fixture",
            geometry_record_content_sha256=_digest("geometry"),
            tokenizer_contract_sha256=contract,
            tokenizer_snapshot_sha256=snapshot,
            input_ids=(10, 11, 12, 13, 14, 15),
            token_to_logical_motif=(0, 0, 0, 1, 1, 1),
            token_role=("identity", "identity", "connection", "identity", "identity", "connection"),
            identity_spans=(Span(0, 2), Span(3, 5)),
            connection_token_indices=((2,), (5,)),
            logical_to_carrier=(0, 3),
            exact_identity_sha256=(_digest("left"), _digest("right")),
            source_atom_count=3,
            full_e3fp_ids=((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)),
            atom_valid_mask=(True, True, True),
            model_to_source_atom_index=(0, 1, 2),
            atom_to_logical_motif=(0, 0, 1),
            atom_is_attachment=(False, True, True),
            connection_token_to_atom=(-1, -1, 1, -1, -1, 2),
        )

    def test_endpoint_addresses_follow_identity_corruption(self) -> None:
        batch = collate_factorized_motif_view_v3(
            (self.record,),
            tokenizer=self.tokenizer,
            objective_mode="cross_view",
            seed=3,
            epoch=0,
            atom_address_provider=_AddressProvider(),
            identity_mask_probability=1.0,
            num_e3fp_embeddings=16,
        )
        self.assertEqual(tuple(batch.input_ids.shape), (1, 4))
        self.assertTrue(
            torch.equal(
                batch.endpoint_token_to_atom,
                torch.tensor([[-1, 1, -1, 2]]),
            )
        )
        self.assertEqual(
            tuple(batch.model_inputs()["endpoint_token_to_atom"].shape),
            tuple(batch.input_ids.shape),
        )

    def test_loader_projection_uses_ordered_cross_bond_endpoints(self) -> None:
        document = {
            "token_domain": {
                "input_ids": [10, 11, 12, 13, 14, 15],
                "token_role": ["identity", "identity", "connection", "identity", "identity", "connection"],
                "token_to_logical_motif": [0, 0, 0, 1, 1, 1],
            },
            "logical_motif_domain": {
                "connection_token_indices": [[2], [5]],
                "cross_motif_bonds": [{
                    "edge_id": 0,
                    "left": {"logical_motif_index": 0, "atom_index": 1, "slot_ordinal": 0},
                    "right": {"logical_motif_index": 1, "atom_index": 2, "slot_ordinal": 0},
                    "bond_type": "single",
                }],
            },
            "atom_domain": {"atom_to_logical_motif": [0, 0, 1]},
        }
        self.assertEqual(
            _connection_token_to_atom_from_document(document),
            (-1, -1, 1, -1, -1, 2),
        )

    def test_single_motif_without_connections_has_an_empty_address_surface(self) -> None:
        document = {
            "token_domain": {
                "input_ids": [10, 11],
                "token_role": ["identity", "identity"],
                "token_to_logical_motif": [0, 0],
            },
            "logical_motif_domain": {
                "connection_token_indices": [[]],
                "cross_motif_bonds": [],
            },
            "atom_domain": {"atom_to_logical_motif": [0]},
        }
        self.assertEqual(
            _connection_token_to_atom_from_document(document),
            (-1, -1),
        )


if __name__ == "__main__":
    unittest.main()
