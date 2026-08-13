from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest
import uuid

import torch

from most_t5_next.p1.bound_record import Span
from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
)
from most_t5_next.p2.factorized_view_collator_v1 import (
    FactorizedViewCollatorError,
)
from most_t5_next.p2.factorized_view_collator_v2 import (
    GraphPortsCanonicalAtomAddressProvider,
    collate_factorized_motif_view_v2,
)
from most_t5_next.r1.adapter.graphports_donor_atom_map_sidecar_v1 import (
    ROW_SCHEMA,
    SIDECAR_SCHEMA,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class FactorizedViewCollatorV2Test(unittest.TestCase):
    def setUp(self) -> None:
        contract = _digest("tokenizer-contract")
        snapshot = _digest("tokenizer-snapshot")
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
            token_role=(
                "identity",
                "identity",
                "connection",
                "identity",
                "identity",
                "connection",
            ),
            identity_spans=(Span(0, 2), Span(3, 5)),
            connection_token_indices=((2,), (5,)),
            logical_to_carrier=(0, 3),
            exact_identity_sha256=(_digest("left"), _digest("right")),
            source_atom_count=3,
            full_e3fp_ids=((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)),
            atom_valid_mask=(True, True, True),
            model_to_source_atom_index=(0, 1, 2),
            atom_to_logical_motif=(0, 0, 1),
            atom_is_attachment=(False, True, False),
        )
        self.sidecar = Path.cwd() / f".test-donor-atom-maps-{uuid.uuid4().hex}.jsonl"
        self._write_sidecar(storage_key="fixture/1")

    def tearDown(self) -> None:
        self.sidecar.unlink(missing_ok=True)

    def _write_sidecar(self, *, storage_key: str) -> None:
        row = {
            "schema_version": ROW_SCHEMA,
            "selection_index": 0,
            "member_id": "molecule-1",
            "sdf_record_index": 7,
            "split": "train",
            "storage_key": storage_key,
            "motif_count": 2,
            "overlay_planning_sidecar": {
                "schema_version": SIDECAR_SCHEMA,
                "source_codec_format_version": "fixture",
                # Canonical motif-local order is deliberately the reverse of
                # model-atom order for motif zero.
                "canonical_local_atom_to_model_atom": [
                    [[1, 1], [2, 0]],
                    [[1, 2]],
                ],
            },
        }
        self.sidecar.write_text(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def test_sidecar_is_inverted_onto_the_model_atom_axis(self) -> None:
        provider = GraphPortsCanonicalAtomAddressProvider(self.sidecar)
        self.assertEqual(provider.record_count, 1)
        self.assertEqual(provider.get(self.record), (1, 0, 0))
        batch = collate_factorized_motif_view_v2(
            (self.record,),
            tokenizer=self.tokenizer,
            objective_mode="state",
            seed=11,
            epoch=0,
            atom_address_provider=provider,
            state_mask_probability=1.0,
            num_e3fp_embeddings=16,
        )
        self.assertTrue(
            torch.equal(batch.atom_local_positions, torch.tensor([[1, 0, 0]]))
        )
        self.assertTrue(
            torch.equal(
                batch.model_inputs()["atom_local_positions"],
                batch.atom_local_positions,
            )
        )

    def test_lineage_mismatch_is_rejected_before_model_forward(self) -> None:
        self._write_sidecar(storage_key="wrong/key")
        provider = GraphPortsCanonicalAtomAddressProvider(self.sidecar)
        with self.assertRaisesRegex(FactorizedViewCollatorError, "storage keys"):
            collate_factorized_motif_view_v2(
                (self.record,),
                tokenizer=self.tokenizer,
                objective_mode="state",
                seed=11,
                epoch=0,
                atom_address_provider=provider,
                num_e3fp_embeddings=16,
            )


if __name__ == "__main__":
    unittest.main()
