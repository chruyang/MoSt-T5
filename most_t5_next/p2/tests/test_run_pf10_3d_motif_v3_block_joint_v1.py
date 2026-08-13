from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import torch

from most_t5_next.p2.factorized_motif_t5_v3 import FactorizedMotifT5V3
from most_t5_next.p2.motif_state_matching_v3 import MotifStateMatchingHeadV3
from most_t5_next.p2.pf10_training_tensor_cache_v1 import CachedV3Batch
from most_t5_next.p2.run_pf10_3d_motif_v3_block_joint_v1 import (
    BLOCK_CYCLE,
    EVALUATION_UPDATES,
    GRAMMAR_PROTOCOL,
    MATCHING_PROTOCOL,
    TOTAL_UPDATES,
    V3BlockJointError,
    _grammar_forward,
    block_for_update,
    configure_block,
    load_matching_only_state,
    optimization_family_for_block,
)
from most_t5_next.p2.run_pf10_3d_motif_v3_matching_only_v1 import SCHEMA_VERSION
from most_t5_next.p2.tests.test_factorized_motif_t5_v3 import _TinyT5


class V3BlockJointTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(617)
        self.model = FactorizedMotifT5V3(
            _TinyT5(),
            num_e3fp_embeddings=16,
            state_embedding_dim=4,
            atom_memory_dim=8,
            max_identity_span_length=8,
            max_atoms_per_motif=4,
            geometry_fraction=0.15,
        )
        self.head = MotifStateMatchingHeadV3(
            hidden_size=8,
            projection_dim=4,
            temperature=0.1,
        )
        inputs = {
            "input_ids": torch.tensor([[2, 3, 4, 5, 6, 7]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "e3fp_mask_token_id": 17,
            "e3fp_input_ids": torch.tensor(
                [[[7, 8, 9, 10], [8, 9, 10, 11], [4, 5, 6, 7]]]
            ),
            "atom_mask": torch.ones((1, 3), dtype=torch.bool),
            "atom_to_motif": torch.tensor([[0, 0, 1]]),
            "atom_local_positions": torch.tensor([[0, 1, 0]]),
            "motif_mask": torch.ones((1, 2), dtype=torch.bool),
            "motif_to_carrier": torch.tensor([[0, 3]]),
            "identity_span_bounds": torch.tensor([[[0, 2], [3, 5]]]),
            "endpoint_token_to_atom": torch.tensor([[-1, -1, 1, -1, -1, 2]]),
            "atom_is_attachment": torch.tensor([[False, True, True]]),
            "labels": torch.tensor([[3, 4, 5, 6, 7, 8]]),
            "objective_mode": "cross_view",
        }
        self.batch = CachedV3Batch(
            view_id="m_plus_g",
            epoch=0,
            record_ids=("record",),
            exact_identity_sha256=(("0" * 64, "1" * 64),),
            inputs=inputs,
            labels=inputs["labels"],
        )

    def test_cycle_has_one_separated_matching_and_three_grammar_updates(self) -> None:
        self.assertEqual(tuple(block_for_update(i) for i in range(1, 5)), BLOCK_CYCLE)
        self.assertEqual(block_for_update(5), BLOCK_CYCLE[0])
        with self.assertRaisesRegex(V3BlockJointError, "positive"):
            block_for_update(0)

    def test_each_parameter_family_owns_its_optimizer_clock(self) -> None:
        families = [
            optimization_family_for_block(block_for_update(update))
            for update in range(1, TOTAL_UPDATES + 1)
        ]
        self.assertEqual(families.count("matching"), MATCHING_PROTOCOL.total_updates)
        self.assertEqual(families.count("grammar"), GRAMMAR_PROTOCOL.total_updates)
        self.assertEqual(MATCHING_PROTOCOL.base_learning_rate, 3.0e-4)
        self.assertEqual(GRAMMAR_PROTOCOL.base_learning_rate, 1.0e-3)
        self.assertEqual(EVALUATION_UPDATES, (0, 800, 1600))
        with self.assertRaisesRegex(V3BlockJointError, "unknown"):
            optimization_family_for_block("not-a-block")

    def test_trainability_is_disjoint_between_blocks(self) -> None:
        configure_block(self.model, self.head, "matching_m_plus_g")
        self.assertFalse(any(p.requires_grad for p in self.model.t5.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.adapter.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.head.parameters()))

        configure_block(self.model, self.head, "grammar_g_only")
        self.assertTrue(all(p.requires_grad for p in self.model.t5.parameters()))
        self.assertFalse(any(p.requires_grad for p in self.model.adapter.parameters()))
        self.assertFalse(any(p.requires_grad for p in self.head.parameters()))

    def test_grammar_component_modes_reach_the_model_contract(self) -> None:
        outputs = {
            mode: _grammar_forward(self.model, self.batch, memory_mode=mode)
            for mode in ("both", "carrier_only", "endpoint_only", "zero")
        }
        baseline = self.model.get_input_embeddings()(self.batch.inputs["input_ids"])
        self.assertTrue(
            torch.equal(outputs["zero"].adapter_encoding.fused_embeddings, baseline)
        )
        for mode in ("both", "carrier_only", "endpoint_only"):
            self.assertFalse(
                torch.equal(outputs[mode].adapter_encoding.fused_embeddings, baseline)
            )
        with self.assertRaisesRegex(V3BlockJointError, "component"):
            _grammar_forward(self.model, self.batch, memory_mode="unknown")

    def test_matching_checkpoint_is_cell_bound_and_exactly_loaded(self) -> None:
        handle = tempfile.NamedTemporaryFile(
            dir=Path.cwd(), suffix=".pt", delete=False
        )
        path = Path(handle.name)
        handle.close()
        try:
            expected = {
                "schema_version": SCHEMA_VERSION,
                "cell": "F3D",
                "completed_updates": 1000,
                "geometry_fraction": 0.15,
                "adapter_state_dict": self.model.adapter.state_dict(),
                "matching_head_state_dict": self.head.state_dict(),
            }
            torch.save(expected, path)
            load_matching_only_state(
                self.model, self.head, checkpoint_path=path, cell="F3D"
            )
            with self.assertRaisesRegex(V3BlockJointError, "cell"):
                load_matching_only_state(
                    self.model, self.head, checkpoint_path=path, cell="B2D"
                )
            with self.assertRaisesRegex(V3BlockJointError, "fraction"):
                load_matching_only_state(
                    self.model,
                    self.head,
                    checkpoint_path=path,
                    cell="F3D",
                    expected_geometry_fraction=0.5,
                )
        finally:
            path.unlink(missing_ok=True)

    def test_legacy_matching_checkpoint_must_bind_fraction_in_manifest(self) -> None:
        root = Path("tmp") / f"v3_block_joint_test_{uuid4().hex}"
        root.mkdir(parents=True)
        checkpoint = root / "adapter_and_matching_state.pt"
        manifest = root / "matching_only_manifest.json"
        try:
            torch.save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "cell": "B2D",
                    "completed_updates": 1000,
                    "adapter_state_dict": self.model.adapter.state_dict(),
                    "matching_head_state_dict": self.head.state_dict(),
                },
                checkpoint,
            )
            with self.assertRaisesRegex(V3BlockJointError, "unbound"):
                load_matching_only_state(
                    self.model, self.head, checkpoint_path=checkpoint, cell="B2D"
                )
            manifest.write_text(
                '{"geometry_fraction": 0.15}\n', encoding="utf-8"
            )
            load_matching_only_state(
                self.model, self.head, checkpoint_path=checkpoint, cell="B2D"
            )
        finally:
            manifest.unlink(missing_ok=True)
            checkpoint.unlink(missing_ok=True)
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
