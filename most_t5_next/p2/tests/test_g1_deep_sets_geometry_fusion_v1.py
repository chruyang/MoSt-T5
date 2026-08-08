import tempfile
from pathlib import Path
import unittest

import torch

from most_t5_next.p1.level_aware_motif_state_v1 import LevelAwareMotifStateEncoder
from most_t5_next.p1.shared_geometry_fusion import GeometryTensorSidecar
from most_t5_next.p2.g1_deep_sets_geometry_fusion_v1 import (
    FrozenG1DeepSetsCarrierFusion,
    G1DeepSetsFusionError,
    load_g1b_encoder,
)


def _checkpoint(path: Path) -> None:
    model = LevelAwareMotifStateEncoder(
        num_e3fp_embeddings=4096,
        embedding_dim=64,
        hidden_dim=128,
        pooling="deep_sets",
    )
    torch.save(
        {
            "schema_version": "most-t5-p1/g1-motif-state-screen/v1",
            "pooling": "deep_sets",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "completed_updates": 500,
        },
        path,
    )


class FrozenG1DeepSetsCarrierFusionTest(unittest.TestCase):
    def test_checkpoint_loads_frozen_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "g1.pt"
            _checkpoint(path)
            encoder = load_g1b_encoder(path)
            self.assertFalse(encoder.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))

    def test_forward_is_atom_permutation_invariant_within_carrier(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "g1.pt"
            _checkpoint(path)
            torch.manual_seed(7)
            fusion = FrozenG1DeepSetsCarrierFusion(
                num_e3fp_embeddings=4096,
                hidden_size=16,
                g1_checkpoint=path,
            )
            embeddings = torch.randn(1, 4, 16)
            attention = torch.ones(1, 4, dtype=torch.long)
            base = GeometryTensorSidecar(
                e3fp_ids=torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]]),
                e3fp_atom_mask=torch.tensor([[True, True]]),
                e3fp_atom_to_token=torch.tensor([[1, 1]]),
                e3fp_atom_is_attachment=torch.tensor([[False, True]]),
            )
            permuted = GeometryTensorSidecar(
                e3fp_ids=base.e3fp_ids[:, [1, 0]],
                e3fp_atom_mask=base.e3fp_atom_mask[:, [1, 0]],
                e3fp_atom_to_token=base.e3fp_atom_to_token[:, [1, 0]],
                e3fp_atom_is_attachment=base.e3fp_atom_is_attachment[:, [1, 0]],
            )
            first = fusion(embeddings, base, attention_mask=attention)
            second = fusion(embeddings, permuted, attention_mask=attention)
            torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
            torch.testing.assert_close(first[:, 0], embeddings[:, 0], rtol=0.0, atol=0.0)
            self.assertFalse(torch.equal(first[:, 1], embeddings[:, 1]))

    def test_attachment_roles_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "g1.pt"
            _checkpoint(path)
            fusion = FrozenG1DeepSetsCarrierFusion(
                num_e3fp_embeddings=4096,
                hidden_size=8,
                g1_checkpoint=path,
            )
            geometry = GeometryTensorSidecar(
                e3fp_ids=torch.tensor([[[1, 2, 3, 4]]]),
                e3fp_atom_mask=torch.tensor([[True]]),
                e3fp_atom_to_token=torch.tensor([[0]]),
            )
            with self.assertRaisesRegex(G1DeepSetsFusionError, "attachment"):
                fusion(
                    torch.zeros(1, 2, 8),
                    geometry,
                    attention_mask=torch.ones(1, 2, dtype=torch.long),
                )


if __name__ == "__main__":
    unittest.main()
