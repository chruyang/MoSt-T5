from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional in the local fixture")
class AnchoredV4ShellScreenTest(unittest.TestCase):
    def test_cells_freeze_views_and_memory(self) -> None:
        from most_t5_next.p2.run_anchored_v4_shell_screen_v1 import cell_contract

        self.assertEqual(cell_contract("B0", "l12_mean")["view_id"], "m_only")
        self.assertEqual(cell_contract("B0", "l12_mean")["memory_mode"], "zero")
        self.assertEqual(cell_contract("B2D", "l12_mean")["state_kind"], "coordinate_blind_morgan")
        self.assertEqual(cell_contract("F3D", "l0_l12_mean")["view_id"], "m_plus_g")

    def test_invalid_cell_and_shell_are_rejected(self) -> None:
        from most_t5_next.p2.run_anchored_v4_shell_screen_v1 import (
            AnchoredV4ShellScreenError,
            cell_contract,
        )

        with self.assertRaisesRegex(AnchoredV4ShellScreenError, "cell"):
            cell_contract("unknown", "l12_mean")
        with self.assertRaisesRegex(AnchoredV4ShellScreenError, "shell"):
            cell_contract("F3D", "unknown")

    def test_matched_state_replaces_only_real_atom_rows(self) -> None:
        from most_t5_next.p2.pf10_training_tensor_cache_v1 import CachedV3Batch
        from most_t5_next.p2.run_anchored_v4_shell_screen_v1 import (
            _replace_state_batches,
        )

        class Provider:
            def get(self, record_id: str):
                self.record_id = record_id
                return ((9, 8, 7, 6), (5, 4, 3, 2))

        inputs = {
            "e3fp_input_ids": torch.tensor([[[1, 1, 1, 1], [2, 2, 2, 2], [-1, -1, -1, -1]]]),
            "atom_mask": torch.tensor([[True, True, False]]),
        }
        batch = CachedV3Batch(
            view_id="m_plus_g",
            epoch=0,
            record_ids=("r",),
            exact_identity_sha256=(("x",),),
            inputs=inputs,
            labels=torch.tensor([[1]]),
        )
        replaced = _replace_state_batches((batch,), provider=Provider())[0]
        self.assertEqual(
            replaced.inputs["e3fp_input_ids"].tolist(),
            [[[9, 8, 7, 6], [5, 4, 3, 2], [-1, -1, -1, -1]]],
        )
        self.assertEqual(inputs["e3fp_input_ids"][0, 0].tolist(), [1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
