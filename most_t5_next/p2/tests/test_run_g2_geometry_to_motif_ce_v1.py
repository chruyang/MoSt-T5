import tempfile
from pathlib import Path
import unittest

from most_t5_next.p2.g1_deep_sets_geometry_fusion_v1 import FUSION_ID
from most_t5_next.p1.run_pf1_four_grid_v1 import PF1TrainingError
from most_t5_next.p2.run_g2_geometry_to_motif_ce_v1 import (
    G2_MASK_PROBABILITY,
    G2_PROTOCOL,
    REPORT_SCHEMA,
    execute_g2_geometry_to_motif_ce,
)


class G2RunnerTest(unittest.TestCase):
    def test_executor_freezes_all_identity_view_and_bridge_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            checkpoint = Path(directory) / "g1.pt"
            checkpoint.write_bytes(b"fixture")
            observed = {}

            def engine(**kwargs):
                observed.update(kwargs)
                return {
                    "schema_version": "engine/v1",
                    "status": "pass",
                    "data": {"mask_probability": 1.0},
                    "optimization": {},
                    "interpretation": {},
                    "conditions": [{"condition": "M1"}],
                }

            report = execute_g2_geometry_to_motif_ce(
                g1_checkpoint=checkpoint,
                engine=engine,
                condition_ids=("M1",),
                protocol=G2_PROTOCOL,
                output_dir=output,
            )
            self.assertEqual(observed["mask_probability"], G2_MASK_PROBABILITY)
            self.assertTrue(callable(observed["wrapper_loader"]))
            self.assertEqual(report["schema_version"], REPORT_SCHEMA)
            self.assertEqual(report["conditions"][0]["g2_cell"], "G2-G")
            self.assertEqual(report["fusion_id"], FUSION_ID)

    def test_executor_rejects_unpaired_condition_request(self):
        with self.assertRaisesRegex(PF1TrainingError, "exactly one"):
            execute_g2_geometry_to_motif_ce(
                g1_checkpoint=Path("missing"),
                engine=lambda **_: {},
                condition_ids=("M0", "M1"),
                protocol=G2_PROTOCOL,
                output_dir=Path("unused"),
            )


if __name__ == "__main__":
    unittest.main()
