from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p2.build_pf10_morgan_overlay_v1 import (
    COMMON_TRAIN_MEMBERSHIP,
    MANIFEST_NAME as OVERLAY_MANIFEST_NAME,
    SCHEMA_VERSION as OVERLAY_SCHEMA_VERSION,
)
from most_t5_next.p2.freeze_pf10_factorized_smoke128_v1 import (
    MEMBERSHIP_NAME,
    PF10SmokeMembershipError,
    evenly_spaced_indices,
    freeze_pf10_factorized_smoke128,
)


class PF10FactorizedSmokeMembershipTest(unittest.TestCase):
    def test_even_schedule_is_unique_and_includes_endpoints(self):
        indices = evenly_spaced_indices(294253)
        self.assertEqual(len(indices), 128)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 294252)
        self.assertEqual(len(set(indices)), 128)

    def test_freezer_preserves_selected_source_rows(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            overlay = root / "overlay"
            output = root / "smoke"
            overlay.mkdir()
            (overlay / OVERLAY_MANIFEST_NAME).write_text(
                json.dumps(
                    {
                        "schema_version": OVERLAY_SCHEMA_VERSION,
                        "status": "pass",
                        "counts": {"common_train_records": 130},
                    }
                ),
                encoding="utf-8",
            )
            with (overlay / COMMON_TRAIN_MEMBERSHIP).open("w", encoding="utf-8") as handle:
                for index in range(130):
                    handle.write(
                        json.dumps(
                            {
                                "split_index": index + 10,
                                "selection_index": index + 100,
                                "record_id": f"record-{index}",
                                "storage_key": f"key-{index}",
                                "eligible_motifs": [{"motif_id": 0, "eligible_atom_indices": [0, 1]}],
                            }
                        )
                        + "\n"
                    )
            manifest = freeze_pf10_factorized_smoke128(
                morgan_overlay=overlay,
                output_dir=output,
            )
            rows = [json.loads(line) for line in (output / MEMBERSHIP_NAME).read_text().splitlines()]
            self.assertEqual(manifest["counts"]["smoke_records"], 128)
            self.assertEqual(rows[0]["record_id"], "record-0")
            self.assertEqual(rows[-1]["record_id"], "record-129")
            self.assertEqual([row["smoke_index"] for row in rows], list(range(128)))

    def test_rejects_too_small_domain(self):
        with self.assertRaises(PF10SmokeMembershipError):
            evenly_spaced_indices(127)


if __name__ == "__main__":
    unittest.main()
