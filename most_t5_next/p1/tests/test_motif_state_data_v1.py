from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import lmdb
import torch

from most_t5_next.p1.motif_state_data_v1 import (
    PF1MotifStateDataset,
    collate_motif_state_records,
)
from most_t5_next.p1.run_g1_motif_state_v1 import run


def _envelope(record_id: str, rows, groups):
    atom_count = len(rows)
    motif_count = max(groups) + 1
    return {
        "receipt": {"member_id": record_id},
        "motif_training_document": {
            "atom_domain": {
                "full_e3fp_ids": rows,
                "atom_valid_mask": [True] * atom_count,
                "atom_is_attachment": [False] * atom_count,
                "atom_to_logical_motif": groups,
            },
            "logical_motif_domain": {
                "motif_atom_indices": [
                    [index for index, group in enumerate(groups) if group == motif]
                    for motif in range(motif_count)
                ]
            },
        },
    }


class MotifStateDataTests(unittest.TestCase):
    def test_lmdb_dataset_and_padding_collator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            members = [
                ("000000001", "row:1", [[1, 2, 3, 4], [5, 6, 7, 8]], [0, 0]),
                ("000000002", "row:2", [[9, 10, 11, 12]], [0]),
            ]
            with (root / "train_membership.jsonl").open("w", encoding="utf-8") as handle:
                for index, (key, member_id, _, _) in enumerate(members):
                    handle.write(
                        json.dumps(
                            {
                                "storage_key": key,
                                "member_id": member_id,
                                "selection_index": index,
                            }
                        )
                        + "\n"
                    )
            (root / "dev_membership.jsonl").write_text(
                (root / "train_membership.jsonl").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            env = lmdb.open(str(root / "paired_records.lmdb"), map_size=1 << 20, subdir=True)
            with env.begin(write=True) as transaction:
                for key, member_id, rows, groups in members:
                    transaction.put(
                        key.encode("ascii"),
                        json.dumps(_envelope(member_id, rows, groups)).encode("utf-8"),
                    )
            env.close()

            dataset = PF1MotifStateDataset(str(root), "train")
            batch = collate_motif_state_records([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["e3fp_ids"].shape), (2, 2, 4))
            self.assertTrue(torch.equal(batch["e3fp_ids"][1, 1], torch.full((4,), -1)))
            self.assertEqual(batch["record_ids"], ("row:1", "row:2"))
            self.assertEqual(batch["num_groups"], 1)
            dataset.close()

            manifest = run(
                SimpleNamespace(
                    paired_release=str(root),
                    output_dir=str(root / "g1-smoke"),
                    pooling="gated",
                    updates=2,
                    batch_size=2,
                    workers=0,
                    embedding_dim=8,
                    hidden_dim=16,
                    learning_rate=1e-3,
                    mask_probability=0.5,
                    seed=1,
                    data_seed=2,
                    train_mask_seed=3,
                    dev_mask_seed=4,
                    require_cuda=False,
                )
            )
            self.assertEqual(manifest["status"], "pass")
            self.assertEqual(set(manifest["evaluations"]), {"0", "1", "2"})
            self.assertTrue((root / "g1-smoke" / "final_state.pt").is_file())


if __name__ == "__main__":
    unittest.main()
