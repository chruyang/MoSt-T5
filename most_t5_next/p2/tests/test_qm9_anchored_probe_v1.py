import unittest
import json
from pathlib import Path
import shutil
import uuid

import torch

from most_t5_next.p1.bound_record import Span
from most_t5_next.p1.production_bridge import ProductionMotifRecord
from most_t5_next.p2.build_qm9_anchored_probe_cache_v1 import (
    _record_document,
    _renumber_anchors_by_edge_order,
    record_from_document,
)
from most_t5_next.p2.run_qm9_anchored_probe_v1 import (
    ProbeCollator,
    QM9ProbeDataset,
    training_target_statistics,
)


def _record(record_id: str, shift: int = 0) -> ProductionMotifRecord:
    return ProductionMotifRecord(
        record_artifact_sha256="a" * 64,
        record_id=record_id,
        storage_key="b" * 64,
        release_id="fixture",
        geometry_record_content_sha256="c" * 64,
        tokenizer_contract_sha256="d" * 64,
        tokenizer_snapshot_sha256="e" * 64,
        input_ids=(10, 11, 12),
        token_to_logical_motif=(0, 0, 1),
        token_role=("identity", "connection", "identity"),
        identity_spans=(Span(0, 1), Span(2, 3)),
        connection_token_indices=((1,), ()),
        logical_to_carrier=(0, 2),
        exact_identity_sha256=("1" * 64, "2" * 64),
        source_atom_count=2,
        full_e3fp_ids=((1 + shift, 2, 3, -1), (4 + shift, 5, 6, 7)),
        atom_valid_mask=(True, True),
        model_to_source_atom_index=(0, 1),
        atom_to_logical_motif=(0, 1),
        atom_is_attachment=(True, False),
        connection_token_to_atom=(-1, 0, -1),
    )


def _row(record_id: str, targets, mask, shift: int = 0):
    return {
        "schema_version": "fixture",
        "split": "train",
        "targets_hartree": targets,
        "target_mask": mask,
        "morgan_state_ids": [[101 + shift, 102, 103, 104], [105 + shift, 106, 107, 108]],
        "record": _record_document(_record(record_id, shift)),
    }


class QM9AnchoredProbeV1Test(unittest.TestCase):
    def test_model_facing_anchor_ids_follow_sorted_edges(self) -> None:
        topology = {
            "logical_motif_domain": {
                "cross_motif_bonds": [
                    {"edge_id": 0, "source_anchor_id": 4},
                    {"edge_id": 1, "source_anchor_id": 2},
                ],
                "motif_slot_anchor_ids": [[2], [4, 2], [4]],
            }
        }
        _renumber_anchors_by_edge_order(topology)
        domain = topology["logical_motif_domain"]
        self.assertEqual(domain["motif_slot_anchor_ids"], [[1], [0, 1], [0]])
        self.assertEqual(
            [row["source_anchor_id"] for row in domain["cross_motif_bonds"]],
            [0, 1],
        )

    def test_record_json_round_trip_and_state_provider_selection(self) -> None:
        rows = [
            _row("a", [-0.2, None, 0.3], [True, False, True]),
            _row("b", [None, 0.1, 0.2], [False, True, True], shift=10),
        ]
        rebuilt = record_from_document(rows[0]["record"])
        self.assertEqual(rebuilt, _record("a"))
        f3d = ProbeCollator(pad_token_id=0, cell="F3D")(rows)
        b2d = ProbeCollator(pad_token_id=0, cell="B2D")(rows)
        self.assertTrue(torch.equal(f3d.inputs["e3fp_input_ids"][0, 0], torch.tensor([1, 2, 3, -1])))
        self.assertTrue(torch.equal(b2d.inputs["e3fp_input_ids"][0, 0], torch.tensor([101, 102, 103, 104])))
        self.assertEqual(int(f3d.target_mask.sum()), 4)

    def test_partial_targets_define_train_only_statistics(self) -> None:
        rows = [
            _row("a", [1.0, 2.0, None], [True, True, False]),
            _row("b", [3.0, None, 5.0], [True, False, True]),
            _row("c", [None, 4.0, 7.0], [False, True, True]),
        ]
        dataset = QM9ProbeDataset.__new__(QM9ProbeDataset)
        dataset.rows = rows
        means, stds = training_target_statistics(dataset)
        self.assertTrue(torch.allclose(means, torch.tensor([2.0, 3.0, 6.0])))
        self.assertTrue(torch.all(stds > 0))

    def test_standard_target_overlay_filters_and_rebinds_rows(self) -> None:
        names = ("mu", "alpha", "r2", "u0", "u0_atom")
        root = Path("tmp") / f"qm9_probe_dataset_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, root, True)
        try:
            source_rows = [
                _row("a", [1.0, 2.0, 3.0], [True, True, True]),
                _row("b", [4.0, 5.0, 6.0], [True, True, True], shift=10),
            ]
            (root / "records.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in source_rows),
                encoding="utf-8",
            )
            overlay = root / "overlay"
            overlay.mkdir()
            target_row = {
                "record_id": "b",
                "storage_key": "b" * 64,
                "split": "train",
                "targets": {name: index + 0.5 for index, name in enumerate(names)},
            }
            (overlay / "targets.jsonl").write_text(
                json.dumps(target_row) + "\n", encoding="utf-8"
            )
            dataset = QM9ProbeDataset(
                root,
                split="train",
                target_overlay_dir=overlay,
                property_names=names,
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.rows[0]["record"]["record_id"], "b")
            self.assertEqual(dataset.rows[0]["targets_hartree"], [0.5, 1.5, 2.5, 3.5, 4.5])
            batch = ProbeCollator(
                pad_token_id=0,
                cell="F3D",
                property_names=names,
            )(dataset.rows)
            self.assertEqual(tuple(batch.targets.shape), (1, 5))
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
