from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

from most_t5_next.p2.matched_motif_state_donor_v1 import (
    DONOR_ATOM_MAP_SIDECAR_SCHEMA,
    MatchedMotifDonorError,
    build_graphports_donor_atom_map_sidecar,
    build_matched_motif_donor_plan,
    extract_motif_occurrences,
    materialize_matched_state_overlay,
)


def _document(
    record_id: str,
    identities: list[str],
    atom_groups: list[list[int]],
    slot_atoms: list[list[int]],
    cross_bonds: list[dict[str, object]],
    state_rows: list[list[int]],
    local_atom_maps: list[list[list[int]]] | None = None,
) -> dict[str, object]:
    if local_atom_maps is None:
        local_atom_maps = [
            [[local_id, model_atom] for local_id, model_atom in enumerate(group, 1)]
            for group in atom_groups
        ]
    return {
        "member": {"member_id": record_id},
        "logical_motif_domain": {
            "exact_identity_sha256": identities,
            "motif_atom_indices": atom_groups,
            "motif_slot_atom_indices": slot_atoms,
            "cross_motif_bonds": cross_bonds,
        },
        "atom_domain": {"full_e3fp_ids": state_rows},
        "overlay_planning_sidecar": {
            "schema_version": DONOR_ATOM_MAP_SIDECAR_SCHEMA,
            "canonical_local_atom_to_model_atom": local_atom_maps,
        },
    }


def _edge(
    left_motif: int,
    left_atom: int,
    right_motif: int,
    right_atom: int,
    bond_type: str = "single",
) -> dict[str, object]:
    return {
        "left": {
            "logical_motif_index": left_motif,
            "atom_index": left_atom,
        },
        "right": {
            "logical_motif_index": right_motif,
            "atom_index": right_atom,
        },
        "bond_type": bond_type,
    }


class MatchedMotifStateDonorTests(unittest.TestCase):
    def test_graphports_source_atom_map_builds_planning_sidecar(self) -> None:
        encoding = SimpleNamespace(
            format_version="graph-ports-v1",
            motifs=(
                SimpleNamespace(motif_id=0, source_atom_map=((1, 3), (2, 1))),
                SimpleNamespace(motif_id=1, source_atom_map=((1, 0),)),
            ),
        )

        sidecar = build_graphports_donor_atom_map_sidecar(encoding)

        self.assertEqual(sidecar["schema_version"], DONOR_ATOM_MAP_SIDECAR_SCHEMA)
        self.assertEqual(sidecar["source_codec_format_version"], "graph-ports-v1")
        self.assertEqual(
            sidecar["canonical_local_atom_to_model_atom"],
            [[[1, 3], [2, 1]], [[1, 0]]],
        )

    def test_cross_record_plan_and_overlay_close_on_fixture(self) -> None:
        left = _document(
            "left",
            ["motif-a"],
            [[0, 1]],
            [[]],
            [],
            [[1, 2, 3, 4], [5, 6, 7, 8]],
        )
        right = _document(
            "right",
            ["motif-a"],
            [[0, 1]],
            [[]],
            [],
            [[101, 102, 103, 104], [105, 106, 107, 108]],
        )
        documents = [left, right]
        before = copy.deepcopy(documents)

        plan = build_matched_motif_donor_plan(documents)

        self.assertEqual(len(plan.assignments), 2)
        self.assertTrue(
            all(
                row.recipient_record_id != row.donor_record_id
                for row in plan.assignments
            )
        )
        self.assertEqual(plan.coverage["motif_occurrence_coverage"], 1.0)
        self.assertEqual(plan.coverage["atom_row_coverage"], 1.0)
        self.assertEqual(plan.coverage["records_with_all_motifs_eligible"], 2)

        overlay = materialize_matched_state_overlay(documents, plan)

        self.assertEqual(
            overlay.state_by_record_id["left"],
            ((101, 102, 103, 104), (105, 106, 107, 108)),
        )
        self.assertEqual(
            overlay.state_by_record_id["right"],
            ((1, 2, 3, 4), (5, 6, 7, 8)),
        )
        self.assertEqual(overlay.changed_motifs_by_record_id, {"left": (0,), "right": (0,)})
        self.assertEqual(overlay.changed_state_slot_count, 16)
        self.assertEqual(documents, before)

    def test_canonical_local_ids_not_model_indices_drive_atom_pairing(self) -> None:
        left = _document(
            "left",
            ["same-motif"],
            [[0, 1]],
            [[]],
            [],
            [[10, 11, 12, 13], [20, 21, 22, 23]],
            local_atom_maps=[[[1, 0], [2, 1]]],
        )
        # The same canonical motif has the opposite model/SDF atom numbering.
        # Row 0 is local atom 2, and row 1 is local atom 1.
        renumbered = _document(
            "renumbered",
            ["same-motif"],
            [[0, 1]],
            [[]],
            [],
            [[200, 201, 202, 203], [100, 101, 102, 103]],
            local_atom_maps=[[[1, 1], [2, 0]]],
        )
        documents = [left, renumbered]
        before = copy.deepcopy(documents)

        plan = build_matched_motif_donor_plan(documents)
        overlay = materialize_matched_state_overlay(documents, plan)

        self.assertEqual(
            overlay.state_by_record_id["left"],
            ((100, 101, 102, 103), (200, 201, 202, 203)),
        )
        self.assertEqual(
            overlay.state_by_record_id["renumbered"],
            ((20, 21, 22, 23), (10, 11, 12, 13)),
        )
        self.assertEqual(documents, before)

    def test_singleton_signature_is_excluded_and_reported(self) -> None:
        documents = [
            _document("a", ["shared"], [[0]], [[]], [], [[1, 2, 3, 4]]),
            _document("b", ["shared"], [[0]], [[]], [], [[5, 6, 7, 8]]),
            _document("c", ["unique"], [[0]], [[]], [], [[9, 10, 11, 12]]),
        ]

        plan = build_matched_motif_donor_plan(documents)

        self.assertEqual(len(plan.assignments), 2)
        self.assertEqual(plan.excluded_occurrences, (("c", 0),))
        self.assertEqual(plan.coverage["eligible_motif_occurrences"], 2)
        self.assertEqual(plan.coverage["excluded_motif_occurrences"], 1)
        self.assertEqual(plan.coverage["records_with_any_eligible_motif"], 2)

    def test_port_pattern_prevents_identity_only_match(self) -> None:
        connected = _document(
            "connected",
            ["focal", "neighbor"],
            [[0], [1]],
            [[0], [1]],
            [_edge(0, 0, 1, 1)],
            [[1, 2, 3, 4], [5, 6, 7, 8]],
        )
        disconnected = _document(
            "disconnected",
            ["focal", "neighbor"],
            [[0], [1]],
            [[], []],
            [],
            [[11, 12, 13, 14], [15, 16, 17, 18]],
        )

        plan = build_matched_motif_donor_plan([connected, disconnected])

        self.assertEqual(plan.assignments, ())
        self.assertEqual(plan.coverage["eligible_motif_occurrences"], 0)
        self.assertEqual(plan.coverage["excluded_motif_occurrences"], 4)

    def test_strict_neighbor_signature_is_an_optional_tighter_gate(self) -> None:
        left = _document(
            "left",
            ["focal", "neighbor-a"],
            [[0], [1]],
            [[0], [1]],
            [_edge(0, 0, 1, 1)],
            [[1, 2, 3, 4], [5, 6, 7, 8]],
        )
        right = _document(
            "right",
            ["focal", "neighbor-b"],
            [[0], [1]],
            [[0], [1]],
            [_edge(0, 0, 1, 1)],
            [[11, 12, 13, 14], [15, 16, 17, 18]],
        )

        loose = build_matched_motif_donor_plan([left, right])
        strict = build_matched_motif_donor_plan(
            [left, right], strict_neighbors=True
        )

        self.assertEqual(loose.coverage["eligible_motif_occurrences"], 2)
        self.assertEqual(strict.coverage["eligible_motif_occurrences"], 0)
        self.assertFalse(loose.strict_neighbor_match)
        self.assertTrue(strict.strict_neighbor_match)

    def test_port_atom_must_belong_to_its_motif(self) -> None:
        malformed = _document(
            "bad",
            ["motif"],
            [[0]],
            [[1]],
            [],
            [[1, 2, 3, 4]],
        )

        with self.assertRaisesRegex(
            MatchedMotifDonorError, "outside its group"
        ):
            extract_motif_occurrences(malformed)

    def test_missing_or_inconsistent_canonical_atom_map_is_rejected(self) -> None:
        missing = _document(
            "missing",
            ["motif"],
            [[0]],
            [[]],
            [],
            [[1, 2, 3, 4]],
        )
        del missing["overlay_planning_sidecar"]
        with self.assertRaisesRegex(
            MatchedMotifDonorError, "fields are malformed"
        ):
            extract_motif_occurrences(missing)

        inconsistent = _document(
            "inconsistent",
            ["motif"],
            [[0, 1]],
            [[]],
            [],
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            local_atom_maps=[[[1, 0], [2, 0]]],
        )
        with self.assertRaisesRegex(
            MatchedMotifDonorError, "disagrees with its atom group"
        ):
            extract_motif_occurrences(inconsistent)


if __name__ == "__main__":
    unittest.main()
