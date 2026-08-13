from __future__ import annotations

import unittest
from types import SimpleNamespace

from most_t5_next.p2.build_pf10_morgan_overlay_v1 import (
    PF10MorganOverlayError,
    _common_eligible_motifs,
)


class _Span:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop


class PF10MorganOverlayTest(unittest.TestCase):
    def test_common_support_requires_two_atoms_in_both_views(self):
        record = SimpleNamespace(
            identity_spans=(_Span(1, 3), _Span(3, 4)),
            full_e3fp_ids=((1, 2, 3, 4), (2, 3, 4, 5), (3, 4, -1, 6)),
            atom_valid_mask=(True, True, True),
            atom_to_logical_motif=(0, 0, 1),
        )
        common = _common_eligible_motifs(
            record,
            ((10, 11, 12, 13), (20, 21, 22, 23), (30, 31, 32, 33)),
        )
        self.assertEqual(
            common,
            (
                {
                    "motif_id": 0,
                    "identity_span_length": 2,
                    "eligible_atom_indices": [0, 1],
                },
            ),
        )

    def test_missing_level_in_either_view_removes_atom(self):
        record = SimpleNamespace(
            identity_spans=(_Span(0, 1),),
            full_e3fp_ids=((1, 2, 3, 4), (2, 3, -1, 5), (3, 4, 5, 6)),
            atom_valid_mask=(True, True, True),
            atom_to_logical_motif=(0, 0, 0),
        )
        common = _common_eligible_motifs(
            record,
            ((10, 11, -1, 13), (20, 21, 22, 23), (30, 31, 32, 33)),
        )
        self.assertEqual(common, ())

    def test_rejects_atom_domain_mismatch(self):
        record = SimpleNamespace(
            identity_spans=(_Span(0, 1),),
            full_e3fp_ids=((1, 2, 3, 4),),
            atom_valid_mask=(True,),
            atom_to_logical_motif=(0,),
        )
        with self.assertRaisesRegex(PF10MorganOverlayError, "domains"):
            _common_eligible_motifs(record, ())


if __name__ == "__main__":
    unittest.main()
