from __future__ import annotations

import copy
import hashlib
import unittest

from most_t5_next.r1.adapter.mol_linearizer import linearize_mol
from most_t5_next.r1.adapter.p1_topology_augmentation_v1 import (
    PROJECTION_POLICY,
    TopologyAugmentationError,
    augmentation_sha256,
    build_topology_augmentation,
    validate_topology_augmentation,
)
from most_t5_next.r1.adapter.tests.test_mol_linearizer import build_synthetic_molecule


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class P1TopologyAugmentationTest(unittest.TestCase):
    def setUp(self):
        self.result = linearize_mol(build_synthetic_molecule())
        self.mapping = tuple(range(9)) + (11,)
        self.expected_digests = tuple(digest(fragment) for fragment in self.result.fragment_sequence)
        self.document = build_topology_augmentation(
            linearization_result=self.result,
            member_id="pcqm4mv2:000000123",
            base_record_content_sha256=digest("production-record"),
            linearizer_spec_sha256=digest("linearizer-spec"),
            expected_motif_atom_indices=self.result.motif_atom_groups,
            expected_motif_lexeme_sha256=self.expected_digests,
            source_atom_count=12,
            model_to_source_atom_index=self.mapping,
        )

    def test_canonical_to_logical_remap_and_ports_are_explicit(self):
        domain = self.document["logical_motif_domain"]
        self.assertEqual(domain["logical_to_canonical_motif_index"], [1, 0, 2, 3])
        self.assertEqual(domain["canonical_to_logical_motif_index"], [1, 0, 2, 3])
        self.assertEqual(len(domain["cross_motif_bonds"]), 2)
        self.assertEqual(sum(map(len, domain["motif_slot_atom_indices"])), 4)
        self.assertFalse(self.document["provenance"]["geometry_or_e3fp_recomputed"])
        self.assertEqual(self.document["provenance"]["projection_policy"], PROJECTION_POLICY)
        self.assertNotIn("e3fp", self.document["logical_motif_domain"])
        validate_topology_augmentation(self.document)
        self.assertEqual(len(augmentation_sha256(self.document)), 64)

    def test_rerun_must_match_existing_record_groups_and_lexeme_digests(self):
        bad_groups = list(self.result.motif_atom_groups)
        bad_groups[0], bad_groups[1] = bad_groups[1], bad_groups[0]
        with self.assertRaises(TopologyAugmentationError):
            self._build(expected_motif_atom_indices=bad_groups)
        bad_digests = list(self.expected_digests)
        bad_digests[0] = digest("wrong-fragment")
        with self.assertRaises(TopologyAugmentationError):
            self._build(expected_motif_lexeme_sha256=bad_digests)

    def _build(self, **overrides):
        arguments = {
            "linearization_result": self.result,
            "member_id": "pcqm4mv2:000000123",
            "base_record_content_sha256": digest("production-record"),
            "linearizer_spec_sha256": digest("linearizer-spec"),
            "expected_motif_atom_indices": self.result.motif_atom_groups,
            "expected_motif_lexeme_sha256": self.expected_digests,
            "source_atom_count": 12,
            "model_to_source_atom_index": self.mapping,
        }
        arguments.update(overrides)
        return build_topology_augmentation(**arguments)

    def test_endpoint_inverse_and_no_recompute_mutations_fail_closed(self):
        broken_endpoint = copy.deepcopy(self.document)
        broken_endpoint["logical_motif_domain"]["cross_motif_bonds"][0]["left"]["model_atom_index"] += 1
        broken_inverse = copy.deepcopy(self.document)
        broken_inverse["logical_motif_domain"]["canonical_to_logical_motif_index"] = [0, 1, 2, 3]
        recomputed = copy.deepcopy(self.document)
        recomputed["provenance"]["geometry_or_e3fp_recomputed"] = True
        for mutation in (broken_endpoint, broken_inverse, recomputed):
            with self.subTest(mutation=mutation):
                with self.assertRaises(TopologyAugmentationError):
                    validate_topology_augmentation(mutation)

    def test_source_atom_mapping_must_be_explicit_strict_and_in_range(self):
        for mapping in (
            (0, 1, 2, 3, 4, 5, 6, 8, 7, 11),
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 12),
        ):
            with self.subTest(mapping=mapping):
                with self.assertRaises(TopologyAugmentationError):
                    self._build(model_to_source_atom_index=mapping)


if __name__ == "__main__":
    unittest.main()
