from __future__ import annotations

import copy
import hashlib
import unittest

from rdkit import Chem

from most_t5_next.r1.adapter.mol_linearizer import linearize_mol
from most_t5_next.r1.tokenizer.stereo_free_anchored_motif_surface_v1 import (
    AnchoredMotifSurfaceError,
    build_stereo_free_anchored_surface,
    build_stereo_free_anchored_surface_from_persisted_pair,
    canonicalize_legacy_fragment,
    decode_rendering,
    project_legacy_fragment,
    reconstruct_stereo_free_molecule,
    restore_canonical_legacy_fragment,
    stereo_free_canonical_smiles,
    surface_document,
    validate_stereo_free_molecule_round_trip,
)
from most_t5_next.r1.adapter.p1_topology_augmentation_v1 import (
    DOCUMENT_KIND,
    PROJECTION_POLICY,
    SCHEMA_VERSION,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _topology_document() -> dict[str, object]:
    pure = ["[C()()N]", "[O()]", "[S]", "[Cl]"]
    return {
        "schema_version": SCHEMA_VERSION,
        "document_kind": DOCUMENT_KIND,
        "training_admission": False,
        "member": {
            "member_id": "fixture:anchored-surface",
            "base_record_content_sha256": _sha("record"),
        },
        "provenance": {
            "linearizer_schema_version": "r1-molecule-native-linearizer/v1",
            "linearizer_spec_sha256": _sha("linearizer"),
            "projection_policy": PROJECTION_POLICY,
            "geometry_or_e3fp_recomputed": False,
        },
        "atom_universe": {
            "model_atom_count": 5,
            "source_atom_count": 7,
            "model_to_source_atom_index": [0, 1, 2, 4, 6],
            "atom_is_attachment": [True, True, True, True, False],
        },
        "logical_motif_domain": {
            "logical_motif_count": 4,
            "logical_to_canonical_motif_index": [0, 1, 2, 3],
            "canonical_to_logical_motif_index": [0, 1, 2, 3],
            "component_logical_motif_ranges": [[0, 3], [3, 4]],
            "motif_atom_indices": [[0, 1], [2], [3], [4]],
            "exact_motif_lexeme_sha256": [_sha(f"exact-{i}") for i in range(4)],
            "pure_motif_token": pure,
            "pure_motif_token_sha256": [_sha(value) for value in pure],
            "motif_slot_anchor_ids": [[0, 1], [0], [1], []],
            "motif_slot_atom_indices": [[0, 1], [2], [3], []],
            "motif_slot_source_atom_indices": [[0, 1], [2], [4], []],
            "cross_motif_bonds": [
                {
                    "edge_id": 0,
                    "source_anchor_id": 0,
                    "left": {
                        "logical_motif_index": 0,
                        "slot_ordinal": 0,
                        "model_atom_index": 0,
                        "source_atom_index": 0,
                    },
                    "right": {
                        "logical_motif_index": 1,
                        "slot_ordinal": 0,
                        "model_atom_index": 2,
                        "source_atom_index": 2,
                    },
                    "bond_type": "single",
                    "source_bond_type": "SINGLE",
                },
                {
                    "edge_id": 1,
                    "source_anchor_id": 1,
                    "left": {
                        "logical_motif_index": 0,
                        "slot_ordinal": 1,
                        "model_atom_index": 1,
                        "source_atom_index": 1,
                    },
                    "right": {
                        "logical_motif_index": 2,
                        "slot_ordinal": 0,
                        "model_atom_index": 3,
                        "source_atom_index": 4,
                    },
                    "bond_type": "single",
                    "source_bond_type": "SINGLE",
                },
            ],
        },
    }


def _surface_from_molecule(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise AssertionError("fixture SMILES did not parse")
    result = linearize_mol(molecule)
    canonical_to_logical = {
        canonical: logical
        for logical, canonical in enumerate(result.metadata.fragment_motif_ids)
    }
    anchor_by_id = {
        anchor.anchor_id: anchor for anchor in result.metadata.cross_motif_bonds
    }
    slot_atoms = []
    lexical_slot_by_anchor = {}
    for logical, (fragment, canonical_motif) in enumerate(
        zip(result.fragment_sequence, result.metadata.fragment_motif_ids)
    ):
        _pure, anchor_ids = project_legacy_fragment(fragment)
        atoms = []
        for lexical_slot, anchor_id in enumerate(anchor_ids):
            anchor = anchor_by_id[anchor_id]
            if canonical_motif == anchor.motif_a:
                atom = anchor.atom_a
            elif canonical_motif == anchor.motif_b:
                atom = anchor.atom_b
            else:
                raise AssertionError("fixture anchor is outside its motif")
            atoms.append(atom)
            lexical_slot_by_anchor[(logical, anchor_id)] = lexical_slot
        slot_atoms.append(tuple(atoms))
    edge_candidates = []
    for anchor in result.metadata.cross_motif_bonds:
        left = (
            canonical_to_logical[anchor.motif_a],
            lexical_slot_by_anchor[(canonical_to_logical[anchor.motif_a], anchor.anchor_id)],
            anchor.atom_a,
        )
        right = (
            canonical_to_logical[anchor.motif_b],
            lexical_slot_by_anchor[(canonical_to_logical[anchor.motif_b], anchor.anchor_id)],
            anchor.atom_b,
        )
        left, right = sorted((left, right))
        edge_candidates.append((left, right))
    edge_candidates.sort()
    bonds = tuple(
        {
            "edge_id": edge_id,
            "bond_type": "single",
            "left": {
                "logical_motif_index": left[0],
                "slot_ordinal": left[1],
                "atom_index": left[2],
            },
            "right": {
                "logical_motif_index": right[0],
                "slot_ordinal": right[1],
                "atom_index": right[2],
            },
        }
        for edge_id, (left, right) in enumerate(edge_candidates)
    )
    attachment_atoms = {endpoint[2] for row in edge_candidates for endpoint in row}
    surface = build_stereo_free_anchored_surface_from_persisted_pair(
        member_id=f"fixture:{smiles}",
        source_atom_count=molecule.GetNumAtoms(),
        model_to_source_atom_index=tuple(range(molecule.GetNumAtoms())),
        atom_is_attachment=tuple(
            atom in attachment_atoms for atom in range(molecule.GetNumAtoms())
        ),
        motif_atom_indices=result.motif_atom_groups,
        exact_motif_lexemes=result.fragment_sequence,
        motif_slot_atom_indices=tuple(slot_atoms),
        cross_motif_bonds=bonds,
    )
    return molecule, surface


class StereoFreeAnchoredMotifSurfaceV1Test(unittest.TestCase):
    def test_both_boundary_renderings_decode_to_one_logical_surface(self) -> None:
        surface = build_stereo_free_anchored_surface(_topology_document())
        explicit = surface.render("explicit")
        implicit = surface.render("implicit")
        self.assertEqual(decode_rendering(explicit.tokens, "explicit")[0], decode_rendering(implicit.tokens, "implicit")[0])
        self.assertEqual(explicit.motif_to_carrier, (3, 6, 9, 12))
        self.assertEqual(implicit.motif_to_carrier, (2, 4, 6, 8))
        self.assertEqual(explicit.anchor_token_positions[0], (1, 2))
        self.assertEqual(implicit.anchor_token_positions[0], (0, 1))
        self.assertEqual(explicit.component_token_ranges, ((0, 10), (11, 13)))
        self.assertEqual(implicit.component_token_ranges, ((0, 7), (8, 9)))
        self.assertEqual(surface.phrases[0].motif_atom_indices, (0, 1))
        self.assertEqual(tuple(anchor.model_atom_index for anchor in surface.phrases[0].anchors), (0, 1))

    def test_document_digest_and_legacy_projection_are_fixed(self) -> None:
        surface = build_stereo_free_anchored_surface(_topology_document())
        document = surface_document(surface)
        self.assertEqual(document["artifact_sha256"], surface.artifact_sha256)
        self.assertFalse(document["geometry_or_e3fp_recomputed"])
        self.assertFalse(document["graphports_exposed_to_model"])
        restored = restore_canonical_legacy_fragment("[C()()N]", (0, 1))
        self.assertEqual(restored, "C(<0*>)(<1*>)N")
        self.assertEqual(project_legacy_fragment(restored), ("[C()()N]", (0, 1)))
        prefix_suffix = restore_canonical_legacy_fragment("[CN]", (3, 8))
        self.assertEqual(prefix_suffix, "<3*>CN<8*>")
        self.assertEqual(project_legacy_fragment(prefix_suffix), ("[CN]", (3, 8)))

    def test_graph_canonicalization_collapses_traversal_aliases(self) -> None:
        self.assertEqual(
            canonicalize_legacy_fragment("<0*>C=N"),
            canonicalize_legacy_fragment("N=C<0*>"),
        )
        self.assertEqual(
            canonicalize_legacy_fragment("c1cc(<7*>)ccc1")[0],
            canonicalize_legacy_fragment("c1ccc(<7*>)cc1")[0],
        )
        pure, anchors = canonicalize_legacy_fragment("<9*>C(<3*>)N")
        self.assertEqual(pure, "[C()N]")
        self.assertEqual(restore_canonical_legacy_fragment(pure, anchors), "<9*>C(<3*>)N")
        with self.assertRaisesRegex(AnchoredMotifSurfaceError, "noncanonical"):
            decode_rendering(("<0*>", "[N=C()]"), "implicit")

    def test_stereo_leak_and_non_single_cross_bond_are_rejected(self) -> None:
        stereo = _topology_document()
        stereo["logical_motif_domain"]["pure_motif_token"][0] = "[C@H()N]"
        stereo["logical_motif_domain"]["pure_motif_token_sha256"][0] = _sha("[C@H()N]")
        with self.assertRaisesRegex(AnchoredMotifSurfaceError, "stereochemical"):
            build_stereo_free_anchored_surface(stereo)

        double = _topology_document()
        double["logical_motif_domain"]["cross_motif_bonds"][0]["bond_type"] = "double"
        with self.assertRaisesRegex(AnchoredMotifSurfaceError, "SINGLE"):
            build_stereo_free_anchored_surface(double)

    def test_topology_validator_rejects_wrong_endpoint_before_projection(self) -> None:
        broken = copy.deepcopy(_topology_document())
        broken["logical_motif_domain"]["cross_motif_bonds"][0]["left"]["model_atom_index"] = 1
        with self.assertRaises(Exception):
            build_stereo_free_anchored_surface(broken)

    def test_decoding_rejects_broken_phrase_boundaries_and_duplicate_anchor(self) -> None:
        with self.assertRaises(AnchoredMotifSurfaceError):
            decode_rendering(("<MOST:MOTIF>", "<0*>", "[C]", "<1*>", "[N]"), "explicit")
        with self.assertRaises(AnchoredMotifSurfaceError):
            decode_rendering(("<0*>", "<0*>", "[C]"), "implicit")

    def test_persisted_pair_join_remaps_source_anchors_to_canonical_edges(self) -> None:
        surface = build_stereo_free_anchored_surface_from_persisted_pair(
            member_id="fixture:persisted-pair",
            source_atom_count=4,
            model_to_source_atom_index=(0, 1, 2, 3),
            atom_is_attachment=(True, True, True, True),
            motif_atom_indices=((0, 1), (2,), (3,)),
            # Source IDs deliberately have the opposite ordering from edge IDs.
            exact_motif_lexemes=("C(<1*>)(<0*>)N", "O<1*>", "S<0*>"),
            motif_slot_atom_indices=((0, 1), (2,), (3,)),
            cross_motif_bonds=(
                {
                    "edge_id": 0,
                    "bond_type": "single",
                    "left": {"logical_motif_index": 0, "slot_ordinal": 1, "atom_index": 1},
                    "right": {"logical_motif_index": 2, "slot_ordinal": 0, "atom_index": 3},
                },
                {
                    "edge_id": 1,
                    "bond_type": "single",
                    "left": {"logical_motif_index": 0, "slot_ordinal": 0, "atom_index": 0},
                    "right": {"logical_motif_index": 1, "slot_ordinal": 0, "atom_index": 2},
                },
            ),
        )
        self.assertEqual(
            tuple(anchor.anchor_id for anchor in surface.phrases[0].anchors),
            (1, 0),
        )
        self.assertEqual(surface.component_motif_ranges, ((0, 3),))
        self.assertEqual(
            decode_rendering(surface.render("implicit").tokens, "implicit")[0][0],
            ((1, 0), "[C()N]"),
        )

    def test_complete_molecule_round_trip_and_atom_renumbering(self) -> None:
        molecule, surface = _surface_from_molecule("O/C=N/S.Cl")
        expected = stereo_free_canonical_smiles(molecule)
        self.assertEqual(validate_stereo_free_molecule_round_trip(surface, molecule), expected)
        self.assertEqual(
            stereo_free_canonical_smiles(reconstruct_stereo_free_molecule(surface)),
            expected,
        )

        renumbered = Chem.RenumberAtoms(
            molecule, tuple(reversed(range(molecule.GetNumAtoms())))
        )
        _renumbered_molecule, renumbered_surface = _surface_from_molecule(
            Chem.MolToSmiles(renumbered, canonical=False, isomericSmiles=True)
        )
        self.assertEqual(
            sorted(phrase.pure_motif for phrase in surface.phrases),
            sorted(phrase.pure_motif for phrase in renumbered_surface.phrases),
        )
        self.assertEqual(
            validate_stereo_free_molecule_round_trip(
                renumbered_surface, _renumbered_molecule
            ),
            expected,
        )

    def test_symmetric_ring_ports_keep_external_connections(self) -> None:
        molecule, surface = _surface_from_molecule("Clc1ccc(Br)cc1")
        self.assertEqual(validate_stereo_free_molecule_round_trip(surface, molecule), "Clc1ccc(Br)cc1")
        ring_phrases = [phrase for phrase in surface.phrases if len(phrase.motif_atom_indices) == 6]
        self.assertEqual(len(ring_phrases), 1)
        self.assertEqual(len(ring_phrases[0].anchors), 2)


if __name__ == "__main__":
    unittest.main()
