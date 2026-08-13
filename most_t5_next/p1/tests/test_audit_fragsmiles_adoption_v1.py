from __future__ import annotations

from pathlib import Path
import unittest

from rdkit import Chem

from most_t5_next.p1.audit_fragsmiles_adoption_v1 import (
    audit_molecules,
    encode_with_sidecar,
    stereo_address_signature,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CHEMICALGOF_ROOT = REPO_ROOT / "reference_repos" / "chemicalgof-master"


class FragSmilesAdoptionAuditTests(unittest.TestCase):
    def test_sidecar_matches_official_multifragment_surface(self):
        mol = Chem.MolFromSmiles("CC(=O)NCCc1ccccc1")
        record = encode_with_sidecar(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        self.assertEqual(record.fragsmiles, "C.C=O.N.C.C.<0>c1ccccc1")
        self.assertEqual(
            sorted(atom for row in record.fragments for atom in row.source_atom_indices),
            list(range(mol.GetNumAtoms())),
        )
        self.assertEqual(len(record.connectors), 5)
        for connector in record.connectors:
            left = record.fragments[connector.left_fragment_index]
            right = record.fragments[connector.right_fragment_index]
            self.assertEqual(
                connector.left_source_atom_index,
                left.source_atom_indices[connector.left_local_atom_index],
            )
            self.assertEqual(
                connector.right_source_atom_index,
                right.source_atom_indices[connector.right_local_atom_index],
            )

    def test_two_connectors_on_same_ring_are_preserved(self):
        mol = Chem.MolFromSmiles("Clc1ccc(Br)cc1")
        record = encode_with_sidecar(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        self.assertEqual(len(record.fragments), 3)
        self.assertEqual(len(record.connectors), 2)
        ring = next(row for row in record.fragments if row.fragment_smiles == "c1ccccc1")
        attached = {
            endpoint
            for connector in record.connectors
            for fragment, endpoint in (
                (connector.left_fragment_index, connector.left_source_atom_index),
                (connector.right_fragment_index, connector.right_source_atom_index),
            )
            if fragment == ring.sequence_fragment_index
        }
        self.assertEqual(len(attached), 2)

    def test_formal_charge_round_trip_and_renumbering(self):
        mol = Chem.MolFromSmiles("C[N+](=O)[O-]")
        report = audit_molecules((mol,), chemicalgof_root=CHEMICALGOF_ROOT)
        self.assertEqual(report["counts"]["failures"], 0)
        self.assertEqual(report["counts"]["identity_round_trip_pass"], 1)
        self.assertEqual(report["counts"]["connectivity_round_trip_pass"], 1)
        self.assertEqual(report["counts"]["sidecar_atom_partition_pass"], 1)
        self.assertEqual(report["counts"]["renumbering_surface_invariant_pass"], 1)

    def test_chiral_fixture_is_audited_without_silent_relaxation(self):
        mol = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
        report = audit_molecules((mol,), chemicalgof_root=CHEMICALGOF_ROOT)
        self.assertEqual(report["counts"]["encoded"], 1)
        self.assertEqual(report["counts"]["identity_round_trip_pass"], 1)
        record = encode_with_sidecar(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        self.assertEqual(len(record.stereo_identity.atom_centers), 1)
        self.assertFalse(report["training_admission"])

    def test_connectivity_policy_is_explicit_and_lossless_for_e_z_input(self):
        mol = Chem.MolFromSmiles("C/C=N/O")
        report = audit_molecules(
            (mol,),
            chemicalgof_root=CHEMICALGOF_ROOT,
            stereo_policy="connectivity_only",
        )
        self.assertEqual(report["representation"]["stereo_policy"], "connectivity_only")
        self.assertEqual(report["counts"]["failures"], 0)
        self.assertEqual(report["counts"]["connectivity_round_trip_pass"], 1)
        record = encode_with_sidecar(
            mol,
            chemicalgof_root=CHEMICALGOF_ROOT,
            stereo_policy="connectivity_only",
        )
        self.assertEqual(len(record.stereo_identity.double_bonds), 1)
        reverse = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        reverse_record = encode_with_sidecar(
            reverse,
            chemicalgof_root=CHEMICALGOF_ROOT,
            stereo_policy="connectivity_only",
        )
        self.assertEqual(
            stereo_address_signature(record), stereo_address_signature(reverse_record)
        )

    def test_disconnected_components_use_one_explicit_boundary(self):
        mol = Chem.MolFromSmiles("N.C=C1C=CC(=O)O1")
        record = encode_with_sidecar(
            mol,
            chemicalgof_root=CHEMICALGOF_ROOT,
            stereo_policy="connectivity_only",
        )
        self.assertEqual(len(record.component_surfaces), 2)
        self.assertEqual(record.tokens.count("<COMP>"), 1)
        self.assertEqual(record.fragsmiles.count("<COMP>"), 1)
        report = audit_molecules(
            (mol,),
            chemicalgof_root=CHEMICALGOF_ROOT,
            stereo_policy="connectivity_only",
        )
        self.assertEqual(report["counts"]["failures"], 0)
        self.assertEqual(report["counts"]["connectivity_round_trip_pass"], 1)

    def test_stereo_defining_hydrogen_is_explicit_in_audit_sidecar(self):
        mol = Chem.MolFromSmiles("[H]/N=C(\\N)C#N")
        record = encode_with_sidecar(
            mol,
            chemicalgof_root=CHEMICALGOF_ROOT,
            stereo_policy="connectivity_only",
        )
        self.assertEqual(len(record.stereo_identity.double_bonds), 1)
        bond = record.stereo_identity.double_bonds[0]
        self.assertIn(
            None,
            (bond.begin_support_atom_index, bond.end_support_atom_index),
        )

    def test_complex_ring_tetrahedral_state_survives_as_addressed_sidecar(self):
        mol = Chem.MolFromSmiles("C[C@H]1[C@H]2C[C@@H]1C2")
        record = encode_with_sidecar(
            mol,
            chemicalgof_root=CHEMICALGOF_ROOT,
            stereo_policy="connectivity_only",
        )
        self.assertGreaterEqual(len(record.stereo_identity.atom_centers), 2)
        self.assertTrue(
            all(
                center.fragment_index is not None
                and center.fragment_local_atom_index is not None
                for center in record.stereo_identity.atom_centers
            )
        )


if __name__ == "__main__":
    unittest.main()
