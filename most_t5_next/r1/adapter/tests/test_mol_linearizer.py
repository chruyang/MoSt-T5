"""Hermetic tests for the R1 molecule-native motif linearizer.

All test molecules are assembled with RDKit's graph API.  The tests do not
parse SMILES and do not need a dataset, network access, or historical code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from rdkit import Chem

from most_t5_next.r1.adapter.mol_linearizer import linearize_mol


def build_synthetic_molecule() -> Chem.Mol:
    """Build ring + double-bond + singleton + disconnected-component coverage."""

    rw_mol = Chem.RWMol()
    aromatic_atoms: list[int] = []
    for _ in range(6):
        atom = Chem.Atom(6)
        atom.SetIsAromatic(True)
        aromatic_atoms.append(rw_mol.AddAtom(atom))
    for atom_idx in range(6):
        rw_mol.AddBond(atom_idx, (atom_idx + 1) % 6, Chem.rdchem.BondType.AROMATIC)
        rw_mol.GetBondBetweenAtoms(atom_idx, (atom_idx + 1) % 6).SetIsAromatic(True)

    carbonyl_carbon = rw_mol.AddAtom(Chem.Atom(6))
    carbonyl_oxygen = rw_mol.AddAtom(Chem.Atom(8))
    terminal_nitrogen = rw_mol.AddAtom(Chem.Atom(7))
    disconnected_chlorine = rw_mol.AddAtom(Chem.Atom(17))
    rw_mol.AddBond(0, carbonyl_carbon, Chem.rdchem.BondType.SINGLE)
    rw_mol.AddBond(carbonyl_carbon, carbonyl_oxygen, Chem.rdchem.BondType.DOUBLE)
    rw_mol.AddBond(carbonyl_carbon, terminal_nitrogen, Chem.rdchem.BondType.SINGLE)

    mol = rw_mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def molecule_snapshot(mol: Chem.Mol) -> tuple[object, ...]:
    """Structural snapshot sufficient to catch any source-Mol mutation."""

    atom_state = tuple(
        (
            atom.GetIdx(),
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetIsotope(),
            atom.GetNoImplicit(),
            atom.GetNumExplicitHs(),
            atom.GetIsAromatic(),
            int(atom.GetChiralTag()),
            atom.GetProp("test_atom_marker") if atom.HasProp("test_atom_marker") else None,
        )
        for atom in mol.GetAtoms()
    )
    bond_state = tuple(
        (
            bond.GetIdx(),
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            str(bond.GetBondType()),
            bond.GetIsAromatic(),
            bond.GetIsConjugated(),
            int(bond.GetStereo()),
        )
        for bond in mol.GetBonds()
    )
    conformer_state: tuple[tuple[float, float, float], ...] = ()
    if mol.GetNumConformers():
        conformer = mol.GetConformer()
        conformer_state = tuple(
            (position.x, position.y, position.z)
            for position in (conformer.GetAtomPosition(atom_idx) for atom_idx in range(mol.GetNumAtoms()))
        )
    return (
        mol.GetNumAtoms(),
        mol.GetNumBonds(),
        atom_state,
        bond_state,
        conformer_state,
        mol.GetProp("test_molecule_marker") if mol.HasProp("test_molecule_marker") else None,
    )


_HASH_SEED_SCRIPT = r'''
import json
from rdkit import Chem
from most_t5_next.r1.adapter.mol_linearizer import linearize_mol

rw_mol = Chem.RWMol()
for _ in range(6):
    atom = Chem.Atom(6)
    atom.SetIsAromatic(True)
    rw_mol.AddAtom(atom)
for atom_idx in range(6):
    rw_mol.AddBond(atom_idx, (atom_idx + 1) % 6, Chem.rdchem.BondType.AROMATIC)
    rw_mol.GetBondBetweenAtoms(atom_idx, (atom_idx + 1) % 6).SetIsAromatic(True)
carbonyl_carbon = rw_mol.AddAtom(Chem.Atom(6))
carbonyl_oxygen = rw_mol.AddAtom(Chem.Atom(8))
terminal_nitrogen = rw_mol.AddAtom(Chem.Atom(7))
rw_mol.AddBond(0, carbonyl_carbon, Chem.rdchem.BondType.SINGLE)
rw_mol.AddBond(carbonyl_carbon, carbonyl_oxygen, Chem.rdchem.BondType.DOUBLE)
rw_mol.AddBond(carbonyl_carbon, terminal_nitrogen, Chem.rdchem.BondType.SINGLE)
mol = rw_mol.GetMol()
Chem.SanitizeMol(mol)
result = linearize_mol(mol)
print(json.dumps({
    "fragments": result.fragment_sequence,
    "groups": result.motif_atom_groups,
    "motif_ids": result.metadata.fragment_motif_ids,
    "anchors": [
        (anchor.anchor_id, anchor.motif_a, anchor.motif_b, anchor.atom_a, anchor.atom_b, anchor.bond_type)
        for anchor in result.metadata.cross_motif_bonds
    ],
}, sort_keys=True))
'''


class MoleculeNativeLinearizerTests(unittest.TestCase):
    def test_mapping_is_identical_for_copied_molecules(self) -> None:
        source = build_synthetic_molecule()
        copied = Chem.Mol(source)

        first = linearize_mol(source)
        second = linearize_mol(copied)

        self.assertEqual(first, second)
        self.assertEqual(first.metadata.canonical_motif_atom_groups, ((0, 1, 2, 3, 4, 5), (6, 7), (8,), (9,)))
        self.assertEqual(first.motif_atom_groups, ((6, 7), (0, 1, 2, 3, 4, 5), (8,), (9,)))
        self.assertEqual(first.metadata.fragment_motif_ids, (1, 0, 2, 3))
        self.assertEqual(first.metadata.component_fragment_ranges, ((0, 3), (3, 4)))
        self.assertEqual(len(first.metadata.cross_motif_bonds), 2)
        self.assertEqual(
            sorted(anchor_id for fragment in first.fragment_sequence for anchor_id in (0, 1) if f"<{anchor_id}*>" in fragment),
            [0, 0, 1, 1],
        )
        self.assertIn("[.]", first.fragment_string)

    def test_motif_groups_are_a_complete_original_index_partition(self) -> None:
        mol = build_synthetic_molecule()
        result = linearize_mol(mol)

        flattened = [atom_idx for group in result.motif_atom_groups for atom_idx in group]
        self.assertEqual(sorted(flattened), list(range(mol.GetNumAtoms())))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(all(tuple(sorted(group)) == group for group in result.motif_atom_groups))
        self.assertEqual(len(result.fragment_sequence), len(result.motif_atom_groups))
        self.assertTrue(
            all(
                0 <= anchor.atom_a < mol.GetNumAtoms()
                and 0 <= anchor.atom_b < mol.GetNumAtoms()
                for anchor in result.metadata.cross_motif_bonds
            )
        )

    def test_source_molecule_is_not_modified_and_smiles_parser_is_not_used(self) -> None:
        mol = build_synthetic_molecule()
        mol.SetProp("test_molecule_marker", "preserve-me")
        mol.GetAtomWithIdx(0).SetProp("test_atom_marker", "preserve-me")
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for atom_idx in range(mol.GetNumAtoms()):
            conformer.SetAtomPosition(atom_idx, (float(atom_idx), 1.5, -2.0))
        mol.AddConformer(conformer, assignId=True)
        before = molecule_snapshot(mol)

        with mock.patch.object(Chem, "MolFromSmiles", side_effect=AssertionError("must not parse SMILES")):
            linearize_mol(mol)

        self.assertEqual(molecule_snapshot(mol), before)

    def test_output_is_hash_seed_invariant_across_fresh_processes(self) -> None:
        workspace_root = Path(__file__).resolve().parents[4]
        outputs: list[str] = []
        for hash_seed in ("1", "47", "20260731"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            environment["PYTHONPATH"] = str(workspace_root)
            completed = subprocess.run(
                [sys.executable, "-c", _HASH_SEED_SCRIPT],
                cwd=workspace_root,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )
            outputs.append(completed.stdout.strip())

        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])
        self.assertEqual(json.loads(outputs[0])["groups"], [[6, 7], [0, 1, 2, 3, 4, 5], [8]])


if __name__ == "__main__":
    unittest.main()
