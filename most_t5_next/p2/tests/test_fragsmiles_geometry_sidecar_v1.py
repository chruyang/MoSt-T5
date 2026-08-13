from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import unittest

from rdkit import Chem

from most_t5_next.p1.fragsmiles_compact_stereo_codec_v1 import strict_round_trip
from most_t5_next.p1.fragsmiles_lossless_fallback_v1 import (
    encode_lossless_fallback,
)
from most_t5_next.p1.fragsmiles_macro_fallback_surface_v1 import (
    CONNECTOR_END,
    encode_compact_model_surface,
)
from most_t5_next.p2.fragsmiles_geometry_sidecar_v1 import (
    AtomAxisAddress,
    FragSmilesGeometrySidecar,
    FragSmilesGeometrySidecarError,
    build_compact_geometry_sidecar,
    build_fallback_geometry_sidecar,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CHEMICALGOF_ROOT = REPO_ROOT / "reference_repos" / "chemicalgof-master"


def _compact(smiles: str, macros=()):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    authoritative = strict_round_trip(mol, chemicalgof_root=CHEMICALGOF_ROOT)
    model = encode_compact_model_surface(
        mol,
        authoritative,
        macros,
        chemicalgof_root=CHEMICALGOF_ROOT,
    )
    atom_axes = tuple(
        AtomAxisAddress(
            source_sdf_atom_index=index,
            projected_atom_index=index,
            e3fp_row=index,
        )
        for index in range(mol.GetNumAtoms())
    )
    return mol, authoritative, model, build_compact_geometry_sidecar(
        authoritative, model, macros, atom_axes
    )


def _fallback_axes(fallback, *, include_source_atoms=None):
    selected = [
        row
        for row in fallback.atom_addresses
        if include_source_atoms is None
        or row.source_atom_index in include_source_atoms
    ]
    return tuple(
        AtomAxisAddress(
            source_sdf_atom_index=row.source_atom_index,
            projected_atom_index=row.projected_atom_index,
            e3fp_row=e3fp_row,
        )
        for e3fp_row, row in enumerate(selected)
    )


class FragSmilesGeometrySidecarV1Tests(unittest.TestCase):
    def test_macro_and_local_fallback_share_one_atom_and_carrier_abi(self) -> None:
        macros = (
            {"fragment_smiles": "C", "surface_token": "<MOST:FM:000000>"},
            {
                "fragment_smiles": "c1ccccc1",
                "surface_token": "<MOST:FM:000001>",
            },
        )
        mol, _authoritative, model, sidecar = _compact(
            "CC(=O)NCCc1ccccc1", macros
        )
        self.assertEqual(sidecar.mode, "compact")
        self.assertEqual(sidecar.model_tokens[0], "<bom>")
        self.assertEqual(sidecar.model_tokens[-1], "<eom>")
        self.assertEqual(sidecar.token_roles[0], "molecule_boundary")
        self.assertEqual(sidecar.token_roles[-1], "molecule_boundary")
        self.assertEqual(len(sidecar.atoms), mol.GetNumAtoms())
        self.assertEqual(
            {row.representation for row in sidecar.fragments},
            {"macro", "fragment_lexer"},
        )
        self.assertEqual(
            tuple(row.carrier_token_index for row in sidecar.fragments),
            tuple(row.carrier_token_index + 1 for row in model.fragment_phrases),
        )
        self.assertEqual(
            sorted(atom.e3fp_row for atom in sidecar.atoms),
            list(range(mol.GetNumAtoms())),
        )
        self.assertTrue(all(row.e3fp_rows for row in sidecar.fragments))

    def test_explicit_and_implicit_endpoints_have_stable_carriers(self) -> None:
        _mol, _authoritative, _model, sidecar = _compact("CC1CCC(CC1)C")
        endpoints = [
            endpoint
            for connector in sidecar.connectors
            for endpoint in (connector.left, connector.right)
        ]
        self.assertEqual(sum(row.explicit_in_surface for row in endpoints), 2)
        self.assertEqual(sum(not row.explicit_in_surface for row in endpoints), 2)
        for endpoint in endpoints:
            fragment = sidecar.fragments[endpoint.fragment_index]
            if endpoint.explicit_in_surface:
                self.assertEqual(
                    sidecar.model_tokens[endpoint.carrier_token_index], CONNECTOR_END
                )
            else:
                self.assertEqual(
                    endpoint.carrier_token_index, fragment.carrier_token_index
                )
            atom = sidecar.atoms[endpoint.e3fp_row]
            self.assertTrue(atom.is_attachment)
            self.assertEqual(atom.fragment_index, endpoint.fragment_index)
            self.assertEqual(
                atom.fragment_local_atom_index, endpoint.fragment_local_atom_index
            )

    def test_branching_multi_endpoint_fragment_retains_each_edge(self) -> None:
        _mol, authoritative, _model, sidecar = _compact("CC(C)(C)C")
        self.assertEqual(len(sidecar.connectors), len(authoritative.connectivity_record.connectors))
        center = max(
            range(len(sidecar.fragments)),
            key=lambda index: sum(
                endpoint.fragment_index == index
                for connector in sidecar.connectors
                for endpoint in (connector.left, connector.right)
            ),
        )
        center_endpoints = [
            endpoint
            for connector in sidecar.connectors
            for endpoint in (connector.left, connector.right)
            if endpoint.fragment_index == center
        ]
        self.assertEqual(len(center_endpoints), 4)
        self.assertEqual({row.fragment_local_atom_index for row in center_endpoints}, {0})

    def test_whole_molecule_fallback_is_zero_motif_molecule_envelope(self) -> None:
        mol = Chem.MolFromSmiles("CC.O")
        assert mol is not None
        fallback = encode_lossless_fallback(mol)
        sidecar = build_fallback_geometry_sidecar(
            fallback, _fallback_axes(fallback)
        )
        self.assertEqual(sidecar.mode, "whole_molecule_fallback")
        self.assertEqual(sidecar.component_count, 2)
        self.assertEqual(sidecar.fragments, ())
        self.assertEqual(sidecar.connectors, ())
        self.assertEqual(sidecar.model_tokens[0], "<bom>")
        self.assertEqual(sidecar.model_tokens[-1], "<eom>")
        self.assertEqual(sidecar.model_tokens[1:-1], fallback.tokens)
        self.assertEqual(
            sidecar.molecule_carrier_token_index, len(sidecar.model_tokens) - 1
        )
        self.assertTrue(all(index == -1 for index in sidecar.token_to_fragment))
        self.assertFalse(any(token == "<MOST:FB:MOL>" for token in sidecar.model_tokens))
        self.assertEqual(
            sorted(atom.e3fp_row for atom in sidecar.atoms), [0, 1, 2]
        )
        self.assertTrue(all(atom.fragment_index is None for atom in sidecar.atoms))
        self.assertTrue(all(atom.has_e3fp_row for atom in sidecar.atoms))
        self.assertTrue(
            all(
                sidecar.token_roles[index] == "atom_glyph"
                for atom in sidecar.atoms
                for index in range(atom.token_start, atom.token_stop)
            )
        )

    def test_fallback_geometry_rows_follow_explicit_external_atom_axis(self) -> None:
        mol = Chem.MolFromSmiles("[2H]O")
        assert mol is not None
        fallback = encode_lossless_fallback(mol)
        sidecar = build_fallback_geometry_sidecar(
            fallback, _fallback_axes(fallback, include_source_atoms={1})
        )
        self.assertEqual(len(sidecar.atoms), 2)
        hydrogen = next(
            row for row in sidecar.atoms if row.source_sdf_atom_index == 0
        )
        oxygen = next(
            row for row in sidecar.atoms if row.source_sdf_atom_index == 1
        )
        self.assertFalse(hydrogen.has_e3fp_row)
        self.assertIsNone(hydrogen.e3fp_row)
        self.assertTrue(oxygen.has_e3fp_row)
        self.assertEqual(oxygen.e3fp_row, 0)
        self.assertEqual(sidecar.fragments, ())

        explicit_h_sidecar = build_fallback_geometry_sidecar(
            fallback, _fallback_axes(fallback)
        )
        explicit_h = next(
            row
            for row in explicit_h_sidecar.atoms
            if row.source_sdf_atom_index == 0
        )
        self.assertTrue(explicit_h.has_e3fp_row)
        self.assertEqual(explicit_h.e3fp_row, 0)

    def test_disconnected_compact_surface_preserves_component_ownership(self) -> None:
        _mol, _authoritative, _model, sidecar = _compact("CC.O")
        self.assertEqual(sidecar.component_count, 2)
        self.assertEqual(
            [row.component_index for row in sidecar.fragments], [0, 0, 1]
        )
        self.assertEqual(sidecar.connectors[0].left.fragment_index, 0)
        self.assertEqual(sidecar.connectors[0].right.fragment_index, 1)
        self.assertTrue(
            any(role == "component" for role in sidecar.token_roles)
        )

    def test_atom_renumbering_preserves_surface_and_local_connector_contract(self) -> None:
        smiles = "CC1CCC(CC1)C"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        _, base_authoritative, _base_model, base = _compact(smiles)
        rng = random.Random(20260812)
        for _ in range(8):
            order = list(range(mol.GetNumAtoms()))
            rng.shuffle(order)
            renumbered = Chem.RenumberAtoms(mol, order)
            authoritative = strict_round_trip(
                renumbered, chemicalgof_root=CHEMICALGOF_ROOT
            )
            model = encode_compact_model_surface(
                renumbered,
                authoritative,
                (),
                chemicalgof_root=CHEMICALGOF_ROOT,
            )
            inverse = [0] * len(order)
            for new_index, old_index in enumerate(order):
                inverse[old_index] = new_index
            atom_axes = tuple(
                AtomAxisAddress(
                    source_sdf_atom_index=old_index,
                    projected_atom_index=inverse[old_index],
                    e3fp_row=old_index,
                )
                for old_index in range(len(order))
            )
            actual = build_compact_geometry_sidecar(
                authoritative, model, (), atom_axes
            )
            self.assertEqual(authoritative.tokens, base_authoritative.tokens)
            self.assertEqual(actual.model_tokens, base.model_tokens)
            self.assertEqual(
                [
                    (
                        row.left.fragment_index,
                        row.left.fragment_local_atom_index,
                        row.left.explicit_in_surface,
                        row.right.fragment_index,
                        row.right.fragment_local_atom_index,
                        row.right.explicit_in_surface,
                    )
                    for row in actual.connectors
                ],
                [
                    (
                        row.left.fragment_index,
                        row.left.fragment_local_atom_index,
                        row.left.explicit_in_surface,
                        row.right.fragment_index,
                        row.right.fragment_local_atom_index,
                        row.right.explicit_in_surface,
                    )
                    for row in base.connectors
                ],
            )
            self.assertEqual(
                [
                    (
                        row.source_sdf_atom_index,
                        row.projected_atom_index,
                        row.e3fp_row,
                    )
                    for row in actual.atoms
                ],
                [
                    (index, inverse[index], index)
                    for index in range(mol.GetNumAtoms())
                ],
            )

    def test_compact_atom_axes_are_required_and_fail_closed(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        authoritative = strict_round_trip(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        model = encode_compact_model_surface(
            mol, authoritative, (), chemicalgof_root=CHEMICALGOF_ROOT
        )
        missing = tuple(
            AtomAxisAddress(index, index, index)
            for index in range(mol.GetNumAtoms() - 1)
        )
        with self.assertRaises(FragSmilesGeometrySidecarError):
            build_compact_geometry_sidecar(authoritative, model, (), missing)

    def test_invalid_explicit_endpoint_span_fails_closed(self) -> None:
        _mol, _authoritative, _model, sidecar = _compact("CC1CCC(CC1)C")
        connector = next(
            row
            for row in sidecar.connectors
            if row.left.explicit_in_surface or row.right.explicit_in_surface
        )
        endpoint = connector.left if connector.left.explicit_in_surface else connector.right
        broken_endpoint = replace(endpoint, carrier_token_index=0)
        broken_connector = replace(
            connector,
            left=broken_endpoint if endpoint.side == "left" else connector.left,
            right=broken_endpoint if endpoint.side == "right" else connector.right,
        )
        connectors = tuple(
            broken_connector if row.connector_index == connector.connector_index else row
            for row in sidecar.connectors
        )
        with self.assertRaises(FragSmilesGeometrySidecarError):
            FragSmilesGeometrySidecar(
                schema_version=sidecar.schema_version,
                mode=sidecar.mode,
                model_tokens=sidecar.model_tokens,
                token_roles=sidecar.token_roles,
                token_to_fragment=sidecar.token_to_fragment,
                fragments=sidecar.fragments,
                atoms=sidecar.atoms,
                connectors=connectors,
                component_count=sidecar.component_count,
                molecule_carrier_token_index=sidecar.molecule_carrier_token_index,
                fallback_mode=sidecar.fallback_mode,
                padding_materialized=sidecar.padding_materialized,
            )


if __name__ == "__main__":
    unittest.main()
