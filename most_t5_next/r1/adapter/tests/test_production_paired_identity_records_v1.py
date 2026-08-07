from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import unittest
from unittest import mock

from rdkit import Chem
import selfies as sf

from most_t5_next.p1.production_bridge import (
    ProductionBridgeError,
    ProductionTokenizerRuntime,
)
from most_t5_next.p1.runtime_bridge import P1ArtifactBindings, P1MemberRef
from most_t5_next.r1.adapter import production_paired_identity_records_v1 as subject
from most_t5_next.r1.tokenizer.production_graph_ports_codec_v1 import (
    CrossEdgeInput,
    FALLBACK_BEGIN,
    GPORTS_UNION_TOKENS,
    ProductionGraphPortsCodecV1,
)
from most_t5_next.r1.tokenizer.production_atom_selfies_codec_v1 import (
    tokenizer_surface_for_selfies_symbol,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class ExactUnionTokenizer:
    """Greedy exact-token fixture, including adversarial merged tokens."""

    def __init__(self, tokens: tuple[str, ...]) -> None:
        ordered = tuple(dict.fromkeys(tokens))
        self.unk_token_id = 0
        self._token_to_id = {token: index + 1 for index, token in enumerate(ordered)}
        self._id_to_token = {value: token for token, value in self._token_to_id.items()}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._id_to_token.get(token_id, "<unk>")

    def encode(self, surface: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        result: list[int] = []
        cursor = 0
        vocabulary = tuple(self._token_to_id)
        while cursor < len(surface):
            candidates = [token for token in vocabulary if surface.startswith(token, cursor)]
            if not candidates:
                result.append(self.unk_token_id)
                cursor += 1
                continue
            selected = max(candidates, key=lambda token: (len(token), token))
            result.append(self._token_to_id[selected])
            cursor += len(selected)
        return result


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def _cross_edges(
    mol: Chem.Mol, groups: tuple[tuple[int, ...], ...]
) -> tuple[CrossEdgeInput, ...]:
    owner = {
        atom_id: motif_id
        for motif_id, group in enumerate(groups)
        for atom_id in group
    }
    return tuple(
        CrossEdgeInput(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            str(bond.GetBondType()).upper(),
        )
        for bond in mol.GetBonds()
        if owner[bond.GetBeginAtomIdx()] != owner[bond.GetEndAtomIdx()]
    )


def _canonical_selfies(mol: Chem.Mol) -> str:
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return sf.encoder(smiles, strict=True)


def _tokenizer(
    mol: Chem.Mol,
    *,
    macro_tokens: tuple[str, ...] = (),
    extra_tokens: tuple[str, ...] = (),
) -> ExactUnionTokenizer:
    symbols = tuple(
        tokenizer_surface_for_selfies_symbol(symbol)
        for symbol in sf.split_selfies(_canonical_selfies(mol))
    )
    return ExactUnionTokenizer(
        (
            "<bom>",
            "<eom>",
            *symbols,
            *GPORTS_UNION_TOKENS,
            *macro_tokens,
            *extra_tokens,
        )
    )


def _bindings() -> tuple[P1ArtifactBindings, ProductionTokenizerRuntime]:
    contract = _digest("union-tokenizer-contract")
    snapshot = _digest("union-tokenizer-snapshot")
    bindings = P1ArtifactBindings(
        release_id="pcqm-inherited-overlay-canary-v1",
        data_release_manifest_sha256=_digest("release-manifest"),
        geometry_record_schema_sha256=_digest("overlay-schema"),
        geometry_record_content_sha256=_digest("effective-inherited-overlay-row"),
        membership_manifest_sha256=_digest("membership"),
        tokenizer_contract_sha256=contract,
        tokenizer_snapshot_sha256=snapshot,
        identity_codec_sha256=_digest("graph-identity-codec-v1"),
        connection_codec_sha256=_digest("graph-connection-codec-v1"),
    )
    tokenizer_binding = ProductionTokenizerRuntime(
        tokenizer_contract_sha256=contract,
        tokenizer_snapshot_sha256=snapshot,
        vocab_size=10000,
        pad_token_id=9000,
        eos_token_id=9001,
        sentinel_token_ids=(9002, 9003, 9004),
    )
    return bindings, tokenizer_binding


def _build(
    mol: Chem.Mol,
    groups: tuple[tuple[int, ...], ...],
    *,
    tokenizer: ExactUnionTokenizer,
    macro_by_identity: dict[str, str] | None = None,
):
    bindings, tokenizer_binding = _bindings()
    e3fp = tuple((100 + atom_id, 200 + atom_id, -1, -1) for atom_id in range(mol.GetNumAtoms()))
    return subject.build_production_paired_identity_records(
        Chem,
        sf,
        projected_mol=mol,
        logical_motif_atom_groups=groups,
        cross_edges=_cross_edges(mol, groups),
        member=P1MemberRef("pcqm4mv2:000000017", "000000017"),
        bindings=bindings,
        base_geometry_record_content_sha256=_digest("base-raw-e3fp-row"),
        effective_inherited_overlay_content_sha256=bindings.geometry_record_content_sha256,
        source_atom_count=mol.GetNumAtoms(),
        model_to_source_atom_index=tuple(range(mol.GetNumAtoms())),
        inherited_e3fp=e3fp,
        union_tokenizer=tokenizer,
        tokenizer_binding=tokenizer_binding,
        macro_by_identity=macro_by_identity,
    )


def _build_from_prepared(
    mol: Chem.Mol,
    groups: tuple[tuple[int, ...], ...],
    *,
    tokenizer: ExactUnionTokenizer,
    macro_by_identity: dict[str, str] | None = None,
):
    prepared = subject.discover_production_paired_identity_surfaces(
        Chem,
        sf,
        mol,
        groups,
        _cross_edges(mol, groups),
    )
    bindings, tokenizer_binding = _bindings()
    e3fp = tuple(
        (100 + atom_id, 200 + atom_id, -1, -1)
        for atom_id in range(mol.GetNumAtoms())
    )
    return subject.build_production_paired_identity_records_from_prepared(
        prepared=prepared,
        member=P1MemberRef("pcqm4mv2:000000017", "000000017"),
        bindings=bindings,
        base_geometry_record_content_sha256=_digest("base-raw-e3fp-row"),
        effective_inherited_overlay_content_sha256=(
            bindings.geometry_record_content_sha256
        ),
        source_atom_count=mol.GetNumAtoms(),
        model_to_source_atom_index=tuple(range(mol.GetNumAtoms())),
        inherited_e3fp=e3fp,
        union_tokenizer=tokenizer,
        tokenizer_binding=tokenizer_binding,
        macro_by_identity=macro_by_identity,
    )


class ProductionPairedIdentityRecordsTest(unittest.TestCase):
    def test_macro_and_fallback_build_one_strict_rs_pair_through_vnext_loader(self) -> None:
        mol = _mol("F[C@H](Cl)Br")
        groups = tuple((atom_id,) for atom_id in range(mol.GetNumAtoms()))
        graph = ProductionGraphPortsCodecV1().encode(mol, groups, _cross_edges(mol, groups))
        macro_token = "<MOST:M:000000>"
        macros = {graph.motifs[0].identity_smiles: macro_token}
        tokenizer = _tokenizer(mol, macro_tokens=(macro_token,))

        captured: dict[str, object] = {}
        real_loader = subject.load_production_motif_record

        def capture_loader(document):
            captured["document"] = copy.deepcopy(document)
            return real_loader(document)

        with mock.patch.object(subject, "load_production_motif_record", side_effect=capture_loader):
            pair = _build(mol, groups, tokenizer=tokenizer, macro_by_identity=macros)

        self.assertEqual(
            pair.receipt.strict_isomeric_identity,
            Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        )
        self.assertIn("@", pair.receipt.strict_isomeric_identity)
        self.assertEqual(
            {surface.mode for surface in pair.motif_identity_surfaces},
            {"macro", "fallback"},
        )
        self.assertEqual(
            pair.atom_record.geometry_record_content_sha256,
            _digest("effective-inherited-overlay-row"),
        )
        self.assertEqual(
            pair.motif_record.geometry_record_content_sha256,
            pair.atom_record.geometry_record_content_sha256,
        )
        self.assertEqual(
            pair.receipt.base_geometry_record_content_sha256,
            _digest("base-raw-e3fp-row"),
        )
        self.assertEqual(pair.atom_record.full_e3fp_ids, pair.motif_record.full_e3fp_ids)
        self.assertEqual(
            pair.atom_record.model_to_source_atom_index,
            pair.motif_record.model_to_source_atom_index,
        )
        self.assertEqual(pair.atom_record.source_atom_count, pair.motif_record.source_atom_count)

        document = captured["document"]
        self.assertEqual(pair.motif_training_document, document)
        externally_mutated = pair.motif_training_document
        externally_mutated["token_domain"]["input_ids"][0] = -1
        self.assertEqual(pair.motif_training_document, document)
        motif_domain = document["logical_motif_domain"]
        self.assertEqual(
            motif_domain["motif_slot_atom_indices"],
            [
                [port.source_atom_index for port in motif.ports]
                for motif in pair.graph_encoding.motifs
            ],
        )
        for edge_index, (declared, connection) in enumerate(
            zip(motif_domain["cross_motif_bonds"], pair.graph_encoding.connections)
        ):
            self.assertEqual(declared["edge_id"], edge_index)
            self.assertEqual(
                declared["left"]["slot_ordinal"], connection.endpoint_a.port_id - 1
            )
            self.assertEqual(
                declared["right"]["slot_ordinal"], connection.endpoint_b.port_id - 1
            )

        tampered_document = copy.deepcopy(document)
        first_connection = next(
            row
            for row in tampered_document["logical_motif_domain"]["connection_token_indices"]
            if row
        )
        first_connection[0] = 0
        with self.assertRaises(ProductionBridgeError):
            real_loader(tampered_document)

    def test_prepared_second_pass_does_not_repeat_chemistry_discovery(self) -> None:
        mol = _mol("CC(C)O")
        groups = ((0,), (1,), (2,), (3,))
        prepared = subject.discover_production_paired_identity_surfaces(
            Chem,
            sf,
            mol,
            groups,
            _cross_edges(mol, groups),
        )
        bindings, tokenizer_binding = _bindings()
        e3fp = tuple(
            (100 + atom_id, 200 + atom_id, -1, -1)
            for atom_id in range(mol.GetNumAtoms())
        )

        with mock.patch.object(
            subject,
            "discover_atom_selfies_surface",
            side_effect=AssertionError("chemistry discovery repeated"),
        ), mock.patch.object(
            subject.ProductionGraphPortsCodecV1,
            "encode",
            side_effect=AssertionError("graph chemistry encoding repeated"),
        ):
            pair = subject.build_production_paired_identity_records_from_prepared(
                prepared=prepared,
                member=P1MemberRef("pcqm4mv2:000000017", "000000017"),
                bindings=bindings,
                base_geometry_record_content_sha256=_digest("base-raw-e3fp-row"),
                effective_inherited_overlay_content_sha256=(
                    bindings.geometry_record_content_sha256
                ),
                source_atom_count=mol.GetNumAtoms(),
                model_to_source_atom_index=tuple(range(mol.GetNumAtoms())),
                inherited_e3fp=e3fp,
                union_tokenizer=_tokenizer(mol),
                tokenizer_binding=tokenizer_binding,
            )

        self.assertEqual(pair.graph_encoding, prepared.graph_encoding)
        self.assertEqual(
            pair.atom_record.selfies,
            prepared.atom_surface.selfies,
        )

    def test_prepared_surface_rejects_disagreeing_strict_identity(self) -> None:
        mol = _mol("CCO")
        groups = ((0,), (1,), (2,))
        prepared = subject.discover_production_paired_identity_surfaces(
            Chem,
            sf,
            mol,
            groups,
            _cross_edges(mol, groups),
        )
        changed_atom_surface = replace(
            prepared.atom_surface,
            canonical_isomeric_smiles="COC",
        )

        with self.assertRaisesRegex(
            subject.ProductionPairedIdentityError,
            "strict identities disagree",
        ):
            subject.PreparedPairedIdentitySurfaces(
                atom_surface=changed_atom_surface,
                graph_encoding=prepared.graph_encoding,
            )

    def test_original_entrypoint_matches_explicit_prepared_two_pass_entrypoint(self) -> None:
        mol = _mol("[Na+].[Cl-]")
        groups = ((0,), (1,))
        tokenizer = _tokenizer(mol)

        self.assertEqual(
            _build(mol, groups, tokenizer=tokenizer),
            _build_from_prepared(mol, groups, tokenizer=tokenizer),
        )

    def test_complete_m_surface_rejects_tokenizer_swallowing_macro_fallback_boundary(self) -> None:
        mol = _mol("CCO")
        groups = ((0,), (1,), (2,))
        graph = ProductionGraphPortsCodecV1().encode(mol, groups, _cross_edges(mol, groups))
        macro_token = "<MOST:M:000001>"
        macros = {graph.motifs[0].identity_smiles: macro_token}
        swallowed = macro_token + FALLBACK_BEGIN
        tokenizer = _tokenizer(
            mol,
            macro_tokens=(macro_token,),
            extra_tokens=(swallowed,),
        )

        with self.assertRaisesRegex(
            subject.ProductionPairedIdentityError,
            "complete M surface token boundaries",
        ):
            _build(mol, groups, tokenizer=tokenizer, macro_by_identity=macros)

    def test_raw_identity_cannot_be_registered_as_a_motif_macro(self) -> None:
        mol = _mol("CCO")
        groups = ((0,), (1,), (2,))
        graph = ProductionGraphPortsCodecV1().encode(mol, groups, _cross_edges(mol, groups))
        raw_identity = graph.motifs[0].identity_smiles
        tokenizer = _tokenizer(mol, macro_tokens=(raw_identity,))

        with self.assertRaisesRegex(
            subject.ProductionPairedIdentityError,
            "opaque <MOST:M:000000> namespace",
        ):
            _build(
                mol,
                groups,
                tokenizer=tokenizer,
                macro_by_identity={raw_identity: raw_identity},
            )

    def test_pair_level_validation_rejects_geometry_tamper(self) -> None:
        mol = _mol("CC(C)O")
        groups = ((0,), (1,), (2,), (3,))
        pair = _build(mol, groups, tokenizer=_tokenizer(mol))
        changed_rows = list(pair.motif_record.full_e3fp_ids)
        changed_rows[0] = (4095, *changed_rows[0][1:])
        tampered_motif = replace(pair.motif_record, full_e3fp_ids=tuple(changed_rows))

        with self.assertRaisesRegex(
            subject.ProductionPairedIdentityError,
            "shared geometry parity failed",
        ):
            replace(pair, motif_record=tampered_motif)

    def test_effective_overlay_digest_cannot_fall_back_to_base_geometry_digest(self) -> None:
        mol = _mol("CO")
        groups = ((0,), (1,))
        bindings, tokenizer_binding = _bindings()
        base = bindings.geometry_record_content_sha256

        with self.assertRaisesRegex(
            subject.ProductionPairedIdentityError,
            "base and effective",
        ):
            subject.build_production_paired_identity_records(
                Chem,
                sf,
                projected_mol=mol,
                logical_motif_atom_groups=groups,
                cross_edges=_cross_edges(mol, groups),
                member=P1MemberRef("pcqm4mv2:000000018", "000000018"),
                bindings=bindings,
                base_geometry_record_content_sha256=base,
                effective_inherited_overlay_content_sha256=base,
                source_atom_count=mol.GetNumAtoms(),
                model_to_source_atom_index=(0, 1),
                inherited_e3fp=((1, 2), (3, 4)),
                union_tokenizer=_tokenizer(mol),
                tokenizer_binding=tokenizer_binding,
            )


if __name__ == "__main__":
    unittest.main()
