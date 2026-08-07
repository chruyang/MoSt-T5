from __future__ import annotations

from dataclasses import replace
from importlib import metadata as importlib_metadata

import pytest
from rdkit import Chem
import selfies as sf
from selfies.mol_graph import Attribution, AttributionMap

from most_t5_next.r1.tokenizer.production_atom_selfies_codec_v1 import (
    ATOM_IDENTITY_ROLE,
    BRANCH_ROLE,
    MOLECULE_BEGIN,
    MOLECULE_END,
    RING_ROLE,
    SEPARATOR_ROLE,
    SELFIES_SEPARATOR_TOKEN,
    SELFIES_DISTRIBUTION_VERSION,
    DEFAULT_SELFIES_220_CONSTRAINTS,
    AtomSelfiesAlignmentError,
    bind_atom_selfies_surface,
    discover_atom_selfies_surface,
    derive_atom_selfies_alignment,
    tokenizer_surface_for_selfies_symbol,
)


SOURCE_TAG = "_test_source_atom_index"


def test_pinned_selfies_220_runtime_and_default_constraints_are_exact() -> None:
    assert importlib_metadata.version("selfies") == SELFIES_DISTRIBUTION_VERSION
    assert dict(sf.get_semantic_constraints()) == DEFAULT_SELFIES_220_CONSTRAINTS


class ExactUnionTokenizer:
    """Small longest-match tokenizer with the production exact-token API."""

    def __init__(self, tokens: tuple[str, ...], *, extra_tokens: tuple[str, ...] = ()) -> None:
        ordered = tuple(dict.fromkeys((*tokens, *extra_tokens)))
        self.unk_token_id = 0
        self._token_to_id = {token: index + 1 for index, token in enumerate(ordered)}
        self._id_to_token = {token_id: token for token, token_id in self._token_to_id.items()}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._id_to_token.get(token_id, "<unk>")

    def encode(self, surface: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        result: list[int] = []
        cursor = 0
        tokens = tuple(self._token_to_id)
        while cursor < len(surface):
            candidates = [token for token in tokens if surface.startswith(token, cursor)]
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


def _canonical_selfies(mol: Chem.Mol) -> str:
    probe = Chem.Mol(mol)
    canonical = Chem.MolToSmiles(probe, canonical=True, isomericSmiles=True, kekuleSmiles=False)
    return sf.encoder(canonical, strict=True)


def _tokenizer_for_selfies(*values: str) -> ExactUnionTokenizer:
    symbols: list[str] = []
    for value in values:
        symbols.extend(
            tokenizer_surface_for_selfies_symbol(symbol)
            for symbol in sf.split_selfies(value)
        )
    return ExactUnionTokenizer((MOLECULE_BEGIN, MOLECULE_END, *symbols))


@pytest.mark.parametrize(
    ("smiles", "required_role"),
    [
        ("CC(C)O", BRANCH_ROLE),
        ("c1ccccc1", RING_ROLE),
        ("[NH4+]", ATOM_IDENTITY_ROLE),
        ("[13CH3]O", ATOM_IDENTITY_ROLE),
        ("[Na+].[Cl-]", SEPARATOR_ROLE),
    ],
)
def test_supported_chemistry_has_exact_bijective_alignment(
    smiles: str, required_role: str
) -> None:
    mol = _mol(smiles)
    selfies = _canonical_selfies(mol)
    alignment = derive_atom_selfies_alignment(
        Chem,
        sf,
        mol,
        _tokenizer_for_selfies(selfies),
    )

    assert required_role in alignment.symbol_role
    assert len(alignment.canonical_position_to_model_atom) == mol.GetNumAtoms()
    assert sorted(atom_id for atom_id in alignment.symbol_to_model_atom if atom_id >= 0) == list(
        range(mol.GetNumAtoms())
    )
    assert len(alignment.input_ids) == len(alignment.selfies_symbols) + 2
    assert alignment.token_to_atom[0] == alignment.token_to_atom[-1] == -1
    assert len(set(alignment.atom_to_carrier)) == mol.GetNumAtoms()
    for atom_id, span in enumerate(alignment.atom_identity_spans):
        assert span.stop == span.start + 1
        assert alignment.token_to_atom[span.start] == atom_id
        assert alignment.token_role[span.start] == ATOM_IDENTITY_ROLE


def test_disconnected_separator_uses_opaque_tokenizer_surface_without_changing_evidence() -> None:
    mol = _mol("[Na+].[Cl-]")
    surface = discover_atom_selfies_surface(Chem, sf, mol)
    tokenizer = _tokenizer_for_selfies(surface.selfies)

    alignment = bind_atom_selfies_surface(surface, tokenizer)
    separator_index = surface.selfies_symbols.index(".")
    separator_token_index = separator_index + 1

    assert surface.selfies_symbols[separator_index] == "."
    assert surface.symbol_role[separator_index] == SEPARATOR_ROLE
    assert tokenizer.convert_ids_to_tokens(alignment.input_ids[separator_token_index]) == (
        SELFIES_SEPARATOR_TOKEN
    )
    assert alignment.selfies == surface.selfies
    assert alignment.token_to_atom[separator_token_index] == -1
    assert alignment.token_role[separator_token_index] == SEPARATOR_ROLE


def test_disconnected_complete_surface_rejects_swallowed_opaque_separator_boundary() -> None:
    mol = _mol("[Na+].[Cl-]")
    surface = discover_atom_selfies_surface(Chem, sf, mol)
    mapped = tuple(
        tokenizer_surface_for_selfies_symbol(symbol)
        for symbol in surface.selfies_symbols
    )
    separator_index = mapped.index(SELFIES_SEPARATOR_TOKEN)
    swallowed = mapped[separator_index] + mapped[separator_index + 1]
    tokenizer = ExactUnionTokenizer(
        (MOLECULE_BEGIN, MOLECULE_END, *mapped),
        extra_tokens=(swallowed,),
    )

    with pytest.raises(
        AtomSelfiesAlignmentError, match="UNION_TOKENIZER_WHOLE_SURFACE_NOT_EXACT"
    ):
        bind_atom_selfies_surface(surface, tokenizer)


def test_derive_is_compatible_with_explicit_discover_then_bind() -> None:
    mol = _mol("CC(C)O")
    surface = discover_atom_selfies_surface(Chem, sf, mol)
    tokenizer = _tokenizer_for_selfies(surface.selfies)

    assert derive_atom_selfies_alignment(Chem, sf, mol, tokenizer) == (
        bind_atom_selfies_surface(surface, tokenizer)
    )


def test_branch_and_ring_argument_symbols_remain_structure_not_atoms() -> None:
    branch_mol = _mol("CC(C)O")
    branch = derive_atom_selfies_alignment(
        Chem,
        sf,
        branch_mol,
        _tokenizer_for_selfies(_canonical_selfies(branch_mol)),
    )
    branch_index = branch.selfies_symbols.index("[Branch1]")
    assert branch.symbol_role[branch_index : branch_index + 2] == (BRANCH_ROLE, BRANCH_ROLE)
    assert branch.symbol_to_model_atom[branch_index : branch_index + 2] == (-1, -1)

    ring_mol = _mol("c1ccccc1")
    ring = derive_atom_selfies_alignment(
        Chem,
        sf,
        ring_mol,
        _tokenizer_for_selfies(_canonical_selfies(ring_mol)),
    )
    ring_index = ring.selfies_symbols.index("[Ring1]")
    assert ring.symbol_role[ring_index : ring_index + 2] == (RING_ROLE, RING_ROLE)
    assert ring.symbol_to_model_atom[ring_index : ring_index + 2] == (-1, -1)


@pytest.mark.parametrize(
    "smiles",
    [
        # PF-1 frozen-cohort ordinals 472552, 1693731 and 3263912.  SELFIES
        # 2.2.0 represents the stereogenic ring closure with ``[-/Ring1]``;
        # its following symbol is the ring-distance argument, not an atom.
        "C1=C/NCCNCCNCCN/1",
        "C1=C\\CCOCCCCCC/1",
        "O=C1CCCCCCCCCCC/C(O)=N/1",
    ],
)
def test_real_directional_ring_closure_is_structure_and_preserves_ez(
    smiles: str,
) -> None:
    mol = _mol(smiles)
    surface = discover_atom_selfies_surface(Chem, sf, mol)

    ring_index = surface.selfies_symbols.index("[-/Ring1]")
    assert surface.symbol_role[ring_index : ring_index + 2] == (
        RING_ROLE,
        RING_ROLE,
    )
    assert surface.symbol_to_model_atom[ring_index : ring_index + 2] == (-1, -1)
    assert surface.canonical_isomeric_smiles == Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
        kekuleSmiles=False,
    )


def test_atom_renumbering_maps_canonical_carriers_back_to_stable_source_atoms() -> None:
    original = _mol("F[C@H](Cl)Br")
    for source_index, atom in enumerate(original.GetAtoms()):
        atom.SetIntProp(SOURCE_TAG, source_index)
    original_atom_order = tuple(atom.GetIdx() for atom in original.GetAtoms())
    original_selfies = _canonical_selfies(original)
    tokenizer = _tokenizer_for_selfies(original_selfies)
    first = derive_atom_selfies_alignment(Chem, sf, original, tokenizer)

    permutation = tuple(reversed(range(original.GetNumAtoms())))
    renumbered = Chem.RenumberAtoms(original, permutation)
    second = derive_atom_selfies_alignment(Chem, sf, renumbered, tokenizer)

    def token_source_sequence(mol: Chem.Mol, alignment) -> tuple[int, ...]:
        return tuple(
            mol.GetAtomWithIdx(atom_id).GetIntProp(SOURCE_TAG)
            for atom_id in alignment.symbol_to_model_atom
            if atom_id >= 0
        )

    assert first.canonical_isomeric_smiles == second.canonical_isomeric_smiles
    assert first.selfies == second.selfies
    assert first.input_ids == second.input_ids
    assert token_source_sequence(original, first) == token_source_sequence(renumbered, second)
    assert tuple(atom.GetIdx() for atom in original.GetAtoms()) == original_atom_order
    assert tuple(atom.GetIntProp(SOURCE_TAG) for atom in original.GetAtoms()) == tuple(
        range(original.GetNumAtoms())
    )


def test_real_bridged_stereo_canonical_parse_boundary_is_idempotent_and_renumber_safe() -> None:
    # Frozen PF-1 ordinal 2,250,716.  RDKit's first canonical serialization of
    # this SDF-derived bridged stereoisomer is chemically exact but not textual
    # fixed point: parsing it once selects an equivalent ring traversal with
    # different local @/@@ marks.  The production boundary must normalize that
    # parse transition and compose its atom permutation back to model rows.
    mol_block = """
     RDKit          3D

 14 15  0  0  0  0  0  0  0  0999 V2000
    1.7493   -0.1646    0.7862 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.6806   -0.8255    2.4577 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.6020   -0.1935   -0.2391 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.1184    0.4076   -3.1249 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.2248    1.9528   -2.9029 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.0315    0.0685   -1.5593 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.1201    1.6116   -1.3121 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.3421    1.0053   -0.7727 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.5123   -0.1744   -1.7457 C   0  0  1  0  0  0  0  0  0  0  0  0
    5.6459    2.0682   -1.4180 C   0  0  2  0  0  0  0  0  0  0  0  0
    5.2714    0.6986    0.7074 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.8941    0.8670   -0.7625 C   0  0  1  0  0  0  0  0  0  0  0  0
    5.3356    1.6129    1.5031 O   0  0  0  0  0  0  0  0  0  0  0  0
    5.4568   -0.5950    1.0570 O   0  0  0  0  0  0  0  0  0  0  0  0
  3  1  2  0
  4  5  1  0
  9  4  1  6
 10  5  1  6
  6  7  1  0
 12  8  1  1
  8  3  1  0
  9  6  1  0
  9 12  1  0
 10  7  1  0
 10 12  1  0
 11 14  1  0
 11 13  2  0
 12 11  1  0
 14  2  1  0
M  END
"""
    original = Chem.MolFromMolBlock(mol_block, sanitize=True, removeHs=False)
    assert original is not None
    for source_index, atom in enumerate(original.GetAtoms()):
        atom.SetIntProp(SOURCE_TAG, source_index)

    first_text = Chem.MolToSmiles(
        original, canonical=True, isomericSmiles=True, kekuleSmiles=False
    )
    reparsed = Chem.MolFromSmiles(first_text)
    assert reparsed is not None
    fixed_text = Chem.MolToSmiles(
        reparsed, canonical=True, isomericSmiles=True, kekuleSmiles=False
    )
    assert first_text != fixed_text

    first = discover_atom_selfies_surface(Chem, sf, original)
    assert first.canonical_isomeric_smiles == fixed_text
    assert sorted(first.canonical_position_to_model_atom) == list(
        range(original.GetNumAtoms())
    )

    permutation = tuple(reversed(range(original.GetNumAtoms())))
    renumbered = Chem.RenumberAtoms(original, permutation)
    second = discover_atom_selfies_surface(Chem, sf, renumbered)

    def source_sequence(mol: Chem.Mol, surface) -> tuple[int, ...]:
        return tuple(
            mol.GetAtomWithIdx(atom_id).GetIntProp(SOURCE_TAG)
            for atom_id in surface.symbol_to_model_atom
            if atom_id >= 0
        )

    assert second.canonical_isomeric_smiles == fixed_text
    assert second.selfies == first.selfies
    assert source_sequence(original, first) == source_sequence(renumbered, second)


@pytest.mark.parametrize(
    "left_smiles,right_smiles",
    [
        ("F[C@H](Cl)Br", "F[C@@H](Cl)Br"),
        ("F/C=C/F", "F/C=C\\F"),
    ],
)
def test_opposite_rs_and_ez_stereoisomers_do_not_collapse(
    left_smiles: str, right_smiles: str
) -> None:
    left_mol = _mol(left_smiles)
    right_mol = _mol(right_smiles)
    left_selfies = _canonical_selfies(left_mol)
    right_selfies = _canonical_selfies(right_mol)
    tokenizer = _tokenizer_for_selfies(left_selfies, right_selfies)

    left = derive_atom_selfies_alignment(Chem, sf, left_mol, tokenizer)
    right = derive_atom_selfies_alignment(Chem, sf, right_mol, tokenizer)

    assert left.canonical_isomeric_smiles != right.canonical_isomeric_smiles
    assert left.selfies != right.selfies


@pytest.mark.parametrize(
    "smiles",
    [
        # PCQM4Mv2 ordinals 0, 184767 and 211162.  SELFIES may write explicit
        # H forms whose RDKit parse changes GetNoImplicit(), although chemical
        # atom fields, CIP assignment and the ordered bond graph agree.
        "Cc1ccc([C@H]2[CH]c3cnccc3[N]C2=O)cc1",
        "CCC1=C[C@]2(NCc3ccccc3)C[C]([CH]1)[C@@H]2C",
        "Cc1ccc2c(c1)[C@H]1CC[CH][CH][C@@]1(C)O2",
    ],
)
def test_rdkit_no_implicit_parser_state_is_not_treated_as_chemical_identity(
    smiles: str,
) -> None:
    mol = _mol(smiles)
    surface = discover_atom_selfies_surface(Chem, sf, mol)

    assert surface.canonical_isomeric_smiles == Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=True, kekuleSmiles=False
    )
    assert len([atom for atom in surface.symbol_to_model_atom if atom >= 0]) == (
        mol.GetNumAtoms()
    )


def test_explicit_h_parser_storage_is_not_treated_as_atom_identity() -> None:
    # PCQM4Mv2 ordinal 1,874,895.  The source SDF silicon atom and the
    # SELFIES-decoded ``[SiH2]`` atom both carry two total hydrogens, although
    # RDKit stores them as implicit versus explicit parser metadata.
    mol = _mol("ClC(Cl)[SiH2]c1ccccc1")
    silicon = next(atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 14)
    silicon.SetNumExplicitHs(0)
    silicon.SetNoImplicit(False)
    Chem.SanitizeMol(mol)

    surface = discover_atom_selfies_surface(Chem, sf, mol)

    assert surface.canonical_isomeric_smiles == "ClC(Cl)[SiH2]c1ccccc1"
    assert len([atom for atom in surface.symbol_to_model_atom if atom >= 0]) == (
        mol.GetNumAtoms()
    )


@pytest.mark.parametrize(
    "smiles",
    [
        # Real PCQM4Mv2 cases exercising nested-branch offsets and duplicate
        # AttributionMap rows in SELFIES 2.2.0.
        "O=C1N=C2C(=CC(F)=C[C@@H]2F)OC1=O",
        "COC(=O)[C@@H]1CS/C(=C/[C](C)C)[N]1",
        "Cc1cc(C)cc(O[P@](C)(=O)Cl)c1",
    ],
)
def test_decoder_attribution_selects_unique_atom_symbol_in_nested_branches(
    smiles: str,
) -> None:
    mol = _mol(smiles)
    surface = discover_atom_selfies_surface(Chem, sf, mol)

    assert sorted(atom for atom in surface.symbol_to_model_atom if atom >= 0) == list(
        range(mol.GetNumAtoms())
    )


class AmbiguousDecoderSelfies:
    """Delegate 2.2.0 except for one deliberately ambiguous branch atom."""

    def __getattr__(self, name: str):
        return getattr(sf, name)

    def decoder(self, selfies: str, attribute: bool = False):
        decoded = sf.decoder(selfies, attribute=attribute)
        if not attribute or "[Branch1]" not in selfies:
            return decoded
        decoded_smiles, attribution = decoded
        altered = list(attribution)
        for index, item in enumerate(altered):
            source_indices = [source.index for source in item.attribution or ()]
            if len(source_indices) > 1 and 4 in source_indices:
                altered[index] = AttributionMap(
                    index=item.index,
                    token=item.token,
                    attribution=[*item.attribution, Attribution(index=3, token="[C]")],
                )
                break
        return decoded_smiles, altered


def test_ambiguous_decoder_attribution_is_rejected_instead_of_using_last_item() -> None:
    mol = _mol("CC(C)O")
    selfies = _canonical_selfies(mol)
    with pytest.raises(AtomSelfiesAlignmentError, match="ATTRIBUTION_NOT_BIJECTIVE"):
        derive_atom_selfies_alignment(
            Chem,
            AmbiguousDecoderSelfies(),
            mol,
            _tokenizer_for_selfies(selfies),
        )


def test_union_tokenizer_cannot_swallow_multiple_selfies_symbols() -> None:
    mol = _mol("CC(C)O")
    selfies = _canonical_selfies(mol)
    symbols = tuple(sf.split_selfies(selfies))
    tokenizer = ExactUnionTokenizer(
        (MOLECULE_BEGIN, MOLECULE_END, *symbols),
        extra_tokens=(selfies,),
    )
    with pytest.raises(
        AtomSelfiesAlignmentError, match="UNION_TOKENIZER_WHOLE_SURFACE_NOT_EXACT"
    ):
        derive_atom_selfies_alignment(Chem, sf, mol, tokenizer)


def test_union_tokenizer_unk_is_rejected_before_alignment_is_returned() -> None:
    mol = _mol("CO")
    selfies = _canonical_selfies(mol)
    symbols = tuple(symbol for symbol in sf.split_selfies(selfies) if symbol != "[O]")
    tokenizer = ExactUnionTokenizer((MOLECULE_BEGIN, MOLECULE_END, *symbols))
    with pytest.raises(AtomSelfiesAlignmentError, match="UNION_TOKENIZER_UNK"):
        derive_atom_selfies_alignment(Chem, sf, mol, tokenizer)


def test_alignment_dataclass_is_frozen() -> None:
    mol = _mol("CO")
    selfies = _canonical_selfies(mol)
    alignment = derive_atom_selfies_alignment(
        Chem,
        sf,
        mol,
        _tokenizer_for_selfies(selfies),
    )
    with pytest.raises(AtomSelfiesAlignmentError, match="ALIGNMENT_INVARIANT_INVALID"):
        replace(alignment, atom_to_carrier=(1, 1))
