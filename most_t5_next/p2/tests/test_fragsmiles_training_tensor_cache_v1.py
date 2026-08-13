from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from rdkit import Chem

from most_t5_next.p1.fragsmiles_compact_stereo_codec_v1 import strict_round_trip
from most_t5_next.p1.fragsmiles_lossless_fallback_v1 import encode_lossless_fallback
from most_t5_next.p1.fragsmiles_macro_fallback_surface_v1 import encode_compact_model_surface
from most_t5_next.p2.fragsmiles_geometry_sidecar_v1 import (
    AtomAxisAddress,
    build_compact_geometry_sidecar,
    build_fallback_geometry_sidecar,
)
from most_t5_next.p2.fragsmiles_training_tensor_cache_v1 import (
    CompiledFragSmilesRecord,
    FragSmilesTrainingTensorCache,
    MODE_TO_ID,
    compile_sidecar_record,
    length_class,
    write_training_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CHEMICALGOF_ROOT = REPO_ROOT / "reference_repos" / "chemicalgof-master"


def _axes(count: int):
    return tuple(
        AtomAxisAddress(index, index, index) for index in range(count)
    )


def _compact_record(ordinal: int = 7) -> CompiledFragSmilesRecord:
    mol = Chem.MolFromSmiles("CC1CCC(CC1)C")
    assert mol is not None
    macros = ({"fragment_smiles": "C", "surface_token": "<MOST:FM:000000>"},)
    compact = strict_round_trip(mol, chemicalgof_root=CHEMICALGOF_ROOT)
    model = encode_compact_model_surface(
        mol, compact, macros, chemicalgof_root=CHEMICALGOF_ROOT
    )
    sidecar = build_compact_geometry_sidecar(
        compact, model, macros, _axes(mol.GetNumAtoms())
    )
    return compile_sidecar_record(
        ordinal=ordinal,
        source_segment=0,
        input_ids=tuple(range(100, 100 + len(sidecar.model_tokens))),
        sidecar=sidecar,
        e3fp=tuple((index, index + 1, -1, -1) for index in range(mol.GetNumAtoms())),
    )


def _fallback_record(ordinal: int = 8) -> CompiledFragSmilesRecord:
    mol = Chem.MolFromSmiles("CC.O")
    assert mol is not None
    fallback = encode_lossless_fallback(mol)
    sidecar = build_fallback_geometry_sidecar(fallback, _axes(mol.GetNumAtoms()))
    return compile_sidecar_record(
        ordinal=ordinal,
        source_segment=1,
        input_ids=tuple(range(500, 500 + len(sidecar.model_tokens))),
        sidecar=sidecar,
        e3fp=tuple((index, index + 1, index + 2, -1) for index in range(mol.GetNumAtoms())),
    )


class FragSmilesTrainingTensorCacheV1Tests(unittest.TestCase):
    def test_length_bins_match_512_model_boundary(self) -> None:
        self.assertEqual(
            [length_class(value) for value in (1, 64, 65, 128, 129, 256, 257, 384, 385, 512, 513)],
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        )

    def test_compact_projection_keeps_two_endpoint_rows_per_connector(self) -> None:
        row = _compact_record()
        self.assertEqual(row.mode, MODE_TO_ID["compact"])
        self.assertGreater(len(row.fragment_spans), 0)
        self.assertEqual(len(row.endpoints) % 2, 0)
        self.assertTrue(any(row.atom_is_attachment))
        self.assertTrue(all(0 <= carrier < len(row.input_ids) for carrier in row.atom_carriers))

    def test_fallback_uses_atom_glyph_carriers_without_fake_motif(self) -> None:
        row = _fallback_record()
        self.assertEqual(row.mode, MODE_TO_ID["whole_molecule_fallback"])
        self.assertEqual(row.fragment_spans, ())
        self.assertEqual(row.endpoints, ())
        self.assertEqual(row.molecule_carrier, len(row.input_ids) - 1)
        self.assertTrue(all(owner == -1 for owner in row.atom_to_fragment))
        self.assertTrue(all(0 <= carrier < len(row.input_ids) for carrier in row.atom_carriers))

    def test_mmap_roundtrip_preserves_minimal_fields_and_reports_lengths(self) -> None:
        compact, fallback = _compact_record(), _fallback_record()
        Path("tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="tmp") as temporary:
            output = Path(temporary) / "cache"
            manifest = write_training_cache(
                (compact, fallback),
                output_dir=output,
                source={"name": "fixture"},
                tokenizer={"vocab_size": 53368},
            )
            self.assertEqual(manifest["counts"]["records"], 2)
            self.assertEqual(manifest["counts"]["modes"]["compact"], 1)
            self.assertEqual(
                manifest["counts"]["modes"]["whole_molecule_fallback"], 1
            )
            self.assertFalse(manifest["storage"]["coordinates_cached"])
            self.assertFalse(manifest["storage"]["record_strings_cached"])
            cache = FragSmilesTrainingTensorCache(output)
            try:
                observed = cache[0]
                self.assertEqual(observed.ordinal, compact.ordinal)
                self.assertEqual(tuple(observed.input_ids), compact.input_ids)
                self.assertEqual(tuple(map(tuple, observed.e3fp)), compact.e3fp)
                self.assertEqual(tuple(map(tuple, observed.endpoints)), compact.endpoints)
                fallback_observed = cache[1]
                self.assertEqual(fallback_observed.mode, "whole_molecule_fallback")
                self.assertEqual(tuple(fallback_observed.atom_carriers), fallback.atom_carriers)
            finally:
                cache.close()

    def test_oversize_is_ledgered_not_truncated(self) -> None:
        base = _fallback_record(ordinal=98)
        oversized = CompiledFragSmilesRecord(
            **{
                **base.__dict__,
                "ordinal": 99,
                "input_ids": tuple(range(513)),
                "token_roles": (0,) * 513,
                "token_to_fragment": (-1,) * 513,
                "atom_carriers": tuple(min(value, 511) for value in base.atom_carriers),
                "molecule_carrier": 512,
            }
        )
        Path("tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="tmp") as temporary:
            output = Path(temporary) / "cache"
            manifest = write_training_cache(
                (base, oversized),
                output_dir=output,
                source={"name": "fixture"},
                tokenizer={"vocab_size": 53368},
            )
            self.assertEqual(manifest["counts"]["records"], 1)
            self.assertEqual(manifest["counts"]["excluded_oversize_records"], 1)
            line = (output / "length_exclusions.jsonl").read_text("utf-8")
            self.assertIn('"ordinal":99', line)
            self.assertIn('"sequence_length":513', line)


if __name__ == "__main__":
    unittest.main()
