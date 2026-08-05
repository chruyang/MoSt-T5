from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem

from most_t5_next.r1.adapter import mol_linearizer
from most_t5_next.r1.adapter import run_p1_topology_canary_v1 as runner
from most_t5_next.r1.adapter.tests.test_mol_linearizer import build_synthetic_molecule


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def selection_document() -> dict:
    return {
        "schema_version": runner.SELECTION_SCHEMA,
        "selection_id": "topology-canary-fixture-v1",
        "release": {
            "release_id": "production-v2-fixture",
            "full_release_manifest_sha256": "a" * 64,
            "logical_release_root_sha256": "b" * 64,
        },
        "groups": {
            "smoke": [
                {"sdf_record_index": ordinal, "selection_tags": ["overfit"]}
                for ordinal in range(runner.SMOKE_COUNT)
            ],
            "canary": [
                {"sdf_record_index": ordinal, "selection_tags": ["coverage"]}
                for ordinal in range(runner.SMOKE_COUNT, runner.SMOKE_COUNT + runner.CANARY_COUNT)
            ],
        },
    }


def molecule_with_conformer() -> Chem.Mol:
    molecule = build_synthetic_molecule()
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(atom_index, (float(atom_index), 0.25, -0.5))
    molecule.AddConformer(conformer, assignId=True)
    return molecule


class P1TopologyCanaryRunnerTest(unittest.TestCase):
    def test_selection_is_exactly_disjoint_32_plus_256(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.json"
            path.write_text(json.dumps(selection_document()), encoding="utf-8")
            document, items = runner.load_selection(path)
            self.assertEqual(document["selection_id"], "topology-canary-fixture-v1")
            self.assertEqual(len(items), 288)
            self.assertEqual([item.group for item in items[:32]], ["smoke"] * 32)
            self.assertEqual([item.group for item in items[32:]], ["canary"] * 256)

            duplicate = selection_document()
            duplicate["groups"]["canary"][0]["sdf_record_index"] = 0
            path.write_text(json.dumps(duplicate), encoding="utf-8")
            with self.assertRaises(runner.TopologyCanaryError):
                runner.load_selection(path)

    def test_selected_sdf_scan_parses_only_targets_and_binds_full_member(self):
        molecule = molecule_with_conformer()
        block = Chem.MolToMolBlock(molecule).encode("utf-8") + b"\n$$$$\n"
        member_bytes = block * 3
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "pcqm-fixture.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                info = tarfile.TarInfo("pcqm4m-v2-train.sdf")
                info.size = len(member_bytes)
                archive.addfile(info, io.BytesIO(member_bytes))
            locked = {
                "tar_member_name": "pcqm4m-v2-train.sdf",
                "member_type": "regular_file",
                "uncompressed_bytes": len(member_bytes),
                "sha256": digest(member_bytes),
            }
            observed, receipt = runner.stream_selected_sdf(
                Chem, archive_path, locked, {0, 2}, expected_record_count=3, progress_every=0
            )
            self.assertEqual(set(observed), {0, 2})
            self.assertEqual(receipt["release_lock"]["sha256"], locked["sha256"])
            self.assertEqual(receipt["prefix_scan"]["sdf_records_scanned"], 3)
            self.assertTrue(receipt["prefix_scan"]["complete_member_rehashed"])
            self.assertEqual(observed[0].GetNumAtoms(), molecule.GetNumAtoms())

            prefix_only, prefix_receipt = runner.stream_selected_sdf(
                Chem, archive_path, locked, {0}, expected_record_count=3, progress_every=0
            )
            self.assertEqual(set(prefix_only), {0})
            self.assertEqual(prefix_receipt["prefix_scan"]["sdf_records_scanned"], 1)
            self.assertTrue(prefix_receipt["prefix_scan"]["stopped_after_maximum_selected_ordinal"])
            self.assertFalse(prefix_receipt["prefix_scan"]["complete_member_rehashed"])

    def test_augmentation_row_replays_mapping_and_never_carries_geometry_features(self):
        source = Chem.Mol(bytes(molecule_with_conformer().ToBinary()))
        tagged, source_count, _ = runner.projection.tag_source_atoms(Chem, source)
        geometry, mapping = runner.projection.project_hydrogens(Chem, tagged, source_count)
        result = mol_linearizer.linearize_mol(geometry)
        linearizer_sha = runner.sha256_file(Path(mol_linearizer.__file__))
        record_hash = digest(b"production-record")
        membership = {
            "sidecar_id": "production-v2-fixture",
            "disposition": "admit",
            "member_id": "ogb_pcqm4mv2_train_row_index:7",
            "sdf_record_index": 7,
            "record_storage_key": "000000007",
            "record_content_sha256": record_hash,
        }
        record = {
            "record_schema_version": runner.PRODUCTION_RECORD_SCHEMA,
            "member": {
                "member_id": membership["member_id"],
                "sdf_record_index": 7,
                "storage_key": membership["record_storage_key"],
            },
            "atom_universe": {
                "source_atom_count": source_count,
                "model_atom_count": geometry.GetNumAtoms(),
                "model_to_source_atom_index": np.asarray(mapping, dtype=np.int32),
            },
            "topology": {
                "linearizer_spec_sha256": linearizer_sha,
                "motif_count": len(result.motif_atom_groups),
                "motif_atom_indices": [np.asarray(group, dtype=np.int32) for group in result.motif_atom_groups],
                "motif_lexeme_sha256": [
                    digest(fragment.encode("utf-8")) for fragment in result.fragment_sequence
                ],
            },
        }
        binding = {"record": record, "membership": membership, "shard_index": 3}
        runner.validate_bound_record(record, membership, 7)
        row = runner.build_augmentation_row(
            Chem,
            np,
            runner.SelectionItem("smoke", 0, 7, ("overfit",)),
            binding,
            source,
            linearizer_sha,
        )
        self.assertEqual(row["member"]["base_record_content_sha256"], record_hash)
        self.assertEqual(row["release"]["shard_index"], 3)
        self.assertFalse(row["augmentation"]["provenance"]["geometry_or_e3fp_recomputed"])
        self.assertNotIn("coordinates", row["augmentation"])
        self.assertNotIn("e3fp", row["augmentation"])

        changed = copy.deepcopy(record)
        changed["topology"]["motif_lexeme_sha256"][0] = digest(b"different")
        with self.assertRaises(runner.topology.TopologyAugmentationError):
            runner.build_augmentation_row(
                Chem,
                np,
                runner.SelectionItem("smoke", 0, 7, ()),
                {"record": changed, "membership": membership, "shard_index": 3},
                source,
                linearizer_sha,
            )

    def test_compact_output_is_separate_and_content_addressed(self):
        row = {
            "schema_version": runner.ROW_SCHEMA,
            "selection": {"group": "smoke", "group_index": 0, "selection_tags": []},
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "topology-canary"
            manifest = runner.write_outputs(
                output,
                [row],
                {"schema_version": runner.MANIFEST_SCHEMA, "status": "pass"},
            )
            rows_path = output / "topology_augmentation.jsonl"
            self.assertTrue(rows_path.is_file())
            self.assertTrue((output / "manifest.json").is_file())
            self.assertEqual(manifest["artifacts"]["topology_augmentation"]["sha256"], runner.sha256_file(rows_path))
            self.assertEqual(manifest["artifacts"]["topology_augmentation"]["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
