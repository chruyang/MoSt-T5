from __future__ import annotations

import pickle
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1 import build_phase2_anchored_pure_motif_census_v1 as p2


class Phase2AnchoredPureMotifCensusTest(unittest.TestCase):
    def _payload(self):
        return {
            "atom_to_motif_map": [[0]],
            "atoms": [6],
            "cid": 17,
            "coordinates": [[0.0, 0.0, 0.0]],
            "description": "ignored",
            "e3fp": [[1, 2, 3, -1]],
            "enriched_description": "ignored",
            "motif_seq": "<bom>[legacy]<eom>",
            "raw_smiles": "C",
            "smiles": " C ",
        }

    def test_payload_uses_smiles_and_ignores_legacy_motif_and_text(self):
        self.assertEqual(p2._payload_smiles("17", pickle.dumps(self._payload())), "C")

    def test_payload_rejects_key_cid_mismatch(self):
        with self.assertRaises(p2.Phase2AnchoredCensusError):
            p2._payload_smiles("18", pickle.dumps(self._payload()))

    def test_payload_schema_is_closed(self):
        payload = self._payload()
        payload["unexpected"] = True
        with self.assertRaises(p2.Phase2AnchoredCensusError):
            p2._payload_smiles("17", pickle.dumps(payload))

    def test_one_record_lmdb_builds_current_surface_cache(self):
        import lmdb

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "phase2.lmdb"
            environment = lmdb.open(str(source), subdir=False, map_size=1 << 20)
            try:
                payload = self._payload()
                payload["smiles"] = "CC"
                with environment.begin(write=True) as transaction:
                    transaction.put(b"17", pickle.dumps(payload))
                    transaction.put(b"__len__", pickle.dumps(1))
            finally:
                environment.close()
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            config = {
                "schema_version": p2.CONFIG_SCHEMA,
                "collection": {"phase": "p2", "split": "train"},
                "source": {
                    "format": "legacy_lmdb_pickle",
                    "expected_bytes": source.stat().st_size,
                    "expected_sha256": source_sha,
                    "format_options": {
                        "lmdb": {
                            "metadata_keys_permitted": ["__len__"],
                            "trusted_pickle_source_sha256": source_sha,
                        }
                    },
                },
                "mapping": {"smiles_field": "smiles"},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "output"
            manifest = p2.build(
                argparse.Namespace(
                    source_lmdb=str(source),
                    source_config=str(config_path),
                    output_dir=str(output),
                    expected_records=1,
                    workers=2,
                    max_pending=2,
                    progress_every=0,
                    legacy_pickle_acknowledgement=p2.ACKNOWLEDGEMENT,
                )
            )
            self.assertEqual(manifest["status"], "pass")
            self.assertEqual(manifest["counts"]["admitted_records"], 1)
            self.assertGreater(manifest["counts"]["pure_motif_occurrences"], 0)
            self.assertTrue((output / "pure_motif_registry.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
