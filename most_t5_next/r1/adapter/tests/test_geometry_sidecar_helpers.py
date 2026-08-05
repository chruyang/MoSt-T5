"""Hermetic tests for the bounded PCQM geometry-sidecar harness helpers.

They open no molecular dataset, do not import RDKit/E3FP/LMDB, and exercise
only deterministic contract helpers plus the immutable harness lock.
"""

from __future__ import print_function

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "adapter" / "build_pcqm_p1_geometry_sidecar.py"
SPEC = importlib.util.spec_from_file_location("r1_geometry_builder_test", str(BUILDER_PATH))
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class GeometrySidecarHelperTest(unittest.TestCase):
    def test_fixed_member_and_storage_identities(self):
        self.assertEqual(builder.member_id(17), "ogb_pcqm4mv2_train_row_index:17")
        self.assertEqual(builder.storage_key(17), "000000017")
        self.assertEqual(builder.storage_key(3_378_605), "003378605")
        with self.assertRaises(ValueError):
            builder.storage_key(-1)
        self.assertEqual(builder.scalar_to_int(23), 23)

    def test_source_address_binds_locked_member_and_csv_row(self):
        member = {
            "tar_member_name": "pcqm4m-v2-train.sdf",
            "member_type": "regular_file",
            "uncompressed_bytes": 123,
            "sha256": "a" * 64,
        }
        first = builder.source_address_sha256("b" * 64, member, 17, 23)
        self.assertEqual(first, builder.source_address_sha256("b" * 64, member, 17, 23))
        self.assertNotEqual(first, builder.source_address_sha256("b" * 64, member, 17, 24))
        changed_member = dict(member, sha256="c" * 64)
        self.assertNotEqual(first, builder.source_address_sha256("b" * 64, changed_member, 17, 23))

    def test_selected_prefix_hash_is_deterministic_and_order_sensitive(self):
        self.assertEqual(
            builder.sha256_selected_ordinals([0, 1, 2]),
            builder.sha256_selected_ordinals([0, 1, 2]),
        )
        self.assertNotEqual(
            builder.sha256_selected_ordinals([0, 1, 2]),
            builder.sha256_selected_ordinals([0, 2, 1]),
        )

    def test_expected_preflight_failures_map_to_closed_ledger_codes(self):
        self.assertEqual(
            builder.classify_preflight_rejection("SDF_CONFORMER_COUNT_NOT_ONE"),
            ("SDF_CONFORMER_INVALID", "sdf_parse"),
        )
        self.assertEqual(
            builder.classify_preflight_rejection("SOURCE_ATOM_TAG_ORDER_NOT_PRESERVED"),
            ("SOURCE_ATOM_INDEX_TAG_INVALID", "source_atom_index"),
        )
        self.assertEqual(
            builder.classify_preflight_rejection("E3FP_LEVEL0_MISSING"),
            ("E3FP_SHAPE_OR_RANGE_INVALID", "e3fp"),
        )
        with self.assertRaises(RuntimeError):
            builder.classify_preflight_rejection("UNDECLARED_REASON")

    def test_current_harness_lock_matches_every_component(self):
        lock_path = ROOT / "contracts" / "p1_pcqm_geometry_adapter_harness_lock.json"
        with open(str(lock_path), "r", encoding="utf-8") as handle:
            lock = json.load(handle)
        observed = builder.validate_adapter_lock(
            lock_path,
            lock,
            BUILDER_PATH,
            ROOT / "adapter" / "mol_linearizer.py",
            ROOT / "gates" / "pcqm_e3fp_preflight.py",
            ROOT / "gates" / "pcqm_identity_smoke.py",
            ROOT / "adapter" / "pcqm_source_integrity.py",
            ROOT / "adapter" / "sidecar_v2_codec.py",
            ROOT / "contracts" / "pcqm4mv2_source_contract.json",
            ROOT / "contracts" / "pcqm4mv2_identity_normalization_contract.json",
            ROOT / "contracts" / "p1_pcqm_geometry_record_schema.json",
            ROOT / "contracts" / "p1_pcqm_geometry_payload_format_contract.json",
        )
        self.assertEqual(observed, builder.sha256_file(lock_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
