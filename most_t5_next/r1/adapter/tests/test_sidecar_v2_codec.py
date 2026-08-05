"""Unit tests for the non-executable bounded-sidecar v2 payload codec."""

from __future__ import print_function

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "adapter" / "sidecar_v2_codec.py"
SPEC = importlib.util.spec_from_file_location("r1_sidecar_v2_codec_test", str(MODULE_PATH))
codec = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(codec)


class SidecarV2CodecTest(unittest.TestCase):
    def record(self, np):
        return {
            "record_schema_version": "test/v2",
            "member": {"ordinal": 7, "nullable": None},
            "atom_universe": {"map": np.ascontiguousarray(np.asarray([0, 2, 5], dtype=np.int32))},
            "topology": {
                "groups": [
                    np.ascontiguousarray(np.asarray([0, 1], dtype=np.int32)),
                    np.ascontiguousarray(np.asarray([2], dtype=np.int32)),
                ]
            },
            "geometry": {
                "coordinates": np.ascontiguousarray(
                    np.asarray([[1.5, 2.5, 3.5], [0.0, -1.0, 4.0], [8.0, 9.0, 10.0]], dtype=np.float32)
                ),
                "valid": np.ascontiguousarray(np.asarray([True, True], dtype=np.bool_)),
            },
        }

    def test_round_trip_preserves_native_arrays_and_logical_hash(self):
        import numpy as np

        source = self.record(np)
        payload = codec.encode_record(np, source)
        restored, header_hash = codec.decode_record(np, payload)
        self.assertEqual(header_hash, codec.logical_record_sha256(np, source))
        self.assertEqual(header_hash, codec.logical_record_sha256(np, restored))
        self.assertTrue(np.array_equal(restored["atom_universe"]["map"], source["atom_universe"]["map"]))
        self.assertTrue(np.array_equal(restored["geometry"]["coordinates"], source["geometry"]["coordinates"]))
        self.assertTrue(restored["geometry"]["coordinates"].flags.c_contiguous)
        self.assertTrue(restored["geometry"]["coordinates"].flags.owndata)

    def test_binary_mutation_is_rejected_before_record_consumption(self):
        import numpy as np

        payload = bytearray(codec.encode_record(np, self.record(np)))
        payload[-1] ^= 0x01
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            codec.decode_record(np, payload)

    def test_wrong_magic_is_rejected(self):
        import numpy as np

        payload = bytearray(codec.encode_record(np, self.record(np)))
        payload[0] ^= 0x01
        with self.assertRaisesRegex(ValueError, "magic/version"):
            codec.decode_record(np, payload)

    def test_object_arrays_are_not_serializable(self):
        import numpy as np

        with self.assertRaisesRegex(ValueError, "unsupported sidecar payload dtype"):
            codec.encode_record(np, {"bad": np.asarray(["not-safe"], dtype=object)})


if __name__ == "__main__":
    unittest.main(verbosity=2)
