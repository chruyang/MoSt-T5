from __future__ import annotations

import hashlib
from pathlib import Path
import unittest
import uuid

from most_t5_next.p1.publish_fragsmiles_tokenizer_v1 import (
    EXPECTED_RDKIT_VERSION,
    EXPECTED_TRANSFORMERS_VERSION,
    _tree_sha256,
)


class PublishFragSmilesTokenizerV1Tests(unittest.TestCase):
    def test_snapshot_tree_hash_binds_paths_and_bytes(self):
        root = Path("tmp/unit_publish_tokenizer") / uuid.uuid4().hex
        root.mkdir(parents=True)
        (root / "a").write_bytes(b"x")
        (root / "nested").mkdir()
        (root / "nested" / "b").write_bytes(b"yz")
        observed = _tree_sha256(root)
        digest = hashlib.sha256()
        for name, payload in ((b"a", b"x"), (b"nested/b", b"yz")):
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        self.assertEqual(observed, digest.hexdigest())

    def test_release_runtime_is_explicitly_frozen(self):
        self.assertEqual(EXPECTED_TRANSFORMERS_VERSION, "4.45.2")
        self.assertEqual(EXPECTED_RDKIT_VERSION, "2024.03.5")


if __name__ == "__main__":
    unittest.main()
