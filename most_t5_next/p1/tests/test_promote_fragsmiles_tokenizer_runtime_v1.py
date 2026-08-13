from __future__ import annotations

import unittest

from most_t5_next.p1.promote_fragsmiles_tokenizer_runtime_v1 import (
    EXPECTED_RDKIT_VERSION,
    EXPECTED_SENTENCEPIECE_VERSION,
    EXPECTED_TRANSFORMERS_VERSION,
)


class PromoteFragSmilesTokenizerRuntimeV1Tests(unittest.TestCase):
    def test_runtime_contract_is_frozen(self):
        self.assertEqual(EXPECTED_TRANSFORMERS_VERSION, "4.45.2")
        self.assertEqual(EXPECTED_SENTENCEPIECE_VERSION, "0.2.0")
        self.assertEqual(EXPECTED_RDKIT_VERSION, "2024.03.5")


if __name__ == "__main__":
    unittest.main()
