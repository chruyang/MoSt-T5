from __future__ import annotations

import unittest

from most_t5_next.p1.validate_fragsmiles_hf_tokenizer_v1 import (
    FragSmilesTokenizerCompatibilityError,
    _load_macro_surfaces,
)


class FragSmilesHFTokenizerV1Tests(unittest.TestCase):
    def test_macro_surface_table_requires_dense_unique_rows(self):
        self.assertEqual(
            _load_macro_surfaces(
                [
                    {"rank": 0, "surface_token": "<MOST:FM:000000>"},
                    {"rank": 1, "surface_token": "<MOST:FM:000001>"},
                ]
            ),
            ("<MOST:FM:000000>", "<MOST:FM:000001>"),
        )
        with self.assertRaises(FragSmilesTokenizerCompatibilityError):
            _load_macro_surfaces([{"rank": 2, "surface_token": "x"}])


if __name__ == "__main__":
    unittest.main()
