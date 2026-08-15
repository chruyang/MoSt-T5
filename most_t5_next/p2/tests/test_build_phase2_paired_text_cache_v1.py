from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import tempfile
import unittest

import lmdb

from most_t5_next.p2.build_phase2_paired_text_cache_v1 import Phase2PairedTextCache, run


class BuildPhase2PairedTextCacheV1Tests(unittest.TestCase):
    def test_complete_text_and_membership_order_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.lmdb"
            env = lmdb.open(str(source), subdir=False, map_size=8 * 1024 * 1024)
            with env.begin(write=True) as transaction:
                for cid in (9, 3, 7):
                    transaction.put(
                        str(cid).encode(),
                        pickle.dumps(
                            {
                                "cid": str(cid),
                                "description": f"raw {100 + cid}",
                                "enriched_description": f"text {cid}",
                            }
                        ),
                    )
            env.close()
            membership = root / "membership.jsonl"
            membership.write_text("".join(json.dumps({"member_id": f"pubchem_cid:{cid}"}) + "\n" for cid in (9, 3, 7)), "utf-8")

            def factory(_root):
                def encode_batch(texts):
                    return [[int(text.split()[-1]), 1] for text in texts]
                return encode_batch, 1, 0, 32100

            output = root / "cache"
            manifest = run(argparse.Namespace(source_lmdb=source, membership=membership, tokenizer_root=root, output_dir=output, expected_records=3, batch_size=2, max_records=None), tokenizer_factory=factory)
            self.assertEqual(manifest["counts"], {"records": 3, "text_tokens": 6})
            self.assertEqual(manifest["source"]["text_field"], "enriched_description")
            cache = Phase2PairedTextCache(output)
            self.assertEqual([cache[index][0] for index in range(3)], [9, 3, 7])
            self.assertEqual(cache[1][1].tolist(), [3, 1])


if __name__ == "__main__":
    unittest.main()
