from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    from most_t5_next.p2.run_pf2_t3mi_v1 import (
        T3MI_MASK_PROBABILITY,
        T3MI_OBJECTIVE_CONTRACT,
        T3MI_PROTOCOL,
    )
    from most_t5_next.p2.validate_pf2_gated_fusion_gpu_smoke_v1 import (
        FGateSmokeError,
        _require_all_motif_identities_masked,
    )
    from most_t5_next.p2.validate_pf2_t3mi_gpu_smoke_v1 import (
        REPORT_SCHEMA,
        build_parser,
        run_smoke,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class PF2T3MIGPUSmokeTest(unittest.TestCase):
    def test_thin_runner_freezes_t3mi_arguments(self):
        args = build_parser().parse_args([
            "--paired-release", "paired",
            "--base-model-snapshot", "model",
            "--base-tokenizer-snapshot", "tokenizer",
            "--union-init-dir", "init",
            "--output", "report.json",
            "--geometry-fusion-seed", "7",
        ])
        captured = {}

        def fake_runner(passed_args, **kwargs):
            captured["args"] = passed_args
            captured.update(kwargs)
            return {"status": "pass"}

        self.assertEqual(run_smoke(args, runner=fake_runner), {"status": "pass"})
        self.assertIs(captured["args"], args)
        self.assertEqual(captured["mask_probability"], T3MI_MASK_PROBABILITY)
        self.assertIs(captured["protocol"], T3MI_PROTOCOL)
        self.assertEqual(captured["report_schema"], REPORT_SCHEMA)
        self.assertEqual(captured["objective_contract"], T3MI_OBJECTIVE_CONTRACT)
        self.assertTrue(captured["require_all_motif_identities_masked"])

    def test_all_identity_audit_counts_sentinels_and_length_reduction(self):
        spans = (SimpleNamespace(start=1, stop=3), SimpleNamespace(start=4, stop=5))
        row = SimpleNamespace(
            motif_record=SimpleNamespace(
                input_ids=(10, 20, 21, 30, 40, 50),
                identity_spans=spans,
            )
        )
        batch = SimpleNamespace(
            ce_batch=SimpleNamespace(
                input_ids=((10, 900, 30, 901, 50),),
                input_lengths=(5,),
            )
        )
        report = _require_all_motif_identities_masked((row,), batch, (900, 901, 902))
        self.assertEqual(report["logical_motif_identities"], 2)
        self.assertEqual(report["original_identity_tokens"], 3)

        bad = SimpleNamespace(
            ce_batch=SimpleNamespace(
                input_ids=((10, 900, 30, 40, 50),),
                input_lengths=(5,),
            )
        )
        with self.assertRaisesRegex(FGateSmokeError, "not every"):
            _require_all_motif_identities_masked((row,), bad, (900, 901, 902))


if __name__ == "__main__":
    unittest.main()
