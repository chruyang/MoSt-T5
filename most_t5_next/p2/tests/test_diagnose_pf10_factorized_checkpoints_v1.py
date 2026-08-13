from __future__ import annotations

from dataclasses import asdict
import unittest

from most_t5_next.p2 import diagnose_pf10_factorized_checkpoints_v1 as diagnostic
from most_t5_next.p2 import run_pf10_factorized_grammar_v1 as grammar
from most_t5_next.p2 import run_pf10_factorized_state_v1 as state


class PF10CheckpointDiagnosticTest(unittest.TestCase):
    def _payload(self, stage_name: str, update: int):
        if stage_name == "S":
            schema = state.CHECKPOINT_SCHEMA
            protocol = asdict(state.S_PROTOCOL)
        else:
            schema = grammar.CHECKPOINT_SCHEMA
            protocol = asdict(grammar._cell_protocol("F3D"))
        return {
            "schema_version": schema,
            "cell": "F3D",
            "stage": stage_name,
            "state_kind": "e3fp",
            "completed_updates": update,
            "protocol": protocol,
            "model_state_dict": {"weight": object()},
        }

    def test_accepts_formal_s_and_g_payloads(self) -> None:
        diagnostic._validate_checkpoint_payload(
            self._payload("S", 2500), stage="S", update=2500
        )
        diagnostic._validate_checkpoint_payload(
            self._payload("G", 5000), stage="G", update=5000
        )

    def test_rejects_stage_or_update_mismatch(self) -> None:
        with self.assertRaises(diagnostic.PF10CheckpointDiagnosticError):
            diagnostic._validate_checkpoint_payload(
                self._payload("G", 5000), stage="G", update=10000
            )
        payload = self._payload("S", 2500)
        payload["cell"] = "B2D"
        with self.assertRaises(diagnostic.PF10CheckpointDiagnosticError):
            diagnostic._validate_checkpoint_payload(payload, stage="S", update=2500)


if __name__ == "__main__":
    unittest.main()
