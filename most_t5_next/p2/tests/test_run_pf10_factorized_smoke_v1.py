from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
from pathlib import Path
import unittest
import uuid

import torch
from torch import nn
from torch.nn import functional as F

from most_t5_next.p1.bound_record import Span
from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
)
from most_t5_next.p2.factorized_motif_t5_v1 import FactorizedMotifT5V1
from most_t5_next.p2.run_pf10_factorized_smoke_v1 import (
    FORMAL_STATE_MASKING,
    PF10FactorizedSmokeError,
    SMOKE_RECORD_COUNT,
    assert_same_factorized_initialization,
    get_smoke_stage_spec,
    run_pf10_factorized_smoke_stage,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@contextmanager
def _workspace_test_directory():
    """Use the writable workspace and remove only explicitly named files."""

    root = Path.cwd() / ("pf10_smoke_test_" + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        for subdir in (root / "S", root / "G", root / "B", root):
            for filename in (
                "s_stage_checkpoint.pt",
                "g_stage_checkpoint.pt",
                "b_stage_checkpoint.pt",
                "smoke_report.json",
            ):
                path = subdir / filename
                if path.is_file():
                    path.unlink()
        for subdir in (root / "G", root / "B", root / "S", root):
            if subdir.is_dir():
                subdir.rmdir()


@dataclass
class _Config:
    vocab_size: int
    d_model: int


@dataclass
class _EncoderOutput:
    last_hidden_state: torch.Tensor


@dataclass
class _T5Output:
    loss: torch.Tensor
    encoder_last_hidden_state: torch.Tensor
    logits: torch.Tensor


class _TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, *, inputs_embeds, attention_mask, return_dict=True):
        hidden = self.projection(inputs_embeds)
        hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        return _EncoderOutput(last_hidden_state=hidden)


class _TinyT5(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = _Config(vocab_size=vocab_size, d_model=hidden_size)
        self.shared = nn.Embedding(vocab_size, hidden_size)
        self.encoder = _TinyEncoder(hidden_size)
        self.decoder = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.shared

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        attention_mask,
        labels,
        return_dict=True,
        **_kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.shared(input_ids)
        hidden = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        lengths = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        context = hidden.sum(dim=1) / lengths.to(hidden.dtype)
        context = self.decoder(context)
        decoder_hidden = context.unsqueeze(1).expand(-1, labels.shape[1], -1)
        logits = self.lm_head(decoder_hidden)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return _T5Output(
            loss=loss,
            encoder_last_hidden_state=hidden,
            logits=logits,
        )


class _B2DProvider:
    state_kind = "mock_morgan_r3_4096"

    def get(self, record_id: str):
        index = int(record_id.rsplit("-", 1)[1])
        return (
            (index % 31, (index + 1) % 31, (index + 2) % 31, (index + 3) % 31),
            ((index + 4) % 31, (index + 5) % 31, (index + 6) % 31, (index + 7) % 31),
        )


class PF10FactorizedSmokeRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        contract = _digest("tokenizer-contract")
        snapshot = _digest("tokenizer-snapshot")
        cls.tokenizer = ProductionTokenizerRuntime(
            tokenizer_contract_sha256=contract,
            tokenizer_snapshot_sha256=snapshot,
            vocab_size=128,
            pad_token_id=0,
            eos_token_id=1,
            sentinel_token_ids=tuple(range(127, 27, -1)),
        )
        records = []
        for index in range(SMOKE_RECORD_COUNT):
            records.append(
                ProductionMotifRecord(
                    record_artifact_sha256=_digest(f"record-{index}"),
                    record_id=f"molecule-{index}",
                    storage_key=f"fixture/{index}",
                    release_id="pf10-smoke-fixture",
                    geometry_record_content_sha256=_digest(f"geometry-{index}"),
                    tokenizer_contract_sha256=contract,
                    tokenizer_snapshot_sha256=snapshot,
                    input_ids=(10, 11, 12),
                    token_to_logical_motif=(0, 0, 0),
                    token_role=("identity", "identity", "connection"),
                    identity_spans=(Span(0, 2),),
                    connection_token_indices=((2,),),
                    logical_to_carrier=(0,),
                    exact_identity_sha256=(_digest("motif"),),
                    source_atom_count=2,
                    full_e3fp_ids=(
                        (index % 31, (index + 1) % 31, (index + 2) % 31, (index + 3) % 31),
                        ((index + 4) % 31, (index + 5) % 31, (index + 6) % 31, (index + 7) % 31),
                    ),
                    atom_valid_mask=(True, True),
                    model_to_source_atom_index=(0, 1),
                    atom_to_logical_motif=(0, 0),
                    atom_is_attachment=(False, True),
                )
            )
        cls.records = tuple(records)

    @staticmethod
    def factorized_from_state(initial_state=None):
        model = FactorizedMotifT5V1(
            _TinyT5(),
            num_e3fp_embeddings=4096,
            state_embedding_dim=4,
            atom_memory_dim=8,
            max_identity_span_length=8,
        )
        if initial_state is not None:
            model.load_state_dict(initial_state, strict=True)
        return model

    def test_stage_grid_is_explicit(self) -> None:
        self.assertEqual(FORMAL_STATE_MASKING, "motif_atom_row")
        self.assertFalse(get_smoke_stage_spec("B2D", "S").t5_trainable)
        self.assertEqual(get_smoke_stage_spec("B0", "G").model_path, "raw_t5")
        self.assertTrue(get_smoke_stage_spec("F3D", "G").requires_s_checkpoint)
        with self.assertRaisesRegex(PF10FactorizedSmokeError, "admits B2D and F3D"):
            get_smoke_stage_spec("B0", "S")

    def test_b2d_and_f3d_must_share_exact_initialization(self) -> None:
        torch.manual_seed(5)
        b2d = self.factorized_from_state()
        f3d = self.factorized_from_state(b2d.state_dict())
        assert_same_factorized_initialization(b2d, f3d)
        with torch.no_grad():
            next(f3d.adapter.parameters()).add_(1.0)
        with self.assertRaisesRegex(PF10FactorizedSmokeError, "differs at initialization"):
            assert_same_factorized_initialization(b2d, f3d)

    def test_b0_g_stage_uses_raw_t5_for_three_updates(self) -> None:
        torch.manual_seed(7)
        model = _TinyT5()
        before = model.encoder.projection.weight.detach().clone()
        with _workspace_test_directory() as directory:
            report = run_pf10_factorized_smoke_stage(
                cell="B0",
                stage="G",
                records=self.records,
                tokenizer=self.tokenizer,
                model=model,
                output_dir=directory,
            )
            self.assertEqual(report["model_path"], "raw_t5")
            self.assertEqual(report["optimizer_updates"], 3)
            self.assertEqual(report["members_seen"], 384)
            self.assertTrue((directory / "smoke_report.json").is_file())
        self.assertFalse(torch.equal(before, model.encoder.projection.weight))

    def test_s_then_g_checkpoint_boundary_for_b2d(self) -> None:
        torch.manual_seed(11)
        initial = self.factorized_from_state()
        initial_state = {
            name: value.detach().clone() for name, value in initial.state_dict().items()
        }
        s_model = self.factorized_from_state(initial_state)
        t5_before = {
            name: value.detach().clone() for name, value in s_model.t5.state_dict().items()
        }
        adapter_before = {
            name: value.detach().clone()
            for name, value in s_model.adapter.state_dict().items()
        }
        provider = _B2DProvider()
        with _workspace_test_directory() as directory:
            s_dir = directory / "S"
            s_report = run_pf10_factorized_smoke_stage(
                cell="B2D",
                stage="S",
                records=self.records,
                tokenizer=self.tokenizer,
                model=s_model,
                output_dir=s_dir,
                atom_state_provider=provider,
            )
            self.assertFalse(s_report["t5_trainable"])
            self.assertEqual(s_report["formal_state_masking"], "motif_atom_row")
            self.assertFalse(s_model.t5.training)
            self.assertTrue(s_model.adapter.training)
            for name, value in s_model.t5.state_dict().items():
                torch.testing.assert_close(value, t5_before[name])
            self.assertTrue(
                any(
                    not torch.equal(value, adapter_before[name])
                    for name, value in s_model.adapter.state_dict().items()
                )
            )

            g_model = self.factorized_from_state(initial_state)
            g_t5_before = {
                name: value.detach().clone() for name, value in g_model.t5.state_dict().items()
            }
            g_report = run_pf10_factorized_smoke_stage(
                cell="B2D",
                stage="G",
                records=self.records,
                tokenizer=self.tokenizer,
                model=g_model,
                output_dir=directory / "G",
                atom_state_provider=provider,
                s_checkpoint=s_dir / "s_stage_checkpoint.pt",
            )
            self.assertTrue(g_report["t5_trainable"])
            self.assertIsNotNone(g_report["loaded_s_checkpoint"])
            self.assertTrue(
                any(
                    not torch.equal(value, g_t5_before[name])
                    for name, value in g_model.t5.state_dict().items()
                )
            )

    def test_f3d_s_stage_uses_persisted_e3fp_without_provider(self) -> None:
        torch.manual_seed(19)
        model = self.factorized_from_state()
        with _workspace_test_directory() as directory:
            report = run_pf10_factorized_smoke_stage(
                cell="F3D",
                stage="S",
                records=self.records,
                tokenizer=self.tokenizer,
                model=model,
                output_dir=directory,
            )
            self.assertEqual(report["state_kind"], "e3fp")
            self.assertEqual(report["objective_mode"], "state")
            self.assertEqual(len(report["updates"]), 3)

    def test_factorized_g_stage_cannot_skip_its_s_checkpoint(self) -> None:
        model = self.factorized_from_state()
        with _workspace_test_directory() as directory:
            with self.assertRaisesRegex(PF10FactorizedSmokeError, "requires its S checkpoint"):
                run_pf10_factorized_smoke_stage(
                    cell="F3D",
                    stage="G",
                    records=self.records,
                    tokenizer=self.tokenizer,
                    model=model,
                    output_dir=directory,
                )


if __name__ == "__main__":
    unittest.main()
