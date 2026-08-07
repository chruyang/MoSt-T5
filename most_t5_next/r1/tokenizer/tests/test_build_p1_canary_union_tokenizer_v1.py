from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest

from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    CanaryUnionTokenizerError,
    NATURAL_TEXT_REGRESSION_SURFACES,
    T5_SENTINELS,
    build_canary_union_tokenizer,
    load_verified_canary_union_tokenizer,
)
from most_t5_next.r1.tokenizer.production_atom_selfies_codec_v1 import (
    SELFIES_DISTRIBUTION_VERSION,
    SELFIES_SEPARATOR_TOKEN,
)


class FakeAddedToken:
    def __init__(
        self,
        content,
        *,
        lstrip=False,
        rstrip=False,
        normalized=False,
        special=False,
    ):
        self.content = content
        self.lstrip = lstrip
        self.rstrip = rstrip
        self.normalized = normalized
        self.special = special

    def __str__(self):
        return self.content


class FakeTokenizer:
    fail_save = False

    def __init__(self, vocab, additional_special_tokens, added_surfaces):
        self._vocab = {str(token): int(token_id) for token, token_id in vocab.items()}
        self._reverse = {token_id: token for token, token_id in self._vocab.items()}
        self.additional_special_tokens = list(additional_special_tokens)
        self._added_surfaces = set(added_surfaces)
        self.pad_token_id = self._vocab["<pad>"]
        self.eos_token_id = self._vocab["</s>"]
        self.unk_token_id = self._vocab["<unk>"]
        self.added_tokens_decoder = {
            token_id: FakeAddedToken(
                token,
                special=token in self.additional_special_tokens,
            )
            for token, token_id in self._vocab.items()
            if token in self._added_surfaces or token in self.additional_special_tokens
        }

    @classmethod
    def base(cls):
        vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2}
        for index, token in enumerate(reversed(T5_SENTINELS), start=3):
            vocab[token] = index
        vocab["."] = len(vocab)
        for index, _surface in enumerate(NATURAL_TEXT_REGRESSION_SURFACES):
            vocab["<TEXT_{}>".format(index)] = len(vocab)
        return cls(vocab, list(T5_SENTINELS), set(T5_SENTINELS))

    @property
    def all_special_tokens(self):
        return ["<pad>", "</s>", "<unk>", *self.additional_special_tokens]

    def __len__(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id):
        return self._reverse.get(token_id, "<unk>")

    def _natural_ids(self, surface):
        index = NATURAL_TEXT_REGRESSION_SURFACES.index(surface)
        return [self._vocab["<TEXT_{}>".format(index)]]

    def encode(self, surface, *, add_special_tokens):
        if surface in NATURAL_TEXT_REGRESSION_SURFACES:
            return self._natural_ids(surface)
        if surface in self.additional_special_tokens or surface in self._added_surfaces:
            return [self._vocab[surface]]
        if surface == ".":
            # Model the real slow T5 boundary behavior: the base piece exists,
            # but a standalone dot is not an exact singleton before promotion.
            return [699, self._vocab["."]]
        registered = sorted(
            self._added_surfaces.union(self.additional_special_tokens),
            key=len,
            reverse=True,
        )
        result = []
        cursor = 0
        while cursor < len(surface):
            match = next(
                (token for token in registered if surface.startswith(token, cursor)),
                None,
            )
            if match is None:
                return [self.unk_token_id]
            result.append(self._vocab[match])
            cursor += len(match)
        return result

    def _add(self, value, *, special):
        token = str(getattr(value, "content", value))
        if token not in self._vocab:
            token_id = len(self._vocab)
            self._vocab[token] = token_id
            self._reverse[token_id] = token
        self._added_surfaces.add(token)
        if special and token not in self.additional_special_tokens:
            self.additional_special_tokens.append(token)
        self.added_tokens_decoder[self._vocab[token]] = FakeAddedToken(
            token, special=special
        )

    def add_special_tokens(self, payload):
        before = len(self)
        for value in payload["additional_special_tokens"]:
            self._add(value, special=True)
        return len(self) - before

    def add_tokens(self, values, special_tokens=False):
        before = len(self)
        for value in values:
            self._add(value, special=bool(special_tokens))
        return len(self) - before

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True)
        if self.fail_save:
            raise RuntimeError("injected save failure")
        payload = {
            "vocab": self._vocab,
            "additional_special_tokens": self.additional_special_tokens,
            "added_surfaces": sorted(self._added_surfaces),
        }
        (path / "fake_tokenizer.json").write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(path, **_kwargs):
        payload = json.loads((Path(path) / "fake_tokenizer.json").read_text("utf-8"))
        return FakeTokenizer(
            payload["vocab"],
            payload["additional_special_tokens"],
            payload["added_surfaces"],
        )


@pytest.fixture
def fake_transformers():
    module = types.ModuleType("transformers")
    module.AddedToken = FakeAddedToken
    module.AutoTokenizer = FakeAutoTokenizer
    with patch.dict(sys.modules, {"transformers": module}):
        yield


@pytest.fixture
def base_snapshot(tmp_path):
    path = tmp_path / "base"
    FakeTokenizer.base().save_pretrained(path)
    return path


def macro_registry():
    # Deliberately reverse input order; normalization is count/identity ranked.
    return [
        {"identity": "O", "token": "<MOST:M:000001>", "occurrence_count": 2},
        {"identity": "C", "token": "<MOST:M:000000>", "occurrence_count": 4},
    ]


def test_build_and_verified_reload_bind_macro_semantics_and_opaque_dot(
    fake_transformers, base_snapshot, tmp_path
):
    output = tmp_path / "candidate"
    built = build_canary_union_tokenizer(
        base_snapshot=base_snapshot,
        output_dir=output,
        selfies_distribution_version=SELFIES_DISTRIBUTION_VERSION,
        robust_selfies_symbols={"[O]", "[C]"},
        observed_selfies_symbols={".", "[C]"},
        motif_macro_registry=macro_registry(),
    )

    contract = built.manifest["contract"]
    assert output.is_dir()
    assert not (tmp_path / "candidate.staging").exists()
    assert "." not in contract["declared_token_order"]
    assert SELFIES_SEPARATOR_TOKEN in contract["declared_token_order"]
    assert contract["selfies_symbol_registry"][0] == {
        "selfies_symbol": ".",
        "token_surface": SELFIES_SEPARATOR_TOKEN,
    }
    assert [row["identity"] for row in contract["motif_macro_registry"]] == ["C", "O"]
    assert contract["motif_macro_registry"][0]["rank"] == 0
    assert len(contract["motif_macro_registry"][0]["identity_sha256"]) == 64
    assert built.tokenizer.encode(
        "C.O", add_special_tokens=False
    ) == contract["natural_text_regression"]["C.O"]

    loaded = load_verified_canary_union_tokenizer(
        base_snapshot=base_snapshot,
        output_dir=output,
    )
    assert loaded.runtime == built.runtime
    assert loaded.manifest == built.manifest
    assert loaded.tokenizer.encode(
        SELFIES_SEPARATOR_TOKEN, add_special_tokens=False
    ) == [built.manifest["token_ids"]["declared"][SELFIES_SEPARATOR_TOKEN]]


def test_macro_token_must_follow_deterministic_rank(
    fake_transformers, base_snapshot, tmp_path
):
    with pytest.raises(CanaryUnionTokenizerError, match="deterministic rank"):
        build_canary_union_tokenizer(
            base_snapshot=base_snapshot,
            output_dir=tmp_path / "bad-rank",
            selfies_distribution_version=SELFIES_DISTRIBUTION_VERSION,
            robust_selfies_symbols={"[C]"},
            observed_selfies_symbols=set(),
            motif_macro_registry=[
                {
                    "identity": "C",
                    "token": "<MOST:M:000001>",
                    "occurrence_count": 2,
                }
            ],
        )


def test_macro_identity_changes_contract_even_when_snapshot_tokens_match(
    fake_transformers, base_snapshot, tmp_path
):
    common = {
        "base_snapshot": base_snapshot,
        "selfies_distribution_version": SELFIES_DISTRIBUTION_VERSION,
        "robust_selfies_symbols": {"[C]"},
        "observed_selfies_symbols": set(),
    }
    first = build_canary_union_tokenizer(
        output_dir=tmp_path / "first",
        motif_macro_registry=macro_registry(),
        **common,
    )
    second = build_canary_union_tokenizer(
        output_dir=tmp_path / "second",
        motif_macro_registry=[
            {"identity": "O", "token": "<MOST:M:000001>", "occurrence_count": 2},
            {"identity": "N", "token": "<MOST:M:000000>", "occurrence_count": 4},
        ],
        **common,
    )
    assert (
        first.runtime.tokenizer_snapshot_sha256
        == second.runtime.tokenizer_snapshot_sha256
    )
    assert (
        first.runtime.tokenizer_contract_sha256
        != second.runtime.tokenizer_contract_sha256
    )


def test_verified_loader_rejects_changed_snapshot(
    fake_transformers, base_snapshot, tmp_path
):
    output = tmp_path / "candidate"
    built = build_canary_union_tokenizer(
        base_snapshot=base_snapshot,
        output_dir=output,
        selfies_distribution_version=SELFIES_DISTRIBUTION_VERSION,
        robust_selfies_symbols={"[C]"},
        observed_selfies_symbols=set(),
        motif_macro_registry=[],
    )
    snapshot_file = built.snapshot_path / "fake_tokenizer.json"
    snapshot_file.write_text(snapshot_file.read_text("utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CanaryUnionTokenizerError, match="saved tokenizer tree"):
        load_verified_canary_union_tokenizer(
            base_snapshot=base_snapshot,
            output_dir=output,
        )


def test_failed_save_leaves_only_explicit_staging_path(
    fake_transformers, base_snapshot, tmp_path
):
    output = tmp_path / "candidate"
    FakeTokenizer.fail_save = True
    try:
        with pytest.raises(RuntimeError, match="injected save failure"):
            build_canary_union_tokenizer(
                base_snapshot=base_snapshot,
                output_dir=output,
                selfies_distribution_version=SELFIES_DISTRIBUTION_VERSION,
                robust_selfies_symbols={"[C]"},
                observed_selfies_symbols=set(),
                motif_macro_registry=[],
            )
    finally:
        FakeTokenizer.fail_save = False
    assert not output.exists()
    assert (tmp_path / "candidate.staging").is_dir()
