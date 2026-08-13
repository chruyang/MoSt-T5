from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    from most_t5_next.p2.build_anchored_v4_matched_state_overlay_v1 import (
        AnchoredV4MatchedStateProvider,
        build_anchored_v4_matched_state_overlay,
    )


@dataclass(frozen=True)
class _Record:
    record_id: str
    cache_index: int
    exact_identity_sha256: tuple[str, ...]
    atom_to_logical_motif: tuple[int, ...]
    full_e3fp_ids: tuple[tuple[int, int, int, int], ...]
    connection_token_to_atom: tuple[int, ...]


class _Cache:
    def __init__(self, _root: Path) -> None:
        self.records = (
            _Record("a", 0, ("x",), (0, 0), ((1, 2, 3, 4), (5, 6, 7, 8)), (0, -1)),
            _Record("b", 1, ("x",), (0, 0), ((11, 12, 13, 14), (15, 16, 17, 18)), (1, -1)),
            _Record("c", 2, ("y",), (0,), ((21, 22, 23, 24),), (-1,)),
        )

    def split_indices(self, split: str):
        assert split == "dev"
        return (0, 1, 2)

    def __getitem__(self, index: int):
        return self.records[index]

    def atom_local_positions(self, record: _Record):
        return (2, 1) if record.record_id == "a" else tuple(range(len(record.full_e3fp_ids)))

    def morgan_state(self, record_id: str, *, cache_index: int):
        assert self.records[cache_index].record_id == record_id
        base = 100 + cache_index * 10
        return tuple((base + i, base + i, base + i, base + i) for i in range(len(self.records[cache_index].full_e3fp_ids)))


@unittest.skipIf(torch is None, "PyTorch is optional in the local fixture")
class AnchoredMatchedOverlayTest(unittest.TestCase):
    def test_same_donor_plan_materializes_both_state_kinds(self) -> None:
        import most_t5_next.p2.build_anchored_v4_matched_state_overlay_v1 as module

        original = module.IndexedPF10TrainingTensorCache
        module.IndexedPF10TrainingTensorCache = _Cache
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "overlay"
                report = build_anchored_v4_matched_state_overlay(
                    cache_root=Path(tmp) / "cache", output_dir=root
                )
                self.assertEqual(report["coverage"]["eligible_motif_occurrences"], 2)
                self.assertEqual(report["coverage"]["excluded_motif_occurrences"], 1)
                e3fp = AnchoredV4MatchedStateProvider(root, state_kind="e3fp")
                morgan = AnchoredV4MatchedStateProvider(root, state_kind="morgan")
                self.assertEqual(e3fp.get("a"), ((15, 16, 17, 18), (11, 12, 13, 14)))
                self.assertEqual(morgan.get("a"), ((111, 111, 111, 111), (110, 110, 110, 110)))
                self.assertEqual(e3fp.get("c"), ((21, 22, 23, 24),))
        finally:
            module.IndexedPF10TrainingTensorCache = original


if __name__ == "__main__":
    unittest.main()
