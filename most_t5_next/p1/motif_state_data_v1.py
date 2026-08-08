"""Multi-worker PF-1 LMDB input path for the standalone G1 state gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


DATASET_VERSION = "most-t5-p1/motif-state-dataset/v1"


class MotifStateDataError(ValueError):
    pass


class PF1MotifStateDataset(Dataset):
    """Read only the E3FP and frozen atom-to-motif fields needed by G1."""

    def __init__(self, paired_release: str, split: str) -> None:
        if split not in ("train", "dev"):
            raise MotifStateDataError("split must be train or dev")
        self.release_root = str(Path(paired_release).resolve())
        membership_path = Path(self.release_root) / "{}_membership.jsonl".format(split)
        if not membership_path.is_file():
            raise MotifStateDataError("membership is missing: {}".format(membership_path))
        self.members: List[Dict[str, Any]] = []
        with membership_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    self.members.append(
                        {
                            "storage_key": str(row["storage_key"]),
                            "member_id": str(row["member_id"]),
                            "selection_index": int(row["selection_index"]),
                        }
                    )
                except Exception as exc:
                    raise MotifStateDataError(
                        "invalid membership row {}:{}".format(membership_path, line_number)
                    ) from exc
        if not self.members:
            raise MotifStateDataError("membership is empty")
        self._env = None

    def __len__(self) -> int:
        return len(self.members)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_env"] = None
        return state

    def close(self) -> None:
        environment = getattr(self, "_env", None)
        if environment is not None:
            environment.close()
            self._env = None

    def __del__(self):
        self.close()

    def _environment(self):
        if self._env is None:
            import lmdb

            self._env = lmdb.open(
                str(Path(self.release_root) / "paired_records.lmdb"),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=512,
            )
        return self._env

    def __getitem__(self, index: int) -> Dict[str, Any]:
        member = self.members[int(index)]
        with self._environment().begin(buffers=False) as transaction:
            payload = transaction.get(member["storage_key"].encode("ascii"))
        if payload is None:
            raise MotifStateDataError("LMDB record is missing: {}".format(member["storage_key"]))
        envelope = json.loads(payload)
        receipt = envelope["receipt"]
        if str(receipt["member_id"]) != member["member_id"]:
            raise MotifStateDataError("membership and receipt member_id disagree")
        document = envelope["motif_training_document"]
        atom_domain = document["atom_domain"]
        e3fp_ids = atom_domain["full_e3fp_ids"]
        atom_valid = atom_domain["atom_valid_mask"]
        atom_is_attachment = atom_domain["atom_is_attachment"]
        atom_to_motif = atom_domain["atom_to_logical_motif"]
        if not (
            len(e3fp_ids)
            == len(atom_valid)
            == len(atom_is_attachment)
            == len(atom_to_motif)
        ):
            raise MotifStateDataError("atom-domain field lengths disagree")
        if not e3fp_ids or any(len(row) != 4 for row in e3fp_ids):
            raise MotifStateDataError("E3FP rows must be nonempty width-four rows")
        motif_count = len(document["logical_motif_domain"]["motif_atom_indices"])
        return {
            **member,
            "e3fp_ids": e3fp_ids,
            "atom_valid": atom_valid,
            "atom_is_attachment": atom_is_attachment,
            "atom_to_motif": atom_to_motif,
            "motif_count": int(motif_count),
        }


def collate_motif_state_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise MotifStateDataError("cannot collate an empty record list")
    batch_size = len(records)
    max_atoms = max(len(record["e3fp_ids"]) for record in records)
    max_motifs = max(int(record["motif_count"]) for record in records)
    e3fp_ids = torch.full((batch_size, max_atoms, 4), -1, dtype=torch.long)
    atom_valid = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    atom_is_attachment = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    atom_to_motif = torch.full((batch_size, max_atoms), -1, dtype=torch.long)
    for batch_index, record in enumerate(records):
        atom_count = len(record["e3fp_ids"])
        e3fp_ids[batch_index, :atom_count] = torch.tensor(
            record["e3fp_ids"], dtype=torch.long
        )
        atom_valid[batch_index, :atom_count] = torch.tensor(
            record["atom_valid"], dtype=torch.bool
        )
        atom_is_attachment[batch_index, :atom_count] = torch.tensor(
            record["atom_is_attachment"], dtype=torch.bool
        )
        atom_to_motif[batch_index, :atom_count] = torch.tensor(
            record["atom_to_motif"], dtype=torch.long
        )
    return {
        "record_ids": tuple(str(record["member_id"]) for record in records),
        "selection_indices": torch.tensor(
            [int(record["selection_index"]) for record in records], dtype=torch.long
        ),
        "e3fp_ids": e3fp_ids,
        "atom_valid": atom_valid,
        "atom_is_attachment": atom_is_attachment,
        "atom_to_motif": atom_to_motif,
        "num_groups": int(max_motifs),
    }
