"""Task-aware sequence-length decisions for MoSt-T5.

Text can be truncated at a documented boundary.  fragSMILES cannot: ordinary
token truncation may split a fragment, endpoint, carrier, or geometry address.
The source record is always retained; only an unsafe task view is ineligible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class LengthDecision:
    record_id: str
    task: str
    input_length: int
    target_length: int
    action: str
    admitted: bool


class LengthPolicy:
    """Classify task views without mutating source records."""

    def __init__(
        self,
        *,
        input_limit: int = 512,
        target_limit: int = 512,
        text_raw_block_length: int = 568,
        text_corruption_target_length: int = 114,
    ) -> None:
        values = (
            input_limit,
            target_limit,
            text_raw_block_length,
            text_corruption_target_length,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("length-policy values must be positive integers")
        if text_raw_block_length < input_limit:
            raise ValueError("raw text block must cover the corrupted input")
        self.input_limit = input_limit
        self.target_limit = target_limit
        self.text_raw_block_length = text_raw_block_length
        self.text_corruption_target_length = text_corruption_target_length

    def decide(
        self,
        *,
        record_id: str,
        task: str,
        input_length: int,
        target_length: int,
    ) -> LengthDecision:
        if task not in {"M", "MG", "SYN", "TXT", "CAP", "T2M"}:
            raise ValueError(f"unknown task: {task}")
        if input_length <= 0 or target_length < 0:
            raise ValueError("sequence lengths are invalid")

        if task == "TXT":
            admitted = (
                input_length == self.text_raw_block_length
                and target_length <= self.text_corruption_target_length
            )
            return LengthDecision(
                record_id,
                task,
                input_length,
                target_length,
                "online_span_corruption_568_to_512_114"
                if admitted
                else "reject_invalid_text_block_contract",
                admitted,
            )
        if task == "T2M":
            if target_length > self.target_limit:
                return LengthDecision(
                    record_id,
                    task,
                    input_length,
                    target_length,
                    "exclude_structural_target_view",
                    False,
                )
            return LengthDecision(
                record_id,
                task,
                input_length,
                target_length,
                "truncate_text_input_right" if input_length > self.input_limit else "keep",
                True,
            )
        if task == "CAP":
            if input_length > self.input_limit:
                return LengthDecision(
                    record_id,
                    task,
                    input_length,
                    target_length,
                    "exclude_structural_input_view",
                    False,
                )
            return LengthDecision(
                record_id,
                task,
                input_length,
                target_length,
                "truncate_text_target_right_keep_eos"
                if target_length > self.target_limit
                else "keep",
                True,
            )
        # M, MG and SYN use the frozen molecular corruption contract.  Neither
        # side may be repaired by ordinary token truncation.
        admitted = (
            input_length <= self.input_limit
            and target_length <= self.text_corruption_target_length
        )
        return LengthDecision(
            record_id,
            task,
            input_length,
            target_length,
            "keep" if admitted else "exclude_structural_task_view",
            admitted,
        )


def write_length_action_ledger(
    decisions: Iterable[LengthDecision], path: str | Path
) -> dict[str, object]:
    """Write every classification and return a compact count receipt."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    admitted = 0
    total = 0
    with destination.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(asdict(decision), sort_keys=True) + "\n")
            total += 1
            admitted += int(decision.admitted)
            counts[decision.action] = counts.get(decision.action, 0) + 1
    return {
        "status": "pass",
        "records": total,
        "admitted": admitted,
        "excluded": total - admitted,
        "actions": dict(sorted(counts.items())),
        "ledger": str(destination.resolve()),
    }


__all__ = ["LengthDecision", "LengthPolicy", "write_length_action_ledger"]
