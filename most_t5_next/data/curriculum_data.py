"""Dataset routing for the two-phase MoSt-T5 curriculum.

PCQM and PubChem remain independent read-only caches.  Molecular denoising
sees their logical concatenation, while geometry-only and paired-text tasks
address the source cache required by the training protocol.  No molecular
payload is copied to construct the union.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence


MolecularSource = Literal["pcqm", "pubchem"]
SUPPORTED_TASKS = frozenset({"M", "MG", "SYN", "TXT", "CAP", "T2M"})


class CurriculumDataError(ValueError):
    """The configured caches do not satisfy the curriculum contract."""


@dataclass(frozen=True)
class MolecularExample:
    """One molecule together with its source-local and union coordinates."""

    source: MolecularSource
    source_index: int
    union_index: int
    record: Any


@dataclass(frozen=True)
class PairedExample:
    """One PubChem molecule and its aligned description token sequence."""

    source_index: int
    record: Any
    text_input_ids: Any


class MolecularCacheUnion(Sequence[MolecularExample]):
    """Zero-copy concatenation of PCQM followed by PubChem.

    The component caches are externally owned.  This view deliberately does
    not deduplicate model-visible identities: repeated observations remain
    valid training exposure and source provenance stays recoverable.
    """

    def __init__(self, pcqm: Sequence[Any], pubchem: Sequence[Any]) -> None:
        self.pcqm = pcqm
        self.pubchem = pubchem
        self.pcqm_size = len(pcqm)
        self.pubchem_size = len(pubchem)
        if self.pcqm_size <= 0 or self.pubchem_size <= 0:
            raise CurriculumDataError("PCQM and PubChem caches must both be non-empty")

    def __len__(self) -> int:
        return self.pcqm_size + self.pubchem_size

    def locate(self, union_index: int) -> tuple[MolecularSource, int]:
        if isinstance(union_index, bool) or not isinstance(union_index, int):
            raise IndexError("union index must be an integer")
        if not 0 <= union_index < len(self):
            raise IndexError(union_index)
        if union_index < self.pcqm_size:
            return "pcqm", union_index
        return "pubchem", union_index - self.pcqm_size

    def __getitem__(self, union_index: int) -> MolecularExample:
        source, source_index = self.locate(union_index)
        cache = self.pcqm if source == "pcqm" else self.pubchem
        return MolecularExample(
            source=source,
            source_index=source_index,
            union_index=union_index,
            record=cache[source_index],
        )


class CurriculumDataRouter:
    """Map each curriculum task onto its frozen data population."""

    def __init__(
        self,
        *,
        pcqm: Sequence[Any],
        pubchem: Sequence[Any],
        pubchem_text: Sequence[Any],
        text: Sequence[Any],
    ) -> None:
        self.pcqm = pcqm
        self.pubchem = pubchem
        self.pubchem_text = pubchem_text
        self.text = text
        self.molecular_union = MolecularCacheUnion(pcqm, pubchem)
        if len(pubchem) != len(pubchem_text):
            raise CurriculumDataError(
                "PubChem molecular and paired-text caches must share one row axis"
            )
        if len(text) <= 0:
            raise CurriculumDataError("the text-denoising population is empty")

    def population_size(self, task: str) -> int:
        if task in {"M", "SYN"}:
            return len(self.molecular_union)
        if task == "MG":
            return len(self.pcqm)
        if task in {"CAP", "T2M"}:
            return len(self.pubchem)
        if task == "TXT":
            return len(self.text)
        raise KeyError(task)

    def get(self, task: str, index: int) -> Any:
        if task in {"M", "SYN"}:
            return self.molecular_union[index]
        if task == "MG":
            return MolecularExample(
                source="pcqm",
                source_index=index,
                union_index=index,
                record=self.pcqm[index],
            )
        if task in {"CAP", "T2M"}:
            record = self.pubchem[index]
            text_identity, text_input_ids = self.pubchem_text[index]
            record_identity = getattr(record, "ordinal", None)
            if record_identity != text_identity:
                raise CurriculumDataError(
                    "PubChem molecular and paired-text identities are not aligned"
                )
            return PairedExample(
                source_index=index,
                record=record,
                text_input_ids=text_input_ids,
            )
        if task == "TXT":
            return self.text[index]
        raise KeyError(task)


__all__ = [
    "CurriculumDataError",
    "CurriculumDataRouter",
    "MolecularCacheUnion",
    "MolecularExample",
    "MolecularSource",
    "PairedExample",
    "SUPPORTED_TASKS",
]
