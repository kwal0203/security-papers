from dataclasses import dataclass
from typing import Literal


Split = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class CaseSource:
    dataset: str
    version: str
    record_id: str | None = None
    conversation_id: str | None = None


@dataclass(frozen=True, slots=True)
class SourceAnnotation:
    source_utterance_index: int
    label: Literal["safe", "offensive"]


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    user_turns: tuple[str, ...]
    source: CaseSource
    source_annotations: tuple[SourceAnnotation, ...] = ()
    template_id: str | None = None
    group_label: str | None = None
    fixture_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExperimentCase:
    case_id: str
    condition: str
    split: Split = "test"


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    experiment_id: str
    version: str
    cases: tuple[ExperimentCase, ...]


