"""Load the extracted ParlAI Bot-Adversarial Dialogue v0.2 text splits."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from case_registry import (
    Case, CaseSource, ExperimentCase, ExperimentManifest, SourceAnnotation
)
from chatbot_runner import Message
from scoring import LabeledUtterance, Utterance

Split = Literal["train", "validation", "valid", "test"]
SafetyLabel = Literal["safe", "offensive"]
_DATASET = "bot_adversarial_dialogue"
_VERSION = "v0.2"
_ARCHIVE_DIR = "bot_adversarial_dialogue_datasets_with_persona"
_LABELS: dict[str, SafetyLabel] = {"__ok__": "safe", "__notok__": "offensive"}


@dataclass(frozen=True, slots=True)
class _BadRow:
    line_number: int
    messages: tuple[Message, ...]
    label: SafetyLabel


def _split_file(bad_dir: str | Path, split: Split) -> Path:
    if split not in {"train", "validation", "valid", "test"}:
        raise ValueError(f"Unknown BAD split: {split!r}")

    filename = "valid.txt" if split in {"valid", "validation"} else f"{split}.txt"
    directory = Path(bad_dir)
    direct = directory / filename
    nested = directory / _ARCHIVE_DIR / filename
    for path in (direct, nested):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing BAD split. Looked for {direct} and {nested}")


def _read_rows(path: Path) -> Iterator[_BadRow]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue

            try:
                fields = dict(field.split(":", 1) for field in line.rstrip("\r\n").split("\t"))
                label = _LABELS[fields["labels"]]
                text = fields["text"]
                speaker = fields["speaker_to_eval"]
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: Invalid BAD record") from exc

            # Each row contains the prefix through its labeled utterance.
            turns = text.split(r"\n")
            if any(not turn.strip() for turn in turns):
                raise ValueError(f"{path}:{line_number}: Empty BAD utterance")

            expected_speaker = "human" if len(turns) % 2 else "bot"
            if speaker != expected_speaker:
                raise ValueError(f"{path}:{line_number}: Unexpected final "
                                  "speaker {speaker!r}")

            messages = tuple(
                Message(
                    "user" if index % 2 == 0 else "assistant",
                    turn.replace(r"\t", "\t").replace("__PIPE__", "|"),
                )
                for index, turn in enumerate(turns)
            )
            yield _BadRow(line_number, messages, label)


def load_bad_examples(
    bad_dir: str | Path, *, split: Split
) -> list[LabeledUtterance]:
    """Make one scorer example per labeled utterance in the published split."""
    return [
        LabeledUtterance(
            utterance=Utterance(history=row.messages[:-1], message=row.messages[-1]),
            offensive=row.label == "offensive",
        )
        for row in _read_rows(_split_file(bad_dir, split))
    ]


def build_bad_opening_cases(
    bad_dir: str | Path, *, split: Split = "test"
) -> tuple[dict[str, Case], ExperimentManifest]:
    """Make one fixed chatbot case per original conversation."""
    path = _split_file(bad_dir, split)
    split_name = path.stem
    manifest_cases: list[ExperimentCase] = []
    cases_by_id: dict[str, Case] = {}
    for row in _read_rows(path):
        if len(row.messages) != 1:
            continue

        # The source file and line identify this conversation's opener.
        record_id = f"{path.name}:{row.line_number}"
        case_id = f"bad-{_VERSION}-{split_name}-line-{row.line_number}"
        cases_by_id[case_id] = Case(
            case_id=case_id,
            user_turns=(row.messages[0].content,),
            source=CaseSource(
                dataset=_DATASET,
                version=_VERSION,
                record_id=record_id,
                conversation_id=record_id,
            ),
            source_annotations=(
                SourceAnnotation(source_utterance_index=0, label=row.label),
            ),
        )
        membership_split = "validation" if split == "valid" else split
        manifest_cases.append(
            ExperimentCase(
                case_id=case_id,
                condition="human_bad",
                split=membership_split
            )
        )

    manifest = ExperimentManifest(
        experiment_id=f"bad-openers-{split_name}",
        version=_VERSION,
        cases=tuple(manifest_cases),
    )
    return cases_by_id, manifest
