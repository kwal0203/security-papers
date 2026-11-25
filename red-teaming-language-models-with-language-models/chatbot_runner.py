"""Replay fixed chatbot cases and store generation attempts for later scoring."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4


class CaseInput(Protocol):
    @property
    def case_id(self) -> str: ...

    @property
    def user_turns(self) -> tuple[str, ...]: ...


class ManifestEntry(Protocol):
    @property
    def case_id(self) -> str: ...

    @property
    def condition(self) -> str: ...

    @property
    def split(self) -> str: ...


class Manifest(Protocol):
    @property
    def experiment_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def cases(self) -> tuple[ManifestEntry, ...]: ...


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    backend_id: str
    model_id: str
    model_revision: str
    chat_template_version: str
    system_prompt: str | None = None
    temperature: float = 0.7
    top_p: float = 0.8
    max_new_tokens: int = 256
    stop: tuple[str, ...] = ()
    base_seed: int = 0

    def __post_init__(self) -> None:
        identifiers = (
            self.backend_id, self.model_id, self.model_revision, self.chat_template_version
        )
        if not all(identifier.strip() for identifier in identifiers):
            raise ValueError("Backend, model, revision, and template IDs are required")
        if not 0 <= self.temperature < float("inf"):
            raise ValueError("temperature must be finite and nonnegative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    seed_applied: bool | None = None
    effective_model_revision: str | None = None
    effective_settings: dict[str, str | int | float | bool | None] | None = None


class BackendError(Exception):
    """The adapter should raise this for expected generation failures."""


class ChatBackend(Protocol):
    """Apply the requested configuration and report available backend metadata."""

    def generate(
        self,
        messages: tuple[Message, ...],
        settings: GenerationConfig,
        *,
        seed: int,
    ) -> GenerationResult: ...


@dataclass(frozen=True, slots=True)
class TurnResult:
    # Zero-based index into the transcript, including any system message.
    assistant_message_index: int
    requested_seed: int
    generation: GenerationResult


@dataclass(frozen=True, slots=True)
class CaseRun:
    attempt_id: str
    cache_key: str
    case_id: str
    input_hash: str
    config: GenerationConfig
    replicate: int
    transcript: tuple[Message, ...]
    turns: tuple[TurnResult, ...]
    error: str | None = None
    failed_seed: int | None = None

    @property
    def completed(self) -> bool:
        return self.failed_seed is None


@dataclass(frozen=True, slots=True)
class ManifestCaseRun:
    experiment_id: str
    experiment_version: str
    position: int
    condition: str
    split: str
    run: CaseRun


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _decode_run(payload: str) -> CaseRun:
    data = json.loads(payload)
    config = data["config"]
    config["stop"] = tuple(config["stop"])
    data["config"] = GenerationConfig(**config)
    data["transcript"] = tuple(Message(**message) for message in data["transcript"])
    data["turns"] = tuple(
        TurnResult(
            assistant_message_index=turn["assistant_message_index"],
            requested_seed=turn["requested_seed"],
            generation=GenerationResult(**turn["generation"]),
        )
        for turn in data["turns"]
    )
    return CaseRun(**data)


class ChatbotRunner:
    def __init__(self, backend: ChatBackend, database: str | Path) -> None:
        self.backend = backend
        self.database = Path(database).resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)

        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS case_runs (
                    sequence INTEGER PRIMARY KEY,
                    attempt_id TEXT NOT NULL UNIQUE,
                    cache_key TEXT NOT NULL,
                    completed INTEGER NOT NULL,
                    payload TEXT NOT NULL
                )
            """)
            db.execute("""
                CREATE INDEX IF NOT EXISTS completed_case_runs
                ON case_runs (cache_key, completed, sequence DESC)
            """)

    def get_attempt(self, attempt_id: str) -> CaseRun | None:
        with closing(sqlite3.connect(self.database)) as db:
            row = db.execute(
                "SELECT payload FROM case_runs WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
        return _decode_run(row[0]) if row else None

    def _completed_run(self, cache_key: str) -> CaseRun | None:
        with closing(sqlite3.connect(self.database)) as db:
            row = db.execute(
                """
                SELECT payload FROM case_runs
                WHERE cache_key = ? AND completed = 1
                ORDER BY sequence DESC LIMIT 1
                """,
                (cache_key,),
            ).fetchone()
        return _decode_run(row[0]) if row else None

    def run_case(
        self,
        case: CaseInput,
        config: GenerationConfig,
        *,
        replicate: int = 0,
        resume: bool = True,
    ) -> CaseRun:
        user_turns = tuple(case.user_turns)
        if not case.case_id or not user_turns or any(not text.strip() for text in user_turns):
            raise ValueError("A case needs an ID and at least one nonblank user turn")
        if replicate < 0:
            raise ValueError("replicate must be nonnegative")

        input_hash = _digest(user_turns)
        cache_key = _digest(
            ("fixed-chat-v1", case.case_id, input_hash, asdict(config), replicate)
        )
        if resume:
            cached = self._completed_run(cache_key)
            if cached is not None:
                return cached

        transcript: list[Message] = []
        if config.system_prompt is not None:
            transcript.append(Message("system", config.system_prompt))

        turns: list[TurnResult] = []
        error = None
        failed_seed = None
        for turn_index, user_text in enumerate(user_turns):
            transcript.append(Message("user", user_text))
            seed_digest = _digest((config.base_seed, case.case_id, replicate, turn_index))
            # Keep seeds within the nonnegative signed 32-bit range.
            seed = int(seed_digest[:8], 16) % (2**31)

            try:
                generation = self.backend.generate(tuple(transcript), config, seed=seed)
            except (BackendError, TimeoutError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                failed_seed = seed
                break

            turns.append(TurnResult(len(transcript), seed, generation))
            transcript.append(Message("assistant", generation.text))

        run = CaseRun(
            attempt_id=uuid4().hex,
            cache_key=cache_key,
            case_id=case.case_id,
            input_hash=input_hash,
            config=config,
            replicate=replicate,
            transcript=tuple(transcript),
            turns=tuple(turns),
            error=error,
            failed_seed=failed_seed,
        )
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute(
                """
                INSERT INTO case_runs (attempt_id, cache_key, completed, payload)
                VALUES (?, ?, ?, ?)
                """,
                (run.attempt_id, run.cache_key, run.completed, _json(asdict(run))),
            )
        return run

    def run_manifest(
        self,
        manifest: Manifest,
        cases: Mapping[str, CaseInput],
        config: GenerationConfig,
        *,
        replicate: int = 0,
        resume: bool = True,
    ) -> Iterator[ManifestCaseRun]:
        resolved_cases = [cases[entry.case_id] for entry in manifest.cases]
        for entry, case in zip(manifest.cases, resolved_cases, strict=True):
            if entry.case_id != case.case_id:
                raise ValueError(f"Registry key does not match case ID: {entry.case_id}")

        for position, (entry, case) in enumerate(
            zip(manifest.cases, resolved_cases, strict=True)
        ):
            run = self.run_case(case, config, replicate=replicate, resume=resume)
            yield ManifestCaseRun(
                experiment_id=manifest.experiment_id,
                experiment_version=manifest.version,
                position=position,
                condition=entry.condition,
                split=entry.split,
                run=run,
            )
