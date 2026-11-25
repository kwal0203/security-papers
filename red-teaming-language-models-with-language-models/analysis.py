from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from nltk.translate.bleu_score import sentence_bleu  # type: ignore[import-untyped]
from nltk.translate.bleu_score import SmoothingFunction  # type: ignore[import-untyped]

from scoring import ReplyScore
from typing import Any


@dataclass(frozen=True, slots=True)
class Rate:
    flagged: int
    total: int

    @property
    def value(self) -> float | None:
        return self.flagged / self.total if self.total else None


def _check_scoring_condition(scores: Sequence[ReplyScore]) -> None:
    versions = {(score.scorer_version, score.threshold) for score in scores}
    if len(versions) > 1:
        raise ValueError("Use one scorer version and threshold per analysis")


def flagged_rate(scores: Iterable[ReplyScore]) -> Rate:
    rows = tuple(scores)
    _check_scoring_condition(rows)
    return Rate(sum(row.flagged for row in rows), len(rows))


@dataclass(frozen=True, slots=True)
class CurvePoint:
    position: int
    rate: Rate
    contributing_dialogues: int
    interval_95: tuple[float, float] | None


def dialogue_curve(
    scores: Iterable[ReplyScore],
    *,
    by: str = "turn",
    bootstrap_samples: int = 1_000,
    seed: int = 0,
) -> list[CurvePoint]:
    """Group by turn or by the number of preceding flagged replies."""
    if by not in {"turn", "streak"} or bootstrap_samples < 0:
        raise ValueError("Invalid curve mode or bootstrap sample count")

    rows = tuple(scores)
    _check_scoring_condition(rows)
    dialogues: dict[str, list[ReplyScore]] = defaultdict(list)
    for score in rows:
        dialogues[score.attempt_id].append(score)

    contributions: list[dict[int, Rate]] = []
    positions: set[int] = set()
    for replies in dialogues.values():
        replies.sort(key=lambda score: score.assistant_turn)
        flags: Counter[int] = Counter()
        totals: Counter[int] = Counter()
        streak = 0
        for turn, reply in enumerate(replies, start=1):
            if reply.assistant_turn != turn:
                raise ValueError("Expected one score for each consecutive turn")

            # A run of three flagged replies also satisfies "at least two"
            # and "at least one", matching the paper's conditioning.
            bins = (turn,) if by == "turn" else range(1, streak + 1)
            for position in bins:
                flags[position] += reply.flagged
                totals[position] += 1
            streak = streak + 1 if reply.flagged else 0

        contributions.append({
            position: Rate(flags[position], total)
            for position, total in totals.items()
        })
        positions.update(totals)

    totals_by_position: dict[int, Rate] = {}
    contributing_counts: Counter[int] = Counter()
    for position in positions:
        eligible = [row[position] for row in contributions if position in row]
        totals_by_position[position] = Rate(
            sum(rate.flagged for rate in eligible),
            sum(rate.total for rate in eligible),
        )
        contributing_counts[position] = len(eligible)

    rng = random.Random(seed)
    bootstrap: dict[int, list[float]] = defaultdict(list)
    for _ in range(bootstrap_samples if positions else 0):
        flagged: Counter[int] = Counter()
        total: Counter[int] = Counter()
        for dialogue in rng.choices(contributions, k=len(contributions)):
            for position, rate in dialogue.items():
                flagged[position] += rate.flagged
                total[position] += rate.total
        for position, count in total.items():
            bootstrap[position].append(flagged[position] / count)

    return [
        CurvePoint(
            position,
            totals_by_position[position],
            contributing_counts[position],
            _interval(bootstrap[position]),
        )
        for position in sorted(positions)
    ]


def _interval(samples: list[float]) -> tuple[float, float] | None:
    if not samples:
        return None
    samples.sort()
    last = len(samples) - 1
    return (
        samples[math.floor(0.025 * last)],
        samples[math.ceil(0.975 * last)],
    )


def paired_group_gap(
    scores: Iterable[ReplyScore],
    labels: Mapping[str, tuple[str, str]],
    group_a: str,
    group_b: str,
) -> tuple[int, float | None]:
    """Return shared template count and flagged-rate gap A minus B.

    labels maps case_id to (template_id, group_label).
    Repeated replies are averaged within each template/group cell.
    """
    rows = tuple(scores)
    _check_scoring_condition(rows)
    cells: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for score in rows:
        template, group = labels[score.case_id]
        cells[template, group].append(score.flagged)

    shared = (
        {template for template, group in cells if group == group_a}
        & {template for template, group in cells if group == group_b}
    )
    gaps: list[float] = []
    for template in sorted(shared):
        a = cells[template, group_a]
        b = cells[template, group_b]
        gaps.append(sum(a) / len(a) - sum(b) / len(b))
    return len(gaps), sum(gaps) / len(gaps) if gaps else None


def _words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", text.casefold()))


def _ngrams(words: tuple[str, ...], n: int):
    for start in range(len(words) - n + 1):
        yield words[start:start + n]


def diversity(
    questions: Sequence[str],
    *,
    comparisons: int = 1_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute lexical diversity. BLEU and unique trigrams use a 0–100 scale."""
    if comparisons < 1:
        raise ValueError("comparisons must be positive")

    docs = [_words(question) for question in questions]
    docs = [doc for doc in docs if doc]
    rng = random.Random(seed)
    smoothing = SmoothingFunction().method1
    maxima = []
    for index, question in enumerate(docs):
        k = min(comparisons, len(docs) - 1)
        if not k:
            break
        # Sample from all indices except the current one.
        others = rng.sample(range(len(docs) - 1), k)
        maxima.append(max(
            sentence_bleu(
                [docs[other + (other >= index)]],
                question,
                smoothing_function=smoothing,
            )
            for other in others
        ))

    trigrams = Counter(gram for doc in docs for gram in _ngrams(doc, 3))
    total = sum(trigrams.values())
    frequencies = Counter(word for doc in docs for word in doc)
    return {
        "questions": len(docs),
        "max_self_bleu": 100 * sum(maxima) / len(maxima) if maxima else None,
        "zipf_coefficient": _zipf_coefficient(frequencies),
        "unique_trigram_pct": 100 * len(trigrams) / total if total else None,
        "trigram_entropy_bits": (
            -sum((count / total) * math.log2(count / total)
                 for count in trigrams.values())
            if total else None
        ),
    }


def _zipf_coefficient(frequencies: Counter[str]) -> float | None:
    counts = sorted(frequencies.values(), reverse=True)
    if len(counts) < 2:
        return None
    x = [math.log(rank) for rank in range(1, len(counts) + 1)]
    y = [math.log(count) for count in counts]
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    return -(
        sum((rank - mean_x) * (count - mean_y)
            for rank, count in zip(x, y))
        / sum((rank - mean_x) ** 2 for rank in x)
    )


@dataclass(frozen=True, slots=True)
class CorpusMatch:
    span: tuple[str, ...]
    occurrences: int
    document_ids: tuple[str, ...]


class CorpusOverlap:
    """Index normalized word spans in a known synthetic fixture corpus."""

    def __init__(self, documents: Mapping[str, str], *, span_words: int = 13):
        if span_words < 1:
            raise ValueError("span_words must be positive")
        self.span_words = span_words
        self.counts: Counter[tuple[str, ...]] = Counter()
        self.sources: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for document_id, text in documents.items():
            for span in _ngrams(_words(text), span_words):
                self.counts[span] += 1
                self.sources[span].add(document_id)

    def matches(self, reply: str) -> list[CorpusMatch]:
        spans = dict.fromkeys(_ngrams(_words(reply), self.span_words))
        return [
            CorpusMatch(span, self.counts[span], tuple(sorted(self.sources[span])))
            for span in spans
            if span in self.counts
        ]
