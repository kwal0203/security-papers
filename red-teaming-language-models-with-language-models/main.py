from dataclasses import replace
from collections import defaultdict
from collections.abc import Iterable, Mapping

from analysis import dialogue_curve, flagged_rate
from olmo_backend import OlmoBackend
from chatbot_runner import ChatBackend, GenerationConfig, ChatbotRunner
from scoring import LabeledUtterance, ContextAwareScorer, ReplyScore
from case_registry import Case, ExperimentManifest

import argparse
from pathlib import Path

import httpx

from bad_data import build_bad_opening_cases, load_bad_examples
from chatbot_runner import GenerationConfig
from olmo_backend import OlmoBackend


def run_experiment(
    *,
    manifest: ExperimentManifest,
    cases_by_id: Mapping[str, Case],
    backend: ChatBackend,
    config: GenerationConfig,
    train_examples: Iterable[LabeledUtterance],
    test_examples: Iterable[LabeledUtterance],
) -> None:
    scorer = ContextAwareScorer(version="bad-tfidf-v1")
    scorer.fit(train_examples)
    print("Held-out scorer metrics", scorer.evaluate(test_examples))

    test_manifest = replace(
        manifest,
        cases=tuple(entry for entry in manifest.cases if entry.split == "test"),
    )

    runner = ChatbotRunner(backend, "runs.sqlite3")
    scores_by_condition: defaultdict[str, list[ReplyScore]] = defaultdict(list)

    for entry in runner.run_manifest(test_manifest, cases_by_id, config):
        scores_by_condition[entry.condition].extend(scorer.score_run(entry.run))

    for condition, scores in scores_by_condition.items():
        print(condition, flagged_rate(scores))
        print(dialogue_curve(scores, by="turn"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bad_dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1/")
    args = parser.parse_args()

    train_examples = load_bad_examples(args.bad_dir, split="train")
    test_examples = load_bad_examples(args.bad_dir, split="test")
    cases_by_id, manifest = build_bad_opening_cases(args.bad_dir, split="test")

    config = GenerationConfig(
        backend_id="llama.cpp",
        model_id="olmo-local",
        model_revision="olmo-3-7b-instruct-q4_k_m",
        chat_template_version="embedded-gguf",
        top_p=0.8,
        max_new_tokens=128,
    )

    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        run_experiment(
            manifest=manifest,
            cases_by_id=cases_by_id,
            backend=OlmoBackend(client),
            config=config,
            train_examples=train_examples,
            test_examples=test_examples,
        )


if __name__ == "__main__":
    main()
