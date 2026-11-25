"""Parse and sanity-check a BAD (Bot-Adversarial Dialogue) file in ParlAI text format.

Each line is one example: tab-separated `key:value` fields. `text` holds the whole
dialogue history, utterances joined by a literal backslash-n. The label applies to
the LAST utterance in `text`.

Usage: python bad_inspect.py path/to/train.txt
"""
import sys
from collections import Counter


def parse_line(line):
    fields = {}
    for part in line.rstrip("\n").split("\t"):
        key, _, val = part.partition(":")  # partition: values may contain colons
        fields[key] = val
    utts = fields["text"].split("\\n")  # literal backslash-n, not a real newline
    return {
        "utts": utts,                          # history; utts[-1] is the labeled utterance
        "ply": len(utts),                      # 1-indexed position of the labeled utterance
        "label": fields["labels"],             # __ok__ or __notok__
        "offensive": int(fields["labels"] == "__notok__"),
        "speaker": fields["speaker_to_eval"],  # "human" or "bot"
        "persona": fields.get("bot_persona", ""),
    }


def load(path):
    with open(path, encoding="utf-8") as f:
        return [parse_line(line) for line in f if line.strip()]


def main(path):
    rows = load(path)
    print(f"{len(rows)} examples")
    print("label values:", dict(Counter(r["label"] for r in rows)))

    # Human should speak on odd plies (human starts); speaker_to_eval should agree.
    mismatches = sum((r["ply"] % 2 == 1) != (r["speaker"] == "human") for r in rows)
    print("parity mismatches (expect 0):", mismatches)

    rates = {}
    for who in ("human", "bot"):
        sub = [r for r in rows if r["speaker"] == who]
        rates[who] = sum(r["offensive"] for r in sub) / max(len(sub), 1)
        print(f"{who}: n={len(sub)}, offensive={rates[who]:.1%}")
    if rates["bot"] > 0:
        print(f"human/bot offensive ratio: {rates['human'] / rates['bot']:.2f}")

    firsts = [r for r in rows if r["ply"] == 1 and r["speaker"] == "human"]
    print(f"first-turn human utterances: {len(firsts)} "
          f"(unique: {len({r['utts'][0] for r in firsts})})")


if __name__ == "__main__":
    main(sys.argv[1])

    print()
    print()
    print()
    print()

    from collections import Counter
    from bad_inspect import load

    d = "data/bad/bot_adversarial_dialogue_datasets_with_persona"
    rows = [r for s in ("train", "valid", "test") for r in load(f"{d}/{s}.txt")]
    firsts = [r["utts"][0] for r in rows if r["ply"] == 1 and r["speaker"] == "human"]
    qs = [u for u in firsts if "?" in u]
    print(len(firsts), len(qs), len(set(qs)))
    print(Counter(firsts).most_common(5))