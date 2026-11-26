"""Fine-tune an encoder to classify BAD utterances as ok/notok given dialogue history.

Each BAD line is (history, label of the last utterance). We join the utterances with
the tokenizer's separator and truncate from the LEFT, so the labeled utterance is
always kept and the oldest turns are dropped first.

Usage:
  python train_classifier.py --data data/bad/bot_adversarial_dialogue_datasets_with_persona
  python train_classifier.py --data <dir> --limit 256 --epochs 1   # smoke test
"""
import argparse

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from bad_inspect import load


def encode(rows, tok, max_len):
    texts = [f" {tok.sep_token} ".join(r["utts"]) for r in rows]
    enc = tok(texts, truncation=True, max_length=max_len)  # tok truncates from the left
    return [{"input_ids": ids, "attention_mask": mask, "labels": r["offensive"]}
            for ids, mask, r in zip(enc["input_ids"], enc["attention_mask"], rows)]


def amp(device, enabled=True):
    on = enabled and device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=on)


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    model.eval()
    probs, labels = [], []
    for b in loader:
        b = dict(b)
        labels.append(b.pop("labels"))
        b = {k: v.to(device) for k, v in b.items()}
        with amp(device, use_amp):
            logits = model(**b).logits
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite logits in eval (bad weights or a numerics/kernel problem)")
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu())
    p, y = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    pred = (p >= 0.5).astype(int)  # same 0.5 threshold the paper uses
    return {"acc": 100 * accuracy_score(y, pred), "f1": 100 * f1_score(y, pred),
            "auc": 100 * roc_auc_score(y, p),
            "pred_pos": 100 * pred.mean(), "label_pos": 100 * y.mean()}


def fmt(m):
    return "  ".join(f"{k} {v:.1f}" for k, v in m.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir containing train/valid/test.txt")
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--out", default="runs/clf")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="use only N rows per split (smoke test)")
    ap.add_argument("--fp32", action="store_true", help="disable bf16 autocast")
    args = ap.parse_args()
    use_amp = not args.fp32

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, truncation_side="left")
    assert tok.sep_token, "needs an encoder tokenizer with a separator token"
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

    sets = {}
    for s in ("train", "valid", "test"):
        rows = load(f"{args.data}/{s}.txt")
        sets[s] = encode(rows[: args.limit] if args.limit else rows, tok, args.max_len)
    coll = DataCollatorWithPadding(tok)
    train_dl = DataLoader(sets["train"], batch_size=args.bs, shuffle=True, collate_fn=coll)
    eval_dl = {s: DataLoader(sets[s], batch_size=args.bs * 2, collate_fn=coll)
               for s in ("valid", "test")}

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    best_f1 = -1.0
    for ep in range(args.epochs):
        model.train()
        for i, b in enumerate(train_dl):
            b = {k: v.to(device) for k, v in b.items()}
            with amp(device, use_amp):
                loss = model(**b).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep} step {i}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            if i % (1 if args.limit else 200) == 0:
                print(f"epoch {ep} step {i}/{len(train_dl)} loss {loss.item():.4f}", flush=True)
        m = evaluate(model, eval_dl["valid"], device, use_amp)
        print(f"epoch {ep} valid: {fmt(m)}", flush=True)
        if m["f1"] > best_f1:  # keep the best epoch by F1 on the offensive class
            best_f1 = m["f1"]
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)

    model = AutoModelForSequenceClassification.from_pretrained(args.out).to(device)
    for s in ("valid", "test"):  # valid again: separates a reload problem from a test-set problem
        print(f"{s}:  ", fmt(evaluate(model, eval_dl[s], device, use_amp)))
    print("paper: acc 84.5  f1 87.5  auc 92.4  (Gopher 1.4B, Table 8)")


if __name__ == "__main__":
    main()