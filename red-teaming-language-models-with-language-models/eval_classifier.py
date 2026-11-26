"""Evaluate a saved classifier checkpoint on BAD splits (no retraining).

Prints acc/F1/AUC plus the share of predicted-positive and labeled-positive examples,
which makes a constant-output model (pred_pos 0 or 100, AUC 50) easy to spot.

--manual bypasses from_pretrained's weight loading: it builds the model from the saved
config and loads the safetensors file by hand, printing any key mismatches and a peek
at the classifier head (an untrained head has a zero bias).

Usage:
  python eval_classifier.py --data data/bad/bot_adversarial_dialogue_datasets_with_persona \
      --ckpt runs/roberta-base [--manual]
"""
import argparse

import torch
import transformers
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)

from bad_inspect import load
from train_classifier import encode, evaluate, fmt


def load_model(ckpt, manual):
    if not manual:
        return AutoModelForSequenceClassification.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(ckpt))
    sd = load_file(f"{ckpt}/model.safetensors")
    bad = [k for k, v in sd.items() if not torch.isfinite(v.float()).all()]
    print("tensors in file:", len(sd), "| non-finite tensors:", bad)
    head = sd.get("classifier.out_proj.bias")
    print("classifier.out_proj.bias:", None if head is None else head.float().tolist())
    res = model.load_state_dict(sd, strict=False)
    print("missing keys:", res.missing_keys)
    print("unexpected keys:", res.unexpected_keys)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir containing the split .txt files")
    ap.add_argument("--ckpt", required=True, help="checkpoint dir saved by train_classifier.py")
    ap.add_argument("--manual", action="store_true", help="load weights by hand (diagnostic)")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    args = ap.parse_args()

    print("torch", torch.__version__, "| transformers", transformers.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.ckpt, truncation_side="left")
    model = load_model(args.ckpt, args.manual).to(device)
    coll = DataCollatorWithPadding(tok)
    for s in args.splits:
        dl = DataLoader(encode(load(f"{args.data}/{s}.txt"), tok, args.max_len),
                        batch_size=args.bs, collate_fn=coll)
        print(f"{s}:  ", fmt(evaluate(model, dl, device)), flush=True)


if __name__ == "__main__":
    main()