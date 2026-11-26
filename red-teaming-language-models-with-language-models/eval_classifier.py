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


def load_model(ckpt, manual, attn_implementation=None):
    kwargs = ({"attn_implementation": attn_implementation} if attn_implementation else {})
    if not manual:
        return AutoModelForSequenceClassification.from_pretrained(ckpt, **kwargs)
    model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(ckpt), **kwargs)
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
    ap.add_argument("--fp32", action="store_true", help="disable bf16 autocast")
    ap.add_argument("--attn_implementation", choices=["eager", "sdpa"], default=None)
    ap.add_argument("--diagnose", action="store_true", help="inspect weights, buffers, and input sensitivity")
    args = ap.parse_args()

    print("torch", torch.__version__, "| transformers", transformers.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.ckpt, truncation_side="left")
    model = load_model(args.ckpt, args.manual, args.attn_implementation).to(device)
    print("checkpoint:", args.ckpt, "| device:", device,
          "| bf16 autocast:", not args.fp32 and device.type == "cuda" and torch.cuda.is_bf16_supported(),
          "| attention:", model.config._attn_implementation)
    coll = DataCollatorWithPadding(tok)
    if args.diagnose:
        print("tokenizer:", type(tok).__name__, "| vocab:", len(tok),
              "| special IDs:", tok.all_special_ids, flush=True)
        for name, value in model.named_parameters():
            if "LayerNorm" in name or "layer_norm" in name or name.startswith("classifier."):
                value = value.detach().float()
                print(f"parameter {name}: min={value.min().item():.8g} "
                      f"max={value.max().item():.8g} norm={value.norm().item():.8g}", flush=True)
        for name, value in model.named_buffers():
            print(f"buffer {name}: shape={tuple(value.shape)} "
                  f"first={value.flatten()[:16].tolist()}", flush=True)
        probes = tok(["Hello there.", "A completely different sentence!"],
                     padding=True, return_tensors="pt")
        print("probe input_ids:", probes["input_ids"].tolist(), flush=True)
        model.eval()
        with torch.no_grad():
            result = model(**{k: v.to(device) for k, v in probes.items()}, output_hidden_states=True)
        print("probe logits (fp32):", result.logits.float().tolist(), flush=True)
        for i, hidden in enumerate(result.hidden_states):
            cls = hidden[:, 0].float()
            print(f"layer {i} CLS norm={cls.norm(dim=-1).tolist()} "
                  f"between-input max_diff={(cls[0] - cls[1]).abs().max().item():.8g}", flush=True)
    for s in args.splits:
        dl = DataLoader(encode(load(f"{args.data}/{s}.txt"), tok, args.max_len),
                        batch_size=args.bs, collate_fn=coll)
        print(f"{s}:  ", fmt(evaluate(model, dl, device, use_amp=not args.fp32)), flush=True)


if __name__ == "__main__":
    main()
