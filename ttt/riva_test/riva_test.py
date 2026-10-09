"""
Check how a Riva-Translate model translates Uzbek -> Russian.

Translates N uz-ru sentences of the test set and writes one row per sentence to a JSONL file:
    {"source": <uzbek>, "target": <reference russian>, "translation": <model output>}

Test file: JSONL with {"pair": "uz-ru", "source": <uz>, "target": <ru>}.

Usage:
  python riva_test.py                                                # base model, 200 sentences
  python riva_test.py --test-file /data/til_uz-ru/test.jsonl --samples 500
  python riva_test.py --model ../outputs/riva-uz-ru/final --out results/finetuned.jsonl

The prompt is identical to train_riva_uzru.py, so base and fine-tuned outputs are comparable.
Requires: torch transformers
"""

import argparse
import json
import random
import sys
from pathlib import Path

from gpu_clean import free_gpu

HERE = Path(__file__).resolve().parent
MODEL_ID = "nvidia/Riva-Translate-4B-Instruct-v2"


def build_prompt(text):
    return ("<s>System\nYou are an expert at translating text from Uzbek to Russian.</s>\n"
            f"<s>User\nWhat is the Russian translation of the sentence: {text.strip()}</s>\n"
            "<s>Assistant\n")


def load_rows(path, n, seed):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("pair") == "uz-ru" and r.get("source") and r.get("target"):
                rows.append({"source": r["source"], "target": r["target"]})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    if not cuda:
        print("WARNING: no CUDA GPU — this will be very slow (try --samples 20).")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto" if cuda else None)
    model.eval()
    return tok, model


def translate(tok, model, texts, batch_size, max_new_tokens):
    import torch

    outs = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            # the prompt already has its <s> markers -> no extra BOS
            enc = tok([build_prompt(t) for t in texts[i:i + batch_size]], return_tensors="pt",
                      padding=True, add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            outs += [t.split("</s>")[0].split("<s>")[0].strip()
                     for t in tok.batch_decode(new, skip_special_tokens=True)]
            print(f"  {min(i + batch_size, len(texts))}/{len(texts)}")
    return outs


def main():
    ap = argparse.ArgumentParser(description="Translate uz->ru test sentences with Riva-Translate.")
    ap.add_argument("--model", default=MODEL_ID, help="HF id or local folder (e.g. a fine-tuned final/)")
    ap.add_argument("--test-file", default=str(HERE / "test_data" / "test.jsonl"))
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "results" / "riva_uz_ru.jsonl"), help="result file (JSONL)")
    args = ap.parse_args()

    test_path = Path(args.test_file)
    if not test_path.exists():
        sys.exit(f"Test file not found: {test_path}")
    rows = load_rows(test_path, args.samples, args.seed)
    if not rows:
        sys.exit(f"No uz-ru rows in {test_path}")
    print(f"Test set: {len(rows)} uz-ru sentences from {test_path}")

    print(f"Loading {args.model} ...")
    tok, model = load_model(args.model)
    try:
        print(f"Translating {len(rows)} sentences uz -> ru ...")
        translations = translate(tok, model, [r["source"] for r in rows], args.batch_size, args.max_new_tokens)
    finally:
        del model
        print("Releasing GPU memory ...")
        free_gpu()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r, t in zip(rows, translations):
            f.write(json.dumps({"source": r["source"], "target": r["target"], "translation": t},
                               ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} rows: {out}")


if __name__ == "__main__":
    main()
