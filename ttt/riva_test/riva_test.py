"""
Check how a Riva-Translate model translates Uzbek -> Russian and Russian -> Uzbek.

Test file: JSONL rows {"pair": "uz-ru" | "ru-uz", "source": .., "target": ..}. Each row is
translated in the direction of its own "pair". Result: one row per sentence,
    {"pair": "uz-ru", "source": .., "target": <reference>, "translation": <model output>}
written as a JSON list (--out *.json) or one row per line (--out *.jsonl).

Usage:
  python riva_test.py --test-file /data/datasets/ttt/uz-ru/test.jsonl --out results/base.json
  python riva_test.py --test-file ... --samples 500                  # 500 rows per direction
  python riva_test.py --test-file ... --pairs ru-uz                  # one direction only
  python riva_test.py --model ../outputs/riva-uz-ru/final --test-file ... --out results/finetuned.json

The prompt is identical to train_riva_uzru.py, so base and fine-tuned outputs are comparable.
Requires: torch transformers
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from gpu_clean import free_gpu

HERE = Path(__file__).resolve().parent
MODEL_ID = "nvidia/Riva-Translate-4B-Instruct-v2"
LANG_NAMES = {"uz": "Uzbek", "ru": "Russian"}


def build_prompt(text, pair):
    s, t = (LANG_NAMES[x] for x in pair.split("-"))
    return (f"<s>System\nYou are an expert at translating text from {s} to {t}.</s>\n"
            f"<s>User\nWhat is the {t} translation of the sentence: {text.strip()}</s>\n"
            "<s>Assistant\n")


def load_rows(path, pairs, n, seed):
    """{pair: [{"source", "target"}]} with at most n random rows per pair."""
    rows = {p: [] for p in pairs}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("pair") in rows and r.get("source") and r.get("target"):
                rows[r["pair"]].append({"source": r["source"], "target": r["target"]})
    for p in rows:
        random.Random(seed).shuffle(rows[p])
        rows[p] = rows[p][:n]
    return rows


def load_model(model_id, gpu):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device(f"cuda:{gpu}")
        torch.backends.cuda.matmul.allow_tf32 = True
        props = torch.cuda.get_device_properties(device)
        print(f"GPU {gpu}: {props.name}, {props.total_memory / 1024**3:.0f} GB  "
              f"(torch {torch.__version__}, CUDA {torch.version.cuda})")
    else:
        device = torch.device("cpu")
        print("WARNING: no CUDA GPU — running on CPU, this will be very slow. Check nvidia-smi and torch install.")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if cuda else torch.float32
    # whole model on ONE GPU: a 4B model fits easily; device_map="auto" would split it across GPUs (slow)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa").to(device)
    model.eval()
    return tok, model


def stop_ids(tok):
    """Token ids that end an answer: the tokenizer's EOS plus Riva's </s> / <s> turn markers."""
    ids = {tok.eos_token_id}
    for marker in ("</s>", "<s>"):
        i = tok.convert_tokens_to_ids(marker)
        if isinstance(i, int) and i != tok.unk_token_id:
            ids.add(i)
    return sorted(i for i in ids if i is not None)


def translate(tok, model, texts, pair, batch_size, max_new_tokens):
    import torch

    # longest first, similar lengths per batch: less padding, no batch waits on one long sentence
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    outs = [None] * len(texts)
    eos = stop_ids(tok)
    t0, done = time.time(), 0
    with torch.inference_mode():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            # the prompt already has its <s> markers -> no extra BOS
            enc = tok([build_prompt(texts[i], pair) for i in idx], return_tensors="pt",
                      padding=True, add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=eos)
            new = gen[:, enc["input_ids"].shape[1]:]
            for i, t in zip(idx, tok.batch_decode(new, skip_special_tokens=True)):
                outs[i] = t.split("</s>")[0].split("<s>")[0].strip()
            done += len(idx)
            secs = time.time() - t0
            print(f"  {done}/{len(texts)}  {secs:.0f}s  {done / secs:.1f} sent/s  "
                  f"(batch: {new.shape[1]} new tokens)")
    return outs


def main():
    ap = argparse.ArgumentParser(description="Translate uz->ru and ru->uz test sentences with Riva-Translate.")
    ap.add_argument("--model", default=MODEL_ID, help="HF id or local folder (e.g. a fine-tuned final/)")
    ap.add_argument("--test-file", default=str(HERE / "test_data" / "test.jsonl"))
    ap.add_argument("--pairs", nargs="+", default=["uz-ru", "ru-uz"], choices=["uz-ru", "ru-uz"])
    ap.add_argument("--samples", type=int, default=200, help="rows per direction")
    ap.add_argument("--batch-size", type=int, default=128, help="sentences per batch (B200: 128-512)")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "results" / "riva_test.json"), help="result file (.json or .jsonl)")
    args = ap.parse_args()

    test_path = Path(args.test_file)
    if not test_path.exists():
        sys.exit(f"Test file not found: {test_path}")
    rows = load_rows(test_path, args.pairs, args.samples, args.seed)
    for p, rs in rows.items():
        print(f"Test set: {len(rs)} {p} rows from {test_path}")
    if not any(rows.values()):
        sys.exit(f"No {' / '.join(args.pairs)} rows in {test_path}")

    print(f"Loading {args.model} ...")
    tok, model = load_model(args.model, args.gpu)
    results = []
    try:
        for pair, rs in rows.items():
            if not rs:
                continue
            print(f"Translating {len(rs)} sentences {pair.replace('-', ' -> ')} ...")
            translations = translate(tok, model, [r["source"] for r in rs], pair,
                                     args.batch_size, args.max_new_tokens)
            results += [{"pair": pair, "source": r["source"], "target": r["target"], "translation": t}
                        for r, t in zip(rs, translations)]
    finally:
        del model
        print("Releasing GPU memory ...")
        free_gpu()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        if out.suffix == ".jsonl":
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
        else:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} rows: {out}")


if __name__ == "__main__":
    main()
