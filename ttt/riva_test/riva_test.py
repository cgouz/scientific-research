"""
Test how well a Riva-Translate model translates Uzbek -> Russian and Russian -> Uzbek.

Translates N sentences of the test set in each direction, scores chrF / BLEU against the
references, checks the output is in the right script (Cyrillic for ru, Latin for uz), and
writes everything (scores + every translation) to ONE JSON file (--out).

Test file: JSONL with {"pair": "uz-ru", "source": <uz>, "target": <ru>} (or "ru-uz" rows).
Each row is used for both directions: uz-ru as written, ru-uz with source/target swapped.

Usage:
  python riva_test.py                                              # base model, both directions
  python riva_test.py --test-file /data/til_uz-ru/test.jsonl --samples 500
  python riva_test.py --model ../outputs/riva-uz-ru/final --out results/finetuned.json
  python riva_test.py --directions uz-ru                           # one direction only

The prompt is identical to train_riva_uzru.py, so base vs fine-tuned scores are comparable.
Requires: torch transformers sacrebleu
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from gpu_clean import free_gpu

HERE = Path(__file__).resolve().parent
MODEL_ID = "nvidia/Riva-Translate-4B-Instruct-v2"
LANG_NAMES = {"uz": "Uzbek", "ru": "Russian"}
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")


def build_prompt(text, src, tgt):
    s, t = LANG_NAMES[src], LANG_NAMES[tgt]
    return (f"<s>System\nYou are an expert at translating text from {s} to {t}.</s>\n"
            f"<s>User\nWhat is the {t} translation of the sentence: {text.strip()}</s>\n"
            f"<s>Assistant\n")


def load_pairs(path, n, seed):
    """Return [(uz, ru, origin)] from uz-ru and ru-uz rows."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (r.get("source") and r.get("target")):
                continue
            if r.get("pair") == "uz-ru":
                pairs.append((r["source"], r["target"], r.get("origin", "")))
            elif r.get("pair") == "ru-uz":
                pairs.append((r["target"], r["source"], r.get("origin", "")))
    random.Random(seed).shuffle(pairs)
    return pairs[:n]


def load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU — this will be very slow (try --samples 20).")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto" if cuda else None)
    model.eval()
    return tok, model


def translate(tok, model, prompts, batch_size, max_new_tokens):
    import torch

    outs = []
    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            # the prompt already has its <s> markers -> no extra BOS
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            outs += [t.split("</s>")[0].split("<s>")[0].strip()
                     for t in tok.batch_decode(new, skip_special_tokens=True)]
            print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return outs


def in_target_script(text, tgt):
    cyr, lat = len(CYRILLIC.findall(text)), len(LATIN.findall(text))
    return cyr > lat if tgt == "ru" else lat > cyr


def main():
    ap = argparse.ArgumentParser(description="Test Riva-Translate on uz->ru and ru->uz.")
    ap.add_argument("--model", default=MODEL_ID, help="HF id or local folder (e.g. a fine-tuned final/)")
    ap.add_argument("--test-file", default=str(HERE / "test_data" / "test.jsonl"))
    ap.add_argument("--directions", nargs="+", default=["uz-ru", "ru-uz"], choices=["uz-ru", "ru-uz"])
    ap.add_argument("--samples", type=int, default=200, help="sentences per direction")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "results" / "riva_test.json"), help="result file (JSON)")
    args = ap.parse_args()

    import sacrebleu

    test_path = Path(args.test_file)
    if not test_path.exists():
        sys.exit(f"Test file not found: {test_path}")
    pairs = load_pairs(test_path, args.samples, args.seed)
    if not pairs:
        sys.exit(f"No uz-ru / ru-uz rows in {test_path}")
    print(f"Test set: {len(pairs)} sentence pairs from {test_path}")

    print(f"Loading {args.model} ...")
    tok, model = load_model(args.model)
    scores, translations = {}, []
    try:
        for direction in args.directions:
            src, tgt = direction.split("-")
            sources = [uz if src == "uz" else ru for uz, ru, _ in pairs]
            refs = [ru if tgt == "ru" else uz for uz, ru, _ in pairs]
            print(f"\nTranslating {len(sources)} sentences {src} -> {tgt} ...")
            t0 = time.time()
            hyps = translate(tok, model, [build_prompt(s, src, tgt) for s in sources],
                             args.batch_size, args.max_new_tokens)
            secs = time.time() - t0
            scores[direction] = {
                "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
                "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
                "target_script_pct": round(100 * sum(in_target_script(h, tgt) for h in hyps) / len(hyps), 1),
                "copied_input": sum(1 for s, h in zip(sources, hyps) if h and h.lower() == s.lower()),
                "empty": sum(1 for h in hyps if not h),
                "samples": len(hyps),
                "seconds": round(secs, 1),
            }
            translations += [{"direction": direction, "source": s, "reference": r, "hypothesis": h, "origin": o}
                             for s, r, h, (_, _, o) in zip(sources, refs, hyps, pairs)]
    finally:
        del model
        print("\nReleasing GPU memory ...")
        free_gpu()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"model": args.model, "test_file": str(test_path), "date": datetime.now().isoformat(timespec="seconds"),
              "scores": scores, "translations": translations}
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Results ===")
    print(f"  {'direction':<10} {'chrF':>6} {'BLEU':>6} {'script%':>8} {'copied':>7} {'empty':>6}")
    for d, s in scores.items():
        print(f"  {d:<10} {s['chrf']:>6.2f} {s['bleu']:>6.2f} {s['target_script_pct']:>8.1f} "
              f"{s['copied_input']:>7} {s['empty']:>6}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
