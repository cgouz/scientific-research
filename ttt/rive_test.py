"""
Test nvidia/Riva-Translate-4B-Instruct-v2 (WITHOUT training) on your uz->ru test set.

It answers: "How well does the original Riva model already translate Uzbek -> Russian?"
  * translates N test sentences
  * scores chrF / BLEU against the real Russian translations
  * checks the output really is Russian (Cyrillic) and not a copy of the Uzbek input
  * saves every translation to a CSV so you can read them

Three ways to run the model (--backend):
  hf       transformers on your GPU (needs ~9 GB VRAM)                     <- simplest
  server   any OpenAI-compatible server: llama.cpp (llama-server), vLLM
  ollama   Ollama (a model you created with `ollama create`)

Usage:
  python scripts/test_riva.py                                   # hf, 200 sentences from TIL uz-ru test
  python scripts/test_riva.py --samples 500 --test-file data/external/uzlpc/test.jsonl
  python scripts/test_riva.py --backend server --url http://localhost:8080          # llama-server / vLLM
  python scripts/test_riva.py --backend ollama --ollama-model riva-uz-ru            # Ollama
  python scripts/test_riva.py --pair en-ru --test-file my_en_ru_test.jsonl          # an official pair

Why a custom prompt for uz-ru: Riva only knows tags like "en-ru". For an unknown pair its chat
template drops the system message, so this script builds the prompt in the model's exact format:
    <s>System\nYou are an expert at translating text from Uzbek to Russian.</s>\n
    <s>User\nWhat is the Russian translation of the sentence: ...</s>\n<s>Assistant\n

Requires: pip install torch transformers sacrebleu pandas requests
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "nvidia/Riva-Translate-4B-Instruct-v2"
LANG_NAMES = {"uz": "Uzbek", "ru": "Russian", "en": "English", "kk": "Kazakh", "tr": "Turkish",
              "de": "German", "fr": "French", "zh": "Simplified Chinese", "ky": "Kyrgyz", "tg": "Tajik"}
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")


def build_prompt(text, src, tgt):
    s, t = LANG_NAMES.get(src, src), LANG_NAMES.get(tgt, tgt)
    return (f"<s>System\nYou are an expert at translating text from {s} to {t}.</s>\n"
            f"<s>User\nWhat is the {t} translation of the sentence: {text.strip()}</s>\n"
            f"<s>Assistant\n")


def clean_output(o):
    o = o.split("</s>")[0].split("<s>")[0]
    return o.strip()


# ---------------------------------------------------------------------------
# Backends — each returns a function: list[str prompt] -> list[str output]
# ---------------------------------------------------------------------------
def hf_backend(model_id, batch_size, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    if not cuda:
        print("  WARNING: no GPU — this will be very slow. Try --samples 20, or use --backend ollama.")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto" if cuda else None)
    model.eval()

    @torch.inference_mode()
    def run(prompts):
        outs = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            # the prompt already contains the <s> markers -> don't add another BOS
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            outs += [clean_output(t) for t in tok.batch_decode(new, skip_special_tokens=True)]
            print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}")
        return outs

    return run


def server_backend(url, max_new_tokens, model_name):
    import requests

    endpoint = url.rstrip("/") + ("/completions" if url.rstrip("/").endswith("/v1") else "/v1/completions")

    def run(prompts):
        outs = []
        for i, p in enumerate(prompts, 1):
            r = requests.post(endpoint, json={"model": model_name, "prompt": p, "max_tokens": max_new_tokens,
                                              "temperature": 0, "stop": ["</s>", "<s>"]}, timeout=300)
            r.raise_for_status()
            outs.append(clean_output(r.json()["choices"][0]["text"]))
            if i % 20 == 0 or i == len(prompts):
                print(f"  {i}/{len(prompts)}")
        return outs

    return run


def ollama_backend(url, model_name, max_new_tokens):
    import requests

    def run(prompts):
        outs = []
        for i, p in enumerate(prompts, 1):
            # raw=True: send our exact prompt, bypass the Modelfile template
            r = requests.post(url.rstrip("/") + "/api/generate", json={
                "model": model_name, "prompt": p, "raw": True, "stream": False,
                "options": {"temperature": 0, "num_predict": max_new_tokens, "stop": ["</s>", "<s>"]}},
                timeout=300)
            r.raise_for_status()
            outs.append(clean_output(r.json()["response"]))
            if i % 20 == 0 or i == len(prompts):
                print(f"  {i}/{len(prompts)}")
        return outs

    return run


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_test(path, pair, n, seed):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("pair") == pair and r.get("source") and r.get("target"):
                rows.append(r)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main():
    ap = argparse.ArgumentParser(description="Zero-shot test of Riva-Translate on your test set.")
    ap.add_argument("--backend", choices=["hf", "server", "ollama"], default="hf")
    ap.add_argument("--model", default=MODEL_ID, help="HF id or local folder (hf backend)")
    ap.add_argument("--test-file", default=str(PROJECT_ROOT / "data/external/til_uz-ru/test.jsonl"))
    ap.add_argument("--pair", default="uz-ru")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--url", default="http://localhost:8080", help="server backend URL")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-model", default="riva-uz-ru")
    ap.add_argument("--server-model", default="riva", help="model name the server expects")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "riva_zero_shot"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import sacrebleu

    src, tgt = args.pair.split("-", 1)
    test_path = Path(args.test_file)
    if not test_path.exists():
        sys.exit(f"Test file not found: {test_path}\nMake it with til_to_manifest.py, or pass --test-file.")
    rows = load_test(test_path, args.pair, args.samples, args.seed)
    if not rows:
        sys.exit(f"No '{args.pair}' rows in {test_path}")
    print(f"Test set: {len(rows)} {args.pair} sentences from {test_path}")
    print("Example prompt:\n" + "-" * 60 + "\n" + build_prompt(rows[0]["source"], src, tgt) + "\n" + "-" * 60)

    if args.backend == "hf":
        print(f"Loading {args.model} (first time downloads ~8 GB) ...")
        run = hf_backend(args.model, args.batch_size, args.max_new_tokens)
    elif args.backend == "server":
        run = server_backend(args.url, args.max_new_tokens, args.server_model)
    else:
        run = ollama_backend(args.ollama_url, args.ollama_model, args.max_new_tokens)

    # ---- quick sanity check on an official pair: proves the setup works ----
    print("\nSanity check (official en->ru pair):")
    check = run([build_prompt("The weather is very good today.", "en", "ru")])[0]
    print(f"  'The weather is very good today.' -> {check}")
    if not CYRILLIC.search(check or ""):
        print("  WARNING: the official pair did not return Russian — check the backend/model setup.")

    print(f"\nTranslating {len(rows)} sentences {src} -> {tgt} ...")
    t0 = time.time()
    hyps = run([build_prompt(r["source"], src, tgt) for r in rows])
    secs = time.time() - t0
    refs = [r["target"] for r in rows]

    # ---- scores + simple checks ----
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    empty = sum(1 for h in hyps if not h)
    copied = sum(1 for r, h in zip(rows, hyps) if h and h.lower() == r["source"].lower())
    if tgt in ("ru", "kk", "ky", "tg"):
        right_script = sum(1 for h in hyps if len(CYRILLIC.findall(h)) > len(LATIN.findall(h)))
    else:
        right_script = sum(1 for h in hyps if len(LATIN.findall(h)) >= len(CYRILLIC.findall(h)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame({"source": [r["source"] for r in rows], "reference": refs, "riva": hyps,
                  "origin": [r.get("origin", "") for r in rows]}).to_csv(
        out / f"predictions_{args.pair}.csv", index=False, encoding="utf-8-sig")
    result = {"model": args.model if args.backend == "hf" else f"{args.backend}:{args.ollama_model if args.backend == 'ollama' else args.url}",
              "pair": args.pair, "samples": len(rows), "chrf": round(chrf, 2), "bleu": round(bleu, 2),
              "output_in_target_script_pct": round(100 * right_script / len(rows), 1),
              "copied_input": copied, "empty": empty, "seconds": round(secs, 1)}
    (out / f"scores_{args.pair}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Riva-Translate zero-shot result ===")
    print(f"  chrF  : {chrf:6.2f}   (0-100, main score)")
    print(f"  BLEU  : {bleu:6.2f}")
    print(f"  output really in {LANG_NAMES.get(tgt, tgt)}: {100 * right_script / len(rows):.0f}%")
    print(f"  copied the input: {copied}   empty: {empty}   time: {secs:.0f}s")
    verdict = ("very weak — the model barely knows this pair; prefer NLLB" if chrf < 30 else
               "partial — training could help a lot; compare with NLLB before deciding" if chrf < 45 else
               "good — fine-tuning Riva on your data is a reasonable plan")
    print(f"  => {verdict}")
    print("\nExamples:")
    for r, h in list(zip(rows, hyps))[:5]:
        print(f"  {src}: {r['source'][:110]}\n  ref: {r['target'][:110]}\n  riva: {h[:110]}\n")
    print(f"All translations: {out / f'predictions_{args.pair}.csv'}")


if __name__ == "__main__":
    main()