"""
Check how a Riva-Translate model translates Uzbek -> Russian and Russian -> Uzbek.

Test file: JSONL rows {"pair": "uz-ru" | "ru-uz", "source": .., "target": ..}. Each row is
translated in the direction of its own "pair". Result: one row per sentence,
    {"pair": "uz-ru", "source": .., "target": <reference>, "translation": <model output>}
written as a JSON list (--out *.json) or one row per line (--out *.jsonl).
A relative --out goes under /data/experiments/riva/results/.

Usage (uz <-> ru model from ../riva_train):
  python riva_test.py --test-file /data/datasets/ttt/uz-ru/test.jsonl --out base.json                 # base model
  python riva_test.py --model /data/experiments/riva/final --test-file /data/datasets/ttt/uz-ru/test.jsonl --out finetuned.json
  python riva_test.py --test-file /data/datasets/ttt/uz-ru/test.jsonl --samples 500 --pairs ru-uz    # 500 rows, one direction

Usage (multilingual model from ../riva_multilingual_train, all 6 directions):
  python riva_test.py --test-file /data/datasets/ttt/multilingual/test.jsonl --pairs uz-ru ru-uz uz-en en-uz en-ru ru-en \
      --samples 500 --out multi_base.json
  python riva_test.py --model /data/experiments/riva_multilingual/final --test-file /data/datasets/ttt/multilingual/test.jsonl \
      --pairs uz-ru ru-uz uz-en en-uz en-ru ru-en --samples 500 --out multi_finetuned.json

  bash run.sh <same arguments>                                      # same, in the background (nohup)
  python riva_test.py --score multi_base.json multi_finetuned.json  # only compare existing results (no GPU)

The prompt is identical to ../riva_train/train.py and ../riva_multilingual_train/train.py,
so base and fine-tuned outputs are comparable.
BLEU and chrF++ (sacrebleu) are printed per direction and saved to <out>.scores.json.
Requires: torch transformers sacrebleu
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, for general/
from general.gpu_clean import free_gpu
RESULTS_DIR = Path("/data/experiments/riva/results")
MODEL_ID = "nvidia/Riva-Translate-4B-Instruct-v2"
LANG_NAMES = {"uz": "Uzbek", "ru": "Russian", "en": "English"}
PAIRS = ["uz-ru", "ru-uz", "uz-en", "en-uz", "en-ru", "ru-en"]


def build_prompt(text, pair, auto=False):
    """auto=True: the source language is left out of the prompt, so the model has to detect it from the text."""
    s, t = (LANG_NAMES[x] for x in pair.split("-"))
    system = f"You are an expert at translating text to {t}." if auto else \
        f"You are an expert at translating text from {s} to {t}."
    return (f"<s>System\n{system}</s>\n"
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


def translate(tok, model, texts, pair, batch_size, max_new_tokens, auto=False):
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
            enc = tok([build_prompt(texts[i], pair, auto) for i in idx], return_tensors="pt",
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


def score(results):
    """{pair: {"rows", "bleu", "chrf"}}: sacrebleu corpus BLEU and chrF++ per direction, or None without sacrebleu."""
    try:
        import sacrebleu
    except ImportError:
        print('Scores skipped: pip install sacrebleu   (then: python riva_test.py --score <result files>)')
        return None
    by_pair = {}
    for r in results:
        key = r["pair"] + (" auto" if r.get("prompt") == "auto" else "")   # auto-source rows are scored separately
        by_pair.setdefault(key, []).append(r)
    scores = {}
    for pair, rs in by_pair.items():
        hyp, ref = [r["translation"] or "" for r in rs], [[r["target"] for r in rs]]
        scores[pair] = {"rows": len(rs),
                        "bleu": round(sacrebleu.corpus_bleu(hyp, ref).score, 2),
                        "chrf": round(sacrebleu.corpus_chrf(hyp, ref, word_order=2).score, 2)}
    for suffix in ("", " auto"):  # unweighted mean over directions, separately for each prompt type
        group = [s for k, s in scores.items() if k.endswith(" auto") == (suffix == " auto")]
        if len(group) > 1:
            scores["mean" + suffix] = {k: round(sum(s[k] for s in group) / len(group), 2) for k in ("bleu", "chrf")}
            scores["mean" + suffix]["rows"] = sum(s["rows"] for s in group)
    return scores


def print_scores(named):
    """Table of {name: scores}; with two result sets, the last column is the change from the first to the second."""
    names = list(named)
    pairs = sorted({p for s in named.values() for p in s if not p.startswith("mean")})
    pairs += [m for m in ("mean", "mean auto") if any(m in s for s in named.values())]
    delta = len(names) == 2
    cols = names + (["change"] if delta else [])
    print("\n" + f"{'':12}{'rows':>6}" + "".join(f" | {n[:15]:^15}" for n in cols))
    print(f"{'':18}" + " | {:>6} {:>8}".format("BLEU", "chrF++") * len(cols))
    for p in pairs:
        cells = [named[n].get(p) for n in names]
        rows = next((c["rows"] for c in cells if c), 0)
        line = f"{p:<12}{rows:>6}"
        for c in cells:
            line += f" | {c['bleu']:>6.1f} {c['chrf']:>8.1f}" if c else f" | {'-':>6} {'-':>8}"
        if delta and all(cells):
            line += f" | {cells[1]['bleu'] - cells[0]['bleu']:>+6.1f} {cells[1]['chrf'] - cells[0]['chrf']:>+8.1f}"
        print(line)
    print()


def result_path(p):
    p = Path(p)
    return p if p.is_absolute() or p.exists() else RESULTS_DIR / p


def score_files(paths):
    named = {}
    for p in map(result_path, paths):
        if not p.exists():
            sys.exit(f"Result file not found: {p}")
        text = p.read_text(encoding="utf-8")
        results = [json.loads(l) for l in text.splitlines() if l.strip()] if p.suffix == ".jsonl" else json.loads(text)
        s = score(results)
        if s is None:
            sys.exit(1)
        named[p.stem] = s
    print_scores(named)


def main():
    ap = argparse.ArgumentParser(description="Translate test sentences with Riva-Translate and score them (BLEU, chrF++).")
    ap.add_argument("--model", default=MODEL_ID, help="HF id or local folder (e.g. a fine-tuned final/)")
    ap.add_argument("--test-file", default=str(HERE / "test_data" / "test.jsonl"))
    ap.add_argument("--pairs", nargs="+", default=["uz-ru", "ru-uz"], choices=PAIRS)
    ap.add_argument("--samples", type=int, default=200, help="rows per direction")
    ap.add_argument("--batch-size", type=int, default=256, help="sentences per batch (B200: 256-512)")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--prompt", choices=["explicit", "auto", "both"], default="explicit",
                    help="explicit: prompt names the source language; auto: it doesn't (model detects it, needs a model "
                         "trained with data.auto_source_ratio); both: run both and score them separately")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="riva_test.json",
                    help=f"result file (.json or .jsonl); relative paths go under {RESULTS_DIR}")
    ap.add_argument("--score", nargs="+", metavar="RESULT",
                    help="only score existing result files (no model), e.g. --score multi_base.json multi_finetuned.json")
    args = ap.parse_args()
    if args.score:
        return score_files(args.score)

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
        modes = ["explicit", "auto"] if args.prompt == "both" else [args.prompt]
        for mode in modes:
            for pair, rs in rows.items():
                if not rs:
                    continue
                print(f"Translating {len(rs)} sentences {pair.replace('-', ' -> ')} ({mode} prompt) ...")
                translations = translate(tok, model, [r["source"] for r in rs], pair,
                                         args.batch_size, args.max_new_tokens, auto=mode == "auto")
                results += [{"pair": pair, "prompt": mode, "source": r["source"], "target": r["target"],
                             "translation": t} for r, t in zip(rs, translations)]
    finally:
        del model
        print("Releasing GPU memory ...")
        free_gpu()

    out = Path(args.out) if Path(args.out).is_absolute() else RESULTS_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        if out.suffix == ".jsonl":
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
        else:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} rows: {out}")

    scores = score(results)
    if scores:
        print_scores({out.stem: scores})
        s_out = out.with_name(out.stem + ".scores.json")
        s_out.write_text(json.dumps({"model": args.model, "test_file": str(test_path), "scores": scores},
                                    indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved scores: {s_out}")


if __name__ == "__main__":
    main()
