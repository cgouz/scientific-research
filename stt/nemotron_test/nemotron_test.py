"""
Check how a Nemotron ASR model transcribes: single audio files, or WER per language on a manifest split.

Data (--bench): <root>/<language>/<split>.jsonl, NeMo manifests with rows
    {"audio_filepath": .., "text": .., "locale": "uz-UZ"}
By default every language is transcribed with the `auto` prompt: the model has to detect the language itself.
--prompt locale uses each language's own locale instead; --prompt both runs both and reports how often they agree.
zh/ja/ko get CER instead of WER. Text is lower-cased and whitespace-normalized before scoring.

Usage:
  python nemotron_test.py --audio a.wav b.wav --lang uz-UZ                  # transcribe files
  python nemotron_test.py --bench                                           # dev WER with `auto`, final.nemo
  python nemotron_test.py --bench --split test --n 500 --prompt both --out finetuned.json
  python nemotron_test.py --model nvidia/nemotron-3.5-asr-streaming-0.6b --bench --out base.json
  bash run.sh --bench --split test --out finetuned.json                     # same, in the background (nohup)

A relative --out goes under /data/experiments/nemotron/results/. Result: one entry per language,
    {"uzbek": {"locale": "uz-UZ", "rows": 200, "metric": "WER", "rate_auto": 0.1234, "rate_locale": .., "agree": ..}}

Requires: nemo_toolkit[asr]
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for general/
from general.gpu_clean import free_gpu
from general.jsonl import read_jsonl

RESULTS_DIR = Path("/data/experiments/nemotron/results")
DEFAULT_MODEL = "/data/experiments/nemotron/final.nemo"

# zh/ja/ko get CER, not WER. Korean has spaces, but they don't mark words the way WER expects.
CER_LOCALES = {"zh-CN", "ja-JP", "ko-KR"}


def edit_distance(a, b):
    """Levenshtein distance between two sequences (words or characters)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,                 # deletion
                           cur[j - 1] + 1,              # insertion
                           prev[j - 1] + (ca != cb)))   # substitution
        prev = cur
    return prev[-1]


def error_rate(refs, hyps, use_cer):
    """WER or CER: total errors / total reference units."""
    err = tot = 0
    for r, h in zip(refs, hyps):
        r_u = list(r.strip()) if use_cer else r.split()
        h_u = list(h.strip()) if use_cer else h.split()
        err += edit_distance(r_u, h_u)
        tot += len(r_u)
    return err / max(tot, 1)


def norm(t):
    """Before comparing: lower case, no extra spaces."""
    t = unicodedata.normalize("NFC", str(t or "")).lower()
    return " ".join(t.split())


def load_model(src):
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        sys.exit("NeMo is not installed: pip install -r requirements.txt")
    print(f"\n  model: {src}")
    if Path(src).exists():
        model = nemo_asr.models.ASRModel.restore_from(src, map_location="cuda")
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(src, map_location="cuda")
    model.eval()
    return model


def make_transcriber(model, batch):
    def transcribe(paths, lang):
        """NeMo versions name the language argument differently: try a few."""
        for kw in ("target_lang", "lang", "prompt"):
            try:
                out = model.transcribe(paths, batch_size=batch, **{kw: lang})
                break
            except TypeError:
                continue
        else:
            out = model.transcribe(paths, batch_size=batch)
        return [getattr(o, "text", o) for o in out]  # NeMo sometimes returns Hypothesis objects
    return transcribe


def run_files(transcribe, paths, lang, prompts):
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        sys.exit(f"not found: {missing[0]}")
    if prompts and lang not in prompts:
        print(f"  WARNING: `{lang}` is not in the model's prompt dictionary.")
        print(f"  Known: {', '.join(sorted(prompts)[:12])} ...")
    print(f"  language: {lang}\n")
    for p, t in zip(paths, transcribe(list(paths), lang)):
        print(f"  {Path(p).name}\n    {t}\n")


def run_bench(transcribe, cfg):
    root = Path(cfg.root)
    langs = cfg.langs or sorted(d.name for d in root.iterdir()
                                if d.is_dir() and (d / f"{cfg.split}.jsonl").exists())
    if not langs:
        sys.exit(f"no {cfg.split}.jsonl under {root}")

    print(f"  {root}/<language>/{cfg.split}.jsonl, up to {cfg.n} rows per language\n")
    results = {}
    for lang in langs:
        p = root / lang / f"{cfg.split}.jsonl"
        if not p.exists():
            continue
        rows = read_jsonl(p)
        if len(rows) > cfg.n:  # every N-th row: the same rows on every run
            step = len(rows) / cfg.n
            rows = [rows[int(i * step)] for i in range(cfg.n)]
        rows = [r for r in rows if Path(r.get("audio_filepath", "")).exists()]
        if not rows:
            print(f"  {lang:<13} no audio found, skipped")
            continue

        loc = rows[0].get("locale", "?")
        paths = [r["audio_filepath"] for r in rows]
        refs = [norm(r.get("text", "")) for r in rows]
        use_cer = loc in CER_LOCALES
        metric = "CER" if use_cer else "WER"

        line = f"  {lang:<13}{loc:<9}{len(rows):>5} rows   {metric}"
        rec = {"locale": loc, "rows": len(rows), "metric": metric}
        hyps = {}
        for prompt in (["auto", "locale"] if cfg.prompt == "both" else [cfg.prompt]):
            hyps[prompt] = [norm(h) for h in transcribe(paths, "auto" if prompt == "auto" else loc)]
            rate = error_rate(refs, hyps[prompt], use_cer)
            rec[f"rate_{prompt}"] = round(rate, 4)
            line += f"   {prompt} {rate:.4f}"
        if len(hyps) == 2:
            # Did the model find the language itself? Same output as with the explicit prompt = yes.
            same = sum(a == b for a, b in zip(hyps["auto"], hyps["locale"]))
            rec["agree"] = round(same / len(rows), 3)
            line += f"   ({100 * same / len(rows):.0f}% identical)"
        rec["rate"] = rec[f"rate_{'auto' if 'auto' in hyps else 'locale'}"]

        print(line)
        results[lang] = rec
        hyps = next(iter(hyps.values()))
        for r, h in list(zip(refs, hyps))[:cfg.show]:
            print(f"      ref: {r[:90]}\n      hyp: {h[:90]}")
        if cfg.show:
            print()

    if results:
        best = min(results, key=lambda k: results[k]["rate"])
        worst = max(results, key=lambda k: results[k]["rate"])
        avg = sum(r["rate"] for r in results.values()) / len(results)
        print("  " + "=" * 62)
        print(f"  best:    {best} {results[best]['rate']:.4f}")
        print(f"  worst:   {worst} {results[worst]['rate']:.4f}")
        print(f"  average: {avg:.4f}  ({len(results)} languages)")
        if cfg.split == "dev":
            print("\n  Note: the checkpoint was picked on dev, so this number looks a bit better than it is.")
            print("  Use --split test for a fair score.")
        print("  " + "=" * 62)
    return results


def main():
    ap = argparse.ArgumentParser(description="Transcribe audio or measure WER with a Nemotron ASR model.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=".nemo file or Hugging Face model name")
    ap.add_argument("--audio", nargs="*", help="one or more audio files to transcribe")
    ap.add_argument("--lang", default="auto", help="language for --audio: uz-UZ, ru-RU, auto ...")
    ap.add_argument("--bench", action="store_true", help="WER per language on <root>/<language>/<split>.jsonl")
    ap.add_argument("--root", default="/data/datasets/stt")
    ap.add_argument("--split", default="dev", choices=("dev", "test", "train"))
    ap.add_argument("--langs", nargs="*", help="language folders to test (default: all)")
    ap.add_argument("--n", type=int, default=200, help="rows per language")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--prompt", default="auto", choices=("auto", "locale", "both"),
                    help="auto = model detects the language (default), locale = each language's own prompt, both = compare")
    ap.add_argument("--show", type=int, default=3, help="examples to print per language")
    ap.add_argument("--out", help="save results as JSON; a relative path goes under " + str(RESULTS_DIR))
    cfg = ap.parse_args()

    if not cfg.audio and not cfg.bench:
        sys.exit("give --audio or --bench")

    model = load_model(cfg.model)
    try:
        try:
            prompts = dict(model.cfg.train_ds.prompt_dictionary)
        except Exception:
            prompts = {}
        transcribe = make_transcriber(model, cfg.batch)
        if cfg.audio:
            run_files(transcribe, cfg.audio, cfg.lang, prompts)
            return 0
        results = run_bench(transcribe, cfg)
    finally:
        del model
        print("  releasing GPU memory ...")
        free_gpu(verbose=False)

    if cfg.out:
        out = Path(cfg.out)
        if not out.is_absolute():
            out = RESULTS_DIR / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  saved: {out}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
