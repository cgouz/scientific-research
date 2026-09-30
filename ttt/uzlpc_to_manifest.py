"""
Convert blinoff/UzLPC (2.33M pairs: English/Russian -> Uzbek) into this project's JSONL manifests.

About the data (from the dataset card):
  * data.csv, 360 MB, one "train" split, CC BY-SA 4.0
  * columns: sentence_id, source_lang (eng/rus), source_texts, translation (Uzbek),
             references (real Uzbek, Wikipedia rows only), confidence_score (LaBSE, Wikipedia only),
             agglutination_index
  * 89.6% Tatoeba sentences, 10.4% Wikipedia
  * the Uzbek "translation" column is MACHINE-GENERATED (Qwen3.5-122B-A10B)

Because the Uzbek side is machine-made and the English/Russian side is human-written, the
recommended training directions are uz -> en and uz -> ru (the model learns to produce the
human side). Add en-uz / ru-uz with --directions if you want them anyway.

Usage:
  python scripts/uzlpc_to_manifest.py                              # download + convert (uz-en, uz-ru)
  python scripts/uzlpc_to_manifest.py --directions uz-en uz-ru en-uz ru-uz
  python scripts/uzlpc_to_manifest.py --langs en                   # only the English part
  python scripts/uzlpc_to_manifest.py --csv data.csv               # already downloaded
  python scripts/uzlpc_to_manifest.py --wiki-uzbek reference --min-confidence 0.8
                                         # Wikipedia rows: use the real Uzbek reference, LaBSE >= 0.8

Output (default data/external/uzlpc/): train.jsonl dev.jsonl test.jsonl stats.json
Requires: pip install pandas huggingface_hub
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent  # everything lives inside riva_train/
REPO_ID = "blinoff/UzLPC"
LANG_MAP = {"eng": "en", "en": "en", "english": "en", "rus": "ru", "ru": "ru", "russian": "ru"}

WS = re.compile(r"\s+")
LATIN = re.compile(r"[A-Za-z]")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
UZ_APOS = re.compile(r"[‘’ʻʼ`´]")
# things an LLM sometimes adds around a translation
LLM_PREFIX = re.compile(r"^(tarjima|translation|перевод|uzbek|o'zbekcha)\s*[:：-]\s*", re.IGNORECASE)
QUOTES = "\"'«»“”„"


def clean(t):
    if not isinstance(t, str):
        return ""
    return WS.sub(" ", t.replace(" ", " ").replace("​", "")).strip()


def clean_uz(uz, src):
    uz = UZ_APOS.sub("'", clean(uz))
    uz = LLM_PREFIX.sub("", uz)
    # model wrapped the whole answer in quotes although the source wasn't quoted
    if len(uz) > 2 and uz[0] in QUOTES and uz[-1] in QUOTES and not (src[:1] in QUOTES and src[-1:] in QUOTES):
        uz = uz[1:-1].strip()
    return uz


def check(src, uz, lang, raw_uz, args):
    if not src or not uz:
        return "empty"
    if "\n" in str(raw_uz).strip():
        return "multi-line output (LLM added extra text)"
    if len(src) < args.min_chars or len(uz) < args.min_chars:
        return "too short"
    if len(src) > args.max_chars or len(uz) > args.max_chars:
        return "too long"
    if src.lower() == uz.lower():
        return "untranslated (identical)"
    n_lat, n_cyr = len(LATIN.findall(uz)), len(CYRILLIC.findall(uz))
    if n_cyr > 0:
        return "Cyrillic in Uzbek side (untranslated Russian / Cyrillic Uzbek)"
    if n_lat < 0.5 * len(uz.replace(" ", "")):
        return "Uzbek side has too few letters"
    s_lat, s_cyr = len(LATIN.findall(src)), len(CYRILLIC.findall(src))
    if lang == "ru" and s_cyr <= s_lat:
        return "Russian source not in Cyrillic"
    if lang == "en" and s_cyr > 0:
        return "English source contains Cyrillic"
    ratio = max(len(src), len(uz)) / max(1, min(len(src), len(uz)))
    if ratio > args.max_ratio and max(len(src), len(uz)) > 15:
        return "length ratio (likely bad translation)"
    return None


def split_of(text, f_test, f_dev):
    u = int.from_bytes(hashlib.blake2b(text.lower().encode(), digest_size=8).digest(), "big") / 2**64
    return "test" if u < f_test else "dev" if u < f_test + f_dev else "train"


def main():
    ap = argparse.ArgumentParser(description="blinoff/UzLPC -> project JSONL manifests")
    ap.add_argument("--csv", default=None, help="local data.csv (skip download)")
    ap.add_argument("--langs", nargs="+", default=["en", "ru"], choices=["en", "ru"])
    ap.add_argument("--directions", nargs="+", default=None,
                    help="default: uz-en uz-ru (machine Uzbek as input, human text as output)")
    ap.add_argument("--wiki-uzbek", choices=["translation", "reference", "both"], default="translation",
                    help="Wikipedia rows: machine translation, the real Uzbek Wikipedia reference, or both")
    ap.add_argument("--min-confidence", type=float, default=None,
                    help="Wikipedia rows: minimum confidence_score (LaBSE), e.g. 0.8")
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--max-ratio", type=float, default=2.5)
    ap.add_argument("--dev", type=int, default=2000, help="approx. dev pairs per language")
    ap.add_argument("--test", type=int, default=2000, help="approx. test pairs per language")
    ap.add_argument("--limit", type=int, default=None, help="read only the first N CSV rows")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "external" / "uzlpc"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    directions = args.directions or [f"uz-{l}" for l in args.langs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- get the CSV ----------
    if args.csv:
        csv_path = Path(args.csv)
    else:
        from huggingface_hub import hf_hub_download

        print(f"Downloading {REPO_ID}/data.csv (~360 MB, cached, resumable) ...")
        csv_path = Path(hf_hub_download(REPO_ID, "data.csv", repo_type="dataset"))
    print(f"Reading {csv_path}")

    # ---------- read in chunks, clean ----------
    pairs = {"en": [], "ru": []}  # (src, uz, origin)
    dropped, seen, n_rows = Counter(), set(), 0
    reader = pd.read_csv(csv_path, dtype=str, keep_default_na=False, chunksize=200_000,
                         nrows=args.limit, on_bad_lines="warn")
    for chunk in reader:
        cols = set(chunk.columns)
        need = {"source_lang", "source_texts", "translation"}
        if not need <= cols:
            sys.exit(f"Unexpected columns: {list(chunk.columns)} (need {sorted(need)})")
        for r in chunk.itertuples(index=False):
            n_rows += 1
            lang = LANG_MAP.get(str(r.source_lang).strip().lower())
            if lang is None:
                dropped[f"unknown source_lang '{r.source_lang}'"] += 1
                continue
            if lang not in args.langs:
                continue
            src = clean(r.source_texts)
            ref = clean(getattr(r, "references", ""))
            conf = getattr(r, "confidence_score", "")
            is_wiki = bool(ref) or conf not in ("", None)
            if is_wiki and args.min_confidence is not None:
                try:
                    if float(conf) < args.min_confidence:
                        dropped[f"Wikipedia confidence < {args.min_confidence}"] += 1
                        continue
                except ValueError:
                    pass
            candidates = [(r.translation, "tatoeba-mt" if not is_wiki else "wiki-mt")]
            if is_wiki and ref and args.wiki_uzbek != "translation":
                ref_c = (ref, "wiki-ref")
                candidates = [ref_c] if args.wiki_uzbek == "reference" else candidates + [ref_c]
            for raw_uz, origin in candidates:
                uz = clean_uz(raw_uz, src)
                why = check(src, uz, lang, raw_uz, args)
                if why:
                    dropped[why] += 1
                    continue
                k = hashlib.blake2b(f"{lang}\t{src.lower()}\t{uz.lower()}".encode(), digest_size=8).digest()
                if k in seen:
                    dropped["duplicate"] += 1
                    continue
                seen.add(k)
                pairs[lang].append((src, uz, origin))
        print(f"  read {n_rows:,} rows  (kept en {len(pairs['en']):,} / ru {len(pairs['ru']):,})")

    # ---------- split by source sentence (same sentence never in two splits) ----------
    rng = random.Random(args.seed)
    splits = {"train": [], "dev": [], "test": []}
    per_lang = {}
    for lang, rows in pairs.items():
        if not rows:
            continue
        f_test, f_dev = min(0.1, args.test / len(rows)), min(0.1, args.dev / len(rows))
        counts = Counter()
        for src, uz, origin in rows:
            s = split_of(src, f_test, f_dev)
            splits[s].append((lang, src, uz, origin))
            counts[s] += 1
        per_lang[lang] = dict(counts)

    written = {}
    for split, rows in splits.items():
        rng.shuffle(rows)
        n = 0
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for lang, src, uz, origin in rows:
                for d in directions:
                    if d == f"uz-{lang}":
                        rec = {"pair": d, "source": uz, "target": src}
                    elif d == f"{lang}-uz":
                        rec = {"pair": d, "source": src, "target": uz}
                    else:
                        continue
                    rec["origin"] = f"uzlpc:{origin}"
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
        written[split] = n

    stats = {
        "dataset": REPO_ID, "csv_rows_read": n_rows, "directions": directions,
        "pairs_per_language": per_lang, "rows_written": written, "dropped": dict(dropped.most_common()),
        "note": "Uzbek side is machine-generated (Qwen3.5-122B-A10B) except wiki-ref rows.",
        "license": "CC BY-SA 4.0 (share-alike); Tatoeba source sentences CC BY 2.0 FR",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    print(f"csv rows read: {n_rows:,}")
    for lang, c in per_lang.items():
        print(f"  {lang}: " + "  ".join(f"{k} {v:,}" for k, v in sorted(c.items())))
    for why, c in dropped.most_common():
        print(f"  dropped {c:>9,}  {why}")
    for split in ("train", "dev", "test"):
        print(f"  {split:<5} {written[split]:>10,} rows -> {out_dir / (split + '.jsonl')}")
    for lang, src, uz, origin in rng.sample(splits["train"], min(4, len(splits["train"]))):
        print(f"\n  [{origin}] {lang}: {src[:120]}\n  {'':>{len(origin) + 3}}uz: {uz[:120]}")


if __name__ == "__main__":
    main()
