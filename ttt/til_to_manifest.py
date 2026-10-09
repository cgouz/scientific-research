"""
Convert a downloaded TIL corpus pair (e.g. data/uz-ru/) into this project's JSONL manifests.

TIL layout (from til-mt/til_corpus/download_data.py):
    data/uz-ru/train/uz-ru/uz-ru.uz   data/uz-ru/train/uz-ru/uz-ru.ru     (line-aligned)
    data/uz-ru/dev/...                data/uz-ru/test/...                 (if they exist)

What it does
  1. finds the aligned file pairs for train / dev / test, checks line counts match
  2. DETOKENIZES the text — TIL is Moses-tokenized:
        "so ‘ ng , oliy o ‘ quv"      -> "so'ng, oliy o'quv"
        "СЭЗ « Сирдарё » :"          -> "СЭЗ «Сирдарё»:"
        "“ Baholash ... ” gi"        -> "“Baholash ...”gi"
  3. cleans: empty, too short/long, untranslated, wrong script, misaligned length ratio,
     duplicates; optional LaBSE meaning check (--labse 0.75, needs a GPU for 1M+ rows)
  4. removes train pairs that also appear in dev/test (no test leakage)
  5. writes {"pair": "uz-ru", "source": ..., "target": ..., "origin": "til"} JSONL

Usage:
    python til_to_manifest.py --data-dir data/uz-ru
    python til_to_manifest.py --data-dir data/uz-ru --directions uz-ru      # one direction
    python til_to_manifest.py --data-dir data/uz-ru --labse 0.75            # + meaning filter
    python til_to_manifest.py --data-dir data/uz-en                         # any TIL pair
    python til_to_manifest.py --data-dir data/uz-ru --limit 1000 --show 10  # quick look

Output (default data/external/til_<pair>/): train.jsonl dev.jsonl test.jsonl stats.json
License of TIL: CC BY-NC-SA 4.0 (non-commercial, share-alike).
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent  # everything lives inside riva_train/

# ---------------------------------------------------------------------------
# Detokenization (reverses Moses-style tokenization used in TIL)
# ---------------------------------------------------------------------------
APOS = "‘’ʻʼ'`´"
LETTER = r"[^\W\d_]"
# o ‘ quv -> o'quv ; qat ’ i -> qat'i ; ma ’ rifiy -> ma'rifiy  (letter + apostrophe + letter)
RE_UZ_APOS = re.compile(rf"({LETTER}) ?[{APOS}] ?(?={LETTER})")
RE_SPACE_BEFORE = re.compile(r" +([.,;:!?%)\]}»”…])")        # "word ," -> "word,"
RE_SPACE_AFTER = re.compile(r"([(\[{«“„]) +")                 # "( word" -> "(word"
RE_ELLIPSIS = re.compile(r"\. \. \.")
RE_NUM_DOT = re.compile(r"(\d) \.(?=\s|$)")                   # "13 ." -> "13."
# Uzbek suffixes glued to a closing quote/bracket:  ”gi, »ning, )dagi
RE_UZ_SUFFIX = re.compile(r"([”»\")]) (gi|ga|da|dan|ni|ning|dagi|dagi|ka|qa|lar|larni|larning|dir)\b")
RE_STRAIGHT_Q = re.compile(r'"')
WS = re.compile(r"\s+")


def fix_straight_quotes(s):
    """Toggle " as opening/closing and attach it to the quoted text."""
    out, opening = [], True
    parts = RE_STRAIGHT_Q.split(s)
    for i, part in enumerate(parts):
        if i:
            out.append('"')
            opening = not opening
        if i and not opening:        # we just opened a quote: strip space after it
            part = part.lstrip()
        if i < len(parts) - 1 and not opening:  # a closing quote follows: strip space before it
            part = part.rstrip()
        out.append(part)
    return "".join(out)


def detok(s, lang, apostrophe="'"):
    s = WS.sub(" ", s.replace(" ", " ")).strip()
    if not s:
        return s
    if lang in ("uz", "kaa", "tk", "az", "crh", "tr", "kk", "ky"):
        s = RE_UZ_APOS.sub(lambda m: m.group(1) + apostrophe, s)
    s = RE_ELLIPSIS.sub("...", s)
    s = RE_NUM_DOT.sub(r"\1.", s)
    s = RE_SPACE_BEFORE.sub(r"\1", s)
    s = RE_SPACE_AFTER.sub(r"\1", s)
    if s.count('"') % 2 == 0 and '"' in s:
        s = fix_straight_quotes(s)
    if lang == "uz":
        s = RE_UZ_SUFFIX.sub(r"\1\2", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
LATIN = re.compile(r"[A-Za-z]")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
SCRIPT = {"uz": "latin", "en": "latin", "tr": "latin", "az": "latin", "tk": "latin", "kaa": "latin",
          "ru": "cyrillic", "kk": "cyrillic", "ky": "cyrillic", "tt": "cyrillic", "ba": "cyrillic",
          "cv": "cyrillic", "sah": "cyrillic"}


def script_ok(text, lang):
    want = SCRIPT.get(lang)
    if not want:
        return True
    n_lat, n_cyr = len(LATIN.findall(text)), len(CYRILLIC.findall(text))
    if n_lat + n_cyr < 2:
        return True  # numbers/symbols only: judged by other rules
    return (n_lat >= n_cyr) if want == "latin" else (n_cyr > n_lat)


def check(a, b, la, lb, args):
    if not a or not b:
        return "empty"
    if len(a) < args.min_chars or len(b) < args.min_chars:
        return "too short"
    if len(a) > args.max_chars or len(b) > args.max_chars:
        return "too long"
    if a.lower() == b.lower():
        return "untranslated (identical sides)"
    if sum(c.isalpha() for c in a) < 2 or sum(c.isalpha() for c in b) < 2:
        return "no real words (numbers/symbols)"
    if not script_ok(a, la):
        return f"{la} side in wrong script"
    if not script_ok(b, lb):
        return f"{lb} side in wrong script"
    ratio = max(len(a), len(b)) / max(1, min(len(a), len(b)))
    if ratio > args.max_ratio and max(len(a), len(b)) > 15:
        return "length ratio (likely misaligned)"
    return None


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def find_pair_files(split_dir, la, lb):
    """Return (file_a, file_b) where file_a ends with .<la> and file_b with .<lb> and same stem."""
    if not split_dir.is_dir():
        return None
    a_files = {p.with_suffix("").relative_to(split_dir): p for p in split_dir.rglob(f"*.{la}")}
    b_files = {p.with_suffix("").relative_to(split_dir): p for p in split_dir.rglob(f"*.{lb}")}
    common = sorted(set(a_files) & set(b_files))
    return [(a_files[k], b_files[k]) for k in common] or None


def count_lines(path):
    with open(path, "rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))


def read_split(files, la, lb, args, dropped, limit):
    rows = []
    for fa, fb in files:
        na, nb = count_lines(fa), count_lines(fb)
        if na != nb:
            sys.exit(f"Line counts differ — files are not aligned:\n  {fa}: {na:,}\n  {fb}: {nb:,}")
        print(f"    {fa.name} / {fb.name}: {na:,} lines")
        with open(fa, encoding="utf-8", errors="replace") as A, open(fb, encoding="utf-8", errors="replace") as B:
            for i, (a, b) in enumerate(zip(A, B)):
                a, b = detok(a, la, args.apostrophe), detok(b, lb, args.apostrophe)
                why = check(a, b, la, lb, args)
                if why:
                    dropped[why] += 1
                    continue
                rows.append((a, b))
                if limit and len(rows) >= limit:
                    return rows
    return rows


def key(text):
    return hashlib.blake2b(text.lower().encode("utf-8"), digest_size=8).digest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="TIL corpus -> project JSONL manifests.")
    ap.add_argument("--data-dir", required=True, help="e.g. data/uz-ru (folder made by download_data.py)")
    ap.add_argument("--langs", nargs=2, default=None, metavar=("A", "B"),
                    help="language codes; default from folder name, e.g. uz ru")
    ap.add_argument("--directions", nargs="+", default=None,
                    help="which pairs to write, e.g. uz-ru ru-uz (default: both)")
    ap.add_argument("--min-chars", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--max-ratio", type=float, default=2.0, help="max length ratio between sides")
    ap.add_argument("--labse", type=float, default=None, help="LaBSE similarity cut-off, e.g. 0.75 (optional)")
    ap.add_argument("--apostrophe", default="'", help="character for Uzbek o'/g' apostrophes (default ')")
    ap.add_argument("--limit", type=int, default=None, help="max rows per split (quick test)")
    ap.add_argument("--show", type=int, default=3, help="print N random examples")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    la, lb = args.langs or data_dir.name.split("-", 1)
    directions = args.directions or [f"{la}-{lb}", f"{lb}-{la}"]
    out_dir = Path(args.out_dir or PROJECT_ROOT / "data" / "external" / f"til_{la}-{lb}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    splits, dropped = {}, {}
    for split in ("train", "dev", "test"):
        files = find_pair_files(data_dir / split, la, lb)
        if not files:
            print(f"[{split}] no *.{la} / *.{lb} files under {data_dir / split} — skipped")
            continue
        print(f"[{split}]")
        dropped[split] = Counter()
        splits[split] = read_split(files, la, lb, args, dropped[split], args.limit)

    if "train" not in splits:
        sys.exit(f"No train files found under {data_dir}/train. Check --data-dir.")

    # ---------- dedupe; keep dev/test out of train ----------
    held = set()
    for split in ("dev", "test"):
        uniq, seen = [], set()
        for a, b in splits.get(split, []):
            k = key(a)
            if k in seen:
                dropped[split]["duplicate"] += 1
                continue
            seen.add(k)
            held.add(k)
            held.add(key(b))
            uniq.append((a, b))
        if split in splits:
            splits[split] = uniq
    uniq, seen = [], set()
    for a, b in splits["train"]:
        ka = key(a)
        if ka in held or key(b) in held:
            dropped["train"]["also in dev/test (removed from train)"] += 1
            continue
        if ka in seen:
            dropped["train"]["duplicate"] += 1
            continue
        seen.add(ka)
        uniq.append((a, b))
    splits["train"] = uniq

    # no dev/test shipped? carve 2,000 each from train
    for split in ("dev", "test"):
        if not splits.get(split):
            rng.shuffle(splits["train"])
            n = min(2000, len(splits["train"]) // 20)
            splits[split], splits["train"] = splits["train"][:n], splits["train"][n:]
            print(f"[{split}] not provided by TIL — took {n:,} random pairs from train")

    # ---------- optional LaBSE ----------
    if args.labse is not None:
        from sentence_transformers import SentenceTransformer

        print(f"LaBSE filter (>= {args.labse}) ...")
        m = SentenceTransformer("sentence-transformers/LaBSE")
        for split, rows in splits.items():
            kept = []
            for i in range(0, len(rows), 50_000):
                chunk = rows[i:i + 50_000]
                ea = m.encode([a for a, _ in chunk], batch_size=512, normalize_embeddings=True)
                eb = m.encode([b for _, b in chunk], batch_size=512, normalize_embeddings=True)
                for (a, b), s in zip(chunk, (ea * eb).sum(1)):
                    if s >= args.labse:
                        kept.append((a, b))
                    else:
                        dropped[split][f"LaBSE < {args.labse}"] += 1
                print(f"  {split}: {min(i + 50_000, len(rows)):,}/{len(rows):,}")
            splits[split] = kept

    # ---------- write ----------
    written = {}
    for split, rows in splits.items():
        rng.shuffle(rows)
        n = 0
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for a, b in rows:
                for d in directions:
                    s, t = (a, b) if d == f"{la}-{lb}" else (b, a)
                    f.write(json.dumps({"pair": d, "source": s, "target": t, "origin": "til"},
                                       ensure_ascii=False) + "\n")
                    n += 1
        written[split] = n

    stats = {
        "source": str(data_dir), "languages": [la, lb], "directions": directions,
        "pairs": {k: len(v) for k, v in splits.items()}, "rows_written": written,
        "dropped": {k: dict(v.most_common()) for k, v in dropped.items()},
        "license": "TIL corpus: CC BY-NC-SA 4.0 (non-commercial, share-alike)",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    for split in ("train", "dev", "test"):
        if split in splits:
            print(f"  {split:<5} {len(splits[split]):>10,} pairs -> {written[split]:>10,} rows  "
                  f"{out_dir / (split + '.jsonl')}")
            for why, c in dropped.get(split, Counter()).most_common():
                print(f"          dropped {c:>9,}  {why}")
    for a, b in rng.sample(splits["train"], min(args.show, len(splits["train"]))):
        print(f"\n  {la}: {a[:150]}\n  {lb}: {b[:150]}")
    print(f"\nSummary: {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
