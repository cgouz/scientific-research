#!/usr/bin/env python3

"""
Uzbek <-> Russian Riva dataset builder.

Verified built-in sources:

1. blinoff/UzLPC
   - filter: source_lang == "rus"
   - Russian: source_texts
   - Uzbek: translation

2. sukhrobnurali/uzbek-legal-corpus-v1
   config: parallel_uz_ru
   - Uzbek: text_uz
   - Russian: text_ru

Additionally, you can pass your own local JSONL files:

    --extra /path/corpus1.jsonl /path/corpus2.jsonl

Supported extra JSONL shapes:

    {"uz": "...", "ru": "..."}
    {"text_uz": "...", "text_ru": "..."}
    {"pair":"uz-ru","source":"...","target":"..."}
    {"pair":"ru-uz","source":"...","target":"..."}

Output:

    output/
    ├── train.jsonl
    ├── dev.jsonl
    └── test.jsonl

By default, each selected parallel pair becomes BOTH:

    uz -> ru
    ru -> uz

If target=1,500,000 parallel pairs:

    train examples ≈ 3,000,000
    because each pair is written in 2 directions.

Example:

    python uz_ru_builder.py \
        --output-dir /workspace/datasets/ttt/uz-ru-1.5m \
        --target 1500000

With additional local corpora:

    python uz_ru_builder.py \
        --output-dir /workspace/datasets/ttt/uz-ru-1.5m \
        --target 1500000 \
        --extra \
            /workspace/datasets/uz-ru/opus.jsonl \
            /workspace/datasets/uz-ru/tatoeba.jsonl
"""

import argparse
import hashlib
import heapq
import json
import math
import re
import tempfile
from pathlib import Path

from datasets import load_dataset


# =========================================================
# Regex
# =========================================================

WHITESPACE_RE = re.compile(r"\s+")

CYRILLIC_RE = re.compile(
    r"[А-Яа-яЁё\u0400-\u052f]"
)

LATIN_RE = re.compile(
    r"[A-Za-zÀ-žʻʼ‘’]"
)

URL_RE = re.compile(
    r"https?://|www\.",
    re.IGNORECASE,
)

HTML_RE = re.compile(
    r"<[^>]+>"
)

REPEATED_RE = re.compile(
    r"(.)\1{6,}"
)


# =========================================================
# Basic text cleaning
# =========================================================

def clean_text(text: str) -> str:
    """
    Basic normalization.

    Full aggressive dataset cleaning can still be done
    before this builder if desired.
    """

    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", " ")

    text = WHITESPACE_RE.sub(
        " ",
        text,
    )

    return text.strip()


# =========================================================
# Alphabet statistics
# =========================================================

def count_letters(text: str) -> int:
    return sum(
        char.isalpha()
        for char in text
    )


def cyrillic_ratio(text: str) -> float:
    total = count_letters(text)

    if total == 0:
        return 0.0

    return (
        len(CYRILLIC_RE.findall(text))
        / total
    )


def latin_ratio(text: str) -> float:
    total = count_letters(text)

    if total == 0:
        return 0.0

    return (
        len(LATIN_RE.findall(text))
        / total
    )


# =========================================================
# Quality checking
# =========================================================

def quality_score(
    uz: str,
    ru: str,
) -> float | None:
    """
    Returns:
        None  -> reject
        float -> higher means better

    This is only a heuristic ranking/filter.
    """

    uz = clean_text(uz)
    ru = clean_text(ru)

    if not uz or not ru:
        return None

    uz_len = len(uz)
    ru_len = len(ru)

    # ---------------------------------------------
    # Sentence length
    # ---------------------------------------------

    if uz_len < 5 or ru_len < 5:
        return None

    if uz_len > 1000 or ru_len > 1000:
        return None

    # ---------------------------------------------
    # Obvious garbage
    # ---------------------------------------------

    if URL_RE.search(uz) or URL_RE.search(ru):
        return None

    if HTML_RE.search(uz) or HTML_RE.search(ru):
        return None

    if REPEATED_RE.search(uz):
        return None

    if REPEATED_RE.search(ru):
        return None

    # ---------------------------------------------
    # Same text check
    # ---------------------------------------------

    if uz.casefold() == ru.casefold():
        return None

    # ---------------------------------------------
    # Russian should mostly contain Cyrillic
    # ---------------------------------------------

    ru_cyr = cyrillic_ratio(ru)

    if ru_cyr < 0.65:
        return None

    # ---------------------------------------------
    # Uzbek Latin should normally contain Latin
    #
    # We keep threshold moderate because numbers,
    # names and symbols are common.
    # ---------------------------------------------

    uz_lat = latin_ratio(uz)

    if uz_lat < 0.55:
        return None

    # ---------------------------------------------
    # Length ratio
    # ---------------------------------------------

    ratio = uz_len / ru_len

    if ratio < 0.35 or ratio > 2.8:
        return None

    ratio_score = math.exp(
        -abs(math.log(ratio))
    )

    # Prefer useful sentence lengths but do not
    # punish longer text too heavily.
    avg_len = (
        uz_len + ru_len
    ) / 2

    length_score = math.exp(
        -abs(avg_len - 100) / 350
    )

    script_score = (
        uz_lat + ru_cyr
    ) / 2

    return (
        ratio_score * 4.0
        + script_score * 3.0
        + length_score * 1.5
    )


# =========================================================
# Stable pair hash
# =========================================================

def pair_hash(
    uz: str,
    ru: str,
) -> bytes:
    """
    Used for exact deduplication.
    """

    value = (
        uz.casefold()
        + "\t"
        + ru.casefold()
    ).encode(
        "utf-8",
        errors="ignore",
    )

    return hashlib.blake2b(
        value,
        digest_size=16,
    ).digest()


# =========================================================
# Normalize a candidate
# =========================================================

def make_candidate(
    uz,
    ru,
    source_name,
):
    uz = clean_text(uz)
    ru = clean_text(ru)

    score = quality_score(
        uz,
        ru,
    )

    if score is None:
        return None

    return {
        "uz": uz,
        "ru": ru,
        "score": score,
        "source": source_name,
    }


# =========================================================
# Source 1: UzLPC
# =========================================================

def iter_uzlpc():
    """
    blinoff/UzLPC

    Russian-source rows only:

        source_lang = rus
        source_texts = Russian
        translation = Uzbek
    """

    print()
    print("Opening blinoff/UzLPC ...")

    ds = load_dataset(
        "blinoff/UzLPC",
        split="train",
        streaming=True,
    )

    for row in ds:

        if row.get("source_lang") != "rus":
            continue

        ru = row.get(
            "source_texts"
        )

        uz = row.get(
            "translation"
        )

        candidate = make_candidate(
            uz,
            ru,
            "UzLPC",
        )

        if candidate:
            yield candidate


# =========================================================
# Source 2: Uzbek legal corpus
# =========================================================

def iter_legal():
    """
    sukhrobnurali/uzbek-legal-corpus-v1

    Config:
        parallel_uz_ru

    Columns:
        text_uz
        text_ru
    """

    print()
    print(
        "Opening Uzbek Legal Corpus..."
    )

    ds = load_dataset(
        "sukhrobnurali/"
        "uzbek-legal-corpus-v1",
        "parallel_uz_ru",
        split="train",
        streaming=True,
    )

    for row in ds:

        candidate = make_candidate(
            row.get("text_uz"),
            row.get("text_ru"),
            "uzbek-legal-corpus",
        )

        if candidate:
            yield candidate


# =========================================================
# Extra local JSONL corpora
# =========================================================

def parse_extra_row(row):
    """
    Accept several common formats.
    """

    # ---------------------------------------------
    # {"uz": ..., "ru": ...}
    # ---------------------------------------------

    if (
        isinstance(row.get("uz"), str)
        and isinstance(row.get("ru"), str)
    ):
        return (
            row["uz"],
            row["ru"],
        )

    # ---------------------------------------------
    # legal-style
    # ---------------------------------------------

    if (
        isinstance(
            row.get("text_uz"),
            str,
        )
        and isinstance(
            row.get("text_ru"),
            str,
        )
    ):
        return (
            row["text_uz"],
            row["text_ru"],
        )

    # ---------------------------------------------
    # Riva format
    # ---------------------------------------------

    pair = row.get("pair")

    source = row.get("source")
    target = row.get("target")

    if (
        isinstance(source, str)
        and isinstance(target, str)
    ):

        if pair == "uz-ru":
            return source, target

        if pair == "ru-uz":
            return target, source

    return None


def iter_extra_file(path: Path):

    print()
    print(
        f"Opening extra corpus: {path}"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(
            f,
            1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError:
                continue

            parsed = parse_extra_row(
                row
            )

            if not parsed:
                continue

            uz, ru = parsed

            candidate = make_candidate(
                uz,
                ru,
                path.name,
            )

            if candidate:
                yield candidate


# =========================================================
# Merge sources into temporary JSONL
# =========================================================

def build_pool(
    temp_path: Path,
    extras,
):
    """
    Streams all sources into a temporary normalized pool.

    Deduplication happens here.

    Temporary row:
        {
          "uz": ...,
          "ru": ...,
          "score": ...
        }
    """

    seen = set()

    total_seen = 0
    saved = 0
    duplicates = 0

    sources = [
        iter_uzlpc(),
        iter_legal(),
    ]

    for extra in extras:
        sources.append(
            iter_extra_file(extra)
        )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for iterator in sources:

            for item in iterator:

                total_seen += 1

                key = pair_hash(
                    item["uz"],
                    item["ru"],
                )

                if key in seen:

                    duplicates += 1
                    continue

                seen.add(key)

                fout.write(
                    json.dumps(
                        {
                            "uz": item["uz"],
                            "ru": item["ru"],
                            "score": item[
                                "score"
                            ],
                            "source": item[
                                "source"
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                saved += 1

                if saved % 100_000 == 0:
                    print(
                        f"pool saved: "
                        f"{saved:,}"
                    )

    print()
    print("Pool completed:")
    print(
        f"  candidates : {total_seen:,}"
    )
    print(
        f"  unique     : {saved:,}"
    )
    print(
        f"  duplicates : {duplicates:,}"
    )

    return saved


# =========================================================
# Select best N without loading whole corpus
# =========================================================

def select_best_offsets(
    temp_path: Path,
    target: int,
):
    """
    Keeps only top-N score + file-offset entries in RAM.
    """

    heap = []

    scanned = 0

    with temp_path.open(
        "rb"
    ) as f:

        while True:

            offset = f.tell()

            raw = f.readline()

            if not raw:
                break

            scanned += 1

            row = json.loads(
                raw.decode("utf-8")
            )

            score = float(
                row["score"]
            )

            item = (
                score,
                offset,
            )

            if len(heap) < target:

                heapq.heappush(
                    heap,
                    item,
                )

            elif score > heap[0][0]:

                heapq.heapreplace(
                    heap,
                    item,
                )

            if scanned % 100_000 == 0:

                min_score = (
                    heap[0][0]
                    if heap
                    else 0
                )

                print(
                    f"select scanned="
                    f"{scanned:,} "
                    f"selected="
                    f"{len(heap):,} "
                    f"min_score="
                    f"{min_score:.4f}"
                )

    offsets = sorted(
        offset
        for _, offset in heap
    )

    return offsets


# =========================================================
# Deterministic dev/test reservation
# =========================================================

def split_selected(
    temp_path: Path,
    offsets,
    dev_pairs: int,
    test_pairs: int,
):
    """
    Reads selected pairs.

    First reserves dev/test deterministically using
    pair hashes, then the rest goes to train.

    We sort by hash to make split reproducible.
    """

    selected = []

    with temp_path.open(
        "rb"
    ) as f:

        for offset in offsets:

            f.seek(offset)

            row = json.loads(
                f.readline().decode(
                    "utf-8"
                )
            )

            uz = row["uz"]
            ru = row["ru"]

            h = pair_hash(
                uz,
                ru,
            )

            selected.append(
                (
                    h,
                    uz,
                    ru,
                    row.get(
                        "source",
                        "",
                    ),
                )
            )

    selected.sort(
        key=lambda x: x[0]
    )

    needed_eval = (
        dev_pairs
        + test_pairs
    )

    if len(selected) <= needed_eval:

        raise RuntimeError(
            "Selected pair count is too small "
            "for requested dev/test sizes."
        )

    dev = selected[
        :dev_pairs
    ]

    test = selected[
        dev_pairs:
        dev_pairs + test_pairs
    ]

    train = selected[
        needed_eval:
    ]

    return (
        train,
        dev,
        test,
    )


# =========================================================
# Output writer
# =========================================================

def write_bidirectional(
    path: Path,
    rows,
):
    """
    Each parallel pair becomes:

        uz-ru
        ru-uz
    """

    uz_ru_count = 0
    ru_uz_count = 0

    with path.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for _, uz, ru, _source in rows:

            fout.write(
                json.dumps(
                    {
                        "pair": "uz-ru",
                        "source": uz,
                        "target": ru,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            fout.write(
                json.dumps(
                    {
                        "pair": "ru-uz",
                        "source": ru,
                        "target": uz,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            uz_ru_count += 1
            ru_uz_count += 1

    return (
        uz_ru_count,
        ru_uz_count,
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build best Uzbek-Russian "
            "bidirectional Riva dataset"
        )
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "/workspace/datasets/"
            "ttt/uz-ru-1.5m"
        ),
    )

    parser.add_argument(
        "--target",
        type=int,
        default=1_500_000,
        help=(
            "Total UNIQUE parallel pairs "
            "selected before direction reversal."
        ),
    )

    parser.add_argument(
        "--dev-pairs",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--test-pairs",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help=(
            "Additional local RU-UZ "
            "JSONL corpora."
        ),
    )

    parser.add_argument(
        "--allow-less",
        action="store_true",
        help=(
            "If fewer than target good unique "
            "pairs exist, use everything "
            "instead of failing."
        ),
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    extras = [
        Path(x)
        for x in args.extra
    ]

    for path in extras:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    temp_path = (
        output_dir
        / "_pool.tmp.jsonl"
    )

    print()
    print(
        "======================================"
    )
    print(
        " UZBEK-RUSSIAN RIVA DATASET BUILDER"
    )
    print(
        "======================================"
    )

    print()
    print(
        f"Target unique pairs : "
        f"{args.target:,}"
    )

    print(
        f"Dev pairs           : "
        f"{args.dev_pairs:,}"
    )

    print(
        f"Test pairs          : "
        f"{args.test_pairs:,}"
    )

    # ---------------------------------------------
    # 1. Build normalized pool
    # ---------------------------------------------

    unique_count = build_pool(
        temp_path,
        extras,
    )

    if unique_count < args.target:

        message = (
            f"\nOnly {unique_count:,} unique "
            f"good pairs were available, "
            f"but target={args.target:,}."
        )

        if not args.allow_less:

            raise RuntimeError(
                message
                + "\nAdd more RU-UZ corpora "
                "via --extra, or run with "
                "--allow-less."
            )

        print(message)

        target = unique_count

    else:
        target = args.target

    # Need room for eval sets while still preserving
    # roughly target selected pairs in total.
    print()
    print(
        f"Selecting best {target:,} pairs..."
    )

    offsets = select_best_offsets(
        temp_path,
        target,
    )

    # ---------------------------------------------
    # 2. Train/dev/test
    # ---------------------------------------------

    train, dev, test = split_selected(
        temp_path,
        offsets,
        args.dev_pairs,
        args.test_pairs,
    )

    # ---------------------------------------------
    # 3. Write Riva format
    # ---------------------------------------------

    train_path = (
        output_dir
        / "train.jsonl"
    )

    dev_path = (
        output_dir
        / "dev.jsonl"
    )

    test_path = (
        output_dir
        / "test.jsonl"
    )

    print()
    print("Writing train...")

    train_counts = write_bidirectional(
        train_path,
        train,
    )

    print("Writing dev...")

    dev_counts = write_bidirectional(
        dev_path,
        dev,
    )

    print("Writing test...")

    test_counts = write_bidirectional(
        test_path,
        test,
    )

    # Remove temp pool after successful write.
    temp_path.unlink(
        missing_ok=True
    )

    # ---------------------------------------------
    # Result
    # ---------------------------------------------

    print()
    print(
        "======================================"
    )
    print(" DONE")
    print(
        "======================================"
    )

    print()
    print("TRAIN")
    print(
        f"  uz-ru : "
        f"{train_counts[0]:,}"
    )
    print(
        f"  ru-uz : "
        f"{train_counts[1]:,}"
    )
    print(
        f"  total : "
        f"{sum(train_counts):,}"
    )

    print()
    print("DEV")
    print(
        f"  uz-ru : "
        f"{dev_counts[0]:,}"
    )
    print(
        f"  ru-uz : "
        f"{dev_counts[1]:,}"
    )
    print(
        f"  total : "
        f"{sum(dev_counts):,}"
    )

    print()
    print("TEST")
    print(
        f"  uz-ru : "
        f"{test_counts[0]:,}"
    )
    print(
        f"  ru-uz : "
        f"{test_counts[1]:,}"
    )
    print(
        f"  total : "
        f"{sum(test_counts):,}"
    )

    print()
    print("Output:")
    print(output_dir)


if __name__ == "__main__":
    main()