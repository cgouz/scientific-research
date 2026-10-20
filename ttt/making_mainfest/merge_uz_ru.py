#!/usr/bin/env python3

"""
Existing UZ-RU corpus + TIL Corpus ni birlashtirib,
Riva Translate uchun final dataset yaratadi.

INPUT 1:
    Existing Riva dataset:

    /workspace/datasets/ttt/uz-ru-1.5m/
        train.jsonl
        dev.jsonl
        test.jsonl

INPUT 2:
    TIL Corpus:

    /workspace/datasets/til-mt/til_corpus/data/uz-ru/

        train/uz-ru/uz-ru.uz
        train/uz-ru/uz-ru.ru

        dev/uz-ru/uz-ru.uz
        dev/uz-ru/uz-ru.ru

        test/uz-ru/uz-ru.uz
        test/uz-ru/uz-ru.ru


OUTPUT:

    /workspace/datasets/ttt/uz-ru-final-1.5m/

        train.jsonl
        dev.jsonl
        test.jsonl


TRAIN:

    1,500,000 unique parallel pair

    har biri:

        uz-ru
        ru-uz

    qilib yoziladi.

Natija:

    3,000,000 train rows
"""

import argparse
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path


# =========================================================
# Regex
# =========================================================

WHITESPACE_RE = re.compile(r"\s+")

CYRILLIC_RE = re.compile(
    r"[\u0400-\u052f]"
)

LATIN_RE = re.compile(
    r"[A-Za-z]"
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
# Text cleaning
# =========================================================

def clean_text(text: str) -> str:
    """
    Basic normalization.

    Newline, tab va ortiqcha whitespace'larni
    bitta space'ga aylantiradi.
    """

    if not isinstance(text, str):
        return ""

    text = text.replace(
        "\u00a0",
        " ",
    )

    text = WHITESPACE_RE.sub(
        " ",
        text,
    )

    return text.strip()


# =========================================================
# Script ratio
# =========================================================

def alphabet_ratio(
    text: str,
    regex: re.Pattern,
) -> float:

    total_letters = sum(
        1
        for c in text
        if c.isalpha()
    )

    if total_letters == 0:
        return 0.0

    return (
        len(regex.findall(text))
        / total_letters
    )


# =========================================================
# Pair hash
# =========================================================

def make_hash(
    uz: str,
    ru: str,
) -> bytes:
    """
    Exact duplicate detection.

    Case-insensitive.
    """

    text = (
        uz.casefold()
        + "\t"
        + ru.casefold()
    )

    return hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=16,
    ).digest()


# =========================================================
# Quality score
# =========================================================

def quality_score(
    uz: str,
    ru: str,
) -> float | None:
    """
    Pair quality heuristic.

    None:
        pair yaroqsiz

    float:
        kattaroq score = yaxshiroq pair
    """

    uz = clean_text(uz)
    ru = clean_text(ru)

    if not uz or not ru:
        return None

    uz_len = len(uz)
    ru_len = len(ru)

    # ---------------------------------------------
    # Length
    # ---------------------------------------------

    if uz_len < 8 or ru_len < 8:
        return None

    if uz_len > 1200 or ru_len > 1200:
        return None

    # ---------------------------------------------
    # Garbage
    # ---------------------------------------------

    if URL_RE.search(uz):
        return None

    if URL_RE.search(ru):
        return None

    if HTML_RE.search(uz):
        return None

    if HTML_RE.search(ru):
        return None

    if REPEATED_RE.search(uz):
        return None

    if REPEATED_RE.search(ru):
        return None

    # ---------------------------------------------
    # Same text
    # ---------------------------------------------

    if uz.casefold() == ru.casefold():
        return None

    # ---------------------------------------------
    # Language check
    # ---------------------------------------------

    uz_latin = alphabet_ratio(
        uz,
        LATIN_RE,
    )

    ru_cyrillic = alphabet_ratio(
        ru,
        CYRILLIC_RE,
    )

    # Uzbek Latin.
    if uz_latin < 0.60:
        return None

    # Russian Cyrillic.
    if ru_cyrillic < 0.70:
        return None

    # Uzbek ichida juda ko'p Cyrillic bo'lmasin.
    uz_cyrillic = alphabet_ratio(
        uz,
        CYRILLIC_RE,
    )

    if uz_cyrillic > 0.30:
        return None

    # Russian ichida juda ko'p Latin bo'lmasin.
    ru_latin = alphabet_ratio(
        ru,
        LATIN_RE,
    )

    if ru_latin > 0.30:
        return None

    # ---------------------------------------------
    # Length ratio
    # ---------------------------------------------

    ratio = uz_len / ru_len

    if ratio < 0.40:
        return None

    if ratio > 2.50:
        return None

    # Ratio 1 ga yaqin bo'lsa yaxshi.
    ratio_score = math.exp(
        -abs(math.log(ratio))
    )

    language_score = (
        uz_latin
        + ru_cyrillic
    ) / 2

    avg_len = (
        uz_len
        + ru_len
    ) / 2

    # O'rtacha uzunlikdagi gaplarga
    # kichik bonus.
    length_score = math.exp(
        -abs(avg_len - 120)
        / 300
    )

    return (
        ratio_score * 4
        + language_score * 4
        + length_score * 2
    )


# =========================================================
# SQLite candidate DB
# =========================================================

def create_db(path: Path):

    if path.exists():
        path.unlink()

    db = sqlite3.connect(path)

    db.execute(
        """
        CREATE TABLE pairs (
            hash BLOB PRIMARY KEY,
            uz TEXT NOT NULL,
            ru TEXT NOT NULL,
            score REAL NOT NULL,
            source TEXT NOT NULL
        )
        """
    )

    db.execute(
        """
        CREATE INDEX idx_score
        ON pairs(score DESC)
        """
    )

    db.commit()

    return db


# =========================================================
# Add pair
# =========================================================

def add_pair(
    db,
    uz: str,
    ru: str,
    source: str,
):
    """
    Pairni:

        clean
        validate
        deduplicate
        score

    qiladi.
    """

    uz = clean_text(uz)
    ru = clean_text(ru)

    score = quality_score(
        uz,
        ru,
    )

    if score is None:
        return "rejected"

    h = make_hash(
        uz,
        ru,
    )

    try:

        db.execute(
            """
            INSERT INTO pairs
            (
                hash,
                uz,
                ru,
                score,
                source
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                h,
                uz,
                ru,
                score,
                source,
            ),
        )

        return "added"

    except sqlite3.IntegrityError:

        return "duplicate"


# =========================================================
# Existing Riva JSONL
# =========================================================

def import_riva_file(
    db,
    path: Path,
    source_name: str,
):
    """
    Existing Riva:

        {"pair":"uz-ru", ...}

    yoki:

        {"pair":"ru-uz", ...}

    ni canonical:

        uz
        ru

    ko'rinishga keltiradi.
    """

    if not path.exists():

        print(
            f"[SKIP] topilmadi: {path}"
        )

        return

    added = 0
    duplicate = 0
    rejected = 0

    print()
    print(
        f"Import Riva: {path}"
    )

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for i, line in enumerate(
            f,
            1,
        ):

            try:
                row = json.loads(line)

            except json.JSONDecodeError:
                continue

            pair = row.get("pair")

            source = row.get("source")
            target = row.get("target")

            if not isinstance(source, str):
                continue

            if not isinstance(target, str):
                continue

            if pair == "uz-ru":

                uz = source
                ru = target

            elif pair == "ru-uz":

                uz = target
                ru = source

            else:
                continue

            result = add_pair(
                db,
                uz,
                ru,
                source_name,
            )

            if result == "added":
                added += 1

            elif result == "duplicate":
                duplicate += 1

            else:
                rejected += 1

            if i % 100_000 == 0:

                db.commit()

                print(
                    f"  read={i:,} "
                    f"added={added:,} "
                    f"duplicate={duplicate:,} "
                    f"rejected={rejected:,}"
                )

    db.commit()

    print(
        f"  added     : {added:,}"
    )

    print(
        f"  duplicate : {duplicate:,}"
    )

    print(
        f"  rejected  : {rejected:,}"
    )


# =========================================================
# TIL parallel line files
# =========================================================

def import_til_parallel(
    db,
    uz_path: Path,
    ru_path: Path,
    source_name: str,
):
    """
    TIL parallel files:

        line 1 Uzbek
        line 1 Russian

    alignmentni saqlagan holda import qiladi.
    """

    if not uz_path.exists():

        print(
            f"[SKIP] topilmadi: {uz_path}"
        )

        return

    if not ru_path.exists():

        print(
            f"[SKIP] topilmadi: {ru_path}"
        )

        return

    print()
    print(
        f"TIL import:"
    )

    print(
        f"  UZ: {uz_path}"
    )

    print(
        f"  RU: {ru_path}"
    )

    added = 0
    duplicate = 0
    rejected = 0

    with (
        uz_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as fuz,

        ru_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as fru,
    ):

        for i, (uz, ru) in enumerate(
            zip(fuz, fru),
            1,
        ):

            result = add_pair(
                db,
                uz,
                ru,
                source_name,
            )

            if result == "added":
                added += 1

            elif result == "duplicate":
                duplicate += 1

            else:
                rejected += 1

            if i % 100_000 == 0:

                db.commit()

                print(
                    f"  read={i:,} "
                    f"added={added:,} "
                    f"duplicate={duplicate:,} "
                    f"rejected={rejected:,}"
                )

    db.commit()

    print(
        f"  added     : {added:,}"
    )

    print(
        f"  duplicate : {duplicate:,}"
    )

    print(
        f"  rejected  : {rejected:,}"
    )


# =========================================================
# Output train
# =========================================================

def write_train(
    db,
    output_path: Path,
    count: int,
):
    """
    Eng yaxshi N unique pairni tanlaydi.

    Har bir pair:

        uz-ru
        ru-uz

    qilib yoziladi.
    """

    total = db.execute(
        "SELECT COUNT(*) FROM pairs"
    ).fetchone()[0]

    print()
    print(
        f"Unique valid candidates: {total:,}"
    )

    limit = min(
        count,
        total,
    )

    if limit < count:

        print(
            f"[WARNING] "
            f"{count:,} o'rniga "
            f"{limit:,} pair mavjud."
        )

    cursor = db.execute(
        """
        SELECT uz, ru, score, source
        FROM pairs
        ORDER BY score DESC
        LIMIT ?
        """,
        (limit,),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_hashes = set()

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for i, (
            uz,
            ru,
            score,
            source,
        ) in enumerate(
            cursor,
            1,
        ):

            selected_hashes.add(
                make_hash(
                    uz,
                    ru,
                )
            )

            # UZ -> RU

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

            # RU -> UZ

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

            if i % 100_000 == 0:

                print(
                    f"train pairs={i:,} "
                    f"rows={i * 2:,}"
                )

    return (
        limit,
        selected_hashes,
    )


# =========================================================
# Eval collection
# =========================================================

def collect_eval_riva(
    path: Path,
    result: dict,
):
    """
    Existing dev/test'dan pairlarni yig'adi.
    """

    if not path.exists():
        return

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line in f:

            try:
                row = json.loads(line)

            except json.JSONDecodeError:
                continue

            pair = row.get("pair")

            source = row.get("source")
            target = row.get("target")

            if not isinstance(source, str):
                continue

            if not isinstance(target, str):
                continue

            if pair == "uz-ru":

                uz = clean_text(source)
                ru = clean_text(target)

            elif pair == "ru-uz":

                uz = clean_text(target)
                ru = clean_text(source)

            else:
                continue

            if not uz or not ru:
                continue

            h = make_hash(
                uz,
                ru,
            )

            result[h] = (
                uz,
                ru,
            )


def collect_eval_til(
    uz_path: Path,
    ru_path: Path,
    result: dict,
):
    """
    TIL dev/test parallel filelarni yig'adi.
    """

    if (
        not uz_path.exists()
        or not ru_path.exists()
    ):
        return

    with (
        uz_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as fuz,

        ru_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as fru,
    ):

        for uz, ru in zip(
            fuz,
            fru,
        ):

            uz = clean_text(uz)
            ru = clean_text(ru)

            if not uz or not ru:
                continue

            h = make_hash(
                uz,
                ru,
            )

            result[h] = (
                uz,
                ru,
            )


# =========================================================
# Write eval
# =========================================================

def write_eval(
    pairs: dict,
    output_path: Path,
    train_hashes: set,
):
    """
    Dev/test yozadi.

    Train ichida mavjud pairlarni skip qiladi.
    """

    pair_count = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for h, (
            uz,
            ru,
        ) in pairs.items():

            # Leakage bo'lmasin.
            if h in train_hashes:
                continue

            # UZ -> RU

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

            # RU -> UZ

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

            pair_count += 1

    return pair_count


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--existing-dir",
        default=(
            "/workspace/datasets/"
            "ttt/uz-ru-1.5m"
        ),
    )

    parser.add_argument(
        "--til-dir",
        default=(
            "/workspace/datasets/"
            "til-mt/til_corpus/data/uz-ru"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "/workspace/datasets/"
            "ttt/uz-ru-final-1.5m"
        ),
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1_500_000,
    )

    args = parser.parse_args()

    existing = Path(
        args.existing_dir
    )

    til = Path(
        args.til_dir
    )

    output = Path(
        args.output_dir
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    db_path = (
        output
        / "merge.sqlite3"
    )

    print()
    print(
        "======================================"
    )
    print(
        " UZ-RU DATASET MERGER"
    )
    print(
        "======================================"
    )

    print()

    print(
        f"Existing : {existing}"
    )

    print(
        f"TIL      : {til}"
    )

    print(
        f"Output   : {output}"
    )

    print(
        f"Target   : {args.count:,} unique pairs"
    )

    # =====================================================
    # Database
    # =====================================================

    db = create_db(
        db_path
    )

    # =====================================================
    # Existing train
    # =====================================================

    import_riva_file(
        db,
        existing / "train.jsonl",
        "existing",
    )

    # =====================================================
    # TIL train
    # =====================================================

    import_til_parallel(
        db,

        til
        / "train"
        / "uz-ru"
        / "uz-ru.uz",

        til
        / "train"
        / "uz-ru"
        / "uz-ru.ru",

        "TIL",
    )

    # =====================================================
    # Final train
    # =====================================================

    train_count, train_hashes = (
        write_train(
            db,
            output / "train.jsonl",
            args.count,
        )
    )

    # =====================================================
    # DEV
    # =====================================================

    dev_pairs = {}

    collect_eval_riva(
        existing / "dev.jsonl",
        dev_pairs,
    )

    collect_eval_til(

        til
        / "dev"
        / "uz-ru"
        / "uz-ru.uz",

        til
        / "dev"
        / "uz-ru"
        / "uz-ru.ru",

        dev_pairs,
    )

    dev_count = write_eval(
        dev_pairs,
        output / "dev.jsonl",
        train_hashes,
    )

    # =====================================================
    # TEST
    # =====================================================

    test_pairs = {}

    collect_eval_riva(
        existing / "test.jsonl",
        test_pairs,
    )

    collect_eval_til(

        til
        / "test"
        / "uz-ru"
        / "uz-ru.uz",

        til
        / "test"
        / "uz-ru"
        / "uz-ru.ru",

        test_pairs,
    )

    test_count = write_eval(
        test_pairs,
        output / "test.jsonl",
        train_hashes,
    )

    db.close()

    # =====================================================
    # Results
    # =====================================================

    print()
    print(
        "======================================"
    )
    print(
        " DONE"
    )
    print(
        "======================================"
    )

    print()

    print("TRAIN")
    print(
        f"  unique pairs : {train_count:,}"
    )
    print(
        f"  uz-ru        : {train_count:,}"
    )
    print(
        f"  ru-uz        : {train_count:,}"
    )
    print(
        f"  total rows   : {train_count * 2:,}"
    )

    print()

    print("DEV")
    print(
        f"  unique pairs : {dev_count:,}"
    )
    print(
        f"  total rows   : {dev_count * 2:,}"
    )

    print()

    print("TEST")
    print(
        f"  unique pairs : {test_count:,}"
    )
    print(
        f"  total rows   : {test_count * 2:,}"
    )

    print()

    print(
        f"Output: {output}"
    )


if __name__ == "__main__":
    main()