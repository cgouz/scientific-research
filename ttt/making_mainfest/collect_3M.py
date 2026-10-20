#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path


WHITESPACE_RE = re.compile(r"\s+")
CYRILLIC_RE = re.compile(r"[\u0400-\u052f]")
LATIN_RE = re.compile(r"[A-Za-z]")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
REPEATED_RE = re.compile(r"(.)\1{6,}")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\u00a0", " ")
    text = WHITESPACE_RE.sub(" ", text)

    return text.strip()


def alphabet_ratio(text: str, regex: re.Pattern) -> float:
    total = sum(1 for c in text if c.isalpha())

    if total == 0:
        return 0.0

    return len(regex.findall(text)) / total


def make_hash(uz: str, ru: str) -> bytes:
    value = (
        uz.casefold().strip()
        + "\t"
        + ru.casefold().strip()
    )

    return hashlib.blake2b(
        value.encode("utf-8"),
        digest_size=16,
    ).digest()


def quality_score(uz: str, ru: str) -> float | None:
    uz = clean_text(uz)
    ru = clean_text(ru)

    if not uz or not ru:
        return None

    uz_len = len(uz)
    ru_len = len(ru)

    if uz_len < 8 or ru_len < 8:
        return None

    if uz_len > 1200 or ru_len > 1200:
        return None

    if URL_RE.search(uz) or URL_RE.search(ru):
        return None

    if HTML_RE.search(uz) or HTML_RE.search(ru):
        return None

    if REPEATED_RE.search(uz) or REPEATED_RE.search(ru):
        return None

    if uz.casefold() == ru.casefold():
        return None

    uz_latin = alphabet_ratio(uz, LATIN_RE)
    uz_cyr = alphabet_ratio(uz, CYRILLIC_RE)

    ru_cyr = alphabet_ratio(ru, CYRILLIC_RE)
    ru_latin = alphabet_ratio(ru, LATIN_RE)

    if uz_latin < 0.60:
        return None

    if ru_cyr < 0.70:
        return None

    if uz_cyr > 0.30:
        return None

    if ru_latin > 0.30:
        return None

    ratio = uz_len / ru_len

    if ratio < 0.40 or ratio > 2.50:
        return None

    ratio_score = math.exp(
        -abs(math.log(ratio))
    )

    language_score = (
        uz_latin + ru_cyr
    ) / 2

    avg_len = (
        uz_len + ru_len
    ) / 2

    length_score = math.exp(
        -abs(avg_len - 120) / 300
    )

    return (
        ratio_score * 4
        + language_score * 4
        + length_score * 2
    )


def create_db(path: Path):
    if path.exists():
        path.unlink()

    db = sqlite3.connect(path)

    db.execute("""
        CREATE TABLE pairs (
            hash BLOB PRIMARY KEY,
            uz TEXT NOT NULL,
            ru TEXT NOT NULL,
            score REAL NOT NULL,
            origin TEXT
        )
    """)

    db.execute("""
        CREATE INDEX idx_score
        ON pairs(score DESC)
    """)

    db.commit()

    return db


def normalize_row(row):
    pair = row.get("pair")
    source = row.get("source")
    target = row.get("target")

    if not isinstance(source, str):
        return None

    if not isinstance(target, str):
        return None

    if pair == "uz-ru":
        uz = source
        ru = target

    elif pair == "ru-uz":
        uz = target
        ru = source

    else:
        return None

    return (
        clean_text(uz),
        clean_text(ru),
    )


def import_train(
    db,
    path: Path,
    dataset_name: str,
):
    print()
    print(f"Importing: {path}")

    added = 0
    duplicate = 0
    rejected = 0
    invalid = 0

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for i, line in enumerate(f, 1):

            try:
                row = json.loads(line)

            except json.JSONDecodeError:
                invalid += 1
                continue

            normalized = normalize_row(row)

            if normalized is None:
                invalid += 1
                continue

            uz, ru = normalized

            score = quality_score(
                uz,
                ru,
            )

            if score is None:
                rejected += 1
                continue

            h = make_hash(
                uz,
                ru,
            )

            origin = row.get(
                "origin",
                dataset_name,
            )

            try:
                db.execute(
                    """
                    INSERT INTO pairs (
                        hash,
                        uz,
                        ru,
                        score,
                        origin
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        h,
                        uz,
                        ru,
                        score,
                        str(origin),
                    ),
                )

                added += 1

            except sqlite3.IntegrityError:
                duplicate += 1

            if i % 100_000 == 0:
                db.commit()

                print(
                    f"  read={i:,} "
                    f"added={added:,} "
                    f"dup={duplicate:,} "
                    f"rejected={rejected:,}"
                )

    db.commit()

    print()
    print(f"  added     : {added:,}")
    print(f"  duplicates: {duplicate:,}")
    print(f"  rejected  : {rejected:,}")
    print(f"  invalid   : {invalid:,}")


def write_train(
    db,
    path: Path,
    count: int,
):
    total = db.execute(
        "SELECT COUNT(*) FROM pairs"
    ).fetchone()[0]

    print()
    print(f"Unique valid pairs: {total:,}")

    if total < count:
        raise RuntimeError(
            f"Need {count:,}, "
            f"but only {total:,} unique valid pairs available."
        )

    cursor = db.execute(
        """
        SELECT hash, uz, ru, origin
        FROM pairs
        ORDER BY score DESC
        LIMIT ?
        """,
        (count,),
    )

    train_hashes = set()

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for i, (
            h,
            uz,
            ru,
            origin,
        ) in enumerate(
            cursor,
            1,
        ):

            train_hashes.add(h)

            f.write(
                json.dumps(
                    {
                        "pair": "uz-ru",
                        "source": uz,
                        "target": ru,
                        "origin": origin,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            f.write(
                json.dumps(
                    {
                        "pair": "ru-uz",
                        "source": ru,
                        "target": uz,
                        "origin": origin,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if i % 100_000 == 0:
                print(
                    f"  selected={i:,} "
                    f"rows={i * 2:,}"
                )

    return train_hashes


def collect_eval_file(
    path: Path,
    pairs: dict,
):
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

            normalized = normalize_row(row)

            if normalized is None:
                continue

            uz, ru = normalized

            if not uz or not ru:
                continue

            h = make_hash(
                uz,
                ru,
            )

            pairs[h] = (
                uz,
                ru,
                row.get("origin"),
            )


def write_eval(
    pairs: dict,
    output_path: Path,
    train_hashes: set,
    forbidden_hashes: set | None = None,
):
    if forbidden_hashes is None:
        forbidden_hashes = set()

    written_pairs = 0
    written_hashes = set()

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for h, (
            uz,
            ru,
            origin,
        ) in pairs.items():

            if h in train_hashes:
                continue

            if h in forbidden_hashes:
                continue

            row1 = {
                "pair": "uz-ru",
                "source": uz,
                "target": ru,
            }

            row2 = {
                "pair": "ru-uz",
                "source": ru,
                "target": uz,
            }

            if origin:
                row1["origin"] = origin
                row2["origin"] = origin

            f.write(
                json.dumps(
                    row1,
                    ensure_ascii=False,
                )
                + "\n"
            )

            f.write(
                json.dumps(
                    row2,
                    ensure_ascii=False,
                )
                + "\n"
            )

            written_pairs += 1
            written_hashes.add(h)

    return (
        written_pairs,
        written_hashes,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-a",
        default="/data/datasets/ttt/uz-ru",
    )

    parser.add_argument(
        "--dataset-b",
        default="/data/datasets/ttt/uz-ru-1.3M",
    )

    parser.add_argument(
        "--output-dir",
        default="/data/datasets/ttt/uz-ru-3m",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1_500_000,
    )

    args = parser.parse_args()

    dataset_a = Path(args.dataset_a)
    dataset_b = Path(args.dataset_b)
    output = Path(args.output_dir)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    db_path = (
        output
        / "merge.sqlite3"
    )

    db = create_db(
        db_path
    )

    # ============================================
    # TRAIN
    # ============================================

    import_train(
        db,
        dataset_a / "train.jsonl",
        dataset_a.name,
    )

    import_train(
        db,
        dataset_b / "train.jsonl",
        dataset_b.name,
    )

    train_hashes = write_train(
        db,
        output / "train.jsonl",
        args.count,
    )

    # ============================================
    # DEV
    # ============================================

    dev_pairs = {}

    collect_eval_file(
        dataset_a / "dev.jsonl",
        dev_pairs,
    )

    collect_eval_file(
        dataset_b / "dev.jsonl",
        dev_pairs,
    )

    dev_count, dev_hashes = write_eval(
        dev_pairs,
        output / "dev.jsonl",
        train_hashes,
    )

    # ============================================
    # TEST
    # ============================================

    test_pairs = {}

    collect_eval_file(
        dataset_a / "test.jsonl",
        test_pairs,
    )

    collect_eval_file(
        dataset_b / "test.jsonl",
        test_pairs,
    )

    test_count, _ = write_eval(
        test_pairs,
        output / "test.jsonl",
        train_hashes,
        forbidden_hashes=dev_hashes,
    )

    db.close()

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print()
    print(f"Output: {output}")

    print()
    print("TRAIN")
    print(f"  unique pairs : {args.count:,}")
    print(f"  uz-ru        : {args.count:,}")
    print(f"  ru-uz        : {args.count:,}")
    print(f"  total rows   : {args.count * 2:,}")

    print()
    print("DEV")
    print(f"  unique pairs : {dev_count:,}")
    print(f"  total rows   : {dev_count * 2:,}")

    print()
    print("TEST")
    print(f"  unique pairs : {test_count:,}")
    print(f"  total rows   : {test_count * 2:,}")


if __name__ == "__main__":
    main()