#!/usr/bin/env python3

"""
WMT19 EN-RU dataset -> Riva training JSONL converter.

Riva format:

    {
        "pair": "en-ru",
        "source": "English sentence",
        "target": "Russian sentence"
    }

Natija:

    output/
    ├── train.jsonl
    ├── dev.jsonl
    └── test.jsonl


Ishlatish:

    python wmt19_to_riva.py \
        --output-dir /data/datasets/ttt/en-ru


Agar WMT19 parquet fayllarini oldindan yuklab olgan bo'lsang:

    python wmt19_to_riva.py \
        --input-dir /workspace/datasets/wmt19/ru-en \
        --output-dir /data/datasets/ttt/en-ru


Test uchun faqat 10000 train row:

    python wmt19_to_riva.py \
        --output-dir /data/datasets/ttt/en-ru \
        --max-train 10000
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------
# Whitespace
# ---------------------------------------------------------

WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Matndagi newline, tab va ortiqcha space'larni
    bitta oddiy space'ga keltiradi.
    """

    if not isinstance(text, str):
        return ""

    text = WHITESPACE_RE.sub(" ", text)

    return text.strip()


# ---------------------------------------------------------
# WMT row -> Riva row
# ---------------------------------------------------------

def convert_row(row):
    """
    Hugging Face WMT19 row:

        {
            "translation": {
                "ru": "...",
                "en": "..."
            }
        }

    ni Riva formatiga o'tkazadi:

        {
            "pair": "en-ru",
            "source": "...",
            "target": "..."
        }
    """

    translation = row.get("translation")

    if not isinstance(translation, dict):
        return None

    en = clean_text(translation.get("en", ""))
    ru = clean_text(translation.get("ru", ""))

    # Bo'sh pairlarni tashlaymiz.
    if not en or not ru:
        return None

    # Source va target aynan bir xil bo'lsa,
    # translation dataset uchun foydasi kam.
    if en.casefold() == ru.casefold():
        return None

    return {
        "pair": "en-ru",
        "source": en,
        "target": ru,
    }


# ---------------------------------------------------------
# Duplicate hash
# ---------------------------------------------------------

def pair_hash(source: str, target: str) -> bytes:
    """
    Duplicate pairlarni aniqlash uchun kichik hash.

    To'liq matnni set ichida saqlashdan ko'ra
    xotirani kamroq ishlatadi.
    """

    raw = (
        source.casefold()
        + "\t"
        + target.casefold()
    ).encode("utf-8")

    return hashlib.blake2b(
        raw,
        digest_size=16,
    ).digest()


# ---------------------------------------------------------
# Local parquet topish
# ---------------------------------------------------------

def find_local_parquet(input_dir: Path):
    """
    Local WMT19 ru-en papkasidan train va validation
    parquet shardlarini avtomatik topadi.
    """

    train_files = sorted(
        input_dir.glob("train-*.parquet")
    )

    validation_files = sorted(
        input_dir.glob("validation-*.parquet")
    )

    # Ba'zi directory strukturalarida ichkarida
    # qo'shimcha folder bo'lishi mumkin.
    if not train_files:
        train_files = sorted(
            input_dir.rglob("train-*.parquet")
        )

    if not validation_files:
        validation_files = sorted(
            input_dir.rglob("validation-*.parquet")
        )

    return train_files, validation_files


# ---------------------------------------------------------
# Dataset load
# ---------------------------------------------------------

def load_wmt19(input_dir=None):
    """
    Agar --input-dir berilgan bo'lsa:
        local parquet fayllarni ishlatadi.

    Aks holda:
        Hugging Face'dan wmt/wmt19 ru-en ni streaming
        rejimida o'qiydi.

    Streaming ishlatishimiz sababi:
        train dataset juda katta va RAM'ga to'liq
        yuklash shart emas.
    """

    if input_dir:
        input_dir = Path(input_dir)

        print(f"Local dataset qidirilmoqda: {input_dir}")

        train_files, validation_files = (
            find_local_parquet(input_dir)
        )

        if not train_files:
            raise FileNotFoundError(
                f"Train parquet topilmadi: {input_dir}"
            )

        if not validation_files:
            raise FileNotFoundError(
                f"Validation parquet topilmadi: {input_dir}"
            )

        print(
            f"Train shardlar      : {len(train_files)}"
        )

        print(
            f"Validation shardlar : {len(validation_files)}"
        )

        ds = load_dataset(
            "parquet",
            data_files={
                "train": [
                    str(x)
                    for x in train_files
                ],
                "validation": [
                    str(x)
                    for x in validation_files
                ],
            },
            streaming=True,
        )

        return ds

    print(
        "Hugging Face'dan wmt/wmt19 ru-en "
        "streaming orqali ochilmoqda..."
    )

    return load_dataset(
        "wmt/wmt19",
        "ru-en",
        streaming=True,
    )


# ---------------------------------------------------------
# Train yozish
# ---------------------------------------------------------

def write_train(
    dataset,
    output_path: Path,
    max_rows=None,
    deduplicate=False,
):
    """
    Train splitni Riva JSONL formatiga yozadi.
    """

    saved = 0
    skipped = 0

    seen = set() if deduplicate else None

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for row in dataset:

            converted = convert_row(row)

            if converted is None:
                skipped += 1
                continue

            if seen is not None:

                key = pair_hash(
                    converted["source"],
                    converted["target"],
                )

                if key in seen:
                    skipped += 1
                    continue

                seen.add(key)

            fout.write(
                json.dumps(
                    converted,
                    ensure_ascii=False,
                )
                + "\n"
            )

            saved += 1

            # Har 100k qatorda progress chiqaramiz.
            if saved % 100_000 == 0:
                print(
                    f"train: {saved:,} row yozildi"
                )

            if (
                max_rows is not None
                and saved >= max_rows
            ):
                break

    print()
    print("Train tayyor:")
    print(f"  saqlandi : {saved:,}")
    print(f"  skip     : {skipped:,}")

    return saved


# ---------------------------------------------------------
# Validation -> dev/test
# ---------------------------------------------------------

def write_dev_test(
    dataset,
    dev_path: Path,
    test_path: Path,
):
    """
    WMT19 ru-en alohida test split bermaydi.

    Shuning uchun validation splitni deterministic
    tarzda 50/50:

        dev
        test

    ga ajratamiz.

    Hash ishlatilgani uchun script qayta ishga
    tushirilganda row har doim bir xil splitga tushadi.
    """

    dev_count = 0
    test_count = 0
    skipped = 0

    with (
        dev_path.open(
            "w",
            encoding="utf-8",
        ) as fdev,

        test_path.open(
            "w",
            encoding="utf-8",
        ) as ftest,
    ):

        for row in dataset:

            converted = convert_row(row)

            if converted is None:
                skipped += 1
                continue

            key = pair_hash(
                converted["source"],
                converted["target"],
            )

            # Birinchi hash byte.
            #
            # Juft bo'lsa dev,
            # toq bo'lsa test.
            if key[0] % 2 == 0:

                fdev.write(
                    json.dumps(
                        converted,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                dev_count += 1

            else:

                ftest.write(
                    json.dumps(
                        converted,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                test_count += 1

    print()
    print("Validation split ajratildi:")
    print(f"  dev  : {dev_count:,}")
    print(f"  test : {test_count:,}")
    print(f"  skip : {skipped:,}")

    return dev_count, test_count


# ---------------------------------------------------------
# main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "WMT19 ru-en datasetni "
            "Riva en-ru JSONL formatiga o'tkazadi"
        )
    )

    parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Oldindan yuklangan WMT19 ru-en parquet "
            "papkasi. Berilmasa HF Hub ishlatiladi."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="/data/datasets/ttt/en-ru",
        help=(
            "Riva JSONL output directory "
            "(default: /data/datasets/ttt/en-ru)"
        ),
    )

    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help=(
            "Faqat N ta train row yozish. "
            "Test/debug uchun foydali."
        ),
    )

    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help=(
            "Train dataset ichidagi exact "
            "duplicate pairlarni olib tashlash. "
            "37M datasetda RAM ko'proq ishlatadi."
        ),
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    test_path = output_dir / "test.jsonl"

    print()
    print("======================================")
    print(" WMT19 RU-EN -> Riva EN-RU")
    print("======================================")
    print()

    # Datasetni ochamiz.
    ds = load_wmt19(
        input_dir=args.input_dir,
    )

    print()
    print("Output:")
    print(f"  train : {train_path}")
    print(f"  dev   : {dev_path}")
    print(f"  test  : {test_path}")
    print()

    # ---------------------------------------------
    # Train
    # ---------------------------------------------

    write_train(
        ds["train"],
        train_path,
        max_rows=args.max_train,
        deduplicate=args.deduplicate,
    )

    # ---------------------------------------------
    # Validation -> dev / test
    # ---------------------------------------------

    write_dev_test(
        ds["validation"],
        dev_path,
        test_path,
    )

    print()
    print("======================================")
    print(" DONE")
    print("======================================")

    print()
    print("Riva dataset:")
    print(output_dir)

    print()
    print("Fayllar:")
    print(f"  {train_path.name}")
    print(f"  {dev_path.name}")
    print(f"  {test_path.name}")


if __name__ == "__main__":
    main()