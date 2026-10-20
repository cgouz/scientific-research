#!/usr/bin/env python3

"""
EN-RU, UZ-EN va UZ-RU Riva datasetlarini
bitta multilingual datasetga merge + shuffle qiladi.

Input:

    /data/datasets/ttt/
    ├── en-ru/
    │   ├── train.jsonl
    │   ├── dev.jsonl
    │   └── test.jsonl
    ├── uz-en/
    │   ├── train.jsonl
    │   ├── dev.jsonl
    │   └── test.jsonl
    └── uz-ru/
        ├── train.jsonl
        ├── dev.jsonl
        └── test.jsonl

Output:

    /data/datasets/ttt/multilingual/
    ├── train.jsonl
    ├── dev.jsonl
    └── test.jsonl

Muhim:

- train faqat train bilan merge qilinadi
- dev faqat dev bilan merge qilinadi
- test faqat test bilan merge qilinadi
- barcha mavjud fieldlar saqlanadi (`origin` ham)
- millionlab qator RAM'ga yuklanmaydi
- shuffle seed orqali reproducible
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------
# Datasetlar
# ---------------------------------------------------------

DATASET_DIRS = [
    "en-ru",
    "uz-en",
    "uz-ru",
]

SPLITS = [
    "train",
    "dev",
    "test",
]

# Biz kutayotgan barcha directionlar.
ALLOWED_PAIRS = {
    "en-ru",
    "ru-en",
    "uz-en",
    "en-uz",
    "uz-ru",
    "ru-uz",
}


# ---------------------------------------------------------
# Shuffle key
# ---------------------------------------------------------

def make_shuffle_key(
    line: bytes,
    seed: int,
    source_index: int,
    line_number: int,
) -> str:
    """
    Har bir row uchun deterministic pseudo-random key yaratadi.

    Bir xil seed bilan script qayta ishlatilsa,
    bir xil shuffle tartibi hosil bo'ladi.

    source_index + line_number qo'shilishining sababi:
    bir xil JSON row bir necha datasetda uchrasa ham
    key collision bo'lmasligi.
    """

    prefix = (
        f"{seed}:{source_index}:{line_number}:"
    ).encode("utf-8")

    digest = hashlib.blake2b(
        prefix + line,
        digest_size=16,
    )

    return digest.hexdigest()


# ---------------------------------------------------------
# Splitni merge qilish
# ---------------------------------------------------------

def build_keyed_file(
    input_files: list[Path],
    keyed_path: Path,
    seed: int,
):
    """
    Barcha input JSONL fayllarni stream qilib o'qiydi.

    Temporary file format:

        RANDOM_HASH<TAB>JSON

    Masalan:

        a82f...    {"pair":"uz-ru", ...}

    Bu key keyinchalik `sort` orqali global shuffle
    qilish uchun ishlatiladi.
    """

    stats = Counter()

    source_stats = Counter()

    total_input = 0
    valid_rows = 0
    invalid_rows = 0

    with keyed_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as fout:

        for source_index, input_path in enumerate(
            input_files
        ):

            print(f"  reading: {input_path}")

            with input_path.open(
                "rb"
            ) as fin:

                for line_number, raw in enumerate(
                    fin,
                    1,
                ):

                    total_input += 1

                    raw = raw.strip()

                    if not raw:
                        continue

                    # -------------------------------------
                    # JSON validligini tekshiramiz
                    # -------------------------------------

                    try:
                        row = json.loads(
                            raw.decode("utf-8")
                        )

                    except (
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                    ):
                        invalid_rows += 1
                        continue

                    if not isinstance(row, dict):
                        invalid_rows += 1
                        continue

                    pair = row.get("pair")
                    source = row.get("source")
                    target = row.get("target")

                    if pair not in ALLOWED_PAIRS:
                        invalid_rows += 1
                        continue

                    if not isinstance(source, str):
                        invalid_rows += 1
                        continue

                    if not isinstance(target, str):
                        invalid_rows += 1
                        continue

                    if not source.strip():
                        invalid_rows += 1
                        continue

                    if not target.strip():
                        invalid_rows += 1
                        continue

                    # -------------------------------------
                    # Statistikalar
                    # -------------------------------------

                    stats[pair] += 1
                    source_stats[input_path.parent.name] += 1

                    # -------------------------------------
                    # Shuffle key
                    # -------------------------------------

                    key = make_shuffle_key(
                        raw,
                        seed=seed,
                        source_index=source_index,
                        line_number=line_number,
                    )

                    # Original JSON row'ni o'zgartirmaymiz.
                    fout.write(
                        key
                        + "\t"
                        + raw.decode("utf-8")
                        + "\n"
                    )

                    valid_rows += 1

                    if valid_rows % 500_000 == 0:
                        print(
                            f"    valid rows: "
                            f"{valid_rows:,}"
                        )

    return {
        "total_input": total_input,
        "valid": valid_rows,
        "invalid": invalid_rows,
        "pairs": stats,
        "sources": source_stats,
    }


# ---------------------------------------------------------
# Disk-based global shuffle
# ---------------------------------------------------------

def external_shuffle(
    keyed_path: Path,
    temp_dir: Path,
    memory: str,
    parallel: int,
):
    """
    GNU `sort` yordamida hash key bo'yicha sort qiladi.

    Hashlar random taqsimlanganligi sababli,
    natija amalda global shuffled dataset bo'ladi.

    `sort` RAM yetmasa temporary disk ishlatadi.
    """

    command = [
        "sort",
        "-T",
        str(temp_dir),
        "-S",
        memory,
        "--parallel",
        str(parallel),

        # Faqat birinchi field (hash) bo'yicha sort.
        "-k1,1",

        # Input va output bir file.
        "-o",
        str(keyed_path),

        str(keyed_path),
    ]

    env = os.environ.copy()

    # Byte-wise sort:
    # tezroq va locale'dan mustaqil.
    env["LC_ALL"] = "C"

    print()
    print("  global shuffle (external sort)...")

    subprocess.run(
        command,
        check=True,
        env=env,
    )


# ---------------------------------------------------------
# Hash keyni olib tashlash
# ---------------------------------------------------------

def write_final_jsonl(
    keyed_path: Path,
    output_path: Path,
):
    """
    Temporary:

        HASH<TAB>JSON

    ni final:

        JSON

    formatga o'tkazadi.
    """

    written = 0

    with (
        keyed_path.open(
            "r",
            encoding="utf-8",
        ) as fin,

        output_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as fout,
    ):

        for line in fin:

            # Faqat birinchi TAB bo'yicha ajratamiz.
            try:
                _, json_line = line.split(
                    "\t",
                    1,
                )

            except ValueError:
                continue

            fout.write(json_line)

            written += 1

            if written % 500_000 == 0:
                print(
                    f"  written: {written:,}"
                )

    return written


# ---------------------------------------------------------
# Bitta splitni tayyorlash
# ---------------------------------------------------------

def process_split(
    base_dir: Path,
    output_dir: Path,
    split: str,
    temp_root: Path,
    seed: int,
    sort_memory: str,
    parallel: int,
):
    """
    Masalan train uchun:

        en-ru/train.jsonl
        uz-en/train.jsonl
        uz-ru/train.jsonl

    -> multilingual/train.jsonl
    """

    print()
    print("=" * 60)
    print(f"{split.upper()} MERGE + SHUFFLE")
    print("=" * 60)

    input_files = []

    for dataset_name in DATASET_DIRS:

        path = (
            base_dir
            / dataset_name
            / f"{split}.jsonl"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Topilmadi: {path}"
            )

        input_files.append(path)

    # Har split uchun alohida temp directory.
    split_temp = (
        temp_root
        / split
    )

    split_temp.mkdir(
        parents=True,
        exist_ok=True,
    )

    keyed_path = (
        split_temp
        / f"{split}.keyed"
    )

    output_path = (
        output_dir
        / f"{split}.jsonl"
    )

    # Splitlarga boshqa seed beramiz.
    split_seed = (
        seed
        + {
            "train": 0,
            "dev": 1,
            "test": 2,
        }[split]
    )

    print()
    print("1. Merge + validate + shuffle key")

    stats = build_keyed_file(
        input_files=input_files,
        keyed_path=keyed_path,
        seed=split_seed,
    )

    print()
    print(
        f"  jami input : "
        f"{stats['total_input']:,}"
    )

    print(
        f"  valid      : "
        f"{stats['valid']:,}"
    )

    print(
        f"  invalid    : "
        f"{stats['invalid']:,}"
    )

    print()
    print("  Pair distribution:")

    for pair, count in sorted(
        stats["pairs"].items()
    ):
        print(
            f"    {pair:6s} : "
            f"{count:,}"
        )

    print()
    print("  Dataset sources:")

    for source, count in sorted(
        stats["sources"].items()
    ):
        print(
            f"    {source:6s} : "
            f"{count:,}"
        )

    # ---------------------------------------------
    # Global shuffle
    # ---------------------------------------------

    print()
    print("2. Shuffle")

    external_shuffle(
        keyed_path=keyed_path,
        temp_dir=split_temp,
        memory=sort_memory,
        parallel=parallel,
    )

    # ---------------------------------------------
    # Final JSONL
    # ---------------------------------------------

    print()
    print("3. Final JSONL")

    written = write_final_jsonl(
        keyed_path=keyed_path,
        output_path=output_path,
    )

    # Temp keyed file endi kerak emas.
    keyed_path.unlink(
        missing_ok=True
    )

    # Agar temp folder bo'sh bo'lsa o'chiramiz.
    try:
        split_temp.rmdir()
    except OSError:
        pass

    print()
    print(
        f"{split}.jsonl tayyor: "
        f"{written:,} rows"
    )

    print(
        f"  {output_path}"
    )

    return stats


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "EN-RU + UZ-EN + UZ-RU datasetlarini "
            "Riva multilingual train/dev/test datasetga "
            "merge va global shuffle qiladi."
        )
    )

    parser.add_argument(
        "--base-dir",
        default="/data/datasets/ttt",
        help=(
            "en-ru, uz-en va uz-ru "
            "papkalari joylashgan directory"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="/data/datasets/ttt/multilingual",
    )

    parser.add_argument(
        "--temp-dir",
        default="/data/datasets/ttt/.shuffle_tmp",
        help=(
            "External sort temporary fayllari "
            "uchun directory"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Shuffle seed. "
            "Bir xil seed -> bir xil tartib."
        ),
    )

    parser.add_argument(
        "--sort-memory",
        default="2G",
        help=(
            "GNU sort ishlatishi mumkin bo'lgan RAM. "
            "Masalan: 1G, 2G, 4G"
        ),
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=max(
            1,
            min(
                os.cpu_count() or 1,
                8,
            ),
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # GNU sort mavjudligini tekshirish
    # -----------------------------------------------------

    if shutil.which("sort") is None:

        raise RuntimeError(
            "GNU sort topilmadi."
        )

    base_dir = Path(
        args.base_dir
    )

    output_dir = Path(
        args.output_dir
    )

    temp_dir = Path(
        args.temp_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 60)
    print("RIVA MULTILINGUAL DATASET BUILDER")
    print("=" * 60)

    print()
    print(f"Input : {base_dir}")
    print(f"Output: {output_dir}")
    print(f"Temp  : {temp_dir}")
    print(f"Seed  : {args.seed}")

    print()
    print("Sources:")

    for dataset_name in DATASET_DIRS:
        print(
            f"  {base_dir / dataset_name}"
        )

    all_stats = {}

    # -----------------------------------------------------
    # Train / dev / test alohida-alohida
    # -----------------------------------------------------

    for split in SPLITS:

        all_stats[split] = (
            process_split(
                base_dir=base_dir,
                output_dir=output_dir,
                split=split,
                temp_root=temp_dir,
                seed=args.seed,
                sort_memory=args.sort_memory,
                parallel=args.parallel,
            )
        )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    for split in SPLITS:

        stats = all_stats[split]

        print()
        print(
            f"{split.upper()}: "
            f"{stats['valid']:,} rows"
        )

        for pair, count in sorted(
            stats["pairs"].items()
        ):
            print(
                f"  {pair:6s}: "
                f"{count:,}"
            )

    print()
    print("Final dataset:")
    print(output_dir)


if __name__ == "__main__":
    main()