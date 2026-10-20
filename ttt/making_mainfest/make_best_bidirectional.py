#!/usr/bin/env python3

"""
Katta EN-RU Riva datasetdan eng yaxshi N ta parallel juftlikni tanlaydi
va ularni ikkala direction'da bitta train.jsonl faylga yozadi.

Input:

    /workspace/datasets/ttt/en-ru/train.jsonl

Input row:

    {
        "pair": "en-ru",
        "source": "English text",
        "target": "Russian text"
    }

Output:

    /workspace/datasets/ttt/en-ru-ru-en-3m/train.jsonl

Natija:

    1,500,000 en-ru
    1,500,000 ru-en

Jami:

    3,000,000 rows

Ishlatish:

    python make_best_bidirectional.py

yoki:

    python make_best_bidirectional.py \
        --input /workspace/datasets/ttt/en-ru/train.jsonl \
        --output-dir /workspace/datasets/ttt/en-ru-ru-en-3m \
        --count 1500000
"""

import argparse
import heapq
import json
import math
import re
from pathlib import Path


# ---------------------------------------------------------
# Regexlar
# ---------------------------------------------------------

EN_RE = re.compile(r"[A-Za-z]")
RU_RE = re.compile(r"[А-Яа-яЁё]")

URL_RE = re.compile(
    r"https?://|www\.",
    re.IGNORECASE,
)

HTML_RE = re.compile(
    r"<[^>]+>"
)

REPEATED_RE = re.compile(
    r"(.)\1{5,}"
)

WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------
# Matnni normalize qilish
# ---------------------------------------------------------

def clean(text: str) -> str:
    """
    Faqat basic normalization.

    Dataset oldindan clean qilingan bo'lsa ham,
    scoring vaqtida space'larni normal holatga keltiramiz.
    """

    text = WHITESPACE_RE.sub(" ", text)

    return text.strip()


# ---------------------------------------------------------
# Harf ulushi
# ---------------------------------------------------------

def letter_ratio(text: str, regex: re.Pattern) -> float:
    """
    Matndagi kerakli alphabet belgilarining
    barcha alphabetic belgilariga nisbati.
    """

    letters = sum(
        1
        for char in text
        if char.isalpha()
    )

    if letters == 0:
        return 0.0

    wanted = len(regex.findall(text))

    return wanted / letters


# ---------------------------------------------------------
# Quality score
# ---------------------------------------------------------

def quality_score(en: str, ru: str) -> float | None:
    """
    Parallel pair uchun quality score qaytaradi.

    Juda yomon pair bo'lsa:
        None

    Yaxshi pair bo'lsa:
        katta score = yaxshiroq.
    """

    en = clean(en)
    ru = clean(ru)

    # -----------------------------------------------------
    # 1. Empty
    # -----------------------------------------------------

    if not en or not ru:
        return None

    en_len = len(en)
    ru_len = len(ru)

    # -----------------------------------------------------
    # 2. Juda qisqa / juda uzun gaplarni olib tashlaymiz
    # -----------------------------------------------------

    # Juda qisqa phrase'lar translation training uchun
    # ko'pincha kamroq foydali.
    if en_len < 15 or ru_len < 15:
        return None

    # Juda katta paragraphlarni ham olmaslik.
    if en_len > 800 or ru_len > 800:
        return None

    # -----------------------------------------------------
    # 3. Language purity
    # -----------------------------------------------------

    en_ratio = letter_ratio(en, EN_RE)
    ru_ratio = letter_ratio(ru, RU_RE)

    # English tomon asosan Latin bo'lishi kerak.
    if en_ratio < 0.80:
        return None

    # Russian tomon asosan Cyrillic bo'lishi kerak.
    if ru_ratio < 0.80:
        return None

    # -----------------------------------------------------
    # 4. URL / HTML
    # -----------------------------------------------------

    if URL_RE.search(en) or URL_RE.search(ru):
        return None

    if HTML_RE.search(en) or HTML_RE.search(ru):
        return None

    # -----------------------------------------------------
    # 5. Buzuq repeated character
    # -----------------------------------------------------

    # Masalan:
    #
    #   !!!!!!!!
    #   aaaaaaaa
    #
    if REPEATED_RE.search(en):
        return None

    if REPEATED_RE.search(ru):
        return None

    # -----------------------------------------------------
    # 6. Source == target
    # -----------------------------------------------------

    if en.casefold() == ru.casefold():
        return None

    # -----------------------------------------------------
    # 7. Length ratio
    # -----------------------------------------------------

    ratio = en_len / ru_len

    # English/Russian uchun haddan tashqari farqli
    # length ko'pincha bad alignment signalidir.
    if ratio < 0.45 or ratio > 2.2:
        return None

    # Ratio 1 ga qancha yaqin bo'lsa score yuqoriroq.
    ratio_score = math.exp(
        -abs(math.log(ratio))
    )

    # -----------------------------------------------------
    # 8. Sentence length preference
    # -----------------------------------------------------

    # Translation training uchun o'rtacha uzunlikdagi
    # sentence'lar foydali.
    avg_len = (en_len + ru_len) / 2

    # ~120 belgi atrofini yaxshi deb olamiz.
    length_score = math.exp(
        -abs(avg_len - 120) / 250
    )

    # -----------------------------------------------------
    # 9. Language purity score
    # -----------------------------------------------------

    language_score = (
        en_ratio + ru_ratio
    ) / 2

    # -----------------------------------------------------
    # Final score
    # -----------------------------------------------------

    score = (
        ratio_score * 4.0
        + language_score * 3.0
        + length_score * 2.0
    )

    return score


# ---------------------------------------------------------
# Eng yaxshi N row offsetlarini topish
# ---------------------------------------------------------

def find_best_offsets(
    input_path: Path,
    count: int,
):
    """
    Faylni bir marta sequential o'qib,
    eng yaxshi N ta row offsetlarini heap orqali saqlaydi.

    Muhim:

    to'liq 35M row RAM'da saqlanmaydi.
    """

    heap = []

    total = 0
    valid = 0

    with input_path.open("rb") as f:

        while True:

            offset = f.tell()

            raw = f.readline()

            if not raw:
                break

            total += 1

            try:
                row = json.loads(
                    raw.decode("utf-8")
                )

            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
            ):
                continue

            en = row.get("source")
            ru = row.get("target")

            if not isinstance(en, str):
                continue

            if not isinstance(ru, str):
                continue

            score = quality_score(en, ru)

            if score is None:
                continue

            valid += 1

            item = (
                score,
                offset,
            )

            # Heap hali to'lmagan.
            if len(heap) < count:

                heapq.heappush(
                    heap,
                    item,
                )

            # Yangi row heap'dagi eng yomon rowdan
            # yaxshiroq bo'lsa almashtiramiz.
            elif score > heap[0][0]:

                heapq.heapreplace(
                    heap,
                    item,
                )

            if total % 500_000 == 0:

                print(
                    f"scanned={total:,} "
                    f"valid={valid:,} "
                    f"selected={len(heap):,} "
                    f"min_score={heap[0][0]:.4f}"
                    if heap
                    else ""
                )

    print()
    print("Selection tugadi")
    print(f"Jami ko'rildi : {total:,}")
    print(f"Valid         : {valid:,}")
    print(f"Tanlandi      : {len(heap):,}")

    # Offsetlar bo'yicha sort qilamiz.
    #
    # Bu second-pass paytida disk seek'ni kamaytiradi.
    offsets = sorted(
        offset
        for _, offset in heap
    )

    return offsets


# ---------------------------------------------------------
# Output yozish
# ---------------------------------------------------------

def write_bidirectional(
    input_path: Path,
    output_path: Path,
    offsets: list[int],
):
    """
    Tanlangan pairlarni:

        en-ru
        ru-en

    qilib bitta train.jsonl ga yozadi.
    """

    en_ru_count = 0
    ru_en_count = 0

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        input_path.open("rb") as fin,
        output_path.open(
            "w",
            encoding="utf-8",
        ) as fout,
    ):

        for index, offset in enumerate(
            offsets,
            1,
        ):

            fin.seek(offset)

            raw = fin.readline()

            row = json.loads(
                raw.decode("utf-8")
            )

            en = clean(row["source"])
            ru = clean(row["target"])

            # ---------------------------------------------
            # EN -> RU
            # ---------------------------------------------

            en_ru = {
                "pair": "en-ru",
                "source": en,
                "target": ru,
            }

            fout.write(
                json.dumps(
                    en_ru,
                    ensure_ascii=False,
                )
                + "\n"
            )

            en_ru_count += 1

            # ---------------------------------------------
            # RU -> EN
            # ---------------------------------------------

            ru_en = {
                "pair": "ru-en",
                "source": ru,
                "target": en,
            }

            fout.write(
                json.dumps(
                    ru_en,
                    ensure_ascii=False,
                )
                + "\n"
            )

            ru_en_count += 1

            if index % 100_000 == 0:

                print(
                    f"written pairs: {index:,} "
                    f"total rows: {index * 2:,}"
                )

    return (
        en_ru_count,
        ru_en_count,
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "/workspace/datasets/"
            "ttt/en-ru/train.jsonl"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "/workspace/datasets/"
            "ttt/en-ru-ru-en-3m"
        ),
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1_500_000,
        help=(
            "Original EN-RU pairlar soni. "
            "Output bundan 2x katta bo'ladi."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    output_dir = Path(
        args.output_dir
    )

    output_path = (
        output_dir
        / "train.jsonl"
    )

    if not input_path.exists():

        raise FileNotFoundError(
            input_path
        )

    print()
    print("==============================")
    print(" BEST EN-RU DATASET SELECTOR")
    print("==============================")
    print()

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    print()
    print(
        f"Tanlanadi : {args.count:,} EN-RU"
    )

    print(
        f"Reverse   : {args.count:,} RU-EN"
    )

    print(
        f"Total     : {args.count * 2:,}"
    )

    print()

    # -----------------------------------------------------
    # 1-pass: best offsets
    # -----------------------------------------------------

    offsets = find_best_offsets(
        input_path,
        args.count,
    )

    if len(offsets) < args.count:

        print()
        print(
            "[WARNING] Filtrlardan o'tgan rowlar "
            f"{args.count:,} dan kam."
        )

    # -----------------------------------------------------
    # 2-pass: output
    # -----------------------------------------------------

    en_ru_count, ru_en_count = (
        write_bidirectional(
            input_path,
            output_path,
            offsets,
        )
    )

    print()
    print("==============================")
    print(" DONE")
    print("==============================")

    print()
    print(
        f"en-ru : {en_ru_count:,}"
    )

    print(
        f"ru-en : {ru_en_count:,}"
    )

    print(
        f"total : "
        f"{en_ru_count + ru_en_count:,}"
    )

    print()
    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()