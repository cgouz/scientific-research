#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


WHITESPACE_RE = re.compile(r"\s+")


def clean(text: str) -> str:
    """Ortiqcha whitespace'larni bitta space'ga keltiradi."""
    return WHITESPACE_RE.sub(" ", text).strip()


def convert_bidirectional(input_path: Path, output_path: Path):
    """
    EN-RU eval faylni ikkala direction'da yozadi:

    en-ru:
        source = English
        target = Russian

    ru-en:
        source = Russian
        target = English
    """

    en_ru = 0
    ru_en = 0
    skipped = 0

    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
    ):
        for line_no, line in enumerate(fin, 1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[!] {line_no}-qator JSON emas")
                skipped += 1
                continue

            # Eski dataset:
            # {"pair":"en-ru","source":"...","target":"..."}
            en = row.get("source")
            ru = row.get("target")

            if not isinstance(en, str) or not isinstance(ru, str):
                skipped += 1
                continue

            en = clean(en)
            ru = clean(ru)

            if not en or not ru:
                skipped += 1
                continue

            # EN -> RU
            fout.write(
                json.dumps(
                    {
                        "pair": "en-ru",
                        "source": en,
                        "target": ru,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            en_ru += 1

            # RU -> EN
            fout.write(
                json.dumps(
                    {
                        "pair": "ru-en",
                        "source": ru,
                        "target": en,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            ru_en += 1

    print(f"\n{input_path.name}")
    print(f"  en-ru : {en_ru:,}")
    print(f"  ru-en : {ru_en:,}")
    print(f"  total : {en_ru + ru_en:,}")
    print(f"  skip  : {skipped:,}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="/workspace/datasets/ttt/en-ru",
    )

    parser.add_argument(
        "--output-dir",
        default="/workspace/datasets/ttt/en-ru-ru-en-3m",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_dev = input_dir / "dev.jsonl"
    input_test = input_dir / "test.jsonl"

    output_dev = output_dir / "dev.jsonl"
    output_test = output_dir / "test.jsonl"

    if not input_dev.exists():
        raise FileNotFoundError(input_dev)

    if not input_test.exists():
        raise FileNotFoundError(input_test)

    print("DEV yaratilmoqda...")
    convert_bidirectional(
        input_dev,
        output_dev,
    )

    print("\nTEST yaratilmoqda...")
    convert_bidirectional(
        input_test,
        output_test,
    )

    print("\nDONE")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()