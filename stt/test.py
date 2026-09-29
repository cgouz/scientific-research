#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

# zh/ja/ko uchun WER emas, CER. Koreys tilida boʻshliq bor, lekin soʻz
# chegarasi WER kutgandek emas.
CER_LOCALES = {"zh-CN", "ja-JP", "ko-KR"}


def edit_distance(a, b):
    """Levenshtein. Ikki qator (soʻzlar yoki belgilar) orasidagi masofa."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,        # oʻchirish
                           cur[j - 1] + 1,     # qoʻshish
                           prev[j - 1] + (ca != cb)))   # almashtirish
        prev = cur
    return prev[-1]


def error_rate(refs, hyps, use_cer):
    """WER yoki CER: jami xato / jami mos yozuv birligi."""
    err = tot = 0
    for r, h in zip(refs, hyps):
        r_u = list(r.strip()) if use_cer else r.split()
        h_u = list(h.strip()) if use_cer else h.split()
        err += edit_distance(r_u, h_u)
        tot += len(r_u)
    return err / max(tot, 1)


def norm(t):
    """Solishtirishdan oldin: kichik harf, ortiqcha boʻshliqsiz."""
    t = unicodedata.normalize("NFC", str(t or "")).lower()
    return " ".join(t.split())


def read_rows(p, limit=None):
    rows = []
    with open(p, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="/data/experiments/nemotron-multi/nemotron-multi-best.nemo",
                    help=".nemo fayl yoki HF nomi")
    ap.add_argument("--audio", nargs="*", help="bitta yoki bir nechta wav")
    ap.add_argument("--lang", default="auto",
                    help="--audio uchun: uz-UZ, ru-RU, auto ...")
    ap.add_argument("--bench", action="store_true",
                    help="dev toʻplamida har til uchun WER")
    ap.add_argument("--root", default="/data/datasets/sst")
    ap.add_argument("--split", default="dev", choices=("dev", "test", "train"))
    ap.add_argument("--langs", nargs="*",
                    help="qaysi til papkalari; berilmasa hammasi")
    ap.add_argument("--n", type=int, default=200, help="har tildan nechta")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--auto", action="store_true",
                    help="`auto` prompt bilan ham oʻlchash va solishtirish")
    ap.add_argument("--show", type=int, default=3,
                    help="har til uchun nechta misol koʻrsatish")
    ap.add_argument("--json", help="natijani shu faylga yozish")
    cfg = ap.parse_args()

    if not cfg.audio and not cfg.bench:
        sys.exit("--audio yoki --bench kerak")

    # ---------------------------------------------------------- model
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        sys.exit("NeMo oʻrnatilmagan")

    src = cfg.model
    print(f"\n  model: {src}")
    if Path(src).exists():
        model = nemo_asr.models.ASRModel.restore_from(src, map_location="cuda")
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(src,
                                                         map_location="cuda")
    model.eval()

    # Modelda qaysi tillar bor?
    try:
        pd = dict(model.cfg.train_ds.prompt_dictionary)
    except Exception:
        pd = {}

    def transcribe(paths, lang):
        """NeMo versiyalari turlicha nomlaydi — bir nechtasini sinaymiz."""
        for kw in ("target_lang", "lang", "prompt"):
            try:
                out = model.transcribe(paths, batch_size=cfg.batch,
                                       **{kw: lang})
                break
            except TypeError:
                continue
        else:
            out = model.transcribe(paths, batch_size=cfg.batch)
        # NeMo ba'zan Hypothesis obyektlarini qaytaradi
        return [getattr(o, "text", o) for o in out]

    # ---------------------------------------------------------- bitta audio
    if cfg.audio:
        missing = [p for p in cfg.audio if not Path(p).exists()]
        if missing:
            sys.exit(f"topilmadi: {missing[0]}")
        if pd and cfg.lang not in pd:
            print(f"  DIQQAT: `{cfg.lang}` modelning prompt lugʻatida yoʻq.")
            print(f"  Mavjudlari: {', '.join(sorted(pd)[:12])} ...")
        print(f"  til: {cfg.lang}\n")
        for p, t in zip(cfg.audio, transcribe(list(cfg.audio), cfg.lang)):
            print(f"  {Path(p).name}")
            print(f"    {t}\n")
        return 0

    # ---------------------------------------------------------- benchmark
    root = Path(cfg.root)
    langs = cfg.langs or sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / f"{cfg.split}.jsonl").exists())
    if not langs:
        sys.exit(f"{root} da {cfg.split}.jsonl topilmadi")

    print(f"  {cfg.split}.jsonl, har tildan {cfg.n} tagacha\n")
    results = {}
    for lang in langs:
        p = root / lang / f"{cfg.split}.jsonl"
        if not p.exists():
            continue
        rows = read_rows(p)
        if len(rows) > cfg.n:
            step = len(rows) / cfg.n
            rows = [rows[int(i * step)] for i in range(cfg.n)]
        rows = [r for r in rows if Path(r.get("audio_filepath", "")).exists()]
        if not rows:
            print(f"  {lang:<13} audio topilmadi, oʻtkazib yuborildi")
            continue

        loc = rows[0].get("locale", "?")
        paths = [r["audio_filepath"] for r in rows]
        refs = [norm(r.get("text", "")) for r in rows]
        use_cer = loc in CER_LOCALES
        metric = "CER" if use_cer else "WER"

        hyps = [norm(h) for h in transcribe(paths, loc)]
        rate = error_rate(refs, hyps, use_cer)
        line = f"  {lang:<13}{loc:<9}{len(rows):>5} ta   {metric} {rate:.4f}"

        rec = {"locale": loc, "rows": len(rows), "metric": metric,
               "rate": round(rate, 4)}

        if cfg.auto:
            hyps_a = [norm(h) for h in transcribe(paths, "auto")]
            rate_a = error_rate(refs, hyps_a, use_cer)
            # Model tilni oʻzi topa oldimi? Aniq til bilan solishtiramiz:
            # natijalar bir xil boʻlsa, prompt toʻgʻri tanlangan.
            same = sum(1 for a, b in zip(hyps, hyps_a) if a == b)
            line += f"   | auto {rate_a:.4f}  ({100*same/len(rows):.0f}% bir xil)"
            rec.update(rate_auto=round(rate_a, 4),
                       agree=round(same / len(rows), 3))

        print(line)
        results[lang] = rec

        for r, h in list(zip(refs, hyps))[:cfg.show]:
            print(f"      ref: {r[:90]}")
            print(f"      hyp: {h[:90]}")
        if cfg.show:
            print()

    # ---------------------------------------------------------- xulosa
    if results:
        print("  " + "=" * 62)
        worst = max(results, key=lambda k: results[k]["rate"])
        best = min(results, key=lambda k: results[k]["rate"])
        print(f"  eng yaxshi: {best} {results[best]['rate']:.4f}")
        print(f"  eng yomon:  {worst} {results[worst]['rate']:.4f}")
        avg = sum(r["rate"] for r in results.values()) / len(results)
        print(f"  oʻrtacha:   {avg:.4f}  ({len(results)} til)")
        print()
        print("  Eslatma: bu dev toʻplami — checkpoint shu boʻyicha")
        print("  tanlangan, shuning uchun raqam biroz chiroyliroq chiqadi.")
        print("  Haqiqiy baho uchun modelga umuman koʻrsatilmagan")
        print("  test toʻplami kerak.")
        print("  " + "=" * 62)

    if cfg.json:
        Path(cfg.json).write_text(json.dumps(results, ensure_ascii=False,
                                             indent=2), encoding="utf-8")
        print(f"\n  natija: {cfg.json}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())