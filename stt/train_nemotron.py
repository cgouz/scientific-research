#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_nemotron.py — Nemotron 3.5 ASR ni koʻp tilli fine-tuning.

    python3 train_nemotron.py

Hamma sozlama pastdagi SOZLAMALAR blokida. Faqat shu yerni oʻzgartiring.
"""

import json
import logging
import os
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict

# ============================== SOZLAMALAR ==============================

MODEL_NAME = "nvidia/nemotron-3.5-asr-streaming-0.6b"
ROOT       = "/data/datasets/stt"
EXP_DIR    = "/data/experiments"
RUN_NAME   = "nemotron-multi"

LANGS = ["english", "uzbek", "russian", "karakalpak", "korean"]

TEMPERATURE = 2.0

# --- Lhotse: batch yozuvlar soni bilan emas, audio davomiyligi bilan ---
# B200'da 600 dan boshlang. OOM boʻlsa 300 ga tushiring, boʻlmasa 900 ga
# koʻtaring. NVIDIA bu modelni 200 bilan oʻqitgan.
BATCH_DURATION     = 600      # bitta batchdagi audio soniyalari
QUADRATIC_DURATION = 30       # uzun audiolarning "narxini" oshiradi
NUM_BUCKETS        = 30       # oʻxshash uzunlikdagilarni guruhlash
NUM_WORKERS        = 32       # B200'ni och qoldirmaslik uchun
MAX_DURATION       = 35.0     # bundan uzun audio tashlab yuboriladi
MIN_DURATION       = 0.5      # bundan qisqasi RNNT loss'ni buzadi

# RNNT joint tensori B x T x U x V. Uni boʻlaklab hisoblash — katta
# batch umuman sigʻishining sababi. Buni oʻchirmang.
FUSED_BATCH_SIZE = 4

# --- Optimizatsiya ---
# DIQQAT: bu yerda sched OʻCHIRILMAYDI. Warmupsiz birinchi qadamlar
# pretrained vaznlarni buzadi. LR — jadval cho'qqisi, Noam koeffitsienti emas.
LR          = 1e-4
WARMUP      = 1000
VAL_EVERY   = 2000            # har shuncha qadamda dev tekshiriladi
GPUS        = 1

# Necha marta ma'lumot ustidan oʻtish ("epoch").
# Qadamlar soni shundan hisoblanadi:
#     qadam/oʻtish = jami_soat * 3600 / BATCH_DURATION
# Dinamik batch'da epoch uzunligi oldindan ma'lum emas, shuning uchun
# Lightning ga max_steps beriladi, max_epochs emas.
# Fine-tuning uchun 3-5 foydali oraliq; 5 dan keyin asosan overfit.
EPOCHS = 5

# Qadamlar sonini QOʻLDA belgilash. None boʻlsa EPOCHS dan hisoblanadi.
MAX_STEPS = None

# Encoder'ni muzlatish. 1800+ soat bor, shuning uchun False.
# Faqat bitta kichik til oʻqitsangiz True qiling.
FREEZE_ENCODER = False

# Manifestdagi til maydonining nomi. Model oʻz configida `target_lang`
# kutadi, sizning manifestlaringizda esa `locale`. Ikkalasini ham shu
# yerda koʻrsatamiz — manifestlarni qayta yozish shart emas.
LANG_FIELD = "locale"

# Model prompt lugʻatida boʻlmagan tillar uchun id biriktirish.
# Embedding jadvalida 128 ta joy bor (num_prompts=128), hammasi
# ishlatilmagan. Boʻsh joyni olish — modelga yangi tilni "tanishtirish".
#   "auto"        — boʻsh id avtomatik tanlanadi
#   {"kaa-UZ": 63} — aniq id
#   {}            — hech narsa qoʻshilmaydi; u holda tilni mavjud biriga
#                   bogʻlash kerak: apply_charmap.py karakalpak --set-locale uz-UZ
#                   (lekin model ikkala tilni farqlay olmaydi)
EXTRA_PROMPTS = "auto"

# --- Avtomatik til aniqlash (language identification) ---
# Modelda `auto` prompt bor (id 101). Agar oʻqitishda ba'zi qatorlar
# `auto` deb belgilansa, model tilni OʻZI aniqlashni oʻrganadi.
#
# Har bir til uchun ikkita manba yaratiladi:
#   train.jsonl       locale=uz-UZ   -> tilni siz aytasiz
#   train_auto.jsonl  locale=auto    -> model oʻzi topadi
# Tilning umumiy ulushi oʻzgarmaydi, faqat ikkiga boʻlinadi.
#
#   0.0  — oʻchirilgan, har doim tilni aytish kerak
#   0.3  — qatorlarning 30% i `auto` (tavsiya etiladi)
#   1.0  — faqat `auto`; tilni majburlash imkoni YOʻQOLADI
AUTO_LANG_RATIO = 0.3
AUTO_LOCALE     = "auto"

# --- Log sozlamalari ---
# Bular GPU vaqtiga TA'SIR QILADI, shunchaki chiroylilik emas.
QUIET_LOGS = True      # har qadam va har namunani yozmaslik
LOG_EVERY  = 100       # necha qadamda bir marta qisqa qator yozish

# NeMo har `LOG_EVERY_NEMO` qadamda TRAIN WER ni hisoblaydi — bu greedy
# dekodlash, ya'ni haqiqiy GPU ishi. NeMo defaulti 25, bu juda tez-tez.
# Train WER baribir shovqinli, undan foyda kam.
LOG_EVERY_NEMO = 500

# Validation narxi: har `VAL_EVERY` qadamda dev'dagi HAR BIR audio
# dekodlanadi. Sizda dev = 13,769 qator = 37.5 soat audio, ya'ni har
# validation 20-70 daqiqa GPU yeydi.
#
# Checkpoint tanlash uchun buncha kerak emas: har tildan 400 qator
# (jami ~2,000) WER ni ishonchli oʻlchash uchun yetarli va 6x tezroq.
# Toʻliq baholashni oxirida alohida qilinadi.
#   0 — cheklovsiz, hamma dev qatorlari
DEV_MAX_PER_LANG = 400

# <unk> topilsa nima qilish kerak.
#   True  — avtomatik tuzatish: qaysi belgi muammo qilayotganini aniqlaydi,
#           almashtiruvni topadi, manifestlarni qayta yozadi va davom etadi.
#           Asl fayllar .bak sifatida saqlanadi.
#   False — faqat xabar berib toʻxtaydi (qoʻlda: tokprobe.py + apply_charmap.py)
AUTO_FIX = True

# Tokenizer tekshiruvi. <unk> chiqsa oʻqitish boshlanmaydi.
CHECK_UNK        = True
CHECK_UNK_SAMPLE = 300
# Bir nechta qator <unk> boʻlishi normal — odatda begona soʻzlar. Shu
# ulushdan kam boʻlsa ogohlantiriladi, koʻp boʻlsa oʻqitish toʻxtaydi.
UNK_MAX_RATE     = 0.01

# ========================================================================

BASE = Path(__file__).resolve().parent
root = Path(ROOT)
exp = Path(EXP_DIR) / RUN_NAME
exp.mkdir(parents=True, exist_ok=True)


# ====================== belgi almashtirish qoidalari ==================
# tokprobe.py dagi bilan bir xil. Skript oʻzi yetarli — boshqa fayl
# kerak emas. (tokprobe.py faqat batafsil koʻrish uchun, ixtiyoriy.)

CANDIDATES = {
    # Uzbek: oʻ / gʻ use U+02BB, the tutuq belgisi uses U+02BC. Both are
    # modifier letters; the vocabulary almost certainly has plain quotes.
    "ʻ": ["‘", "'", "’", "`"],
    "ʼ": ["’", "'", "‘", "`"],
    "‘": ["'", "’"],
    "’": ["'", "‘"],
    "ʹ": ["'", "’"],
    "'": ["’", "‘"],
    # Karakalpak Latin: precomposed letters with diacritics. Turkish and
    # Azerbaijani equivalents come FIRST - those languages are in the model's
    # prompt dictionary, so their letters are likely in the vocabulary, and
    # they carry the same sound. That keeps the distinction instead of
    # collapsing two letters into one.
    "ǵ": ["ğ", "ǵ", "g"],      # ǵ -> ğ (Turkish g-breve)
    "Ǵ": ["Ğ", "Ǵ", "G"],      # Ǵ -> Ğ
    "á": ["á", "a"], "ó": ["ó", "o"],
    "ú": ["ú", "u"], "ı": ["i"],
    "ń": ["ń", "n"], "ś": ["ś", "s"],
    "ų": ["u"],
    # Typography that carries no meaning.
    "–": ["-"], "—": ["-"], "−": ["-"],
    " ": [" "], "​": [""], "‌": [""], "‍": [""],
    "“": ['"'], "”": ['"'], "„": ['"'],
    "«": ['"'], "»": ['"'], "…": ["..."],
}

# Punctuation nobody pronounces. An ASR transcript should not contain a
# bracket at all - the model has no sound to map it to, so it is noise in
# the target sequence. A space is tried before deletion so that removing
# "a: b" does not silently become "ab".
UNSPOKEN = "()[]{}<>:;\"«»“”„/\\|_*#@~^+=%$&"
for _c in UNSPOKEN:
    CANDIDATES.setdefault(_c, [" ", ""])


def auto_candidates(c):
    """Decomposition and case folding, tried before anything lossy."""
    out = []
    for form in ("NFD", "NFKD", "NFKC"):
        n = unicodedata.normalize(form, c)
        if n != c and n not in out:
            out.append(n)
    # the base letter with its marks stripped
    base = "".join(ch for ch in unicodedata.normalize("NFD", c)
                   if not unicodedata.combining(ch))
    if base and base != c and base not in out:
        out.append(base)
    return out


WS_RE = re.compile(r"\s+")


def apply_map(text: str, cmap: dict) -> str:
    """Apply a character map and tidy the whitespace it leaves behind."""
    for a, b in cmap.items():
        if a in text:
            text = text.replace(a, b)
    return WS_RE.sub(" ", text).strip()


def build_charmap(is_ok, chars):
    """Work out what each unrepresentable character should become.

    `is_ok(s)` must return True when s tokenises without <unk>. `chars` is
    any iterable of characters. Returns (mapping, unfixable).

    tokprobe.py da ham shu funksiya bor - ikkalasi bir xil natija beradi.
    """
    bad = [c for c in chars
           if not c.isspace()
           and not (is_ok(c) and is_ok(f"aa{c}aa") and is_ok(f"{c}a"))]
    mapping, unfixable = {}, []
    for c in bad:
        seen = set()
        for cand in CANDIDATES.get(c, []) + auto_candidates(c):
            if cand in seen:
                continue
            seen.add(cand)
            if is_ok(f"aa{cand}aa") and is_ok(cand if cand else "a"):
                mapping[c] = cand
                break
        else:
            unfixable.append(c)
    return mapping, unfixable


# ======================================================================

def read_rows(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
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


def autofix(lang, is_ok, sample=5000):
    """Find and apply the character map for one language, in place.

    Returns (mapping, unfixable, changed_rows) or None when nothing needs
    fixing. Originals are kept as <split>.jsonl.bak and every rewrite reads
    FROM the backup, so running this twice cannot apply the map twice.
    """
    d = root / lang
    chars = Counter()
    for r in read_rows(d / "train.jsonl", sample):
        chars.update(str(r.get("text", "") or ""))

    cmap, unfixable = build_charmap(is_ok, chars)
    if not cmap:
        return None

    (d / "charmap.json").write_text(
        json.dumps(cmap, ensure_ascii=False, indent=2), encoding="utf-8")

    changed = 0
    for split in ("train", "dev", "test"):
        f = d / f"{split}.jsonl"
        if not f.exists():
            continue
        bak = d / f"{split}.jsonl.bak"
        if not bak.exists():
            shutil.copy2(f, bak)
        out = []
        for ln in bak.open(encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            old = str(r.get("text", "") or "")
            new = apply_map(old, cmap)
            if not new:
                continue
            if new != old:
                changed += 1
            r["text"] = new
            out.append(json.dumps(r, ensure_ascii=False))
        f.write_text("\n".join(out) + "\n", encoding="utf-8")
    return cmap, unfixable, changed


def unk_rate(lang, tok, unk, n=None):
    rows = read_rows(root / lang / "train.jsonl", n or CHECK_UNK_SAMPLE)
    bad = sum(1 for r in rows
              if unk is not None
              and unk in tok.text_to_ids(str(r.get("text", "") or "")))
    return bad, len(rows)


# ---------------------------------------------------- 1. manifestlar
print("\n1. Manifestlar tekshirilmoqda...")
hours, manifests, locales = {}, {}, {}
for lang in LANGS:
    tr = root / lang / "train.jsonl"
    dv = root / lang / "dev.jsonl"
    if not tr.exists():
        sys.exit(f"   XATO: {tr} yoʻq")
    if not dv.exists():
        sys.exit(f"   XATO: {dv} yoʻq — dev'siz checkpoint tanlab boʻlmaydi")
    rows = read_rows(tr)
    hours[lang] = sum(r.get("duration", 0) or 0 for r in rows) / 3600
    manifests[lang] = str(tr)
    locales[lang] = rows[0].get(LANG_FIELD) if rows else None
    if not locales[lang]:
        sys.exit(f"   XATO: {lang} da `{LANG_FIELD}` maydoni yoʻq. "
                 f"Nemotron prompt bilan ishlaydi, u shu maydonni oʻqiydi.")
    print(f"   {lang:<13}{locales[lang]:<9}{len(rows):>9,} qator "
          f"{hours[lang]:>8.1f} soat")

total_h = sum(hours.values())

# --- qadamlar soni ---------------------------------------------------
STEPS_PER_PASS = max(1, int(total_h * 3600 / BATCH_DURATION))
if MAX_STEPS is None:
    MAX_STEPS = int(EPOCHS * STEPS_PER_PASS)
    _how = f"{EPOCHS} oʻtish x {STEPS_PER_PASS:,} qadam"
else:
    _how = f"qoʻlda belgilangan ({MAX_STEPS / STEPS_PER_PASS:.1f} oʻtish)"
print(f"\n   jami {total_h:,.1f} soat -> {STEPS_PER_PASS:,} qadam/oʻtish")
print(f"   max_steps = {MAX_STEPS:,}   ({_how})")
if WARMUP > MAX_STEPS * 0.1:
    print(f"   DIQQAT: warmup {WARMUP:,} — max_steps ning "
          f"{100*WARMUP/MAX_STEPS:.0f}% i. Juda koʻp, kamaytiring.")

# ---------------------------------------------------- 2. og'irliklar
# p_i = (soat_i / jami) ** (1/T), keyin normallashtiriladi.
raw = {k: v / total_h for k, v in hours.items()}
w = {k: p ** (1.0 / TEMPERATURE) for k, p in raw.items()}
s = sum(w.values())
weights = {k: v / s for k, v in w.items()}

print(f"\n2. Aralashtirish nisbati (temperature={TEMPERATURE})")
print(f"   {'TIL':<13}{'TABIIY':>9}{'OLINADI':>10}{'FARQ':>8}")
for lang in LANGS:
    print(f"   {lang:<13}{100*raw[lang]:>8.1f}%{100*weights[lang]:>9.1f}%"
          f"{weights[lang]/raw[lang]:>7.1f}x")

# ---------------------------------------------------- 3. dev birlashtirish
# Bitta dev fayl: bir nechta validation dataloader boʻlsa NeMo val_wer_0,
# val_wer_1 deb yozadi va monitor="val_wer" hech narsani koʻrmay qoladi.
dev_path = exp / "dev_combined.jsonl"
n_dev = 0
dev_sec = 0.0
per_lang = {}
with open(dev_path, "w", encoding="utf-8") as out:
    for lang in LANGS:
        rows = read_rows(root / lang / "dev.jsonl")
        if DEV_MAX_PER_LANG and len(rows) > DEV_MAX_PER_LANG:
            # Har N-chi qatorni olamiz: tasodifiy emas, shuning uchun
            # har ishga tushirishda AYNAN bir xil dev — checkpointlarni
            # solishtirganda bu shart.
            step = len(rows) / DEV_MAX_PER_LANG
            rows = [rows[int(i * step)] for i in range(DEV_MAX_PER_LANG)]
        per_lang[lang] = len(rows)
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_dev += 1
            dev_sec += r.get("duration", 0) or 0
print(f"\n3. Dev: {n_dev:,} qator, {dev_sec/3600:.1f} soat audio")
print("   " + "  ".join(f"{k}:{v}" for k, v in per_lang.items()))
if DEV_MAX_PER_LANG:
    print(f"   (har tildan {DEV_MAX_PER_LANG} tagacha — validation tez "
          f"boʻlishi uchun)")

auto_manifests = {}

# ---------------------------------------------------- 4. model
print(f"\n4. Baza model yuklanmoqda: {MODEL_NAME}")
model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)

# ---------------------------------------------------- 5. tokenizer nazorati
# Eng qimmat xato shu yerda ushlanadi: agar tokenizer sizning harflaringizni
# bilmasa, model <unk> yozishni oʻrganadi va butun oʻqitish behuda ketadi.
if CHECK_UNK:
    print("\n5. Tokenizer tekshirilmoqda...")
    tok = model.tokenizer
    unk = getattr(tok, "unk_id", None)
    prompts = None
    try:
        prompts = dict(model.cfg.train_ds.prompt_dictionary)
    except Exception:
        pass

    def is_ok(t):
        if unk is None or not t:
            return True
        try:
            return unk not in tok.text_to_ids(t)
        except Exception:
            return False

    problem = False
    warn_rows = []
    fixed_any = False

    for lang in LANGS:
        loc = locales[lang]
        bad, n = unk_rate(lang, tok, unk)
        rate = bad / max(n, 1)

        # --- avtomatik tuzatish --------------------------------------
        if bad and rate > UNK_MAX_RATE and AUTO_FIX:
            print(f"   {lang:<13}{loc:<9}<unk> {bad}/{n} — tuzatilmoqda...")
            res = autofix(lang, is_ok)
            if res:
                cmap, unfixable, changed = res
                fixed_any = True
                def _show(b):
                    if b.strip():
                        return repr(b)[1:-1]
                    return "(probel)" if b else "(oʻchirildi)"
                shown = "  ".join(f"{repr(a)[1:-1]}->{_show(b)}"
                                  for a, b in list(cmap.items())[:6])
                print(f"   {'':<13}{'':<9}  {len(cmap)} qoida: {shown}")
                print(f"   {'':<13}{'':<9}  {changed:,} qator oʻzgartirildi")
                if unfixable:
                    print(f"   {'':<13}{'':<9}  almashtirib boʻlmadi: " +
                          " ".join("U+%04X" % ord(c) for c in unfixable))
                bad, n = unk_rate(lang, tok, unk)   # qayta tekshirish
                rate = bad / max(n, 1)
            else:
                print(f"   {'':<13}{'':<9}  almashtiruv topilmadi")

        # --- yakuniy holat -------------------------------------------
        in_pd = (prompts is None) or (loc in prompts) or (
            EXTRA_PROMPTS == "auto") or (loc in (EXTRA_PROMPTS or {}))
        if not bad:
            mark = "ok"
        elif rate <= UNK_MAX_RATE:
            mark = f"<unk> {bad}/{n} — kam, zarar qilmaydi"
            warn_rows.append(lang)
        else:
            mark = f"<unk> {bad}/{n} = {100*rate:.0f}% !!!"
            problem = True
        if not in_pd:
            mark += "  |  prompt lugʻatida YOʻQ !!!"
            problem = True
        print(f"   {lang:<13}{loc:<9}{mark}")

    if fixed_any:
        print("\n   Asl manifestlar <split>.jsonl.bak sifatida saqlandi.")
        print("   Qaytarish:  python3 apply_charmap.py <til> --revert")

    if warn_rows and not problem:
        print(f"\n   Ogohlantirish: {', '.join(warn_rows)} da bir nechta")
        print("   qator <unk> beradi — bu ulush zarar qilmaydi.")

    if problem:
        print("\n   OʻQITISH BOSHLANMADI.")
        if not AUTO_FIX:
            print("   AUTO_FIX = True qilsangiz avtomatik tuzatishga urinadi.")
        print("   Batafsil koʻrish uchun:  python3 tokprobe.py <til>")
        sys.exit(1)

# ------------------------------------------- 5b. avtomatik til manbai
# Audio nusxalanmaydi — faqat manifest qatorlari, `locale` maydoni
# `auto` ga oʻzgartirilib. Bir xil audio ikki xil prompt bilan
# koʻrsatiladi, model ikkalasini ham oʻrganadi.
auto_manifests = {}
if AUTO_LANG_RATIO > 0:
    auto_dir = exp / "auto_manifests"
    auto_dir.mkdir(exist_ok=True)
    for lang in LANGS:
        src = root / lang / "train.jsonl"
        dst = auto_dir / f"{lang}.jsonl"
        n = 0
        with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as o:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                r[LANG_FIELD] = AUTO_LOCALE
                o.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        auto_manifests[lang] = str(dst)
    print(f"\n5b. Avtomatik til aniqlash: har bir tilning "
          f"{100*AUTO_LANG_RATIO:.0f}% i `{AUTO_LOCALE}` sifatida")
    print(f"    manbalar: {auto_dir}")

# ---------------------------------------------------- 6. sozlash
print("\n6. Konfiguratsiya...")
# --- manbalar roʻyxati: aniq til + avtomatik til ------------------
# Tilning umumiy ulushi oʻzgarmaydi: w -> w*(1-ratio) aniq til uchun,
# w*ratio `auto` uchun. Shunday qilib aralashtirish nisbati buzilmaydi.
input_cfg = []
for _l in LANGS:
    _w = weights[_l]
    _explicit = _w * (1.0 - AUTO_LANG_RATIO)
    if _explicit > 0:
        input_cfg.append({"type": "nemo",
                          "manifest_filepath": manifests[_l],
                          "weight": round(_explicit, 6)})
    if AUTO_LANG_RATIO > 0 and _l in auto_manifests:
        input_cfg.append({"type": "nemo",
                          "manifest_filepath": auto_manifests[_l],
                          "weight": round(_w * AUTO_LANG_RATIO, 6)})

with open_dict(model.cfg):
    tds = model.cfg.train_ds
    tds.use_lhotse = True              # busiz quyidagilar E'TIBORGA OLINMAYDI
    tds.input_cfg = input_cfg          # og'irlikli aralashtirish
    tds.manifest_filepath = None
    tds.is_tarred = False
    tds.tarred_audio_filepaths = None
    tds.shard_manifests = False
    tds.lang_field = LANG_FIELD        # model default: target_lang
    tds.prompt_field = LANG_FIELD      # promptni tanlaydigan maydon
    tds.shuffle = True
    tds.num_workers = NUM_WORKERS
    tds.pin_memory = True
    tds.batch_size = None              # davomiylik bilan boshqariladi
    tds.batch_duration = BATCH_DURATION
    tds.quadratic_duration = QUADRATIC_DURATION
    tds.use_bucketing = True
    tds.num_buckets = NUM_BUCKETS
    tds.max_duration = MAX_DURATION
    tds.min_duration = MIN_DURATION

    vds = model.cfg.validation_ds
    vds.use_lhotse = True
    vds.input_cfg = None
    vds.manifest_filepath = str(dev_path)
    vds.is_tarred = False
    vds.lang_field = LANG_FIELD
    vds.prompt_field = LANG_FIELD
    vds.shuffle = False
    vds.num_workers = max(4, NUM_WORKERS // 4)
    vds.pin_memory = True
    vds.batch_size = None
    vds.batch_duration = min(BATCH_DURATION, 200)
    vds.use_bucketing = False

    # Yangi tillarni prompt lugʻatiga qoʻshish. Embedding jadvali 128 ta
    # qatorli, shuning uchun boʻsh id allaqachon mavjud — u faqat
    # oʻqitilmagan. Bitta vektorni oʻrgatish kichik masala.
    # Prompt lugʻatida boʻlmagan tillarga id biriktirish. Embedding
    # jadvali 128 qatorli, shuning uchun boʻsh id allaqachon mavjud — u
    # faqat oʻqitilmagan. Bitta vektorni oʻrgatish kichik masala.
    _extra = dict(EXTRA_PROMPTS) if isinstance(EXTRA_PROMPTS, dict) else {}
    _base = dict(model.cfg.train_ds.prompt_dictionary)

    # `auto` modelning oʻz lugʻatida boʻlishi shart - uni biz yarata
    # olmaymiz, chunki u allaqachon oʻqitilgan prompt.
    if AUTO_LANG_RATIO > 0 and AUTO_LOCALE not in _base:
        sys.exit(f"   XATO: `{AUTO_LOCALE}` prompt lugʻatida yoʻq. "
                 f"AUTO_LANG_RATIO = 0.0 qiling.")
    if AUTO_LANG_RATIO > 0:
        print(f"   `{AUTO_LOCALE}` prompt id: {_base[AUTO_LOCALE]}")
    _used = set(_base.values())
    _n = model.cfg.train_ds.get("num_prompts", 128)

    if EXTRA_PROMPTS == "auto":
        # Eng kichik boʻsh id'dan boshlab avtomatik tanlash.
        for _lang in LANGS:
            _loc = locales[_lang]
            if _loc in _base:
                continue
            _free = next((i for i in range(_n)
                          if i not in _used and i not in _extra.values()), None)
            if _free is None:
                sys.exit(f"   XATO: {_n} ta prompt id ning hammasi band.")
            _extra[_loc] = _free
            print(f"   {_loc} uchun boʻsh prompt id tanlandi: {_free}")

    for _loc, _id in _extra.items():
        if _loc in _base:
            continue
        if _id in _used:
            sys.exit(f"   XATO: prompt id {_id} band ({_loc} uchun). "
                     f"Boshqasini tanlang.")
        if _id >= _n:
            sys.exit(f"   XATO: prompt id {_id} >= num_prompts {_n}")
        _used.add(_id)

    if _extra:
        for _name in ("train_ds", "validation_ds", "test_ds"):
            if _name not in model.cfg:
                continue
            _ds = model.cfg[_name]
            if "prompt_dictionary" not in _ds:
                continue
            _pd = dict(_ds.prompt_dictionary)
            _pd.update({k: v for k, v in _extra.items() if k not in _pd})
            _ds.prompt_dictionary = _pd
        print(f"   prompt lugʻatiga qoʻshildi: {_extra}")

    model.cfg.optim.name = "adamw"
    model.cfg.optim.lr = LR
    model.cfg.optim.weight_decay = 1e-3
    model.cfg.optim.sched = {
        "name": "CosineAnnealing",
        "warmup_steps": WARMUP,
        "min_lr": LR / 100.0,
        "max_steps": MAX_STEPS,
    }

    if "joint" in model.cfg:
        model.cfg.joint.fused_batch_size = FUSED_BATCH_SIZE
        model.cfg.joint.fuse_loss_wer = True

# NeMo har bir dev namunasi uchun reference/predicted yozadi — buni
# oʻchiramiz, aks holda log oʻqib boʻlmaydigan boʻlади.
if QUIET_LOGS:
    with open_dict(model.cfg):
        if "decoding" in model.cfg:
            model.cfg.decoding.log_prediction = False
    for _attr in ("wer", "_wer"):
        _w = getattr(model, _attr, None)
        if _w is not None and hasattr(_w, "log_prediction"):
            _w.log_prediction = False
    logging.getLogger("nemo_logger").setLevel(logging.WARNING)

model.setup_training_data(model.cfg.train_ds)
model.setup_validation_data(model.cfg.validation_ds)

if FREEZE_ENCODER:
    model.encoder.freeze()
    print("   Encoder muzlatildi: faqat decoder + joint oʻqitiladi")

def _hms(sec):
    sec = int(max(sec, 0))
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


class Progress(pl.Callback):
    """Qisqa va foydali log: qadam, loss, lr, oʻtgan va qolgan vaqt.

    NeMo ning oʻz progress bari va har namunali WER chiqishi oʻrniga.
    """

    def __init__(self, total_steps, every=100):
        self.total = total_steps
        self.every = every
        self.t0 = None
        self.best = None

    def on_train_start(self, trainer, pl_module):
        self.t0 = time.time()
        print(f"\n  oʻqitish boshlandi — jami {self.total:,} qadam\n", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.every:
            return
        el = time.time() - self.t0
        # Qolgan vaqt: shu paytgacha boʻlgan oʻrtacha tezlik boʻyicha.
        eta = el / step * (self.total - step) if step else 0
        m = trainer.callback_metrics
        loss = m.get("train_loss", m.get("loss"))
        loss = float(loss) if loss is not None else float("nan")
        try:
            lr = trainer.optimizers[0].param_groups[0]["lr"]
        except Exception:
            lr = float("nan")
        print(f"  {step:>7,}/{self.total:,}  {100*step/self.total:>5.1f}%  "
              f"loss {loss:7.4f}  lr {lr:.2e}  "
              f"oʻtdi {_hms(el)}  qoldi ~{_hms(eta)}", flush=True)

    def on_validation_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        wer = m.get("val_wer")
        if wer is None:
            return
        wer = float(wer)
        flag = ""
        if self.best is None or wer < self.best:
            self.best = wer
            flag = "  <-- eng yaxshi"
        el = time.time() - self.t0 if self.t0 else 0
        print(f"\n  === qadam {trainer.global_step:,}   "
              f"val_wer {wer:.4f}   (eng yaxshi {self.best:.4f}){flag}"
              f"   [{_hms(el)}]\n", flush=True)


# ---------------------------------------------------- 7. trainer
# Checkpoint OʻCHIRILMAYDI. 30000 qadamdan keyin faqat oxirgi modelni
# saqlash — eng yomon (overfit boʻlgan) modelni saqlash demak.
ckpt = ModelCheckpoint(
    dirpath=str(exp / "checkpoints"),
    filename="{step}-{val_wer:.4f}",
    monitor="val_wer", mode="min", save_top_k=3,
    save_last=True, every_n_train_steps=VAL_EVERY,
)

trainer = pl.Trainer(
    devices=GPUS,
    accelerator="gpu",
    strategy="ddp" if GPUS > 1 else "auto",
    precision="bf16-mixed",
    max_steps=MAX_STEPS,          # epoch emas: dinamik batch'da epoch
    max_epochs=-1,                # uzunligi oldindan ma'lum emas
    val_check_interval=VAL_EVERY,
    check_val_every_n_epoch=None,
    limit_train_batches=1.0,      # NeMo buni jimgina kesib qoʻyadi
    limit_val_batches=1.0,
    log_every_n_steps=LOG_EVERY_NEMO,
    gradient_clip_val=1.0,
    enable_checkpointing=True,
    enable_progress_bar=not QUIET_LOGS,
    # LearningRateMonitor "step" rejimida har qadam yozadi; Progress
    # baribir lr ni koʻrsatadi, shuning uchun kerak emas.
    callbacks=[ckpt] + ([Progress(MAX_STEPS, LOG_EVERY)] if QUIET_LOGS
                        else [LearningRateMonitor(logging_interval="step")]),
    logger=TensorBoardLogger(save_dir=str(exp), name="tb"),
    default_root_dir=str(exp),
)
model.set_trainer(trainer)
model.setup_optimization(model.cfg.optim)

print(f"""
   jami            {total_h:,.1f} soat, {len(LANGS)} til
   batch_duration  {BATCH_DURATION}s   workers {NUM_WORKERS}
   lr              {LR}  (warmup {WARMUP})
   max_steps       {MAX_STEPS:,}
   natija          {exp}
   tensorboard     tensorboard --logdir {exp}/tb
""")

# ---------------------------------------------------- 8. oʻqitish
print("Fine-tuning boshlandi...\n")
trainer.fit(model)

out = exp / f"{RUN_NAME}-last.nemo"
model.save_to(str(out))

print(f"""
========================================================
  Oxirgi model : {out}
  ENG YAXSHI   : {ckpt.best_model_path}
  best val_wer : {ckpt.best_model_score}

  Ishlatish uchun ENG YAXSHISINI oling. Oxirgi qadam
  deyarli hech qachon eng yaxshi qadam emas.
========================================================
""")