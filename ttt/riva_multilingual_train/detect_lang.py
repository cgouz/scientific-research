"""
Detect whether a sentence is Uzbek, Russian or English (the three languages of the multilingual model).

No model, no downloads: it uses the alphabet and a few letters and words that only one of the languages has.
  Cyrillic text  -> ru, or uz when it has Uzbek-only Cyrillic letters (ў қ ғ ҳ) or common Uzbek words
  Latin text     -> uz or en, by Uzbek signs (o' g' sh-words, q x, suffixes, common words)
                    vs English signs (w, th, c not in "ch", common words)
Short or mixed text can be ambiguous; the source language can always be given explicitly instead.

Usage:
  python detect_lang.py "Bugun havo juda yaxshi."        # -> uz
  python detect_lang.py -v "Сегодня хорошая погода."     # -> ru, with the scores
"""

import re
import sys

CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")
UZ_CYRILLIC_LETTERS = re.compile(r"[ўқғҳЎҚҒҲ]")
APOSTROPHES = "'‘’ʻʼ`"  # ' ‘ ’ ʻ ʼ `

UZ_CYRILLIC_WORDS = {"ва", "бу", "билан", "учун", "ҳам", "бир", "эди", "эмас", "жуда", "мен", "сен", "биз",
                     "сиз", "улар", "керак", "бўлади", "қилиш", "ҳақида", "лекин", "агар", "ёки", "шу"}
UZ_WORDS = {"va", "bu", "bilan", "uchun", "ham", "bir", "edi", "emas", "juda", "men", "sen", "biz", "siz", "ular",
            "kerak", "bo'ladi", "qilish", "haqida", "lekin", "agar", "yoki", "shu", "bugun", "ertalab", "nima",
            "qanday", "yaxshi", "katta", "kichik", "yil", "kun", "uy", "ish", "har", "hamma", "barcha", "endi",
            "hozir", "kim", "nega", "qayerda", "salom", "rahmat", "ha", "yo'q", "o'z", "bo'lgan", "etilgan", "qilib",
            "deb", "esa", "u", "unga", "uni", "sizga", "menga"}
EN_WORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "and", "or", "of", "to", "in", "on", "at",
            "for", "with", "this", "that", "these", "those", "it", "i", "you", "he", "she", "we", "they", "my",
            "your", "his", "her", "our", "their", "have", "has", "had", "do", "does", "did", "not", "will", "would",
            "can", "could", "should", "what", "who", "how", "why", "where", "when", "from", "by", "as", "but",
            "if", "there", "here", "very", "all", "some", "any", "me", "him", "them", "us", "am", "today"}
UZ_SUFFIXES = ("lar", "larni", "larga", "lari", "ning", "dagi", "dan", "ga", "da", "ni", "imiz", "ingiz", "yapti",
               "yapman", "moqda", "gan", "ish", "lik", "chi")


def words(text):
    t = text.lower()
    for a in APOSTROPHES:
        t = t.replace(a, "'")
    return re.findall(r"[a-zЀ-ӿ']+", t)


def scores(text):
    """{"ru": .., "uz": .., "en": ..}: higher = more likely."""
    c, l = len(CYRILLIC.findall(text)), len(LATIN.findall(text))
    ws = words(text)
    if c > l:  # Cyrillic: Russian, or Uzbek written in Cyrillic
        uz = 3 * len(UZ_CYRILLIC_LETTERS.findall(text)) + 2 * sum(w in UZ_CYRILLIC_WORDS for w in ws)
        return {"ru": 1.0, "uz": float(uz), "en": 0.0}

    low = text.lower()
    for a in APOSTROPHES:
        low = low.replace(a, "'")
    uz = 3 * len(re.findall(r"[og]'", low))                 # o' g' (oʻ gʻ)
    uz += 2 * sum(w in UZ_WORDS for w in ws)
    uz += sum(1 for w in ws if len(w) > 4 and w.endswith(UZ_SUFFIXES))
    uz += 0.5 * (low.count("q") + low.count("x"))
    en = 2 * sum(w in EN_WORDS for w in ws)
    en += 1.5 * low.count("w") + low.count("th")
    en += len(re.findall(r"c(?!h)", low))                    # Uzbek Latin has c only in "ch"
    return {"ru": 0.0, "uz": float(uz), "en": float(en)}


def detect(text):
    """'uz', 'ru' or 'en'. Ties and text without letters fall back to 'uz' for Latin, 'ru' for Cyrillic."""
    s = scores(text)
    if s["ru"]:  # Cyrillic
        return "uz" if s["uz"] >= 3 else "ru"
    return "en" if s["en"] > s["uz"] else "uz"


if __name__ == "__main__":
    args = sys.argv[1:]
    verbose = "-v" in args
    text = " ".join(a for a in args if a != "-v") or sys.stdin.read()
    if not text.strip():
        sys.exit(__doc__)
    print(detect(text) + (f"  {scores(text)}" if verbose else ""))
