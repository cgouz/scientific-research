"""
Fine-tune nvidia/nemotron-3.5-asr-streaming-0.6b on several languages at once
(English, Uzbek, Russian, Karakalpak, Korean by default). All settings live in config.yaml.

Data: <data.root>/<language>/{train,dev}.jsonl, NeMo manifests with rows
    {"audio_filepath": .., "duration": .., "text": .., "locale": "uz-UZ"}
The language field (data.lang_field) picks the model's language prompt.

Steps:
  1. tokenizer check: rows that tokenize to <unk> stop training, or are fixed with a character map
     (tokenizer.auto_fix; originals kept as <split>.jsonl.bak)
  2. language mix by temperature: p ~ (hours / total) ** (1/T), so small languages are seen more often
  3. language detection: data.auto_lang_ratio (default 70%) of every language is trained with the `auto` prompt,
     the rest with its own locale (so a language can still be forced)
  4. one dev file with every N-th row of each language (the same rows on every run), scored with the `auto` prompt
     by default (data.dev_prompt), so the best checkpoint is the best language detector
  5. languages missing from the prompt dictionary get a free prompt id
  6. Lhotse batches by audio duration, bf16, cosine LR with warmup; checkpoints picked by dev WER

Usage:
  python train.py                                                            # config.yaml, all GPUs
  python train.py --set data.langs=[uzbek] train.max_steps=500 train.val_every=100   # quick test
  python train.py --resume                                                   # continue from checkpoints/last.ckpt
  bash run.sh                                                                # in the background with nohup
Use python, not torchrun: Lightning starts the other GPU processes itself, after the manifests are ready.

Output (output.dir, default /data/experiments/nemotron):
  final.nemo         best model by dev WER  ->  python ../nemotron_test/nemotron_test.py --model <dir>/final.nemo --bench
  last.nemo          model at the last step
  checkpoints/       top-k by val_wer + last.ckpt
  tb/                TensorBoard logs (tensorboard --logdir <dir>/tb)
  logs/              nohup logs from run.sh
  dev_combined.jsonl, auto_manifests/, config_used.yaml

Requires: nemo_toolkit[asr] lightning omegaconf tensorboard pyyaml
"""

import argparse
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

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, for general/
from general.config import load_config
from general.jsonl import read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (HERE / p).resolve()


# ---------------------------------------------------------------------------
# Character map: what to replace characters the tokenizer doesn't know with
# ---------------------------------------------------------------------------
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
    "ǵ": ["ğ", "ǵ", "g"],      # ǵ -> ğ (Turkish g-breve)
    "Ǵ": ["Ğ", "Ǵ", "G"],      # Ǵ -> Ğ
    "á": ["á", "a"], "ó": ["ó", "o"],
    "ú": ["ú", "u"], "ı": ["i"],
    "ń": ["ń", "n"], "ś": ["ś", "s"],
    "ų": ["u"],
    # Typography that carries no meaning.
    "–": ["-"], "—": ["-"], "−": ["-"],
    " ": [" "], "​": [""], "‌": [""], "‍": [""],
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

WS_RE = re.compile(r"\s+")


def auto_candidates(c):
    """Decomposition and case folding, tried before anything lossy."""
    out = []
    for form in ("NFD", "NFKD", "NFKC"):
        n = unicodedata.normalize(form, c)
        if n != c and n not in out:
            out.append(n)
    # the base letter with its marks stripped
    base = "".join(ch for ch in unicodedata.normalize("NFD", c) if not unicodedata.combining(ch))
    if base and base != c and base not in out:
        out.append(base)
    return out


def apply_map(text, cmap):
    """Apply a character map and tidy the whitespace it leaves behind."""
    for a, b in cmap.items():
        if a in text:
            text = text.replace(a, b)
    return WS_RE.sub(" ", text).strip()


def build_charmap(is_ok, chars):
    """{bad char: replacement} for every char that tokenizes to <unk>, plus the ones with no fix.
    `is_ok(s)` is True when s tokenizes without <unk>."""
    bad = [c for c in chars
           if not c.isspace() and not (is_ok(c) and is_ok(f"aa{c}aa") and is_ok(f"{c}a"))]
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


def autofix(lang_dir, is_ok, sample=5000):
    """Find and apply the character map for one language, in place.

    Returns (mapping, unfixable, changed_rows) or None when nothing needs fixing.
    Originals are kept as <split>.jsonl.bak and every rewrite reads FROM the backup,
    so running this twice cannot apply the map twice.
    """
    chars = Counter()
    for r in read_jsonl(lang_dir / "train.jsonl", sample):
        chars.update(str(r.get("text", "") or ""))
    cmap, unfixable = build_charmap(is_ok, chars)
    if not cmap:
        return None
    (lang_dir / "charmap.json").write_text(json.dumps(cmap, ensure_ascii=False, indent=2), encoding="utf-8")

    changed = 0
    for split in ("train", "dev", "test"):
        f = lang_dir / f"{split}.jsonl"
        if not f.exists():
            continue
        bak = lang_dir / f"{split}.jsonl.bak"
        if not bak.exists():
            shutil.copy2(f, bak)
        out = []
        for r in read_jsonl(bak):
            old = str(r.get("text", "") or "")
            new = apply_map(old, cmap)
            if not new:
                continue
            changed += new != old
            r["text"] = new
            out.append(r)
        write_jsonl(f, out)
    return cmap, unfixable, changed


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def scan_manifests(root, langs, lang_field, log):
    """{lang: hours}, {lang: locale}; stops if a manifest or the language field is missing."""
    hours, locales = {}, {}
    for lang in langs:
        tr, dv = root / lang / "train.jsonl", root / lang / "dev.jsonl"
        if not tr.exists():
            sys.exit(f"Missing {tr}")
        if not dv.exists():
            sys.exit(f"Missing {dv} (needed to pick the best checkpoint)")
        rows = read_jsonl(tr)
        hours[lang] = sum(r.get("duration", 0) or 0 for r in rows) / 3600
        locales[lang] = rows[0].get(lang_field) if rows else None
        if not locales[lang]:
            sys.exit(f"{tr}: no `{lang_field}` field. Nemotron reads the language prompt from it.")
        log(f"  {lang:<13}{locales[lang]:<9}{len(rows):>10,} rows {hours[lang]:>9.1f} h")
    return hours, locales


def mix_weights(hours, temperature):
    total = sum(hours.values())
    natural = {k: v / total for k, v in hours.items()}
    w = {k: p ** (1.0 / temperature) for k, p in natural.items()}
    s = sum(w.values())
    return natural, {k: v / s for k, v in w.items()}


def build_dev(root, langs, max_per_lang, prompt, lang_field, auto_locale, path, write):
    """One dev manifest for all languages. With several validation dataloaders NeMo logs
    val_wer_0, val_wer_1 ... and the checkpoint monitor on `val_wer` sees nothing.
    prompt: auto = every row with the auto prompt, locale = its own, both = each row twice."""
    if prompt not in ("auto", "locale", "both"):
        sys.exit(f"data.dev_prompt must be auto, locale or both, not {prompt!r}")
    rows, per_lang = [], {}
    for lang in langs:
        lr = read_jsonl(root / lang / "dev.jsonl")
        if max_per_lang and len(lr) > max_per_lang:
            step = len(lr) / max_per_lang  # every N-th row, not random: the same dev on every run
            lr = [lr[int(i * step)] for i in range(max_per_lang)]
        per_lang[lang] = len(lr)
        auto = [{**r, lang_field: auto_locale} for r in lr]
        rows += {"auto": auto, "locale": lr, "both": lr + auto}[prompt]
    if write:
        write_jsonl(path, rows)
    return rows, per_lang


def build_auto_manifests(root, langs, lang_field, auto_locale, out_dir, write):
    """Copy of each train manifest with the language set to `auto`. Audio is not copied: the
    same audio is shown with both prompts, so the model learns both."""
    out_dir.mkdir(exist_ok=True)
    paths = {}
    for lang in langs:
        dst = out_dir / f"{lang}.jsonl"
        if write:
            rows = read_jsonl(root / lang / "train.jsonl")
            for r in rows:
                r[lang_field] = auto_locale
            write_jsonl(dst, rows)
        paths[lang] = str(dst)
    return paths


def check_tokenizer(model, tcfg, dcfg, root, locales, log):
    """Count rows that give <unk>; fix them if allowed. Returns True if any manifest was rewritten."""
    tok = model.tokenizer
    unk = getattr(tok, "unk_id", None)
    try:
        prompts = dict(model.cfg.train_ds.prompt_dictionary)
    except Exception:
        prompts = None
    extra = dcfg["extra_prompts"]

    def is_ok(t):
        if unk is None or not t:
            return True
        try:
            return unk not in tok.text_to_ids(t)
        except Exception:
            return False

    def unk_rows(lang):
        rows = read_jsonl(root / lang / "train.jsonl", int(tcfg["sample"]))
        return sum(not is_ok(str(r.get("text", "") or "")) for r in rows), len(rows)

    def show(b):
        return repr(b)[1:-1] if b.strip() else ("(space)" if b else "(deleted)")

    max_rate = float(tcfg["max_unk_rate"])
    problem, fixed, warn = False, False, []
    for lang, loc in locales.items():
        bad, n = unk_rows(lang)
        rate = bad / max(n, 1)
        if bad and rate > max_rate and tcfg["auto_fix"]:
            log(f"  {lang:<13}{loc:<9}<unk> in {bad}/{n} rows, fixing ...")
            res = autofix(root / lang, is_ok)
            if res:
                cmap, unfixable, changed = res
                fixed = True
                log(f"  {'':<22}{len(cmap)} rules: " +
                    "  ".join(f"{repr(a)[1:-1]}->{show(b)}" for a, b in list(cmap.items())[:6]))
                log(f"  {'':<22}{changed:,} rows changed")
                if unfixable:
                    log(f"  {'':<22}no replacement for: " + " ".join("U+%04X" % ord(c) for c in unfixable))
                bad, n = unk_rows(lang)
                rate = bad / max(n, 1)
            else:
                log(f"  {'':<22}no replacement found")

        in_dict = prompts is None or loc in prompts or extra == "auto" or loc in (extra or {})
        if not bad:
            mark = "ok"
        elif rate <= max_rate:
            mark = f"<unk> in {bad}/{n} rows (few, harmless)"
            warn.append(lang)
        else:
            mark = f"<unk> in {bad}/{n} rows = {100 * rate:.0f}% !!!"
            problem = True
        if not in_dict:
            mark += "  |  not in the prompt dictionary !!!"
            problem = True
        log(f"  {lang:<13}{loc:<9}{mark}")

    if fixed:
        log(f"\n  Originals kept as <split>.jsonl.bak. To undo, in {root}/<language>/:")
        log("    for f in *.jsonl.bak; do mv \"$f\" \"${f%.bak}\"; done")
    if warn and not problem:
        log(f"\n  Warning: a few rows in {', '.join(warn)} give <unk>; that share is harmless.")
    if problem:
        log("\n  TRAINING NOT STARTED.")
        if not tcfg["auto_fix"]:
            log("  Set tokenizer.auto_fix=true to try fixing it automatically.")
        sys.exit(1)
    return fixed


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def add_prompts(model, dcfg, langs, locales, log):
    """Give languages missing from the prompt dictionary a free id. The prompt embedding table has
    num_prompts (128) rows and not all are used, so a free row already exists: it just has to be trained."""
    base = dict(model.cfg.train_ds.prompt_dictionary)
    auto_locale = dcfg["auto_locale"]
    if float(dcfg["auto_lang_ratio"]) > 0 or dcfg["dev_prompt"] != "locale":
        # `auto` must already be in the model: it is a trained prompt, we cannot create it
        if auto_locale not in base:
            sys.exit(f"`{auto_locale}` is not in the prompt dictionary. "
                     "Set data.auto_lang_ratio=0 and data.dev_prompt=locale.")
        log(f"  `{auto_locale}` prompt id: {base[auto_locale]}")

    extra = dict(dcfg["extra_prompts"]) if isinstance(dcfg["extra_prompts"], dict) else {}
    used = set(base.values())
    n = model.cfg.train_ds.get("num_prompts", 128)
    if dcfg["extra_prompts"] == "auto":
        for lang in langs:
            loc = locales[lang]
            if loc in base:
                continue
            free = next((i for i in range(n) if i not in used and i not in extra.values()), None)
            if free is None:
                sys.exit(f"All {n} prompt ids are taken.")
            extra[loc] = free
            log(f"  free prompt id for {loc}: {free}")
    for loc, i in extra.items():
        if loc in base:
            continue
        if i in used:
            sys.exit(f"Prompt id {i} for {loc} is already taken. Pick another.")
        if i >= n:
            sys.exit(f"Prompt id {i} >= num_prompts {n}")
        used.add(i)
    if not extra:
        return
    for name in ("train_ds", "validation_ds", "test_ds"):
        if name in model.cfg and "prompt_dictionary" in model.cfg[name]:
            pd = dict(model.cfg[name].prompt_dictionary)
            pd.update({k: v for k, v in extra.items() if k not in pd})
            model.cfg[name].prompt_dictionary = pd
    log(f"  added to the prompt dictionary: {extra}")


def configure(model, cfg, locales, input_cfg, dev_path, max_steps, log):
    from omegaconf import open_dict

    dcfg, tcfg = cfg["data"], cfg["train"]
    lang_field = dcfg["lang_field"]
    with open_dict(model.cfg):
        tds = model.cfg.train_ds
        tds.use_lhotse = True              # without it the settings below are ignored
        tds.input_cfg = input_cfg          # weighted language mix
        tds.manifest_filepath = None
        tds.is_tarred = False
        tds.tarred_audio_filepaths = None
        tds.shard_manifests = False
        tds.lang_field = lang_field
        tds.prompt_field = lang_field      # field that picks the prompt
        tds.shuffle = True
        tds.num_workers = int(tcfg["num_workers"])
        tds.pin_memory = True
        tds.batch_size = None              # batches are sized by duration
        tds.batch_duration = tcfg["batch_duration"]
        tds.quadratic_duration = tcfg["quadratic_duration"]
        tds.use_bucketing = True
        tds.num_buckets = int(tcfg["num_buckets"])
        tds.max_duration = float(dcfg["max_duration"])
        tds.min_duration = float(dcfg["min_duration"])

        vds = model.cfg.validation_ds
        vds.use_lhotse = True
        vds.input_cfg = None
        vds.manifest_filepath = str(dev_path)
        vds.is_tarred = False
        vds.lang_field = lang_field
        vds.prompt_field = lang_field
        vds.shuffle = False
        vds.num_workers = max(4, int(tcfg["num_workers"]) // 4)
        vds.pin_memory = True
        vds.batch_size = None
        vds.batch_duration = min(tcfg["batch_duration"], 200)
        vds.use_bucketing = False

        add_prompts(model, dcfg, dcfg["langs"], locales, log)

        lr = float(tcfg["lr"])
        model.cfg.optim.name = "adamw"
        model.cfg.optim.lr = lr
        model.cfg.optim.weight_decay = float(tcfg["weight_decay"])
        model.cfg.optim.sched = {"name": "CosineAnnealing", "warmup_steps": int(tcfg["warmup_steps"]),
                                 "min_lr": lr / 100.0, "max_steps": max_steps}
        if "joint" in model.cfg:
            model.cfg.joint.fused_batch_size = int(tcfg["fused_batch_size"])
            model.cfg.joint.fuse_loss_wer = True

    if tcfg["quiet_logs"]:  # NeMo otherwise prints reference/prediction for every dev sample
        with open_dict(model.cfg):
            if "decoding" in model.cfg:
                model.cfg.decoding.log_prediction = False
        for attr in ("wer", "_wer"):
            w = getattr(model, attr, None)
            if w is not None and hasattr(w, "log_prediction"):
                w.log_prediction = False
        logging.getLogger("nemo_logger").setLevel(logging.WARNING)

    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    if tcfg["freeze_encoder"]:
        model.encoder.freeze()
        log("  encoder frozen: only decoder + joint are trained")


def hms(sec):
    sec = int(max(sec, 0))
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def make_progress(total, every, dev_prompt):
    import lightning.pytorch as pl

    class Progress(pl.Callback):
        """Short log line: step, loss, lr, elapsed and remaining time (instead of NeMo's progress bar)."""

        def __init__(self):
            self.t0, self.best = None, None

        def on_train_start(self, trainer, pl_module):
            self.t0 = time.time()
            if trainer.is_global_zero:
                print(f"\n  training started, {total:,} steps\n", flush=True)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = trainer.global_step
            if not trainer.is_global_zero or step == 0 or step % every:
                return
            el = time.time() - self.t0
            eta = el / step * (total - step)
            m = trainer.callback_metrics
            loss = m.get("train_loss", m.get("loss"))
            loss = float(loss) if loss is not None else float("nan")
            try:
                lr = trainer.optimizers[0].param_groups[0]["lr"]
            except Exception:
                lr = float("nan")
            print(f"  {step:>7,}/{total:,}  {100 * step / total:>5.1f}%  loss {loss:7.4f}  lr {lr:.2e}  "
                  f"elapsed {hms(el)}  left ~{hms(eta)}", flush=True)

        def on_validation_end(self, trainer, pl_module):
            wer = trainer.callback_metrics.get("val_wer")
            if wer is None or not trainer.is_global_zero:
                return
            wer = float(wer)
            flag = ""
            if self.best is None or wer < self.best:
                self.best, flag = wer, "  <-- best"
            el = time.time() - self.t0 if self.t0 else 0
            print(f"\n  === step {trainer.global_step:,}   val_wer ({dev_prompt}) {wer:.4f}   (best {self.best:.4f}){flag}"
                  f"   [{hms(el)}]\n", flush=True)

    return Progress()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune Nemotron ASR on several languages.")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config, e.g. train.lr=5e-5")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/last.ckpt in output.dir")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    dcfg, tcfg = cfg["data"], cfg["train"]
    # Lightning re-runs this script for GPUs 1..N with LOCAL_RANK set, after rank 0 has written the
    # manifests. Only rank 0 writes files and prints.
    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
    log = print if is_main else (lambda *a, **k: None)

    import torch
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found (check nvidia-smi and that torch is a CUDA build).")
    torch.backends.cuda.matmul.allow_tf32 = True
    gpus = torch.cuda.device_count() if str(tcfg["gpus"]).lower() == "all" else int(tcfg["gpus"])
    for i in range(gpus):
        p = torch.cuda.get_device_properties(i)
        log(f"GPU {i}: {p.name}, {p.total_memory / 1024**3:.0f} GB")

    root = resolve(dcfg["root"])
    langs = list(dcfg["langs"])
    out_dir = resolve(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    import lightning.pytorch as pl
    import nemo.collections.asr as nemo_asr
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    pl.seed_everything(int(tcfg["seed"]), workers=True)

    # ---------- manifests ----------
    log(f"\nManifests in {root} ...")
    hours, locales = scan_manifests(root, langs, dcfg["lang_field"], log)

    log(f"\nLoading {cfg['model']} ...")
    model = nemo_asr.models.ASRModel.from_pretrained(cfg["model"])

    # The costliest mistake is caught here: if the tokenizer doesn't know your letters, the model
    # learns to write <unk> and the whole run is wasted.
    if is_main and cfg["tokenizer"]["check_unk"]:
        log("\nChecking the tokenizer ...")
        if check_tokenizer(model, cfg["tokenizer"], dcfg, root, locales, log):
            hours, locales = scan_manifests(root, langs, dcfg["lang_field"], lambda *a, **k: None)

    # ---------- language mix + steps ----------
    total_h = sum(hours.values())
    natural, weights = mix_weights(hours, float(dcfg["temperature"]))
    log(f"\nLanguage mix (temperature {dcfg['temperature']})")
    log(f"  {'LANGUAGE':<13}{'NATURAL':>9}{'TRAINED':>10}{'x':>7}")
    for lang in langs:
        log(f"  {lang:<13}{100 * natural[lang]:>8.1f}%{100 * weights[lang]:>9.1f}%{weights[lang] / natural[lang]:>7.1f}")

    # Dynamic batches have no fixed epoch length, so Lightning gets max_steps, not max_epochs.
    steps_per_pass = max(1, int(total_h * 3600 / (float(tcfg["batch_duration"]) * gpus)))
    max_steps = tcfg["max_steps"]
    if max_steps is None:
        max_steps = int(float(tcfg["epochs"]) * steps_per_pass)
        how = f"{tcfg['epochs']} passes x {steps_per_pass:,} steps"
    else:
        max_steps = int(max_steps)
        how = f"set by hand ({max_steps / steps_per_pass:.1f} passes)"
    log(f"\n  {total_h:,.1f} h -> {steps_per_pass:,} steps per pass on {gpus} GPU(s)")
    log(f"  max_steps = {max_steps:,}   ({how})")
    if int(tcfg["warmup_steps"]) > max_steps * 0.1:
        log(f"  WARNING: warmup {tcfg['warmup_steps']:,} is {100 * int(tcfg['warmup_steps']) / max_steps:.0f}% "
            "of max_steps. Too much, lower it.")

    # ---------- dev + auto manifests ----------
    dev_path = out_dir / "dev_combined.jsonl"
    dev_rows, per_lang = build_dev(root, langs, int(dcfg["dev_max_per_lang"]), dcfg["dev_prompt"],
                                   dcfg["lang_field"], dcfg["auto_locale"], dev_path, is_main)
    dev_h = sum(r.get("duration", 0) or 0 for r in dev_rows) / 3600
    log(f"\nDev: {len(dev_rows):,} rows, {dev_h:.1f} h, prompt: {dcfg['dev_prompt']}  ("
        + "  ".join(f"{k}: {v}" for k, v in per_lang.items()) + ")")

    ratio = float(dcfg["auto_lang_ratio"])
    auto_paths = {}
    if ratio > 0:
        auto_paths = build_auto_manifests(root, langs, dcfg["lang_field"], dcfg["auto_locale"],
                                          out_dir / "auto_manifests", is_main)
        log(f"Language detection: {100 * ratio:.0f}% of each language trained as `{dcfg['auto_locale']}`, "
            f"{100 * (1 - ratio):.0f}% with its own locale")

    # a language keeps its total share: w * (1 - ratio) with its own prompt, w * ratio with `auto`
    input_cfg = []
    for lang in langs:
        w = weights[lang]
        if w * (1 - ratio) > 0:
            input_cfg.append({"type": "nemo", "manifest_filepath": str(root / lang / "train.jsonl"),
                              "weight": round(w * (1 - ratio), 6)})
        if lang in auto_paths:
            input_cfg.append({"type": "nemo", "manifest_filepath": auto_paths[lang], "weight": round(w * ratio, 6)})

    # ---------- model config ----------
    log("\nConfiguring the model ...")
    configure(model, cfg, locales, input_cfg, dev_path, max_steps, log)

    # ---------- trainer ----------
    # Keep the best checkpoints, not only the last: after many steps the last one is usually overfit.
    ckpt = ModelCheckpoint(dirpath=str(out_dir / "checkpoints"), filename="{step}-{val_wer:.4f}",
                           monitor="val_wer", mode="min", save_top_k=int(tcfg["save_top_k"]),
                           save_last=True, every_n_train_steps=int(tcfg["val_every"]))
    quiet = bool(tcfg["quiet_logs"])
    trainer = pl.Trainer(
        devices=gpus, accelerator="gpu", strategy="ddp" if gpus > 1 else "auto",
        precision="bf16-mixed", max_steps=max_steps, max_epochs=-1,
        val_check_interval=int(tcfg["val_every"]), check_val_every_n_epoch=None,
        limit_train_batches=1.0, limit_val_batches=1.0,  # NeMo silently lowers these otherwise
        log_every_n_steps=int(tcfg["log_every_nemo"]), gradient_clip_val=1.0,
        enable_checkpointing=True, enable_progress_bar=not quiet,
        callbacks=[ckpt, make_progress(max_steps, int(tcfg["log_every"]), dcfg["dev_prompt"]) if quiet
                   else LearningRateMonitor(logging_interval="step")],
        logger=TensorBoardLogger(save_dir=str(out_dir), name="tb"),
        default_root_dir=str(out_dir),
    )
    model.set_trainer(trainer)
    model.setup_optimization(model.cfg.optim)

    resume = None
    if args.resume:
        resume = out_dir / "checkpoints" / "last.ckpt"
        if not resume.exists():
            sys.exit(f"--resume: {resume} not found")
    log(f"""
  data            {total_h:,.1f} h, {len(langs)} languages
  batch_duration  {tcfg['batch_duration']} s per GPU, {gpus} GPU(s), {tcfg['num_workers']} workers
  lr              {tcfg['lr']}  (warmup {tcfg['warmup_steps']})
  max_steps       {max_steps:,}
  output          {out_dir}
  tensorboard     tensorboard --logdir {out_dir}/tb
""")

    # ---------- train ----------
    log("Training ..." + (f" (resuming from {resume})" if resume else ""))
    t0 = time.time()
    trainer.fit(model, ckpt_path=str(resume) if resume else None)
    log(f"Training time: {(time.time() - t0) / 3600:.2f} h")

    # ---------- save ----------
    if trainer.is_global_zero:
        last = out_dir / "last.nemo"
        model.save_to(str(last))
        log(f"\nLast step:  {last}")
        if ckpt.best_model_path:
            state = torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)["state_dict"]
            model.load_state_dict(state)
            final = out_dir / "final.nemo"
            model.save_to(str(final))
            log(f"Best model (val_wer {float(ckpt.best_model_score):.4f}): {final}")
            log(f"Test it:  cd ../nemotron_test && python nemotron_test.py --model {final} --bench --split test")


if __name__ == "__main__":
    main()
