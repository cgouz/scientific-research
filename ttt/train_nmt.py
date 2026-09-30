"""
Train an Uzbek -> Russian translation model (text-to-text / seq2seq) on CUDA GPUs.
All settings live in a YAML config: configs/nmt_uz_ru.yaml

Built to use the GPU fully:
  * mixed precision bf16 + TF32, fused AdamW, SDPA / FlashAttention
  * batch_size: auto  -> probes the GPU with the LONGEST sentences and picks the biggest batch that
                         fits (incl. optimizer memory), so no guessing and no OOM mid-run
  * length-grouped batches (little padding), pinned-memory multi-worker data loading
  * multi-GPU: launch with torchrun, everything else is automatic
  * resumable, early stopping on dev chrF, best checkpoint kept

Usage:
  python scripts/train_nmt.py --config configs/nmt_uz_ru.yaml
  python scripts/train_nmt.py --config configs/nmt_uz_ru.yaml --set model.name=nllb-600m train.max_steps=200
  python scripts/train_nmt.py --config configs/nmt_uz_ru.yaml --resume
  torchrun --nproc_per_node=gpu scripts/train_nmt.py --config configs/nmt_uz_ru.yaml     # all GPUs

Output (output.dir, default outputs/nmt-uz-ru/):
  final/                trained model + tokenizer + nmt_config.json  (use with translate_nmt.py)
  checkpoint-*/         best + last checkpoints
  test_results.json     BLEU / chrF: original model vs your model, per data source
  test_predictions.csv  source | reference | original | yours
  config_used.yaml      the exact settings of this run

Requires: pip install -r requirements.txt   (torch with CUDA, transformers, datasets, sentencepiece, sacrebleu, pyyaml)
"""

import argparse
import copy
import csv
import inspect
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent  # everything lives inside riva_train/
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache" / "datasets"))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "nmt_uz_ru.yaml"
MODELS = {
    "nllb-600m": ("facebook/nllb-200-distilled-600M", "nllb"),
    "nllb-1.3b": ("facebook/nllb-200-distilled-1.3B", "nllb"),
    "nllb-3.3b": ("facebook/nllb-200-3.3B", "nllb"),
    "madlad-3b": ("google/madlad400-3b-mt", "madlad"),
}
NLLB_CODES = {"uz": "uzn_Latn", "ru": "rus_Cyrl", "en": "eng_Latn", "kk": "kaz_Cyrl",
              "tr": "tur_Latn", "ky": "kir_Cyrl", "tg": "tgk_Cyrl", "kaa": "kaa_Latn"}

DEFAULTS = {
    "model": {"name": "nllb-600m", "type": None, "attn": "sdpa"},
    "data": {"pair": "uz-ru", "dirs": ["data/external/til_uz-ru", "data/external/uzlpc"],
             "max_per_source": None, "max_len": 256, "eval_samples": 500, "test_samples": 2000},
    "train": {"epochs": 1.0, "max_steps": -1, "lr": 5e-5, "warmup": 0.02, "scheduler": "inverse_sqrt",
              "weight_decay": 0.01, "label_smoothing": 0.1, "batch_size": "auto", "target_batch": 256,
              "grad_accum": "auto", "memory_fraction": 0.85, "bf16": True, "tf32": True,
              "optim": "adamw_torch_fused", "grad_checkpointing": False, "torch_compile": False,
              "num_workers": "auto", "tokenize_procs": "auto", "eval_steps": 2000, "save_steps": 2000,
              "save_total_limit": 2, "early_stopping_patience": 5, "logging_steps": 50, "beams": 4, "seed": 42},
    "output": {"dir": "outputs/nmt-uz-ru", "baseline": True, "require_gpu": True},
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def deep_merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path, overrides):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        p = Path(path)
        if not p.exists():
            sys.exit(f"Config not found: {p}")
        cfg = deep_merge(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    for item in overrides or []:  # --set train.lr=3e-5 model.name=nllb-600m
        if "=" not in item:
            sys.exit(f"--set expects key=value, got '{item}'")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def resolve(p):
    p = Path(p)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p  # relative paths always mean riva_train/<path>


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def read_manifest(path, pair, limit, rng):
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("pair") == pair and r.get("source") and r.get("target"):
                rows.append({"source": r["source"], "target": r["target"],
                             "origin": str(r.get("origin", path.parent.name))})
    if limit and len(rows) > limit:
        rng.shuffle(rows)
        rows = rows[:limit]
    return rows


def load_data(dcfg, seed, log):
    rng = random.Random(seed)
    splits = {"train": [], "dev": [], "test": []}
    for d in dcfg["dirs"]:
        d = resolve(d)
        if not d.exists():
            log(f"  (skip) {d} not found")
            continue
        for split in splits:
            rows = read_manifest(d / f"{split}.jsonl", dcfg["pair"],
                                 dcfg["max_per_source"] if split == "train" else None, rng)
            splits[split] += rows
            if rows:
                log(f"  {d.name:<20} {split:<5} {len(rows):>10,} {dcfg['pair']} pairs")
    held = {r["source"].lower() for s in ("dev", "test") for r in splits[s]}
    before = len(splits["train"])
    splits["train"] = [r for r in splits["train"] if r["source"].lower() not in held]
    if before != len(splits["train"]):
        log(f"  removed {before - len(splits['train']):,} train rows that also appear in dev/test")
    keep_eval = max(dcfg["eval_samples"], dcfg["test_samples"])
    for s in ("dev", "test"):
        rng.shuffle(splits[s])
        splits[s] = splits[s][:keep_eval]
    rng.shuffle(splits["train"])
    if not splits["dev"]:
        n = min(2000, len(splits["train"]) // 20)
        splits["dev"], splits["train"] = splits["train"][:n], splits["train"][n:]
    if not splits["test"]:
        splits["test"] = splits["dev"]
    return splits


# ---------------------------------------------------------------------------
# Model-specific encoding (NLLB language tokens vs MADLAD <2xx> prefix)
# ---------------------------------------------------------------------------
class Codec:
    def __init__(self, tokenizer, model_type, src, tgt):
        self.tok, self.type, self.src, self.tgt = tokenizer, model_type, src, tgt
        if model_type == "nllb":
            if src not in NLLB_CODES or tgt not in NLLB_CODES:
                sys.exit(f"Add NLLB codes for {src}/{tgt} to NLLB_CODES")
            tokenizer.src_lang, tokenizer.tgt_lang = NLLB_CODES[src], NLLB_CODES[tgt]

    def prefix(self, texts):
        return [f"<2{self.tgt}> {t}" for t in texts] if self.type == "madlad" else list(texts)

    def encode(self, batch, max_len):
        enc = self.tok(self.prefix(batch["source"]), text_target=batch["target"], max_length=max_len, truncation=True)
        enc["length"] = [len(x) for x in enc["input_ids"]]
        return enc

    def gen_kwargs(self):
        if self.type == "nllb":
            return {"forced_bos_token_id": self.tok.convert_tokens_to_ids(NLLB_CODES[self.tgt])}
        return {}

    @torch.inference_mode()
    def translate(self, model, texts, batch_size=32, beams=4, max_new_tokens=256):
        model.eval()
        dev = next(model.parameters()).device
        amp = (torch.autocast("cuda", dtype=torch.bfloat16)
               if dev.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext())
        res = [None] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))  # similar lengths -> less padding
        with amp:
            for b in range(0, len(order), batch_size):
                idx = order[b:b + batch_size]
                enc = self.tok(self.prefix([texts[i] for i in idx]), return_tensors="pt", padding=True,
                               truncation=True, max_length=max_new_tokens).to(dev)
                gen = model.generate(**enc, num_beams=beams, max_length=max_new_tokens, max_new_tokens=None,
                                     **self.gen_kwargs())
                for i, t in zip(idx, self.tok.batch_decode(gen, skip_special_tokens=True)):
                    res[i] = t.strip()
        return res


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def gpu_report():
    n = torch.cuda.device_count()
    lines = [f"  GPU {i}: {torch.cuda.get_device_name(i)}  "
             f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.0f} GB" for i in range(n)]
    return n, lines


def probe_batch_size(model, collator, examples, tcfg, log):
    """Find the largest per-GPU batch that fits, using the LONGEST training examples.
    Measures real forward+backward memory and adds the optimizer-state memory that
    the probe itself does not allocate."""
    dev = next(model.parameters()).device
    total = torch.cuda.get_device_properties(dev).total_memory
    budget = total * float(tcfg["memory_fraction"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optim = str(tcfg["optim"])
    opt_bytes = n_params * (8 if "adamw" in optim and "8bit" not in optim else 2 if "8bit" in optim else 1)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if tcfg["bf16"] else nullcontext()
    model.train()
    best, b = 0, 8
    while b <= 2048:
        batch_ex = (examples * math.ceil(b / len(examples)))[:b]
        try:
            batch = {k: v.to(dev) for k, v in collator(batch_ex).items()}
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            with amp:
                loss = model(**batch).loss
            loss.backward()
            peak = torch.cuda.max_memory_allocated(dev)
            need = peak + opt_bytes
            model.zero_grad(set_to_none=True)
            del batch, loss
            log(f"    batch {b:>5}: {need / 1024**3:6.1f} GB of {budget / 1024**3:.1f} GB budget")
            if need > budget:
                break
            best = b
            b *= 2
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            log(f"    batch {b:>5}: out of memory")
            break
    torch.cuda.empty_cache()
    if best == 0:
        sys.exit("Even batch 8 does not fit. Set train.grad_checkpointing: true, train.optim: adafactor, "
                 "or a smaller model / data.max_len.")
    # between best and 2*best there may be room: try 1.5x once
    mid = int(best * 1.5) // 8 * 8
    if mid > best:
        try:
            batch = {k: v.to(dev) for k, v in collator((examples * math.ceil(mid / len(examples)))[:mid]).items()}
            torch.cuda.reset_peak_memory_stats(dev)
            with amp:
                loss = model(**batch).loss
            loss.backward()
            if torch.cuda.max_memory_allocated(dev) + opt_bytes <= budget:
                best = mid
            model.zero_grad(set_to_none=True)
            del batch, loss
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    return best


def training_args(**kw):
    """Seq2SeqTrainingArguments across transformers 4.x / 5.x (renamed options)."""
    from transformers import Seq2SeqTrainingArguments

    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "warmup_ratio" in params:
        kw["warmup_ratio"] = kw.pop("warmup_steps")
    if "train_sampling_strategy" not in params and kw.pop("train_sampling_strategy", None):
        kw["group_by_length"] = True
    if "eval_strategy" not in params:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    dropped = [k for k in kw if k not in params and k != "evaluation_strategy"]
    return Seq2SeqTrainingArguments(**{k: v for k, v in kw.items() if k not in dropped}), dropped


def score(hyps, refs):
    import sacrebleu

    return {"bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2)}


def auto_int(v, default):
    return default if v in (None, "auto") else int(v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train a seq2seq translation model (config-driven, CUDA).")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config, e.g. train.lr=3e-5")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    mcfg, dcfg, tcfg, ocfg = cfg["model"], cfg["data"], cfg["train"], cfg["output"]

    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = int(os.environ.get("RANK", 0)) == 0
    log = print if is_main else (lambda *a, **k: None)

    # ---------- GPU ----------
    cuda = torch.cuda.is_available()
    if not cuda and ocfg["require_gpu"]:
        sys.exit("No CUDA GPU found. Check `nvidia-smi` and torch.cuda.is_available(). "
                 "(For a CPU test only: --set output.require_gpu=false)")
    if cuda:
        torch.cuda.set_device(local_rank)
        if tcfg["tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        n_gpu, lines = gpu_report()
        log(f"CUDA GPUs: {n_gpu}   processes: {world}")
        for line in lines:
            log(line)
        if n_gpu > 1 and world == 1:
            log(f"  NOTE: {n_gpu} GPUs found but only 1 is used. For all of them run:\n"
                f"        torchrun --nproc_per_node=gpu scripts/train_nmt.py --config {args.config}")
    bf16 = bool(tcfg["bf16"]) and cuda and torch.cuda.is_bf16_supported()
    if tcfg["bf16"] and cuda and not bf16:
        log("  this GPU has no bf16 -> using fp16")

    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              EarlyStoppingCallback, Seq2SeqTrainer, set_seed)

    set_seed(int(tcfg["seed"]))
    src, tgt = dcfg["pair"].split("-", 1)
    model_id, mtype = MODELS.get(mcfg["name"], (mcfg["name"], mcfg["type"]))
    if mtype is None:
        sys.exit("Custom model: set model.type to nllb or madlad in the config")
    out_dir = resolve(ocfg["dir"]) if Path(ocfg["dir"]).is_absolute() else PROJECT_ROOT / ocfg["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    log(f"Model: {model_id} [{mtype}]   direction: {src} -> {tgt}   output: {out_dir}")

    # ---------- data ----------
    log("\nLoading manifests ...")
    data = load_data(dcfg, int(tcfg["seed"]), log)
    if not data["train"]:
        sys.exit(f"No '{dcfg['pair']}' rows in {dcfg['dirs']}. Check data.dirs / data.pair.")
    log(f"  train {len(data['train']):,} | dev {len(data['dev']):,} | test {len(data['test']):,}")
    log("  train sources: " + ", ".join(f"{k} {v:,}" for k, v in Counter(r["origin"] for r in data["train"]).most_common()))

    # ---------- model (fp32 master weights + bf16 autocast = stable mixed precision) ----------
    tok = AutoTokenizer.from_pretrained(model_id)
    codec = Codec(tok, mtype, src, tgt)
    load_kw = {"dtype": torch.float32}
    if cuda and mcfg.get("attn"):
        load_kw["attn_implementation"] = mcfg["attn"]
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **load_kw)
    except (ValueError, ImportError) as e:
        log(f"  attention '{mcfg.get('attn')}' not available ({e}); using default")
        load_kw.pop("attn_implementation", None)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **load_kw)
    gk = codec.gen_kwargs()
    if gk.get("forced_bos_token_id") is not None:
        model.generation_config.forced_bos_token_id = gk["forced_bos_token_id"]
    model.generation_config.max_length = int(dcfg["max_len"])
    model.generation_config.max_new_tokens = None
    model.generation_config.num_beams = int(tcfg["beams"])
    if tcfg["grad_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    model.to(device)
    log(f"  parameters: {sum(p.numel() for p in model.parameters()) / 1e6:,.0f}M   "
        f"attention: {getattr(model.config, '_attn_implementation', 'default')}")

    # ---------- baseline (main process only) ----------
    n_test = min(int(dcfg["test_samples"]), len(data["test"]))
    test_rows = data["test"][:n_test]
    test_src, test_ref = [r["source"] for r in test_rows], [r["target"] for r in test_rows]
    results = {"pair": dcfg["pair"], "model": model_id, "test_sentences": n_test}
    base_hyp = None
    if is_main and ocfg["baseline"] and not args.resume:
        log(f"\nScoring the ORIGINAL model on {n_test:,} test sentences ...")
        t0 = time.time()
        base_hyp = codec.translate(model, test_src, batch_size=64 if cuda else 8,
                                   beams=int(tcfg["beams"]), max_new_tokens=int(dcfg["max_len"]))
        results["baseline"] = score(base_hyp, test_ref)
        log(f"  baseline: {results['baseline']}  ({time.time() - t0:.0f}s)")

    # ---------- tokenize (main process first; others reuse the cache) ----------
    from accelerate import PartialState
    from datasets import Dataset

    procs = auto_int(tcfg["tokenize_procs"], min(32, os.cpu_count() or 1))
    enc = lambda b: codec.encode(b, int(dcfg["max_len"]))  # noqa: E731
    cols = ["source", "target", "origin"]
    log("\nTokenizing ...")
    with PartialState().main_process_first():
        train_ds = Dataset.from_list(data["train"]).map(
            enc, batched=True, remove_columns=cols, num_proc=procs if len(data["train"]) > 50_000 else None,
            desc="tokenize train")
    dev_rows = data["dev"][:int(dcfg["eval_samples"])]
    dev_ds = Dataset.from_list(dev_rows).map(enc, batched=True, remove_columns=cols)
    collator = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

    # ---------- batch size ----------
    if tcfg["batch_size"] in (None, "auto"):
        if cuda:
            log("\nFinding the largest batch that fits (longest sentences) ...")
            longest = train_ds.sort("length", reverse=True).select(range(min(256, len(train_ds))))
            examples = [{k: longest[i][k] for k in ("input_ids", "attention_mask", "labels")}
                        for i in range(len(longest))]
            bs = probe_batch_size(model, collator, examples, tcfg, log)
            log(f"  -> batch_size {bs} per GPU")
        else:
            bs = 8
    else:
        bs = int(tcfg["batch_size"])
    accum = auto_int(tcfg["grad_accum"], max(1, round(int(tcfg["target_batch"]) / (bs * world))))
    eff = bs * accum * world
    steps_per_epoch = math.ceil(len(train_ds) / eff)
    total = int(tcfg["max_steps"]) if int(tcfg["max_steps"]) > 0 else math.ceil(steps_per_epoch * float(tcfg["epochs"]))
    eval_steps = max(1, min(int(tcfg["eval_steps"]), total))
    save_steps = eval_steps  # save at every evaluation so the BEST checkpoint always exists on disk
    log(f"  batch {bs} x grad_accum {accum} x {world} GPU = {eff} sentences/step   "
        f"{total:,} steps   eval every {eval_steps}")

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = [[t for t in seq if t >= 0] for seq in preds.tolist()]
        hyps = [h.strip() for h in tok.batch_decode(preds, skip_special_tokens=True)]
        return score(hyps, [r["target"] for r in dev_rows])

    workers = auto_int(tcfg["num_workers"], min(8, os.cpu_count() or 1)) if cuda else 0
    targs, dropped = training_args(
        output_dir=str(out_dir), per_device_train_batch_size=bs, per_device_eval_batch_size=max(8, bs // 2),
        gradient_accumulation_steps=accum, learning_rate=float(tcfg["lr"]), num_train_epochs=float(tcfg["epochs"]),
        max_steps=int(tcfg["max_steps"]), warmup_steps=float(tcfg["warmup"]), lr_scheduler_type=tcfg["scheduler"],
        weight_decay=float(tcfg["weight_decay"]), max_grad_norm=1.0,
        label_smoothing_factor=float(tcfg["label_smoothing"]),
        optim=tcfg["optim"] if cuda else "adamw_torch", bf16=bf16, fp16=cuda and not bf16 and bool(tcfg["bf16"]),
        tf32=bool(tcfg["tf32"]) if cuda else None, gradient_checkpointing=bool(tcfg["grad_checkpointing"]),
        torch_compile=bool(tcfg["torch_compile"]) and cuda,
        eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=save_steps,
        save_total_limit=int(tcfg["save_total_limit"]), load_best_model_at_end=True,
        metric_for_best_model="chrf", greater_is_better=True,
        predict_with_generate=True, generation_num_beams=int(tcfg["beams"]),
        logging_steps=max(1, min(int(tcfg["logging_steps"]), eval_steps)), logging_first_step=True,
        report_to="none", seed=int(tcfg["seed"]), train_sampling_strategy="group_by_length",
        dataloader_num_workers=workers, dataloader_pin_memory=cuda,
        dataloader_persistent_workers=workers > 0, dataloader_prefetch_factor=4 if workers > 0 else None,
        ddp_find_unused_parameters=False if world > 1 else None, include_num_input_tokens_seen=True,
    )
    if dropped:
        log(f"  (options not supported by this transformers version, ignored: {', '.join(dropped)})")
    callbacks = []
    if int(tcfg["early_stopping_patience"]) > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(tcfg["early_stopping_patience"])))
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                             data_collator=collator, processing_class=tok, compute_metrics=compute_metrics,
                             callbacks=callbacks)

    # ---------- train ----------
    log("\nTraining ...  (progress bar shows steps; dev chrF is printed at every evaluation)")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    log(f"Training time: {(time.time() - t0) / 3600:.2f} h")

    final = out_dir / "final"
    trainer.save_model(str(final))
    if not trainer.is_world_process_zero():
        return
    tok.save_pretrained(str(final))
    (final / "nmt_config.json").write_text(json.dumps(
        {"model_type": mtype, "src": src, "tgt": tgt, "base_model": model_id,
         "forced_bos_token_id": gk.get("forced_bos_token_id")}, indent=2), encoding="utf-8")
    log(f"\nSaved best model to {final}")

    # ---------- final test ----------
    log(f"\nScoring YOUR model on {n_test:,} test sentences ...")
    best_model = trainer.accelerator.unwrap_model(trainer.model)
    hyp = codec.translate(best_model, test_src, batch_size=64 if cuda else 8,
                          beams=int(tcfg["beams"]), max_new_tokens=int(dcfg["max_len"]))
    results["finetuned"] = score(hyp, test_ref)
    by_origin = defaultdict(lambda: ([], []))
    for r, h in zip(test_rows, hyp):
        by_origin[r["origin"]][0].append(h)
        by_origin[r["origin"]][1].append(r["target"])
    results["finetuned_by_source"] = {k: score(h, rf) for k, (h, rf) in by_origin.items()}
    results["batch"] = {"per_gpu": bs, "grad_accum": accum, "gpus": world, "effective": eff}
    (out_dir / "test_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out_dir / "test_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source", "reference", "original_model", "your_model"])
        for i, (s, r, h) in enumerate(zip(test_src, test_ref, hyp)):
            w.writerow([s, r, base_hyp[i] if base_hyp else "", h])

    log("\n=== Test results (higher is better) ===")
    if "baseline" in results:
        log(f"  original model : BLEU {results['baseline']['bleu']:>6}  chrF {results['baseline']['chrf']:>6}")
    log(f"  your model     : BLEU {results['finetuned']['bleu']:>6}  chrF {results['finetuned']['chrf']:>6}")
    for k, v in results["finetuned_by_source"].items():
        log(f"     {k:<22} BLEU {v['bleu']:>6}  chrF {v['chrf']:>6}")
    log(f"\nDetails: {out_dir / 'test_results.json'}  |  {out_dir / 'test_predictions.csv'}")
    log(f"Try it:  python scripts/translate_nmt.py --model {final} --text \"Salom, qalaysiz?\"")


if __name__ == "__main__":
    main()
