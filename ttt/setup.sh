#!/usr/bin/env bash
# =====================================================================
#  setup.sh - creates /workspace/riva_train/ (files + folders only)
#  Plain text: every file is written below with a heredoc.
#  No checksum, no base64, no tar, no python, no downloads, no tokenization.
#  Run:  bash setup.sh          (other place:  RIVA_DIR=/path bash setup.sh)
# =====================================================================
set -e
RIVA_DIR="."

mkdir -p "$RIVA_DIR"/data/external/til_uz-ru \
         "$RIVA_DIR"/data/external/uzlpc \
         "$RIVA_DIR"/test_data \
         "$RIVA_DIR"/outputs/riva-uz-ru \
         "$RIVA_DIR"/results \
         "$RIVA_DIR"/.cache/datasets
cd "$RIVA_DIR"
echo "Folders created in $RIVA_DIR"

# ---------------------------------------------------------------------
cat > train_riva_uzru.py << 'RIVA_FILE_EOF'
"""
Fine-tune nvidia/Riva-Translate-4B-Instruct-v2 on Uzbek -> Russian.
ONE self-contained file: every setting is in CONFIG below. No config file, no other scripts needed.

Prompt (identical to riva_test.py, so before/after scores are comparable):
    <s>System\nYou are an expert at translating text from Uzbek to Russian.</s>\n
    <s>User\nWhat is the Russian translation of the sentence: {uzbek}</s>\n
    <s>Assistant\n{russian}</s>
The loss is computed ONLY on the Russian translation.

GPU features: bf16 + TF32, fused AdamW, SDPA, auto batch size (probes the GPU with the longest
examples), length-grouped batches, multi-GPU via torchrun, gradient checkpointing.

Usage (paths in CONFIG are relative to THIS file's folder):
  python train_riva_uzru.py                                          # full training
  python train_riva_uzru.py --set train.max_steps=200 train.eval_steps=100 data.test_samples=200   # quick test
  python train_riva_uzru.py --set method=lora train.lr=1e-4          # change any setting for one run
  python train_riva_uzru.py --resume                                 # continue after a crash
  torchrun --nproc_per_node=gpu train_riva_uzru.py                   # several GPUs

Output (CONFIG["output"]["dir"], default outputs/riva-uz-ru/):
  final/     full model (method full) or merged model (method lora)  -> riva_test.py --model
  adapter/   the LoRA adapter (method lora)
  test_results.json, test_predictions.csv, config_used.json

Requires: torch transformers datasets accelerate sacrebleu  (+ peft only for method=lora)
"""

import argparse
import ast
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

PROJECT_ROOT = Path(__file__).resolve().parent  # everything lives next to this file (riva_train/)
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache" / "datasets"))

# =====================================================================
#  ALL SETTINGS  (edit here, or override for one run:  --set train.lr=5e-6)
# =====================================================================
CONFIG = {
    "model": "nvidia/Riva-Translate-4B-Instruct-v2",   # HF id or local folder
    "attn": "sdpa",                 # sdpa | flash_attention_2 (if installed) | eager
    # full = train every weight (best for a NEW language, needs an 80 GB+ GPU)
    # lora = train a large adapter (less memory, weaker for a new language)
    # auto = full if the GPU has >= 80 GB, else lora
    "method": "auto",
    "lora": {
        "r": 128,                   # large rank: a new language needs capacity
        "alpha": 256,
        "dropout": 0.05,
        "target_modules": "all-linear",
        "train_embeddings": False,  # True = also train embed_tokens + lm_head
    },
    "data": {
        "pair": "uz-ru",            # only rows with "pair": "uz-ru" are used
        "dirs": [                   # folders with train.jsonl (+ optional dev.jsonl / test.jsonl)
            "data/external/til_uz-ru",
            # add more folders here, e.g. "data/external/my_data",
        ],
        "max_per_source": None,     # cap train rows per folder, e.g. 500000 (None = all)
        "max_len": 384,             # tokens per example (prompt + translation)
        "eval_samples": 300,        # dev sentences translated at every evaluation
        "test_samples": 1000,       # test sentences for the final score
    },
    "train": {
        "epochs": 1.0,
        "max_steps": -1,            # > 0 overrides epochs (quick tests)
        "lr": "auto",               # auto = 1e-5 (full) or 2e-4 (lora)
        "warmup": 0.03,             # fraction of steps
        "scheduler": "cosine",
        "weight_decay": 0.0,
        "batch_size": "auto",       # per GPU; auto = probe the GPU with the longest examples
        "target_batch": 128,        # sentences per optimizer step (all GPUs)
        "grad_accum": "auto",       # auto = target_batch / (batch_size x GPUs)
        "memory_fraction": 0.85,    # auto batch uses at most this share of GPU memory
        "bf16": True,
        "tf32": True,
        "optim": "adamw_torch_fused",   # adamw_torch_fused | adafactor | adamw_8bit (bitsandbytes)
        "grad_checkpointing": True, # much less memory, ~30% slower
        "num_workers": "auto",      # dataloader workers (auto = min(8, CPU cores))
        "tokenize_procs": "auto",   # tokenization processes (auto = CPU cores, max 32)
        "eval_steps": 1000,         # evaluate AND save a checkpoint every N steps
        "save_total_limit": 2,      # keeps best + latest checkpoint
        "early_stopping_patience": 5,   # stop if dev chrF doesn't improve N evals in a row (0 = off)
        "logging_steps": 25,
        "gen_max_new_tokens": 256,
        "gen_batch_size": 32,
        "seed": 42,
    },
    "output": {
        "dir": "outputs/riva-uz-ru",
        "baseline": True,           # score the untrained Riva on the test set first
        "merge": True,              # lora: merge into a standalone model in final/
        "require_gpu": True,
    },
}

LANG_NAMES = {"uz": "Uzbek", "ru": "Russian", "en": "English", "kk": "Kazakh", "tr": "Turkish",
              "ky": "Kyrgyz", "tg": "Tajik", "kaa": "Karakalpak"}


# ---------------------------------------------------------------------------
# One-run overrides:  --set train.lr=5e-6 method=lora 'data.dirs=["data/external/my_data"]'
# ---------------------------------------------------------------------------
def parse_value(text):
    low = text.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null", "~"):
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load_config(overrides):
    cfg = copy.deepcopy(CONFIG)
    for item in overrides or []:
        if "=" not in item:
            sys.exit(f"--set expects key=value, got '{item}'")
        key, val = item.split("=", 1)
        node, parts = cfg, key.split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                sys.exit(f"--set: unknown setting '{key}'")
            node = node[part]
        if parts[-1] not in node:
            sys.exit(f"--set: unknown setting '{key}' (check the spelling against CONFIG)")
        node[parts[-1]] = parse_value(val)
    return cfg


# ---------------------------------------------------------------------------
# Data + GPU helpers
# ---------------------------------------------------------------------------
def resolve(p):
    p = Path(p)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p  # relative paths always mean riva_train/<path>


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
        sys.exit("Even batch 8 does not fit. Try --set method=lora, train.optim=adafactor or data.max_len=256.")
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


def score(hyps, refs):
    import sacrebleu

    return {"bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2)}


def auto_int(v, default):
    return default if v in (None, "auto") else int(v)


def training_args(**kw):
    """TrainingArguments across transformers 4.x / 5.x (renamed options)."""
    from transformers import TrainingArguments

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in params:
        kw["warmup_ratio"] = kw.pop("warmup_steps")
    if "train_sampling_strategy" not in params and kw.pop("train_sampling_strategy", None):
        kw["group_by_length"] = True
    if "eval_strategy" not in params:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    dropped = [k for k in kw if k not in params and k != "evaluation_strategy"]
    return TrainingArguments(**{k: v for k, v in kw.items() if k not in dropped}), dropped


# ---------------------------------------------------------------------------
# Prompt, tokenization, generation, trainer
# ---------------------------------------------------------------------------
def build_prompt(text, src, tgt):
    s, t = LANG_NAMES.get(src, src), LANG_NAMES.get(tgt, tgt)
    return (f"<s>System\nYou are an expert at translating text from {s} to {t}.</s>\n"
            f"<s>User\nWhat is the {t} translation of the sentence: {text.strip()}</s>\n"
            f"<s>Assistant\n")


def clean_output(o):
    return o.split("</s>")[0].split("<s>")[0].strip()


# ---------------------------------------------------------------------------
# Tokenization + collator (loss only on the translation)
# ---------------------------------------------------------------------------
class Encoder:
    def __init__(self, tok, src, tgt, max_len):
        self.tok, self.src, self.tgt, self.max_len = tok, src, tgt, max_len
        self.eos = tok.eos_token_id

    def __call__(self, batch):
        out = {"input_ids": [], "attention_mask": [], "labels": [], "length": []}
        prompts = [build_prompt(s, self.src, self.tgt) for s in batch["source"]]
        p_ids = self.tok(prompts, add_special_tokens=False)["input_ids"]
        t_ids = self.tok([t.strip() for t in batch["target"]], add_special_tokens=False)["input_ids"]
        for p, t in zip(p_ids, t_ids):
            t = t[: max(1, self.max_len - len(p) - 1)] + [self.eos]
            ids = (p + t)[: self.max_len]
            labels = ([-100] * len(p) + t)[: self.max_len]
            out["input_ids"].append(ids)
            out["attention_mask"].append([1] * len(ids))
            out["labels"].append(labels)
            out["length"].append(len(ids))
        return out


class Collator:
    """Right-pads input_ids / attention_mask / labels; ignores any other keys."""

    def __init__(self, pad_id, multiple=8):
        self.pad_id, self.multiple = pad_id, multiple

    def __call__(self, features):
        n = max(len(f["input_ids"]) for f in features)
        n = math.ceil(n / self.multiple) * self.multiple
        ids = torch.full((len(features), n), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(features), n), dtype=torch.long)
        labels = torch.full((len(features), n), -100, dtype=torch.long)
        for i, f in enumerate(features):
            k = len(f["input_ids"])
            ids[i, :k] = torch.tensor(f["input_ids"])
            mask[i, :k] = 1
            labels[i, :k] = torch.tensor(f["labels"])
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# ---------------------------------------------------------------------------
# Generation (for dev chrF during training and the final test)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def translate(model, tok, texts, src, tgt, batch_size, max_new_tokens):
    was_training = model.training
    model.eval()
    use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    dev = next(model.parameters()).device
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if dev.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext())
    side = tok.padding_side
    tok.padding_side = "left"
    res = [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    try:
        with amp:
            for b in range(0, len(order), batch_size):
                idx = order[b:b + batch_size]
                enc = tok([build_prompt(texts[i], src, tgt) for i in idx], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(dev)
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                for i, t in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                    res[i] = clean_output(t)
    finally:
        tok.padding_side = side
        model.config.use_cache = use_cache
        if was_training:
            model.train()
    return res


def make_trainer_class():
    from transformers import Trainer

    class RivaTrainer(Trainer):
        """Adds dev chrF (by real generation) to every evaluation; used to pick the best checkpoint."""

        def _get_train_sampler(self, *args, **kwargs):
            # transformers' own length-grouped sampler reads the "length" column one row at a time
            # (lazy datasets Column): ~40 min of 100% CPU before step 1 on 1.2M rows.
            # Same sampler, but with the lengths loaded as a plain list first (~1 s).
            ds = args[0] if args and args[0] is not None else self.train_dataset
            strategy = getattr(self.args, "train_sampling_strategy", None)
            grouped = strategy == "group_by_length" or getattr(self.args, "group_by_length", False) is True
            if grouped and ds is not None and "length" in getattr(ds, "column_names", []):
                from transformers.trainer_pt_utils import LengthGroupedSampler

                lengths = ds.select_columns(["length"]).to_pandas()["length"].tolist()
                return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                                            lengths=lengths)
            return super()._get_train_sampler(*args, **kwargs)

        def setup_eval(self, tok, dev_rows, src, tgt, gen_bs, max_new):
            self._eval = (tok, dev_rows, src, tgt, gen_bs, max_new)

        def evaluate(self, *args, **kwargs):
            metrics = super().evaluate(*args, **kwargs)
            tok, dev_rows, src, tgt, gen_bs, max_new = self._eval
            chrf = torch.zeros(2, device=self.args.device)
            if self.is_world_process_zero():
                model = self.accelerator.unwrap_model(self.model)
                hyps = translate(model, tok, [r["source"] for r in dev_rows], src, tgt, gen_bs, max_new)
                s = score(hyps, [r["target"] for r in dev_rows])
                chrf[0], chrf[1] = s["chrf"], s["bleu"]
                print(f"\n  [eval] step {self.state.global_step}: dev chrF {s['chrf']}  BLEU {s['bleu']}  "
                      f"| e.g. {dev_rows[0]['source'][:50]!r} -> {hyps[0][:60]!r}")
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast(chrf, src=0)
            metrics["eval_chrf"], metrics["eval_bleu"] = round(float(chrf[0]), 2), round(float(chrf[1]), 2)
            self.log({"eval_chrf": metrics["eval_chrf"], "eval_bleu": metrics["eval_bleu"]})
            return metrics

    return RivaTrainer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune Riva-Translate on uz->ru.")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.set)
    dcfg, tcfg, ocfg, lcfg = cfg["data"], cfg["train"], cfg["output"], cfg["lora"]
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = int(os.environ.get("RANK", 0)) == 0
    log = print if is_main else (lambda *a, **k: None)

    # ---------- GPU ----------
    cuda = torch.cuda.is_available()
    if not cuda and ocfg["require_gpu"]:
        sys.exit("No CUDA GPU found (check nvidia-smi). CPU test only: --set output.require_gpu=false")
    vram = 0
    if cuda:
        torch.cuda.set_device(local_rank)
        if tcfg["tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        n_gpu, lines = gpu_report()
        vram = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
        log(f"CUDA GPUs: {n_gpu}   processes: {world}")
        for line in lines:
            log(line)
        if n_gpu > 1 and world == 1:
            log(f"  NOTE: {n_gpu} GPUs but 1 used. All GPUs: torchrun --nproc_per_node=gpu "
                f"{Path(__file__).name}")
    bf16 = bool(tcfg["bf16"]) and cuda and torch.cuda.is_bf16_supported()

    method = cfg["method"]
    if method == "auto":
        method = "full" if vram >= 80 else "lora"
    lr = float(tcfg["lr"]) if tcfg["lr"] not in (None, "auto") else (1e-5 if method == "full" else 2e-4)
    src, tgt = dcfg["pair"].split("-", 1)
    out_dir = resolve(ocfg["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        (out_dir / "config_used.json").write_text(json.dumps({**cfg, "method_resolved": method, "lr_resolved": lr},
                                                             indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Model: {cfg['model']}   method: {method}   lr: {lr}   direction: {src} -> {tgt}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, set_seed

    set_seed(int(tcfg["seed"]))

    # ---------- data ----------
    log("\nLoading manifests ...")
    data = load_data(dcfg, int(tcfg["seed"]), log)
    if not data["train"]:
        sys.exit(f"No '{dcfg['pair']}' rows found in {dcfg['dirs']}")
    log(f"  train {len(data['train']):,} | dev {len(data['dev']):,} | test {len(data['test']):,}")
    log("  sources: " + ", ".join(f"{k} {v:,}" for k, v in Counter(r['origin'] for r in data['train']).most_common()))

    # ---------- model ----------
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kw = {"dtype": torch.float32 if method == "full" else (torch.bfloat16 if bf16 else torch.float32)}
    if cuda and cfg.get("attn"):
        load_kw["attn_implementation"] = cfg["attn"]
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], **load_kw)
    except (ValueError, ImportError) as e:
        log(f"  attention '{cfg.get('attn')}' unavailable ({e}); using default")
        load_kw.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], **load_kw)
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    model.to(device)

    # ---------- baseline ----------
    n_test = min(int(dcfg["test_samples"]), len(data["test"]))
    test_rows = data["test"][:n_test]
    test_src, test_ref = [r["source"] for r in test_rows], [r["target"] for r in test_rows]
    gen_bs, max_new = int(tcfg["gen_batch_size"]), int(tcfg["gen_max_new_tokens"])
    results, base_hyp = {"pair": dcfg["pair"], "model": cfg["model"], "method": method, "test_sentences": n_test}, None
    if is_main and ocfg["baseline"] and not args.resume:
        log(f"\nScoring the ORIGINAL Riva on {n_test:,} test sentences ...")
        t0 = time.time()
        base_hyp = translate(model, tok, test_src, src, tgt, gen_bs, max_new)
        results["baseline"] = score(base_hyp, test_ref)
        log(f"  baseline: {results['baseline']}  ({time.time() - t0:.0f}s)")

    if tcfg["grad_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    if method == "lora":
        from peft import LoraConfig, get_peft_model

        if tcfg["grad_checkpointing"]:
            model.enable_input_require_grads()
        pcfg = LoraConfig(r=int(lcfg["r"]), lora_alpha=int(lcfg["alpha"]), lora_dropout=float(lcfg["dropout"]),
                          target_modules=lcfg["target_modules"], task_type="CAUSAL_LM",
                          modules_to_save=["embed_tokens", "lm_head"] if lcfg["train_embeddings"] else None)
        model = get_peft_model(model, pcfg)
        if is_main:
            model.print_trainable_parameters()
    else:
        log(f"  training all {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    # ---------- tokenize ----------
    from accelerate import PartialState
    from datasets import Dataset

    enc = Encoder(tok, src, tgt, int(dcfg["max_len"]))
    procs = auto_int(tcfg["tokenize_procs"], min(32, os.cpu_count() or 1))
    log("\nTokenizing ...")
    with PartialState().main_process_first():
        train_ds = Dataset.from_list(data["train"]).map(
            enc, batched=True, remove_columns=["source", "target", "origin"],
            num_proc=procs if len(data["train"]) > 50_000 else None, desc="tokenize train")
    dev_rows = data["dev"][:int(dcfg["eval_samples"])]
    dev_ds = Dataset.from_list(dev_rows[:200]).map(enc, batched=True, remove_columns=["source", "target", "origin"])
    collator = Collator(tok.pad_token_id)

    # ---------- batch size ----------
    if tcfg["batch_size"] in (None, "auto"):
        if cuda:
            log("\nFinding the largest batch that fits (longest examples) ...")
            longest = train_ds.sort("length", reverse=True).select(range(min(256, len(train_ds))))
            examples = [{k: longest[i][k] for k in ("input_ids", "labels")} for i in range(len(longest))]
            bs = probe_batch_size(model, collator, examples, {**tcfg, "bf16": bf16}, log)
            log(f"  -> batch_size {bs} per GPU")
        else:
            bs = 4
    else:
        bs = int(tcfg["batch_size"])
    accum = auto_int(tcfg["grad_accum"], max(1, round(int(tcfg["target_batch"]) / (bs * world))))
    eff = bs * accum * world
    total = (int(tcfg["max_steps"]) if int(tcfg["max_steps"]) > 0
             else math.ceil(len(train_ds) / eff * float(tcfg["epochs"])))
    eval_steps = max(1, min(int(tcfg["eval_steps"]), total))
    save_steps = eval_steps  # save at every evaluation so the BEST checkpoint always exists on disk
    log(f"  batch {bs} x accum {accum} x {world} GPU = {eff} sentences/step   {total:,} steps   eval every {eval_steps}")

    workers = auto_int(tcfg["num_workers"], min(8, os.cpu_count() or 1)) if cuda else 0
    targs, dropped = training_args(
        output_dir=str(out_dir), per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
        gradient_accumulation_steps=accum, learning_rate=lr, num_train_epochs=float(tcfg["epochs"]),
        max_steps=int(tcfg["max_steps"]), warmup_steps=float(tcfg["warmup"]), lr_scheduler_type=tcfg["scheduler"],
        weight_decay=float(tcfg["weight_decay"]), max_grad_norm=1.0,
        optim=tcfg["optim"] if cuda else "adamw_torch", bf16=bf16,
        tf32=bool(tcfg["tf32"]) if cuda else None,
        eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=save_steps,
        save_total_limit=int(tcfg["save_total_limit"]), load_best_model_at_end=True,
        metric_for_best_model="chrf", greater_is_better=True,
        logging_steps=max(1, min(int(tcfg["logging_steps"]), eval_steps)), logging_first_step=True,
        report_to="none", seed=int(tcfg["seed"]), train_sampling_strategy="group_by_length",
        remove_unused_columns=False, dataloader_num_workers=workers, dataloader_pin_memory=cuda,
        dataloader_persistent_workers=workers > 0, dataloader_prefetch_factor=4 if workers > 0 else None,
        ddp_find_unused_parameters=False if world > 1 else None, include_num_input_tokens_seen=True,
    )
    if dropped:
        log(f"  (ignored, not in this transformers version: {', '.join(dropped)})")
    RivaTrainer = make_trainer_class()
    callbacks = ([EarlyStoppingCallback(early_stopping_patience=int(tcfg["early_stopping_patience"]))]
                 if int(tcfg["early_stopping_patience"]) > 0 else [])
    trainer = RivaTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                          data_collator=collator, processing_class=tok, callbacks=callbacks)
    trainer.setup_eval(tok, dev_rows, src, tgt, gen_bs, max_new)

    # ---------- train ----------
    log("\nTraining ...")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    log(f"Training time: {(time.time() - t0) / 3600:.2f} h")
    if not trainer.is_world_process_zero():
        return

    # ---------- save ----------
    best = trainer.accelerator.unwrap_model(trainer.model)
    final = out_dir / "final"
    if method == "lora":
        best.save_pretrained(str(out_dir / "adapter"))
        log(f"Saved LoRA adapter to {out_dir / 'adapter'}")
        if ocfg["merge"]:
            best = best.merge_and_unload()
    if method == "full" or ocfg["merge"]:
        best.config.use_cache = True
        to_save = best.to(torch.bfloat16) if bf16 else best
        to_save.save_pretrained(str(final), safe_serialization=True)
        tok.save_pretrained(str(final))
        log(f"Saved standalone model to {final}")

    # ---------- final test ----------
    log(f"\nScoring YOUR Riva on {n_test:,} test sentences ...")
    hyp = translate(best, tok, test_src, src, tgt, gen_bs, max_new)
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
        w.writerow(["source", "reference", "original_riva", "your_riva"])
        for i, (s, r, h) in enumerate(zip(test_src, test_ref, hyp)):
            w.writerow([s, r, base_hyp[i] if base_hyp else "", h])

    log("\n=== Test results (higher is better) ===")
    if "baseline" in results:
        log(f"  original Riva : BLEU {results['baseline']['bleu']:>6}  chrF {results['baseline']['chrf']:>6}")
    log(f"  your Riva     : BLEU {results['finetuned']['bleu']:>6}  chrF {results['finetuned']['chrf']:>6}")
    for k, v in results["finetuned_by_source"].items():
        log(f"     {k:<22} BLEU {v['bleu']:>6}  chrF {v['chrf']:>6}")
    log("\nExamples:")
    for s, r, h in list(zip(test_src, test_ref, hyp))[:3]:
        log(f"  uz  : {s[:100]}\n  ref : {r[:100]}\n  riva: {h[:100]}\n")
    if final.exists():
        log(f"Test again any time:  python riva_test.py --model {final} --test-file test_data/test.jsonl")


if __name__ == "__main__":
    main()
RIVA_FILE_EOF
echo "  wrote train_riva_uzru.py"

# ---------------------------------------------------------------------
cat > train_nmt.py << 'RIVA_FILE_EOF'
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
RIVA_FILE_EOF
echo "  wrote train_nmt.py"

# ---------------------------------------------------------------------
cat > til_to_manifest.py << 'RIVA_FILE_EOF'
"""
Convert a downloaded TIL corpus pair (e.g. data/uz-ru/) into this project's JSONL manifests.

TIL layout (from til-mt/til_corpus/download_data.py):
    data/uz-ru/train/uz-ru/uz-ru.uz   data/uz-ru/train/uz-ru/uz-ru.ru     (line-aligned)
    data/uz-ru/dev/...                data/uz-ru/test/...                 (if they exist)

What it does
  1. finds the aligned file pairs for train / dev / test, checks line counts match
  2. DETOKENIZES the text — TIL is Moses-tokenized:
        "so ‘ ng , oliy o ‘ quv"      -> "so'ng, oliy o'quv"
        "СЭЗ « Сирдарё » :"          -> "СЭЗ «Сирдарё»:"
        "“ Baholash ... ” gi"        -> "“Baholash ...”gi"
  3. cleans: empty, too short/long, untranslated, wrong script, misaligned length ratio,
     duplicates; optional LaBSE meaning check (--labse 0.75, needs a GPU for 1M+ rows)
  4. removes train pairs that also appear in dev/test (no test leakage)
  5. writes {"pair": "uz-ru", "source": ..., "target": ..., "origin": "til"} JSONL

Usage:
    python scripts/til_to_manifest.py --data-dir data/uz-ru
    python scripts/til_to_manifest.py --data-dir data/uz-ru --directions uz-ru      # one direction
    python scripts/til_to_manifest.py --data-dir data/uz-ru --labse 0.75            # + meaning filter
    python scripts/til_to_manifest.py --data-dir data/uz-en                         # any TIL pair
    python scripts/til_to_manifest.py --data-dir data/uz-ru --limit 1000 --show 10  # quick look

Output (default data/external/til_<pair>/): train.jsonl dev.jsonl test.jsonl stats.json
License of TIL: CC BY-NC-SA 4.0 (non-commercial, share-alike).
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent  # everything lives inside riva_train/

# ---------------------------------------------------------------------------
# Detokenization (reverses Moses-style tokenization used in TIL)
# ---------------------------------------------------------------------------
APOS = "‘’ʻʼ'`´"
LETTER = r"[^\W\d_]"
# o ‘ quv -> o'quv ; qat ’ i -> qat'i ; ma ’ rifiy -> ma'rifiy  (letter + apostrophe + letter)
RE_UZ_APOS = re.compile(rf"({LETTER}) ?[{APOS}] ?(?={LETTER})")
RE_SPACE_BEFORE = re.compile(r" +([.,;:!?%)\]}»”…])")        # "word ," -> "word,"
RE_SPACE_AFTER = re.compile(r"([(\[{«“„]) +")                 # "( word" -> "(word"
RE_ELLIPSIS = re.compile(r"\. \. \.")
RE_NUM_DOT = re.compile(r"(\d) \.(?=\s|$)")                   # "13 ." -> "13."
# Uzbek suffixes glued to a closing quote/bracket:  ”gi, »ning, )dagi
RE_UZ_SUFFIX = re.compile(r"([”»\")]) (gi|ga|da|dan|ni|ning|dagi|dagi|ka|qa|lar|larni|larning|dir)\b")
RE_STRAIGHT_Q = re.compile(r'"')
WS = re.compile(r"\s+")


def fix_straight_quotes(s):
    """Toggle " as opening/closing and attach it to the quoted text."""
    out, opening = [], True
    parts = RE_STRAIGHT_Q.split(s)
    for i, part in enumerate(parts):
        if i:
            out.append('"')
            opening = not opening
        if i and not opening:        # we just opened a quote: strip space after it
            part = part.lstrip()
        if i < len(parts) - 1 and not opening:  # a closing quote follows: strip space before it
            part = part.rstrip()
        out.append(part)
    return "".join(out)


def detok(s, lang, apostrophe="'"):
    s = WS.sub(" ", s.replace(" ", " ")).strip()
    if not s:
        return s
    if lang in ("uz", "kaa", "tk", "az", "crh", "tr", "kk", "ky"):
        s = RE_UZ_APOS.sub(lambda m: m.group(1) + apostrophe, s)
    s = RE_ELLIPSIS.sub("...", s)
    s = RE_NUM_DOT.sub(r"\1.", s)
    s = RE_SPACE_BEFORE.sub(r"\1", s)
    s = RE_SPACE_AFTER.sub(r"\1", s)
    if s.count('"') % 2 == 0 and '"' in s:
        s = fix_straight_quotes(s)
    if lang == "uz":
        s = RE_UZ_SUFFIX.sub(r"\1\2", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
LATIN = re.compile(r"[A-Za-z]")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
SCRIPT = {"uz": "latin", "en": "latin", "tr": "latin", "az": "latin", "tk": "latin", "kaa": "latin",
          "ru": "cyrillic", "kk": "cyrillic", "ky": "cyrillic", "tt": "cyrillic", "ba": "cyrillic",
          "cv": "cyrillic", "sah": "cyrillic"}


def script_ok(text, lang):
    want = SCRIPT.get(lang)
    if not want:
        return True
    n_lat, n_cyr = len(LATIN.findall(text)), len(CYRILLIC.findall(text))
    if n_lat + n_cyr < 2:
        return True  # numbers/symbols only: judged by other rules
    return (n_lat >= n_cyr) if want == "latin" else (n_cyr > n_lat)


def check(a, b, la, lb, args):
    if not a or not b:
        return "empty"
    if len(a) < args.min_chars or len(b) < args.min_chars:
        return "too short"
    if len(a) > args.max_chars or len(b) > args.max_chars:
        return "too long"
    if a.lower() == b.lower():
        return "untranslated (identical sides)"
    if sum(c.isalpha() for c in a) < 2 or sum(c.isalpha() for c in b) < 2:
        return "no real words (numbers/symbols)"
    if not script_ok(a, la):
        return f"{la} side in wrong script"
    if not script_ok(b, lb):
        return f"{lb} side in wrong script"
    ratio = max(len(a), len(b)) / max(1, min(len(a), len(b)))
    if ratio > args.max_ratio and max(len(a), len(b)) > 15:
        return "length ratio (likely misaligned)"
    return None


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def find_pair_files(split_dir, la, lb):
    """Return (file_a, file_b) where file_a ends with .<la> and file_b with .<lb> and same stem."""
    if not split_dir.is_dir():
        return None
    a_files = {p.with_suffix("").relative_to(split_dir): p for p in split_dir.rglob(f"*.{la}")}
    b_files = {p.with_suffix("").relative_to(split_dir): p for p in split_dir.rglob(f"*.{lb}")}
    common = sorted(set(a_files) & set(b_files))
    return [(a_files[k], b_files[k]) for k in common] or None


def count_lines(path):
    with open(path, "rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))


def read_split(files, la, lb, args, dropped, limit):
    rows = []
    for fa, fb in files:
        na, nb = count_lines(fa), count_lines(fb)
        if na != nb:
            sys.exit(f"Line counts differ — files are not aligned:\n  {fa}: {na:,}\n  {fb}: {nb:,}")
        print(f"    {fa.name} / {fb.name}: {na:,} lines")
        with open(fa, encoding="utf-8", errors="replace") as A, open(fb, encoding="utf-8", errors="replace") as B:
            for i, (a, b) in enumerate(zip(A, B)):
                a, b = detok(a, la, args.apostrophe), detok(b, lb, args.apostrophe)
                why = check(a, b, la, lb, args)
                if why:
                    dropped[why] += 1
                    continue
                rows.append((a, b))
                if limit and len(rows) >= limit:
                    return rows
    return rows


def key(text):
    return hashlib.blake2b(text.lower().encode("utf-8"), digest_size=8).digest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="TIL corpus -> project JSONL manifests.")
    ap.add_argument("--data-dir", required=True, help="e.g. data/uz-ru (folder made by download_data.py)")
    ap.add_argument("--langs", nargs=2, default=None, metavar=("A", "B"),
                    help="language codes; default from folder name, e.g. uz ru")
    ap.add_argument("--directions", nargs="+", default=None,
                    help="which pairs to write, e.g. uz-ru ru-uz (default: both)")
    ap.add_argument("--min-chars", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--max-ratio", type=float, default=2.0, help="max length ratio between sides")
    ap.add_argument("--labse", type=float, default=None, help="LaBSE similarity cut-off, e.g. 0.75 (optional)")
    ap.add_argument("--apostrophe", default="'", help="character for Uzbek o'/g' apostrophes (default ')")
    ap.add_argument("--limit", type=int, default=None, help="max rows per split (quick test)")
    ap.add_argument("--show", type=int, default=3, help="print N random examples")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    la, lb = args.langs or data_dir.name.split("-", 1)
    directions = args.directions or [f"{la}-{lb}", f"{lb}-{la}"]
    out_dir = Path(args.out_dir or PROJECT_ROOT / "data" / "external" / f"til_{la}-{lb}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    splits, dropped = {}, {}
    for split in ("train", "dev", "test"):
        files = find_pair_files(data_dir / split, la, lb)
        if not files:
            print(f"[{split}] no *.{la} / *.{lb} files under {data_dir / split} — skipped")
            continue
        print(f"[{split}]")
        dropped[split] = Counter()
        splits[split] = read_split(files, la, lb, args, dropped[split], args.limit)

    if "train" not in splits:
        sys.exit(f"No train files found under {data_dir}/train. Check --data-dir.")

    # ---------- dedupe; keep dev/test out of train ----------
    held = set()
    for split in ("dev", "test"):
        uniq, seen = [], set()
        for a, b in splits.get(split, []):
            k = key(a)
            if k in seen:
                dropped[split]["duplicate"] += 1
                continue
            seen.add(k)
            held.add(k)
            held.add(key(b))
            uniq.append((a, b))
        if split in splits:
            splits[split] = uniq
    uniq, seen = [], set()
    for a, b in splits["train"]:
        ka = key(a)
        if ka in held or key(b) in held:
            dropped["train"]["also in dev/test (removed from train)"] += 1
            continue
        if ka in seen:
            dropped["train"]["duplicate"] += 1
            continue
        seen.add(ka)
        uniq.append((a, b))
    splits["train"] = uniq

    # no dev/test shipped? carve 2,000 each from train
    for split in ("dev", "test"):
        if not splits.get(split):
            rng.shuffle(splits["train"])
            n = min(2000, len(splits["train"]) // 20)
            splits[split], splits["train"] = splits["train"][:n], splits["train"][n:]
            print(f"[{split}] not provided by TIL — took {n:,} random pairs from train")

    # ---------- optional LaBSE ----------
    if args.labse is not None:
        from sentence_transformers import SentenceTransformer

        print(f"LaBSE filter (>= {args.labse}) ...")
        m = SentenceTransformer("sentence-transformers/LaBSE")
        for split, rows in splits.items():
            kept = []
            for i in range(0, len(rows), 50_000):
                chunk = rows[i:i + 50_000]
                ea = m.encode([a for a, _ in chunk], batch_size=512, normalize_embeddings=True)
                eb = m.encode([b for _, b in chunk], batch_size=512, normalize_embeddings=True)
                for (a, b), s in zip(chunk, (ea * eb).sum(1)):
                    if s >= args.labse:
                        kept.append((a, b))
                    else:
                        dropped[split][f"LaBSE < {args.labse}"] += 1
                print(f"  {split}: {min(i + 50_000, len(rows)):,}/{len(rows):,}")
            splits[split] = kept

    # ---------- write ----------
    written = {}
    for split, rows in splits.items():
        rng.shuffle(rows)
        n = 0
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for a, b in rows:
                for d in directions:
                    s, t = (a, b) if d == f"{la}-{lb}" else (b, a)
                    f.write(json.dumps({"pair": d, "source": s, "target": t, "origin": "til"},
                                       ensure_ascii=False) + "\n")
                    n += 1
        written[split] = n

    stats = {
        "source": str(data_dir), "languages": [la, lb], "directions": directions,
        "pairs": {k: len(v) for k, v in splits.items()}, "rows_written": written,
        "dropped": {k: dict(v.most_common()) for k, v in dropped.items()},
        "license": "TIL corpus: CC BY-NC-SA 4.0 (non-commercial, share-alike)",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    for split in ("train", "dev", "test"):
        if split in splits:
            print(f"  {split:<5} {len(splits[split]):>10,} pairs -> {written[split]:>10,} rows  "
                  f"{out_dir / (split + '.jsonl')}")
            for why, c in dropped.get(split, Counter()).most_common():
                print(f"          dropped {c:>9,}  {why}")
    for a, b in rng.sample(splits["train"], min(args.show, len(splits["train"]))):
        print(f"\n  {la}: {a[:150]}\n  {lb}: {b[:150]}")
    print(f"\nSummary: {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
RIVA_FILE_EOF
echo "  wrote til_to_manifest.py"

# ---------------------------------------------------------------------
cat > uzlpc_to_manifest.py << 'RIVA_FILE_EOF'
"""
Convert blinoff/UzLPC (2.33M pairs: English/Russian -> Uzbek) into this project's JSONL manifests.

About the data (from the dataset card):
  * data.csv, 360 MB, one "train" split, CC BY-SA 4.0
  * columns: sentence_id, source_lang (eng/rus), source_texts, translation (Uzbek),
             references (real Uzbek, Wikipedia rows only), confidence_score (LaBSE, Wikipedia only),
             agglutination_index
  * 89.6% Tatoeba sentences, 10.4% Wikipedia
  * the Uzbek "translation" column is MACHINE-GENERATED (Qwen3.5-122B-A10B)

Because the Uzbek side is machine-made and the English/Russian side is human-written, the
recommended training directions are uz -> en and uz -> ru (the model learns to produce the
human side). Add en-uz / ru-uz with --directions if you want them anyway.

Usage:
  python scripts/uzlpc_to_manifest.py                              # download + convert (uz-en, uz-ru)
  python scripts/uzlpc_to_manifest.py --directions uz-en uz-ru en-uz ru-uz
  python scripts/uzlpc_to_manifest.py --langs en                   # only the English part
  python scripts/uzlpc_to_manifest.py --csv data.csv               # already downloaded
  python scripts/uzlpc_to_manifest.py --wiki-uzbek reference --min-confidence 0.8
                                         # Wikipedia rows: use the real Uzbek reference, LaBSE >= 0.8

Output (default data/external/uzlpc/): train.jsonl dev.jsonl test.jsonl stats.json
Requires: pip install pandas huggingface_hub
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent  # everything lives inside riva_train/
REPO_ID = "blinoff/UzLPC"
LANG_MAP = {"eng": "en", "en": "en", "english": "en", "rus": "ru", "ru": "ru", "russian": "ru"}

WS = re.compile(r"\s+")
LATIN = re.compile(r"[A-Za-z]")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
UZ_APOS = re.compile(r"[‘’ʻʼ`´]")
# things an LLM sometimes adds around a translation
LLM_PREFIX = re.compile(r"^(tarjima|translation|перевод|uzbek|o'zbekcha)\s*[:：-]\s*", re.IGNORECASE)
QUOTES = "\"'«»“”„"


def clean(t):
    if not isinstance(t, str):
        return ""
    return WS.sub(" ", t.replace(" ", " ").replace("​", "")).strip()


def clean_uz(uz, src):
    uz = UZ_APOS.sub("'", clean(uz))
    uz = LLM_PREFIX.sub("", uz)
    # model wrapped the whole answer in quotes although the source wasn't quoted
    if len(uz) > 2 and uz[0] in QUOTES and uz[-1] in QUOTES and not (src[:1] in QUOTES and src[-1:] in QUOTES):
        uz = uz[1:-1].strip()
    return uz


def check(src, uz, lang, raw_uz, args):
    if not src or not uz:
        return "empty"
    if "\n" in str(raw_uz).strip():
        return "multi-line output (LLM added extra text)"
    if len(src) < args.min_chars or len(uz) < args.min_chars:
        return "too short"
    if len(src) > args.max_chars or len(uz) > args.max_chars:
        return "too long"
    if src.lower() == uz.lower():
        return "untranslated (identical)"
    n_lat, n_cyr = len(LATIN.findall(uz)), len(CYRILLIC.findall(uz))
    if n_cyr > 0:
        return "Cyrillic in Uzbek side (untranslated Russian / Cyrillic Uzbek)"
    if n_lat < 0.5 * len(uz.replace(" ", "")):
        return "Uzbek side has too few letters"
    s_lat, s_cyr = len(LATIN.findall(src)), len(CYRILLIC.findall(src))
    if lang == "ru" and s_cyr <= s_lat:
        return "Russian source not in Cyrillic"
    if lang == "en" and s_cyr > 0:
        return "English source contains Cyrillic"
    ratio = max(len(src), len(uz)) / max(1, min(len(src), len(uz)))
    if ratio > args.max_ratio and max(len(src), len(uz)) > 15:
        return "length ratio (likely bad translation)"
    return None


def split_of(text, f_test, f_dev):
    u = int.from_bytes(hashlib.blake2b(text.lower().encode(), digest_size=8).digest(), "big") / 2**64
    return "test" if u < f_test else "dev" if u < f_test + f_dev else "train"


def main():
    ap = argparse.ArgumentParser(description="blinoff/UzLPC -> project JSONL manifests")
    ap.add_argument("--csv", default=None, help="local data.csv (skip download)")
    ap.add_argument("--langs", nargs="+", default=["en", "ru"], choices=["en", "ru"])
    ap.add_argument("--directions", nargs="+", default=None,
                    help="default: uz-en uz-ru (machine Uzbek as input, human text as output)")
    ap.add_argument("--wiki-uzbek", choices=["translation", "reference", "both"], default="translation",
                    help="Wikipedia rows: machine translation, the real Uzbek Wikipedia reference, or both")
    ap.add_argument("--min-confidence", type=float, default=None,
                    help="Wikipedia rows: minimum confidence_score (LaBSE), e.g. 0.8")
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--max-ratio", type=float, default=2.5)
    ap.add_argument("--dev", type=int, default=2000, help="approx. dev pairs per language")
    ap.add_argument("--test", type=int, default=2000, help="approx. test pairs per language")
    ap.add_argument("--limit", type=int, default=None, help="read only the first N CSV rows")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "external" / "uzlpc"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    directions = args.directions or [f"uz-{l}" for l in args.langs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- get the CSV ----------
    if args.csv:
        csv_path = Path(args.csv)
    else:
        from huggingface_hub import hf_hub_download

        print(f"Downloading {REPO_ID}/data.csv (~360 MB, cached, resumable) ...")
        csv_path = Path(hf_hub_download(REPO_ID, "data.csv", repo_type="dataset"))
    print(f"Reading {csv_path}")

    # ---------- read in chunks, clean ----------
    pairs = {"en": [], "ru": []}  # (src, uz, origin)
    dropped, seen, n_rows = Counter(), set(), 0
    reader = pd.read_csv(csv_path, dtype=str, keep_default_na=False, chunksize=200_000,
                         nrows=args.limit, on_bad_lines="warn")
    for chunk in reader:
        cols = set(chunk.columns)
        need = {"source_lang", "source_texts", "translation"}
        if not need <= cols:
            sys.exit(f"Unexpected columns: {list(chunk.columns)} (need {sorted(need)})")
        for r in chunk.itertuples(index=False):
            n_rows += 1
            lang = LANG_MAP.get(str(r.source_lang).strip().lower())
            if lang is None:
                dropped[f"unknown source_lang '{r.source_lang}'"] += 1
                continue
            if lang not in args.langs:
                continue
            src = clean(r.source_texts)
            ref = clean(getattr(r, "references", ""))
            conf = getattr(r, "confidence_score", "")
            is_wiki = bool(ref) or conf not in ("", None)
            if is_wiki and args.min_confidence is not None:
                try:
                    if float(conf) < args.min_confidence:
                        dropped[f"Wikipedia confidence < {args.min_confidence}"] += 1
                        continue
                except ValueError:
                    pass
            candidates = [(r.translation, "tatoeba-mt" if not is_wiki else "wiki-mt")]
            if is_wiki and ref and args.wiki_uzbek != "translation":
                ref_c = (ref, "wiki-ref")
                candidates = [ref_c] if args.wiki_uzbek == "reference" else candidates + [ref_c]
            for raw_uz, origin in candidates:
                uz = clean_uz(raw_uz, src)
                why = check(src, uz, lang, raw_uz, args)
                if why:
                    dropped[why] += 1
                    continue
                k = hashlib.blake2b(f"{lang}\t{src.lower()}\t{uz.lower()}".encode(), digest_size=8).digest()
                if k in seen:
                    dropped["duplicate"] += 1
                    continue
                seen.add(k)
                pairs[lang].append((src, uz, origin))
        print(f"  read {n_rows:,} rows  (kept en {len(pairs['en']):,} / ru {len(pairs['ru']):,})")

    # ---------- split by source sentence (same sentence never in two splits) ----------
    rng = random.Random(args.seed)
    splits = {"train": [], "dev": [], "test": []}
    per_lang = {}
    for lang, rows in pairs.items():
        if not rows:
            continue
        f_test, f_dev = min(0.1, args.test / len(rows)), min(0.1, args.dev / len(rows))
        counts = Counter()
        for src, uz, origin in rows:
            s = split_of(src, f_test, f_dev)
            splits[s].append((lang, src, uz, origin))
            counts[s] += 1
        per_lang[lang] = dict(counts)

    written = {}
    for split, rows in splits.items():
        rng.shuffle(rows)
        n = 0
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for lang, src, uz, origin in rows:
                for d in directions:
                    if d == f"uz-{lang}":
                        rec = {"pair": d, "source": uz, "target": src}
                    elif d == f"{lang}-uz":
                        rec = {"pair": d, "source": src, "target": uz}
                    else:
                        continue
                    rec["origin"] = f"uzlpc:{origin}"
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
        written[split] = n

    stats = {
        "dataset": REPO_ID, "csv_rows_read": n_rows, "directions": directions,
        "pairs_per_language": per_lang, "rows_written": written, "dropped": dict(dropped.most_common()),
        "note": "Uzbek side is machine-generated (Qwen3.5-122B-A10B) except wiki-ref rows.",
        "license": "CC BY-SA 4.0 (share-alike); Tatoeba source sentences CC BY 2.0 FR",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    print(f"csv rows read: {n_rows:,}")
    for lang, c in per_lang.items():
        print(f"  {lang}: " + "  ".join(f"{k} {v:,}" for k, v in sorted(c.items())))
    for why, c in dropped.most_common():
        print(f"  dropped {c:>9,}  {why}")
    for split in ("train", "dev", "test"):
        print(f"  {split:<5} {written[split]:>10,} rows -> {out_dir / (split + '.jsonl')}")
    for lang, src, uz, origin in rng.sample(splits["train"], min(4, len(splits["train"]))):
        print(f"\n  [{origin}] {lang}: {src[:120]}\n  {'':>{len(origin) + 3}}uz: {uz[:120]}")


if __name__ == "__main__":
    main()
RIVA_FILE_EOF
echo "  wrote uzlpc_to_manifest.py"

# ---------------------------------------------------------------------
cat > riva_test.py << 'RIVA_FILE_EOF'
"""
Test nvidia/Riva-Translate-4B-Instruct-v2 (WITHOUT training) on your uz->ru test set.

It answers: "How well does the original Riva model already translate Uzbek -> Russian?"
  * translates N test sentences
  * scores chrF / BLEU against the real Russian translations
  * checks the output really is Russian (Cyrillic) and not a copy of the Uzbek input
  * saves every translation to a CSV so you can read them

Three ways to run the model (--backend):
  hf       transformers on your GPU (needs ~9 GB VRAM)                     <- simplest
  server   any OpenAI-compatible server: llama.cpp (llama-server), vLLM
  ollama   Ollama (a model you created with `ollama create`)

Usage:
  python riva_test.py                                   # hf, 200 sentences from TIL uz-ru test
  python riva_test.py --samples 500 --test-file data/external/uzlpc/test.jsonl
  python riva_test.py --backend server --url http://localhost:8080          # llama-server / vLLM
  python riva_test.py --backend ollama --ollama-model riva-uz-ru            # Ollama
  python riva_test.py --pair en-ru --test-file my_en_ru_test.jsonl          # an official pair

Why a custom prompt for uz-ru: Riva only knows tags like "en-ru". For an unknown pair its chat
template drops the system message, so this script builds the prompt in the model's exact format:
    <s>System\nYou are an expert at translating text from Uzbek to Russian.</s>\n
    <s>User\nWhat is the Russian translation of the sentence: ...</s>\n<s>Assistant\n

Requires: pip install torch transformers sacrebleu pandas requests
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent  # everything lives inside riva_train/
MODEL_ID = "nvidia/Riva-Translate-4B-Instruct-v2"
LANG_NAMES = {"uz": "Uzbek", "ru": "Russian", "en": "English", "kk": "Kazakh", "tr": "Turkish",
              "de": "German", "fr": "French", "zh": "Simplified Chinese", "ky": "Kyrgyz", "tg": "Tajik"}
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")


def build_prompt(text, src, tgt):
    s, t = LANG_NAMES.get(src, src), LANG_NAMES.get(tgt, tgt)
    return (f"<s>System\nYou are an expert at translating text from {s} to {t}.</s>\n"
            f"<s>User\nWhat is the {t} translation of the sentence: {text.strip()}</s>\n"
            f"<s>Assistant\n")


def clean_output(o):
    o = o.split("</s>")[0].split("<s>")[0]
    return o.strip()


# ---------------------------------------------------------------------------
# Backends — each returns a function: list[str prompt] -> list[str output]
# ---------------------------------------------------------------------------
def hf_backend(model_id, batch_size, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    if not cuda:
        print("  WARNING: no GPU — this will be very slow. Try --samples 20, or use --backend ollama.")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto" if cuda else None)
    model.eval()

    @torch.inference_mode()
    def run(prompts):
        outs = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            # the prompt already contains the <s> markers -> don't add another BOS
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            outs += [clean_output(t) for t in tok.batch_decode(new, skip_special_tokens=True)]
            print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}")
        return outs

    return run


def server_backend(url, max_new_tokens, model_name):
    import requests

    endpoint = url.rstrip("/") + ("/completions" if url.rstrip("/").endswith("/v1") else "/v1/completions")

    def run(prompts):
        outs = []
        for i, p in enumerate(prompts, 1):
            r = requests.post(endpoint, json={"model": model_name, "prompt": p, "max_tokens": max_new_tokens,
                                              "temperature": 0, "stop": ["</s>", "<s>"]}, timeout=300)
            r.raise_for_status()
            outs.append(clean_output(r.json()["choices"][0]["text"]))
            if i % 20 == 0 or i == len(prompts):
                print(f"  {i}/{len(prompts)}")
        return outs

    return run


def ollama_backend(url, model_name, max_new_tokens):
    import requests

    def run(prompts):
        outs = []
        for i, p in enumerate(prompts, 1):
            # raw=True: send our exact prompt, bypass the Modelfile template
            r = requests.post(url.rstrip("/") + "/api/generate", json={
                "model": model_name, "prompt": p, "raw": True, "stream": False,
                "options": {"temperature": 0, "num_predict": max_new_tokens, "stop": ["</s>", "<s>"]}},
                timeout=300)
            r.raise_for_status()
            outs.append(clean_output(r.json()["response"]))
            if i % 20 == 0 or i == len(prompts):
                print(f"  {i}/{len(prompts)}")
        return outs

    return run


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_test(path, pair, n, seed):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("pair") == pair and r.get("source") and r.get("target"):
                rows.append(r)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main():
    ap = argparse.ArgumentParser(description="Zero-shot test of Riva-Translate on your test set.")
    ap.add_argument("--backend", choices=["hf", "server", "ollama"], default="hf")
    ap.add_argument("--model", default=MODEL_ID, help="HF id or local folder (hf backend)")
    ap.add_argument("--test-file", default=str(PROJECT_ROOT / "test_data" / "test.jsonl"))
    ap.add_argument("--pair", default="uz-ru")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--url", default="http://localhost:8080", help="server backend URL")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-model", default="riva-uz-ru")
    ap.add_argument("--server-model", default="riva", help="model name the server expects")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "riva_zero_shot"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import sacrebleu

    src, tgt = args.pair.split("-", 1)
    test_path = Path(args.test_file)
    if not test_path.exists():
        sys.exit(f"Test file not found: {test_path}\nMake it with til_to_manifest.py, or pass --test-file.")
    rows = load_test(test_path, args.pair, args.samples, args.seed)
    if not rows:
        sys.exit(f"No '{args.pair}' rows in {test_path}")
    print(f"Test set: {len(rows)} {args.pair} sentences from {test_path}")
    print("Example prompt:\n" + "-" * 60 + "\n" + build_prompt(rows[0]["source"], src, tgt) + "\n" + "-" * 60)

    if args.backend == "hf":
        print(f"Loading {args.model} (first time downloads ~8 GB) ...")
        run = hf_backend(args.model, args.batch_size, args.max_new_tokens)
    elif args.backend == "server":
        run = server_backend(args.url, args.max_new_tokens, args.server_model)
    else:
        run = ollama_backend(args.ollama_url, args.ollama_model, args.max_new_tokens)

    # ---- quick sanity check on an official pair: proves the setup works ----
    print("\nSanity check (official en->ru pair):")
    check = run([build_prompt("The weather is very good today.", "en", "ru")])[0]
    print(f"  'The weather is very good today.' -> {check}")
    if not CYRILLIC.search(check or ""):
        print("  WARNING: the official pair did not return Russian — check the backend/model setup.")

    print(f"\nTranslating {len(rows)} sentences {src} -> {tgt} ...")
    t0 = time.time()
    hyps = run([build_prompt(r["source"], src, tgt) for r in rows])
    secs = time.time() - t0
    refs = [r["target"] for r in rows]

    # ---- scores + simple checks ----
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    empty = sum(1 for h in hyps if not h)
    copied = sum(1 for r, h in zip(rows, hyps) if h and h.lower() == r["source"].lower())
    if tgt in ("ru", "kk", "ky", "tg"):
        right_script = sum(1 for h in hyps if len(CYRILLIC.findall(h)) > len(LATIN.findall(h)))
    else:
        right_script = sum(1 for h in hyps if len(LATIN.findall(h)) >= len(CYRILLIC.findall(h)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame({"source": [r["source"] for r in rows], "reference": refs, "riva": hyps,
                  "origin": [r.get("origin", "") for r in rows]}).to_csv(
        out / f"predictions_{args.pair}.csv", index=False, encoding="utf-8-sig")
    result = {"model": args.model if args.backend == "hf" else f"{args.backend}:{args.ollama_model if args.backend == 'ollama' else args.url}",
              "pair": args.pair, "samples": len(rows), "chrf": round(chrf, 2), "bleu": round(bleu, 2),
              "output_in_target_script_pct": round(100 * right_script / len(rows), 1),
              "copied_input": copied, "empty": empty, "seconds": round(secs, 1)}
    (out / f"scores_{args.pair}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Riva-Translate zero-shot result ===")
    print(f"  chrF  : {chrf:6.2f}   (0-100, main score)")
    print(f"  BLEU  : {bleu:6.2f}")
    print(f"  output really in {LANG_NAMES.get(tgt, tgt)}: {100 * right_script / len(rows):.0f}%")
    print(f"  copied the input: {copied}   empty: {empty}   time: {secs:.0f}s")
    verdict = ("very weak — the model barely knows this pair; prefer NLLB" if chrf < 30 else
               "partial — training could help a lot; compare with NLLB before deciding" if chrf < 45 else
               "good — fine-tuning Riva on your data is a reasonable plan")
    print(f"  => {verdict}")
    print("\nExamples:")
    for r, h in list(zip(rows, hyps))[:5]:
        print(f"  {src}: {r['source'][:110]}\n  ref: {r['target'][:110]}\n  riva: {h[:110]}\n")
    print(f"All translations: {out / f'predictions_{args.pair}.csv'}")


if __name__ == "__main__":
    main()
RIVA_FILE_EOF
echo "  wrote riva_test.py"

# test file: only written if you do not have one yet
if [ ! -s test_data/test.jsonl ]; then
cat > test_data/test.jsonl << 'RIVA_FILE_EOF'
{"pair": "uz-ru", "source": "Xalqaro mutaxassislarni jalb etgan holda yo'nalishlar bo'yicha mashg'ulotlar o'tkazish.", "target": "Проведение мастер-классов по направлениям с привлечением международных специалистов.", "origin": "til"}
{"pair": "uz-ru", "source": "13. Tender bo'yicha takliflar topshirilgandan keyin qatnashchilar ularga biron-bir tuzatishlar kiritish huquqiga ega emas.", "target": "13. После сдачи тендерных предложений участники не имеют права вносить в них какие-либо поправки.", "origin": "til"}
{"pair": "uz-ru", "source": "I. Biologik xilma-xillik mavzularini davlat hokimiyati va boshqaruvi organlari va jamiyat faoliyatiga kiritish", "target": "I. Включение тематики биологического разнообразия в деятельность органов государственной власти и управления и общества", "origin": "til"}
{"pair": "uz-ru", "source": "Ovqatlanish anjomlarini kimyoviy moddalar va metallni korroziyaga uchratadigan boshqa materiallar bilan birgalikda tashishga ruxsat berilmaydi.", "target": "Совместная перевозка столовых приборов с химикатами и другими материалами, которые вызывают коррозию металла, не разрешается.", "origin": "til"}
{"pair": "uz-ru", "source": "Inson huquqlariga rioya etish va himoya qilish masalalari bo'yicha axborot-ma'rifiy tadbirlar rejasini ishlab chiqish.", "target": "Разработка плана информационно-просветительских мероприятий по вопросам соблюдения и защиты прав человека.", "origin": "til"}
{"pair": "uz-ru", "source": "3. Quyidagilar “Sirdaryo” erkin iqtisodiy zonasi faoliyatining asosiy vazifalari va yo'nalishlari etib belgilansin:", "target": "3. Определить основными задачами и направлениями деятельности СЭЗ «Сирдарё»:", "origin": "til"}
{"pair": "uz-ru", "source": "Bugun havo juda yaxshi.", "target": "Сегодня очень хорошая погода.", "origin": "manual"}
{"pair": "uz-ru", "source": "Men har kuni ertalab kitob o'qiyman.", "target": "Я каждое утро читаю книгу.", "origin": "manual"}
{"pair": "uz-ru", "source": "Iltimos, eshikni yoping.", "target": "Пожалуйста, закройте дверь.", "origin": "manual"}
{"pair": "uz-ru", "source": "Toshkentga qanday borsam bo'ladi?", "target": "Как мне добраться до Ташкента?", "origin": "manual"}
RIVA_FILE_EOF
echo "  wrote test_data/test.jsonl"
else
echo "  kept your test_data/test.jsonl"
fi

echo
echo "Done:"
find "$RIVA_DIR" -maxdepth 3 | sort | sed "s|^$RIVA_DIR|  riva_train|"