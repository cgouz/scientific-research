"""
Fine-tune nvidia/Riva-Translate-4B-Instruct-v2 on Uzbek -> Russian.
Defaults are in CONFIG below; a YAML file (default configs/riva_uz_ru.yaml) overrides them,
and --set overrides both. Manifest paths: data.root + data.manifests / data.dirs (see the YAML).

Prompt (identical to riva_test/riva_test.py, so before/after scores are comparable):
    <s>System\nYou are an expert at translating text from Uzbek to Russian.</s>\n
    <s>User\nWhat is the Russian translation of the sentence: {uzbek}</s>\n
    <s>Assistant\n{russian}</s>
The loss is computed ONLY on the Russian translation.

GPU features: bf16 + TF32, fused AdamW, SDPA, auto batch size (probes the GPU with the longest
examples), length-grouped batches, multi-GPU via torchrun, gradient checkpointing.

Usage:
  python train_riva_uzru.py                                          # full training (configs/riva_uz_ru.yaml)
  python train_riva_uzru.py --config configs/other.yaml              # another config
  python train_riva_uzru.py --set train.max_steps=200 train.eval_steps=100 data.test_samples=200   # quick test
  python train_riva_uzru.py --set method=lora train.lr=1e-4          # change any setting for one run
  python train_riva_uzru.py --resume                                 # continue after a crash
  torchrun --nproc_per_node=gpu train_riva_uzru.py                   # several GPUs

Output (CONFIG["output"]["dir"], default outputs/riva-uz-ru/):
  final/     full model (method full) or merged model (method lora)  -> riva_test/riva_test.py --model
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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent  # everything lives next to this file (riva_train/)
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache" / "datasets"))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "riva_uz_ru.yaml"

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
        "root": None,               # base folder for relative manifest paths (None = this file's folder)
        "manifests": [],            # explicit files: [{"name": .., "train": .., "dev": .., "test": ..}]
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
        "epochs": 5.0,
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


def merge_yaml(base, extra, prefix=""):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if k not in out:
            sys.exit(f"config: unknown setting '{prefix}{k}' (check the spelling against CONFIG)")
        out[k] = merge_yaml(out[k], v, f"{prefix}{k}.") if isinstance(v, dict) and isinstance(out[k], dict) else v
    return out


def load_config(path, overrides):
    cfg = copy.deepcopy(CONFIG)
    if path:
        p = Path(path)
        if not p.exists():
            sys.exit(f"Config not found: {p}")
        cfg = merge_yaml(cfg, yaml.safe_load(p.read_text(encoding="utf-8")))
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
def resolve(p, root=None):
    p = Path(p)
    if p.is_absolute():
        return p
    return Path(root) / p if root else PROJECT_ROOT / p  # relative = data.root/<path>, else ttt/<path>


def manifest_sources(dcfg):
    """[(name, {split: path}, explicit)] from data.manifests (explicit files) + data.dirs (<dir>/<split>.jsonl)."""
    root = dcfg.get("root")
    sources = []
    for m in dcfg.get("manifests") or []:
        files = {s: resolve(m[s], root) for s in ("train", "dev", "test") if m.get(s)}
        if not files:
            sys.exit(f"data.manifests entry has no train/dev/test path: {m}")
        sources.append((m.get("name") or next(iter(files.values())).parent.name, files, True))
    for d in dcfg.get("dirs") or []:
        d = resolve(d, root)
        sources.append((d.name, {s: d / f"{s}.jsonl" for s in ("train", "dev", "test")}, False))
    return sources


def read_manifest(path, pair, limit, rng, origin=None):
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
                             "origin": str(r.get("origin", origin or path.parent.name))})
    if limit and len(rows) > limit:
        rng.shuffle(rows)
        rows = rows[:limit]
    return rows


def load_data(dcfg, seed, log):
    rng = random.Random(seed)
    splits = {"train": [], "dev": [], "test": []}
    for name, files, explicit in manifest_sources(dcfg):
        if not explicit and not next(iter(files.values())).parent.exists():
            log(f"  (skip) {next(iter(files.values())).parent} not found")
            continue
        for split, path in files.items():
            if explicit and not path.exists():
                log(f"  (missing) {name} {split}: {path}")
                continue
            rows = read_manifest(path, dcfg["pair"],
                                 dcfg["max_per_source"] if split == "train" else None, rng, name)
            splits[split] += rows
            if rows:
                log(f"  {name:<20} {split:<5} {len(rows):>10,} {dcfg['pair']} pairs")
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
    ap.add_argument("--config", default=str(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else None,
                    help="YAML that overrides CONFIG (default configs/riva_uz_ru.yaml)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
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
        sys.exit(f"No '{dcfg['pair']}' train rows found. Check data.root / data.manifests / data.dirs / data.pair.")
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
        log(f"Test again any time:  python riva_test/riva_test.py --model {final}")


if __name__ == "__main__":
    main()
