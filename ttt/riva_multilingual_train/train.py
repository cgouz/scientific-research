"""
Fine-tune nvidia/Riva-Translate-4B-Instruct-v2 on several directions at once
(uz-ru, ru-uz, uz-en, en-uz, en-ru, ru-en by default). All settings live in config.yaml.

Data: JSONL rows {"pair": "uz-en", "source": .., "target": .., "origin": ..}; every row is trained in
the direction of its own "pair" (other fields are ignored).

Auto mode (the `auto` values in config.yaml):
  data.pairs        auto = every direction found in the train file whose languages are in LANG_NAMES
  data.max_len      auto = 99.5th percentile of token lengths on 20k train rows (multiple of 32, 128..1024)
  train.batch_size  auto = largest per-GPU batch whose full training step fits in train.memory_fraction
  train.precision   auto = bf16 on Ampere or newer (compute capability >= 8.0), fp16 on older GPUs
  train.num_workers auto = CPU cores / GPUs, at most 8
  GPUs              run.sh starts torchrun on every GPU nvidia-smi shows

data.script_filter drops rows whose text is in the wrong alphabet for its pair, e.g. an "en-uz" row
whose source is Russian, or a "uz-en" row whose target is Russian.

Prompt (identical to ../riva_test/riva_test.py), loss only on the translation:
    <s>System\\nYou are an expert at translating text from {src} to {tgt}.</s>\\n
    <s>User\\nWhat is the {tgt} translation of the sentence: {source}</s>\\n
    <s>Assistant\\n{target}</s>

Full fine-tuning in mixed precision, best checkpoint picked by dev loss (dev has every direction).

Usage:
  python train.py                                              # config.yaml, one GPU
  torchrun --nproc_per_node=gpu train.py                       # all GPUs
  python train.py --set data.max_train_rows=20000 train.eval_steps=100   # quick test
  python train.py --resume                                     # continue from the last checkpoint
  bash run.sh                                                  # in the background with nohup (all GPUs)

Output (output.dir, default /data/experiments/riva_multilingual):
  final/             trained model + tokenizer  ->  python ../riva_test/riva_test.py --model <dir>/final
  logs/              nohup logs from run.sh
  checkpoint-*/      best + latest checkpoints
  config_used.yaml   the settings of this run (with the auto values filled in)

Requires: torch transformers accelerate datasets pyyaml
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, for general/
os.environ.setdefault("HF_DATASETS_CACHE", str(HERE / ".cache" / "datasets"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less memory fragmentation

import torch
import yaml

from general.config import load_config

LANG_NAMES = {"uz": "Uzbek", "ru": "Russian", "en": "English"}


def is_auto(v):
    return str(v).lower() == "auto"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (HERE / p).resolve()


def pick_precision(setting):
    """bf16 needs Ampere (compute capability 8.0) or newer on every visible GPU; older GPUs get fp16."""
    if not is_auto(setting):
        return str(setting).lower()
    cap = min(torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))
    return "bf16" if cap >= (8, 0) else "fp16"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_prompt(text, pair):
    s, t = (LANG_NAMES[x] for x in pair.split("-"))
    return (f"<s>System\nYou are an expert at translating text from {s} to {t}.</s>\n"
            f"<s>User\nWhat is the {t} translation of the sentence: {text.strip()}</s>\n"
            "<s>Assistant\n")


def known_pair(p):
    parts = p.split("-") if isinstance(p, str) else []
    return len(parts) == 2 and parts[0] != parts[1] and all(x in LANG_NAMES for x in parts)


def read_jsonl(path, pairs=None):
    """Rows with a known pair (and in `pairs`, if given); returns (rows, Counter of skipped pairs)."""
    rows, skipped = [], Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = r.get("pair")
            if not (r.get("source") and r.get("target")):
                continue
            if not known_pair(p) or (pairs is not None and p not in pairs):
                skipped[p] += 1
                continue
            rows.append({"pair": p, "source": r["source"].strip(), "target": r["target"].strip()})
    return rows, skipped


CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")
UZ_CYRILLIC = re.compile(r"[ўқғҳЎҚҒҲ]")


def script_ok(text, lang):
    """Is the text written in an alphabet this language uses? Only Latin vs Cyrillic is checked."""
    c, l = len(CYRILLIC.findall(text)), len(LATIN.findall(text))
    if c + l < 3:                  # numbers, initials, punctuation: nothing to check
        return True
    if lang == "ru":
        return c > l
    if lang == "en":
        return l > c
    return l > c or bool(UZ_CYRILLIC.search(text))   # uz: Latin, or Cyrillic with Uzbek-only letters


def row_ok(r):
    s, t = r["pair"].split("-")
    return script_ok(r["source"], s) and script_ok(r["target"], t)


def count_pairs(rows, pairs):
    return "  ".join(f"{p}: {sum(r['pair'] == p for r in rows):,}" for p in pairs)


def load_data(dcfg, seed, log):
    rng = random.Random(seed)
    want = None if is_auto(dcfg["pairs"]) else set(dcfg["pairs"])
    train, skipped = read_jsonl(resolve(dcfg["train"]), want)
    pairs = sorted({r["pair"] for r in train}) if want is None else sorted(want)
    if skipped:
        log("  skipped train rows with other pairs: " + "  ".join(f"{p}: {n:,}" for p, n in skipped.most_common()))
    log(f"  directions: {', '.join(pairs)}")
    dev = read_jsonl(resolve(dcfg["dev"]), set(pairs))[0] if dcfg.get("dev") else []
    test_path = resolve(dcfg["test"]) if dcfg.get("test") else None
    test = read_jsonl(test_path, set(pairs))[0] if test_path and test_path.exists() else []

    if dcfg["script_filter"]:
        for name, rows in (("train", train), ("dev", dev)):
            bad = Counter(r["pair"] for r in rows if not row_ok(r))
            if bad:
                log(f"  script filter dropped {sum(bad.values()):,} {name} rows (wrong alphabet): "
                    + "  ".join(f"{p}: {n:,}" for p, n in sorted(bad.items())))
        train = [r for r in train if row_ok(r)]
        dev = [r for r in dev if row_ok(r)]

    # a sentence seen in training must not be scored in dev/test
    held = {r["source"].lower() for r in dev + test} | {r["target"].lower() for r in dev + test}
    before = len(train)
    train = [r for r in train if r["source"].lower() not in held]
    if before != len(train):
        log(f"  dropped {before - len(train):,} train rows that also appear in dev/test")

    rng.shuffle(train)
    if dcfg.get("max_rows_per_pair"):
        cap, seen, kept = int(dcfg["max_rows_per_pair"]), Counter(), []
        for r in train:
            seen[r["pair"]] += 1
            if seen[r["pair"]] <= cap:
                kept.append(r)
        train = kept
    if dcfg.get("max_train_rows"):
        train = train[:int(dcfg["max_train_rows"])]
    if not dev:  # no dev file: hold out a small slice of train
        n = min(2000 * len(pairs), len(train) // 50)
        dev, train = train[:n], train[n:]

    # every direction gets the same number of dev rows, so dev loss is not dominated by the biggest one
    rng.shuffle(dev)
    per_pair, picked, kept = int(dcfg["dev_samples_per_pair"]), Counter(), []
    for r in dev:
        picked[r["pair"]] += 1
        if picked[r["pair"]] <= per_pair:
            kept.append(r)
    dev = kept
    for name, rows in (("train", train), ("dev", dev)):
        log(f"  {name:<5} {len(rows):>10,} rows  {count_pairs(rows, pairs)}")
    return train, dev, pairs


def auto_max_len(rows, tok, log, n=20000, pct=0.995):
    """Token length (prompt + target + </s>) that covers `pct` of a random sample of train rows."""
    sample = random.Random(0).sample(rows, min(n, len(rows)))
    p = tok([build_prompt(r["source"], r["pair"]) for r in sample], add_special_tokens=False)["input_ids"]
    t = tok([r["target"] for r in sample], add_special_tokens=False)["input_ids"]
    lens = sorted(len(a) + len(b) + 1 for a, b in zip(p, t))
    v = lens[min(len(lens) - 1, int(pct * len(lens)))]
    m = min(1024, max(128, math.ceil(v / 32) * 32))
    log(f"  max_len auto: {m}  (median {lens[len(lens) // 2]}, {pct:.1%} of rows <= {v} tokens)")
    return m


class Encoder:
    """prompt + target + </s>; labels are -100 on the prompt so the loss is only on the translation."""

    def __init__(self, tok, max_len):
        self.tok, self.max_len = tok, max_len
        end = tok.convert_tokens_to_ids("</s>")
        self.end = end if isinstance(end, int) and end != tok.unk_token_id else tok.eos_token_id

    def __call__(self, batch):
        prompts = [build_prompt(s, p) for s, p in zip(batch["source"], batch["pair"])]
        p_ids = self.tok(prompts, add_special_tokens=False)["input_ids"]
        t_ids = self.tok(batch["target"], add_special_tokens=False)["input_ids"]
        out = {"input_ids": [], "labels": [], "length": []}
        for p, t in zip(p_ids, t_ids):
            t = t[:max(1, self.max_len - len(p) - 1)] + [self.end]
            ids = (p + t)[:self.max_len]
            out["input_ids"].append(ids)
            out["labels"].append(([-100] * len(p) + t)[:self.max_len])
            out["length"].append(len(ids))
        return out


class Collator:
    def __init__(self, pad_id, multiple=8):
        self.pad_id, self.multiple = pad_id, multiple

    def __call__(self, features):
        n = math.ceil(max(len(f["input_ids"]) for f in features) / self.multiple) * self.multiple
        ids = torch.full((len(features), n), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(features), n), dtype=torch.long)
        labels = torch.full((len(features), n), -100, dtype=torch.long)
        for i, f in enumerate(features):
            k = len(f["input_ids"])
            ids[i, :k] = torch.tensor(f["input_ids"])
            mask[i, :k] = 1
            labels[i, :k] = torch.tensor(f["labels"])
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def probe_batch_size(model, collator, examples, fraction, amp_dtype, log):
    """Largest per-GPU batch (multiple of 8) whose full training step on the LONGEST examples
    (forward + backward + AdamW step) stays within `fraction` of GPU memory.
    The probe optimizer uses lr=0, so the weights are not changed."""
    device = next(model.parameters()).device
    total = torch.cuda.get_device_properties(device).total_memory
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0, weight_decay=0.0, fused=True)
    model.train()

    def fits(bs):
        batch = collator((examples * math.ceil(bs / len(examples)))[:bs])
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = model(**batch).loss
            loss.backward()
            opt.step()
            need = torch.cuda.max_memory_allocated(device)
            ok = need <= fraction * total
            log(f"  batch {bs:>4}: {need / 1024**3:5.0f} / {total / 1024**3:.0f} GB  {'ok' if ok else 'too big'}")
            return ok
        except torch.cuda.OutOfMemoryError:
            log(f"  batch {bs:>4}: out of memory")
            return False
        finally:
            del batch, loss
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    try:
        lo, bs = 0, 8                      # grow 8, 16, 32 ... until it doesn't fit
        while bs <= 512 and fits(bs):
            lo, bs = bs, bs * 2
        hi = bs
        if lo == 0:                        # not even 8: try 4, 2, 1
            lo = next((b for b in (4, 2, 1) if fits(b)), 0)
            if lo == 0:
                sys.exit("Even batch 1 does not fit: set train.grad_checkpointing=true or lower data.max_len")
            return lo
        while hi - lo > 8:                 # then narrow down in steps of 8
            mid = (lo + hi) // 2 // 8 * 8
            lo, hi = (mid, hi) if fits(mid) else (lo, mid)
        return lo
    finally:
        opt.state.clear()
        del opt
        torch.cuda.empty_cache()


def hours(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def make_trainer(lengths, **kwargs):
    from transformers import Trainer, TrainerCallback
    from transformers.trainer_pt_utils import LengthGroupedSampler

    class RivaTrainer(Trainer):
        # batches of similar length = little padding. Lengths are passed as a plain list:
        # transformers' built-in grouping reads them row by row (very slow on millions of rows).
        def _get_train_sampler(self, *args, **kw):
            return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                                        lengths=lengths)

    class TimeLeft(TrainerCallback):
        """With every log line: elapsed time, s/step and time left (eval + checkpoint time included, since it is
        averaged over the steps so far). The first few lines overestimate: the first steps are slower."""

        def on_train_begin(self, args, state, control, **kw):
            self.t0, self.step0 = time.time(), state.global_step  # step0 > 0 after --resume

        def on_log(self, args, state, control, logs=None, **kw):
            done = state.global_step - self.step0
            if not state.is_world_process_zero or done <= 0 or not state.max_steps:
                return
            elapsed = time.time() - self.t0
            left = (state.max_steps - state.global_step) * elapsed / done
            end = time.strftime("%m-%d %H:%M", time.localtime(time.time() + left))
            print(f"[time] step {state.global_step:,}/{state.max_steps:,} ({state.global_step / state.max_steps:.1%})"
                  f"  elapsed {hours(elapsed)}  {elapsed / done:.2f} s/step  left ~{hours(left)}  ends ~{end}",
                  flush=True)

    return RivaTrainer(callbacks=[TimeLeft()], **kwargs)


def training_args(log, **kw):
    """TrainingArguments with only the keys this transformers version knows (dropped ones are printed)."""
    import inspect

    from transformers import TrainingArguments

    known = inspect.signature(TrainingArguments.__init__).parameters
    dropped = [k for k, v in kw.items() if k not in known and v is not None]
    if dropped:
        log(f"  not supported by this transformers version, ignored: {', '.join(dropped)}")
    return TrainingArguments(**{k: v for k, v in kw.items() if k in known and v is not None})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune Riva-Translate on several language directions.")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config, e.g. train.lr=5e-6")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in output.dir")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    dcfg, tcfg = cfg["data"], cfg["train"]
    world = int(os.environ.get("WORLD_SIZE", 1))
    is_main = int(os.environ.get("RANK", 0)) == 0
    log = print if is_main else (lambda *a, **k: None)

    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found (check nvidia-smi and that torch is a CUDA build).")
    try:  # the Trainer needs torch <-> numpy at every eval: fail now, not after the first eval_steps
        torch.zeros(1).numpy()
    except RuntimeError as e:
        import numpy
        sys.exit(f"torch can't use numpy {numpy.__version__} ({e}). torch was built for NumPy 1.x: "
                 f'run  pip install "numpy<2"  in this venv, then start again.')
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        log(f"GPU {i}: {p.name}, {p.total_memory / 1024**3:.0f} GB, compute capability {p.major}.{p.minor}")
    log(f"processes: {world}  (torch {torch.__version__}, CUDA {torch.version.cuda})")

    precision = pick_precision(tcfg["precision"])
    if precision not in ("bf16", "fp16"):
        sys.exit(f"train.precision must be auto, bf16 or fp16, not {tcfg['precision']}")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    tf32 = min(torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())) >= (8, 0)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    tcfg["precision"] = precision
    log(f"precision: {precision} mixed{', tf32 matmuls' if tf32 else ''}")
    if is_auto(tcfg["num_workers"]):
        tcfg["num_workers"] = max(2, min(8, (os.cpu_count() or 8) // world))

    out_dir = resolve(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    from accelerate import PartialState
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(int(tcfg["seed"]))

    # ---------- data ----------
    log("\nLoading data ...")
    train_rows, dev_rows, pairs = load_data(dcfg, int(tcfg["seed"]), log)
    if not train_rows:
        sys.exit(f"No usable rows in {dcfg['train']}")
    dcfg["pairs"] = pairs

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if is_auto(dcfg["max_len"]):
        dcfg["max_len"] = auto_max_len(train_rows, tok, log)
    enc = Encoder(tok, int(dcfg["max_len"]))
    log(f"Tokenizing (answer ends with token id {enc.end}) ...")
    with PartialState().main_process_first():
        train_ds = Dataset.from_list(train_rows).map(
            enc, batched=True, remove_columns=["pair", "source", "target"],
            num_proc=min(32, os.cpu_count() or 1) if len(train_rows) > 50_000 else None, desc="tokenize train")
        dev_ds = Dataset.from_list(dev_rows).map(enc, batched=True, remove_columns=["pair", "source", "target"])
    del train_rows
    lengths = train_ds["length"]
    lengths = list(lengths) if not isinstance(lengths, list) else lengths
    cut = sum(n >= int(dcfg["max_len"]) for n in lengths)
    if cut:
        log(f"  {cut:,} train examples cut to max_len {dcfg['max_len']}")

    # ---------- model ----------
    log(f"\nLoading {cfg['model']} ...")
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32, attn_implementation="sdpa")
    if tcfg["grad_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    log(f"  training all {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters ({precision} mixed precision)")
    model.to(torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))))
    collator = Collator(tok.pad_token_id)

    if is_auto(tcfg["batch_size"]):
        log("\nFinding the largest batch that fits (longest examples) ...")
        longest = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)[:64]
        examples = [{k: train_ds[i][k] for k in ("input_ids", "labels")} for i in longest]
        bs = probe_batch_size(model, collator, examples, float(tcfg["memory_fraction"]), amp_dtype, log)
        if world > 1:  # GPUs may differ in free memory: every process uses the smallest batch
            t = torch.tensor([bs], device=next(model.parameters()).device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)  # group set up by PartialState
            bs = int(t.item())
    else:
        bs = int(tcfg["batch_size"])
    tcfg["batch_size"] = bs
    accum = max(1, round(int(tcfg["target_batch"]) / (bs * world)))
    per_step = bs * accum * world
    steps = math.ceil(len(train_ds) / per_step * float(tcfg["epochs"]))
    # warmup as a step count: newer transformers dropped warmup_ratio, and an unknown key is silently skipped
    warmup = math.ceil(float(tcfg["warmup_ratio"]) * steps)
    log(f"  {bs} x {accum} accum x {world} GPU = {per_step} sentences/step, {steps:,} steps, {warmup:,} warmup")
    if is_main:
        (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    targs = training_args(
        log, output_dir=str(out_dir), per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
        gradient_accumulation_steps=accum, num_train_epochs=float(tcfg["epochs"]),
        learning_rate=float(tcfg["lr"]), lr_scheduler_type=tcfg["scheduler"],
        warmup_steps=warmup, weight_decay=float(tcfg["weight_decay"]), max_grad_norm=1.0,
        optim="adamw_torch_fused", bf16=precision == "bf16", fp16=precision == "fp16", tf32=tf32 or None,
        eval_strategy="steps", eval_steps=int(tcfg["eval_steps"]),
        save_strategy="steps", save_steps=int(tcfg["eval_steps"]), save_total_limit=int(tcfg["save_total_limit"]),
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=int(tcfg["logging_steps"]), logging_first_step=True, report_to="none",
        disable_tqdm=not sys.stderr.isatty(),  # nohup log: no progress bar, [time] lines show the ETA instead
        seed=int(tcfg["seed"]), remove_unused_columns=False,
        dataloader_num_workers=int(tcfg["num_workers"]), dataloader_pin_memory=True,
        dataloader_persistent_workers=int(tcfg["num_workers"]) > 0,
        dataloader_prefetch_factor=4 if int(tcfg["num_workers"]) > 0 else None,
        ddp_find_unused_parameters=False if world > 1 else None,
    )
    trainer = make_trainer(lengths, model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                           data_collator=collator, processing_class=tok)

    # ---------- train ----------
    log("\nTraining ...")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    log(f"Training time: {hours(time.time() - t0)}")

    # ---------- save the best model ----------
    if trainer.is_world_process_zero():
        final = out_dir / "final"
        best = trainer.accelerator.unwrap_model(trainer.model)
        best.config.use_cache = True
        best.to(amp_dtype).save_pretrained(str(final), safe_serialization=True)
        tok.save_pretrained(str(final))
        log(f"\nSaved best model (dev loss {trainer.state.best_metric:.4f}) to {final}")
        log(f"Test it:  cd ../riva_test && python riva_test.py --model {final} "
            f"--test-file {resolve(dcfg['test'])} --pairs {' '.join(pairs)} --out multilingual.json")


if __name__ == "__main__":
    main()
