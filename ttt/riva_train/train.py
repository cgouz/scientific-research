"""
Fine-tune nvidia/Riva-Translate-4B-Instruct-v2 on Uzbek -> Russian and Russian -> Uzbek.

Data: JSONL rows {"pair": "uz-ru" | "ru-uz", "source": .., "target": ..}; every row is trained in
the direction of its own "pair". All settings live in config.yaml.

Prompt (identical to ../riva_test/riva_test.py), loss only on the translation:
    <s>System\\nYou are an expert at translating text from {src} to {tgt}.</s>\\n
    <s>User\\nWhat is the {tgt} translation of the sentence: {source}</s>\\n
    <s>Assistant\\n{target}</s>

Full fine-tuning in bf16, best checkpoint picked by dev loss.

Usage:
  python train.py                                              # config.yaml, one GPU
  torchrun --nproc_per_node=gpu train.py                       # all GPUs
  python train.py --set data.max_train_rows=20000 train.eval_steps=100   # quick test
  python train.py --resume                                     # continue from the last checkpoint
  bash run.sh                                                  # in the background with nohup (all GPUs)

Output (output.dir, default /data/experiments/riva):
  final/             trained model + tokenizer  ->  python ../riva_test/riva_test.py --model <dir>/final
  logs/              nohup logs from run.sh
  checkpoint-*/      best + latest checkpoints
  config_used.yaml   the settings of this run

Requires: torch transformers accelerate datasets pyyaml
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_DATASETS_CACHE", str(HERE / ".cache" / "datasets"))

import torch
import yaml

LANG_NAMES = {"uz": "Uzbek", "ru": "Russian"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path, overrides):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for item in overrides:  # --set train.lr=5e-6 data.max_train_rows=1000
        key, sep, val = item.partition("=")
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if not sep or not isinstance(node, dict) or leaf not in node:
            sys.exit(f"--set: unknown setting '{key}' (see config.yaml)")
        node[leaf] = yaml.safe_load(val)
    return cfg


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (HERE / p).resolve()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_prompt(text, pair):
    s, t = (LANG_NAMES[x] for x in pair.split("-"))
    return (f"<s>System\nYou are an expert at translating text from {s} to {t}.</s>\n"
            f"<s>User\nWhat is the {t} translation of the sentence: {text.strip()}</s>\n"
            "<s>Assistant\n")


def read_jsonl(path, pairs):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("pair") in pairs and r.get("source") and r.get("target"):
                rows.append({"pair": r["pair"], "source": r["source"].strip(), "target": r["target"].strip()})
    return rows


def load_data(dcfg, seed, log):
    pairs = set(dcfg["pairs"])
    rng = random.Random(seed)
    train = read_jsonl(resolve(dcfg["train"]), pairs)
    dev = read_jsonl(resolve(dcfg["dev"]), pairs) if dcfg.get("dev") else []
    test = read_jsonl(resolve(dcfg["test"]), pairs) if dcfg.get("test") and resolve(dcfg["test"]).exists() else []

    # a sentence seen in training must not be scored in dev/test
    held = {r["source"].lower() for r in dev + test} | {r["target"].lower() for r in dev + test}
    before = len(train)
    train = [r for r in train if r["source"].lower() not in held]
    if before != len(train):
        log(f"  dropped {before - len(train):,} train rows that also appear in dev/test")

    rng.shuffle(train)
    if dcfg.get("max_train_rows"):
        train = train[:int(dcfg["max_train_rows"])]
    if not dev:  # no dev file: hold out a small slice of train
        n = min(2000, len(train) // 50)
        dev, train = train[:n], train[n:]
    rng.shuffle(dev)
    dev = dev[:int(dcfg["dev_samples"])]
    for name, rows in (("train", train), ("dev", dev)):
        counts = {p: sum(r["pair"] == p for r in rows) for p in sorted(pairs)}
        log(f"  {name:<5} {len(rows):>10,} rows  " + "  ".join(f"{p}: {n:,}" for p, n in counts.items()))
    return train, dev


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


def make_trainer(lengths, **kwargs):
    from transformers import Trainer
    from transformers.trainer_pt_utils import LengthGroupedSampler

    class RivaTrainer(Trainer):
        # batches of similar length = little padding. Lengths are passed as a plain list:
        # transformers' built-in grouping reads them row by row (very slow on millions of rows).
        def _get_train_sampler(self, *args, **kw):
            return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                                        lengths=lengths)

    return RivaTrainer(**kwargs)


def training_args(**kw):
    """TrainingArguments with only the keys this transformers version knows."""
    import inspect

    from transformers import TrainingArguments

    known = inspect.signature(TrainingArguments.__init__).parameters
    return TrainingArguments(**{k: v for k, v in kw.items() if k in known and v is not None})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune Riva-Translate on uz<->ru.")
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        log(f"GPU {i}: {p.name}, {p.total_memory / 1024**3:.0f} GB")
    log(f"processes: {world}  (torch {torch.__version__}, CUDA {torch.version.cuda})")

    out_dir = resolve(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    from accelerate import PartialState
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(int(tcfg["seed"]))

    # ---------- data ----------
    log("\nLoading data ...")
    train_rows, dev_rows = load_data(dcfg, int(tcfg["seed"]), log)
    if not train_rows:
        sys.exit(f"No {dcfg['pairs']} rows in {dcfg['train']}")

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    enc = Encoder(tok, int(dcfg["max_len"]))
    log(f"Tokenizing (answer ends with token id {enc.end}) ...")
    with PartialState().main_process_first():
        train_ds = Dataset.from_list(train_rows).map(
            enc, batched=True, remove_columns=["pair", "source", "target"],
            num_proc=min(32, os.cpu_count() or 1) if len(train_rows) > 50_000 else None, desc="tokenize train")
        dev_ds = Dataset.from_list(dev_rows).map(enc, batched=True, remove_columns=["pair", "source", "target"])
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
    log(f"  training all {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters (bf16 mixed precision)")

    bs, accum = int(tcfg["batch_size"]), int(tcfg["grad_accum"])
    per_step = bs * accum * world
    steps = math.ceil(len(train_ds) / per_step * float(tcfg["epochs"]))
    log(f"  {bs} x {accum} accum x {world} GPU = {per_step} sentences/step, {steps:,} steps")

    targs = training_args(
        output_dir=str(out_dir), per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
        gradient_accumulation_steps=accum, num_train_epochs=float(tcfg["epochs"]),
        learning_rate=float(tcfg["lr"]), lr_scheduler_type=tcfg["scheduler"],
        warmup_ratio=float(tcfg["warmup_ratio"]), weight_decay=float(tcfg["weight_decay"]), max_grad_norm=1.0,
        optim="adamw_torch_fused", bf16=True, tf32=True,
        eval_strategy="steps", eval_steps=int(tcfg["eval_steps"]),
        save_strategy="steps", save_steps=int(tcfg["eval_steps"]), save_total_limit=int(tcfg["save_total_limit"]),
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=int(tcfg["logging_steps"]), logging_first_step=True, report_to="none",
        seed=int(tcfg["seed"]), remove_unused_columns=False,
        dataloader_num_workers=int(tcfg["num_workers"]), dataloader_pin_memory=True,
        ddp_find_unused_parameters=False if world > 1 else None,
    )
    trainer = make_trainer(lengths, model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                           data_collator=Collator(tok.pad_token_id), processing_class=tok)

    # ---------- train ----------
    log("\nTraining ...")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    log(f"Training time: {(time.time() - t0) / 3600:.2f} h")

    # ---------- save the best model ----------
    if trainer.is_world_process_zero():
        final = out_dir / "final"
        best = trainer.accelerator.unwrap_model(trainer.model)
        best.config.use_cache = True
        best.to(torch.bfloat16).save_pretrained(str(final), safe_serialization=True)
        tok.save_pretrained(str(final))
        log(f"\nSaved best model (dev loss {trainer.state.best_metric:.4f}) to {final}")
        log(f"Test it:  cd ../riva_test && python riva_test.py --model {final} "
            f"--test-file {resolve(dcfg['test'])} --out finetuned.json")


if __name__ == "__main__":
    main()
