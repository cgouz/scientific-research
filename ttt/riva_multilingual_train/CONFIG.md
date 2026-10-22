# How riva_multilingual_train is configured

All settings are in [config.yaml](config.yaml). This page explains each one and what `train.py` does with it.
See [README.md](README.md) for how to run it.

## Changing settings

- Edit `config.yaml`, or override any key for one run: `python train.py --set train.lr=5.0e-6 data.max_rows_per_pair=1000000`.
  `bash run.sh --set ...` passes the overrides on the same way.
- A key that isn't in `config.yaml` stops the run with an error, so typos don't go unnoticed.
- `--set` values are read as YAML: write floats as `5.0e-6` (YAML reads `5e-6` as a string; the script converts it anyway).
  Lists are written as `[uz-ru, ru-uz]`.
- Relative paths are resolved from this folder.
- At start, every `auto` value is replaced with the value that was picked, and the result is saved to
  `<output.dir>/config_used.yaml`. Use this file to see exactly what a run used.

## `model`

`nvidia/Riva-Translate-4B-Instruct-v2`: a Hugging Face id or a local folder. All ~4B parameters are trained (full fine-tuning, no LoRA).
The weights are loaded in fp32 and trained with mixed precision (`train.precision`), using PyTorch SDPA attention.

## `data`

| Key | Default | Meaning |
|---|---|---|
| `train` | `/data/datasets/ttt/multilingual/train.jsonl` | training rows |
| `dev` | `.../dev.jsonl` | rows for eval loss; picks the best checkpoint. If missing, a slice of train is held out (2000 rows per direction, at most 2% of train) |
| `test` | `.../test.jsonl` | not trained or scored here; only used to remove train rows that also appear in test |
| `pairs` | `auto` | directions to train. `auto` = every pair in the train file whose two languages are in `LANG_NAMES` (`uz`, `ru`, `en`). Or a list, e.g. `[uz-en, en-uz]` |
| `script_filter` | `true` | drop rows written in the wrong alphabet for their pair (see below) |
| `max_len` | `auto` | tokens per example (prompt + translation + `</s>`); longer examples are cut |
| `max_train_rows` | `null` | use only N random train rows (for quick tests) |
| `max_rows_per_pair` | `null` | keep at most N random rows per direction, so one large direction doesn't dominate |
| `dev_samples_per_pair` | `500` | dev rows per direction used for eval loss |
| `auto_source_ratio` | `0.0` | share of train and dev rows whose prompt leaves out the source language (`You are an expert at translating text to English.`), so the model learns to detect it. `config_auto.yaml` uses `0.5` |

Row format: `{"pair": "uz-en", "source": "...", "target": "...", "origin": "..."}`. Only `pair`, `source` and `target`
are used. Rows with an empty source/target, broken JSON, or an unknown pair (e.g. `uz-zh`) are skipped; the log shows how many per pair.

### Data pipeline (in this order)

1. Read train, dev and test, keeping only the chosen `pairs`.
2. **Script filter** (train and dev): count Latin and Cyrillic letters in each side.
   - `ru` must be mostly Cyrillic; `en` must be mostly Latin.
   - `uz` must be mostly Latin, **or** contain an Uzbek-only Cyrillic letter (ў қ ғ ҳ), so Uzbek written in Cyrillic is kept.
   - Text with fewer than 3 letters (numbers, initials) is not checked.
   - This catches rows like `en-uz` with a Russian source, or `uz-en` with a Russian target. It can't tell Uzbek
     from English (both Latin).
3. **Overlap removal**: a train row is dropped if its source (lower-cased) equals any source or target in dev or test.
4. Shuffle train (seed `train.seed`), then apply `max_rows_per_pair`, then `max_train_rows`.
5. Dev: shuffle and keep `dev_samples_per_pair` rows of each direction.
6. **Auto source** (`auto_source_ratio` > 0): that share of train and dev rows, chosen at random (seed), gets the prompt without the source language. Dev loss then covers both prompts.

The log prints the row count per direction for train and dev, so you can check the mix.

### `max_len: auto`

Tokenizes a random 20,000-row sample of train (prompt + target + `</s>`) and takes the length that 99.5% of rows fit in,
rounded up to a multiple of 32 and clamped to 128..1024. The log shows the median and the chosen value, plus how many
train examples were cut.

## `train`

| Key | Default | Meaning |
|---|---|---|
| `epochs` | `2` | passes over the train set |
| `lr` | `1.0e-5` | peak learning rate (AdamW, fused) |
| `scheduler` | `cosine` | LR schedule after warmup |
| `warmup_ratio` | `0.03` | share of steps used for linear warmup |
| `weight_decay` | `0.0` | |
| `precision` | `auto` | `bf16`, `fp16` or `auto` (see below) |
| `batch_size` | `auto` | sentences per GPU per forward pass |
| `memory_fraction` | `0.85` | for `batch_size: auto`: share of GPU memory one training step may use |
| `target_batch` | `256` | sentences per optimizer step across all GPUs |
| `grad_checkpointing` | `false` | `true` uses much less memory but is ~30% slower |
| `eval_steps` | `1000` | eval dev loss and save a checkpoint every N optimizer steps |
| `save_total_limit` | `2` | checkpoints kept on disk (best + latest) |
| `logging_steps` | `25` | log loss / LR every N steps |
| `num_workers` | `auto` | dataloader processes per GPU |
| `seed` | `42` | shuffling, sampling and training seed |

Gradient clipping is fixed at 1.0.

### `precision: auto`

Looks at the compute capability of every visible GPU and uses the lowest:
- **8.0 or newer** (A100, A10, L4, L40, H100, H200, B200, RTX 30xx/40xx/50xx): `bf16`, with TF32 matmuls turned on.
- **older** (V100, T4, RTX 20xx): `fp16` with loss scaling; TF32 off (these GPUs don't have it).

The final model is saved in the same dtype. fp16 can overflow on a model this size: if the loss turns into `nan`,
lower `lr` or use a bf16 GPU.

### `batch_size: auto`

Before training, on each GPU:
1. Take the 64 longest train examples (the worst case for memory).
2. Run a full training step (forward, backward, AdamW step with `lr=0`, so the weights don't change) with batch
   8, 16, 32 ... up to 512 until one doesn't fit within `memory_fraction` of GPU memory or runs out of memory.
3. Narrow down between the last batch that fit and the first that didn't, in steps of 8. If even 8 doesn't fit, try 4, 2, 1.
4. With several GPUs, every process uses the smallest result.

If even batch 1 doesn't fit, the run stops and asks for `grad_checkpointing: true` or a smaller `data.max_len`.

### Epochs and steps

You don't set the number of steps. You set `epochs` (how many times every train row is seen) and `target_batch`
(how many sentences one optimizer step should use). The script works out the rest:

```
batch_size        = auto (what fits on one GPU) or the number you set
grad_accum        = max(1, round(target_batch / (batch_size × GPUs)))     # forward/backward passes per step
sentences / step  = batch_size × grad_accum × GPUs                         # the real batch, close to target_batch
steps / epoch     = ceil(train_rows / sentences_per_step)
total steps       = ceil(train_rows / sentences_per_step × epochs)
warmup steps      = warmup_ratio × total steps
evals/checkpoints = total steps / eval_steps
```

One **step** = one optimizer update (`grad_accum` forward/backward passes on every GPU, then one AdamW update).
`eval_steps`, `logging_steps` and the LR schedule all count these steps.

**Example 1: 8 GPUs, 3,000,000 train rows (after filtering), auto batch found 64**
```
grad_accum       = max(1, round(256 / (64 × 8))) = max(1, round(0.5)) = 1
sentences / step = 64 × 1 × 8 = 512          # bigger than target_batch: 8 GPUs × 64 is already 512
steps / epoch    = ceil(3,000,000 / 512) = 5,860
total (2 epochs) = 11,719 steps,  warmup ≈ 352 steps,  eval + checkpoint ≈ 11 times
```

**Example 2: 1 GPU, same data, auto batch found 48**
```
grad_accum       = round(256 / 48) = round(5.33) = 5
sentences / step = 48 × 5 × 1 = 240
steps / epoch    = ceil(3,000,000 / 240) = 12,500
total (2 epochs) = 25,000 steps,  warmup = 750 steps,  eval + checkpoint = 25 times
```

The log prints this before training starts, e.g. `64 x 1 accum x 8 GPU = 512 sentences/step, 11,719 steps`.

How to change it:
- **Shorter or longer training:** change `epochs`. It may be a fraction: `--set train.epochs=0.5` stops halfway through the data.
- **Fewer, bigger steps:** raise `target_batch` (e.g. 512). If you double it, consider raising `lr` a little too.
  It can't go below `batch_size × GPUs` (accumulation is at least 1).
- **Eval more or less often:** `eval_steps`. Aim for 10–30 evals per run; every eval also saves a checkpoint (~50 GB: fp32 weights + AdamW state), and only `save_total_limit` of them are kept.
- **Fewer rows:** `max_rows_per_pair` or `max_train_rows` lower `train_rows`, which lowers steps per epoch.

Batches are built from examples of similar length, so there is little padding.

### `num_workers: auto`

`CPU cores / GPUs`, clamped to 2..8. Tokenizing the train set at start uses up to 32 processes (when there are over 50k rows).

## GPUs

- `bash run.sh` counts the GPUs with `nvidia-smi -L`: more than one → `torchrun --nproc_per_node=<count>`, one → plain `python`.
  To use only some GPUs: `CUDA_VISIBLE_DEVICES=0,1 bash run.sh` (the count still comes from `nvidia-smi`, so for this run
  `torchrun --nproc_per_node=2 train.py` directly).
- `run.sh` refuses to start if this folder's `train.py` is already running, and prints the `pkill` command to stop it.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set to reduce memory fragmentation.
- The datasets cache goes to `.cache/datasets/` in this folder.

## `output`

`dir: /data/experiments/riva_multilingual`

| Path | Content |
|---|---|
| `checkpoint-*/` | best (lowest dev loss) + latest checkpoint; `--resume` continues from the latest |
| `final/` | best model + tokenizer, for `riva_test.py --model` |
| `logs/train_<time>.log` | run.sh log; `logs/train.log` links to the latest |
| `config_used.yaml` | settings with the auto values filled in |
| `train.pid` | process id from run.sh |

## Prompt

Same as `riva_train` and `riva_test.py` (`build_prompt` and `LANG_NAMES` are copied in each, so change all three together).
Loss is only computed on the translation:
```
<s>System\nYou are an expert at translating text from {Source} to {Target}.</s>\n
<s>User\nWhat is the {Target} translation of the sentence: {source}</s>\n
<s>Assistant\n{target}</s>
```
To add a language: add it to `LANG_NAMES` in both `train.py` files and `riva_test.py`, add its pairs to `PAIRS` in
`riva_test.py`, and (if its alphabet isn't covered) extend `script_ok` or set `data.script_filter: false`.
