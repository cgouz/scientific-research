# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research scripts (not a package) for fine-tuning and benchmarking NVIDIA speech/translation models for Uzbek and nearby languages:

- `stt/` — ASR with `nvidia/nemotron-3.5-asr-streaming-0.6b` (NeMo + Lightning). Multilingual: en, uz, ru, Karakalpak, ko.
- `ttt/` — translation with `nvidia/Riva-Translate-4B-Instruct-v2` (transformers). `riva_train/` is Uzbek ↔ Russian; `riva_multilingual_train/` is a copy of it for uz/ru/en in all 6 directions with `auto` settings (pairs, max_len, batch size, bf16/fp16 by GPU arch, workers) and an alphabet filter for mislabeled rows. `data.auto_source_ratio` (used by its `config_auto.yaml`, which continues from the finished model) trains part of the rows with a prompt that omits the source language, so the model detects it; `riva_test.py --prompt auto|both` evaluates that.
- `translate_app/` — stdlib-only web app (`app.py` + `static/index.html`) that starts `llama-server` with a Riva GGUF on GPU or CPU (`config.yaml: device`) and sends the training prompt; its `build_prompt` is another copy that must match.
- `tts/` — empty placeholder.
- `general/` — shared helpers (`config.py`, `jsonl.py`, `gpu_clean.py`).

There is no test suite, linter, or build step. "Testing" means running the `*_test.py` benchmark scripts on a GPU machine. All work assumes Linux + NVIDIA GPU; datasets and outputs live outside the repo under `/data/` (some older scripts/guides reference `/workspace/...`, the path on the training server).

## Setup

```sh
git submodule update --init            # ttt/third_party/llama.cpp (used only for GGUF conversion)
python -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

Every train/test folder has the same shape: `train.py` or `*_test.py`, a `config.yaml` (train only), and a `run.sh` that launches the script in the background with `nohup`, writing logs to `<output.dir>/logs/train.log` (symlink to latest) and refusing to start if a run is already alive.

```sh
# Translation (ttt)
python ttt/riva_train/train.py --set data.max_train_rows=20000 train.eval_steps=100   # quick smoke run
bash ttt/riva_train/run.sh [--resume] [--set key=val ...]                             # full run, torchrun on all GPUs
bash ttt/riva_test/run.sh --model /data/experiments/riva/final --test-file /data/datasets/ttt/uz-ru/test.jsonl --out finetuned.json

# ASR (stt)
python stt/nemotron_train/train.py --set data.langs=[uzbek] train.max_steps=500 train.val_every=100   # quick test
bash stt/nemotron_train/run.sh
python stt/nemotron_test/nemotron_test.py --bench --split test --prompt both --out finetuned.json
python stt/nemotron_test/nemotron_test.py --audio a.wav            # language auto-detected

# GPU memory
python general/gpu_clean.py [--kill [--gpu 0] [-y]]
```

A relative `--out` in the test scripts is written under `/data/experiments/{riva,nemotron}/results/`.

## Architecture / conventions that span files

- **Imports of `general/`**: scripts are run directly (not as modules) and do `sys.path.insert(0, str(HERE.parents[1]))` to put the repo root on the path, then `from general.config import load_config`. Keep this pattern in new scripts under `<task>/<folder>/`.
- **Config**: all training settings live in each folder's `config.yaml`; `--set a.b=value` overrides any existing key (unknown keys exit with an error). Values are parsed as YAML, so write floats as `5.0e-6` (PyYAML reads `5e-6` as a string); scripts call `float()` on them anyway. Relative paths in a config resolve against that config's folder. Each run saves `config_used.yaml` to its output dir.
- **Riva prompt must match across `ttt/riva_train/train.py`, `ttt/riva_multilingual_train/train.py`, `ttt/riva_test/riva_test.py` and `translate_app/app.py`** (`build_prompt` and `LANG_NAMES`, duplicated in each). Change both together, or base vs fine-tuned results stop being comparable. Loss is computed only on the target tokens.
- **Translation manifest format** (all of `ttt/`): JSONL rows `{"pair": "uz-ru", "source": ..., "target": ...}`. Each row is trained/translated in the direction of its own `pair`; bidirectional datasets contain both `uz-ru` and `ru-uz` rows. Training drops train rows that also appear in the test file.
- **ASR manifest format**: NeMo JSONL at `/data/datasets/stt/<language>/{train,dev}.jsonl` with `audio_filepath`, `duration`, `text`, `locale` (e.g. `uz-UZ`). The locale selects the model's language prompt; most training uses the `auto` prompt (`data.auto_lang_ratio`) so the model learns language detection, and the best checkpoint is picked by dev WER under `auto`. The tokenizer check may rewrite manifests in place (originals kept as `*.jsonl.bak`).
- **Multi-GPU differs per task**: Riva uses `torchrun --nproc_per_node=N`; Nemotron must be started with plain `python` because Lightning spawns the per-GPU processes itself after manifests are prepared.
- Riva batch size defaults to `auto` (probes the largest batch that fits using the longest examples; `train.memory_fraction`), with grad accumulation derived from `train.target_batch`.
- Test scripts call `free_gpu()` on exit, even after errors.
- `ttt/making_mainfest/` holds standalone dataset builders (HF datasets, WMT19, TIL corpus, merging/dedup with sqlite, picking top-N pairs). They don't use `general/`, have hard-coded input/output paths in their docstrings, and their comments/docstrings are mostly in Uzbek.

## Outputs

```
/data/experiments/riva/       checkpoint-*/, final/, logs/, results/, config_used.yaml
/data/experiments/nemotron/   final.nemo (best dev WER), last.nemo, checkpoints/, tb/, logs/, results/
```

GGUF conversion and llama.cpp inference of the trained Riva model are documented in `riva_translate_4b_to_gguf_guide.md` and `riva_translate_4b_gguf_run_and_test_guide.md` (separate venv).
