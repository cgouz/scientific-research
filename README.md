# scientific-research

Experiments with speech and translation models for Uzbek and nearby languages:

| Folder | Task | Model |
|---|---|---|
| [stt/](stt/) | Speech-to-text (ASR): fine-tune and benchmark | `nvidia/nemotron-3.5-asr-streaming-0.6b` (NeMo) |
| [ttt/](ttt/) | Text-to-text translation, Uzbek ↔ Russian | `nvidia/Riva-Translate-4B-Instruct-v2` (transformers) |
| [tts/](tts/) | Text-to-speech | not started yet |
| [general/](general/README.md) | Shared helpers: GPU cleaner, config loading with `--set`, JSONL read/write | – |
| [hugging_face/](hugging_face/setup.md) | Notes on the `hf` CLI: login, browsing and downloading datasets | – |

Datasets and experiment outputs are kept outside the repo, under `/data/` (see [Data layout](#data-layout)).

## Setup

Python 3.10–3.12, Linux, NVIDIA GPU.

```sh
git clone --recurse-submodules <repo-url>     # or, in an existing clone: git submodule update --init
cd scientific-research

python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
pip check                                     # should print "No broken requirements found"

python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

If CUDA shows `False`, reinstall torch with the command for your CUDA version from
[pytorch.org](https://pytorch.org/get-started/locally/). To download gated models and datasets,
log in with `hf auth login` (see [hugging_face/setup.md](hugging_face/setup.md)).

## STT: Nemotron ASR (multilingual)
- [stt/nemotron_train/](stt/nemotron_train/README.md): fine-tune on English, Uzbek, Russian, Karakalpak and Korean from
  `/data/datasets/stt`. It checks and fixes the tokenizer, mixes languages by temperature, trains mostly with the `auto`
  prompt so the model detects the language itself, uses all GPUs and runs in the background.
- [stt/nemotron_test/](stt/nemotron_test/README.md): transcribe audio files, or measure WER/CER per language for the
  base or fine-tuned model

```sh
bash stt/nemotron_train/run.sh                                         # nohup, all GPUs -> /data/experiments/nemotron
bash stt/nemotron_test/run.sh --bench --split test --prompt both --out finetuned.json
python stt/nemotron_test/nemotron_test.py --audio a.wav               # language detected automatically
```

## TTT: Riva-Translate Uzbek ↔ Russian

| Path | What it does |
|---|---|
| [ttt/riva_train/](ttt/riva_train/README.md) | Fine-tunes on both directions. Finds the batch size automatically, uses all GPUs via `torchrun` and runs in the background |
| [ttt/riva_multilingual_train/](ttt/riva_multilingual_train/README.md) | Same, for uz / ru / en in all 6 directions (`/data/datasets/ttt/multilingual`). Pairs, max length, batch size, precision and workers are `auto` |
| [ttt/riva_test/](ttt/riva_test/README.md) | Translates a test set with the base or fine-tuned model. |
| [ttt/til_to_manifest.py](ttt/til_to_manifest.py) | TIL corpus → JSONL: detokenizes, cleans, removes train/test overlap, optional LaBSE filter |
| [ttt/uzlpc_to_manifest.py](ttt/uzlpc_to_manifest.py) | [blinoff/UzLPC](https://huggingface.co/datasets/blinoff/UzLPC) (en/ru → uz) → JSONL |
| `ttt/third_party/llama.cpp` | git submodule: [llama.cpp](https://github.com/ggml-org/llama.cpp) |

Typical workflow:

```sh
# 1. Build manifests (output defaults to data/external/...; copy or point config.yaml at them)
python ttt/til_to_manifest.py --data-dir data/uz-ru

# 2. Baseline
cd ttt/riva_test
python ../../general/gpu_clean.py
bash run.sh --test-file /data/datasets/ttt/uz-ru/test.jsonl --out base.json

# 3. Fine-tune (edit ttt/riva_train/config.yaml first)
cd ../riva_train
python train.py --set data.max_train_rows=20000 train.eval_steps=100   # quick test
bash run.sh                                                            # full run
tail -f /data/experiments/riva/logs/train.log

# 4. Compare
cd ../riva_test
bash run.sh --model /data/experiments/riva/final --test-file /data/datasets/ttt/uz-ru/test.jsonl --out finetuned.json
```

Each translation manifest row looks like this:

```json
{"pair": "uz-ru", "source": "Men har kuni ertalab kitob o'qiyman.", "target": "Я каждое утро читаю книгу."}
```

## Data layout

Not in git (see [.gitignore](.gitignore)):

```
/data/datasets/stt/<language>/{train,dev}.jsonl     # NeMo ASR manifests, language in the `locale` field
/data/datasets/ttt/uz-ru/{train,dev,test}.jsonl     # translation manifests
/data/experiments/nemotron/                         # final.nemo, last.nemo, checkpoints/, tb/, logs/, results/
/data/experiments/riva/                             # checkpoint-*/, final/, logs/, results/
```
