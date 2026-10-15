# scientific-research

Experiments with speech and translation models for Uzbek and nearby languages:

| Folder | Task | Model |
|---|---|---|
| [stt/](stt/) | Speech-to-text (ASR): fine-tune and benchmark | `nvidia/nemotron-3.5-asr-streaming-0.6b` (NeMo) |
| [ttt/](ttt/) | Text-to-text translation, Uzbek ↔ Russian | `nvidia/Riva-Translate-4B-Instruct-v2` (transformers) |
| [tts/](tts/) | Text-to-speech | not started yet |
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

## STT: Nemotron ASR

[stt/train_nemotron.py](stt/train_nemotron.py) fine-tunes Nemotron ASR on several languages at once:
English, Uzbek, Russian, Karakalpak and Korean. All settings are in the `SOZLAMALAR` block at the top of
the file (batch duration, LR, epochs, languages, share of rows trained with the `auto` language prompt, and so on).
Before training it checks that the tokenizer gives no `<unk>` and can fix the manifests itself (`AUTO_FIX`).

[stt/test.py](stt/test.py) transcribes audio files, or measures WER for each language on dev/test
(CER for zh/ja/ko).

```sh
cd stt
python train_nemotron.py                                       # outputs -> /data/experiments/nemotron-multi

python test.py --audio a.wav b.wav --lang uz-UZ                # transcribe files
python test.py --bench --root /data/datasets/stt --split dev --n 200 --auto --json wer.json
```

`test.py` loads `/data/experiments/nemotron-multi/nemotron-multi-best.nemo` by default (change it with `--model`).
Its default `--root` is `/data/datasets/sst`, while training reads `/data/datasets/stt`, so pass `--root` explicitly.

## TTT: Riva-Translate Uzbek ↔ Russian

| Path | What it does |
|---|---|
| [ttt/riva_train/](ttt/riva_train/README.md) | Fine-tunes on both directions. Finds the batch size automatically, uses all GPUs via `torchrun` and runs in the background |
| [ttt/riva_test/](ttt/riva_test/README.md) | Translates a test set with the base or fine-tuned model. Also has `gpu_clean.py` to see and free GPU memory |
| [ttt/til_to_manifest.py](ttt/til_to_manifest.py) | TIL corpus → JSONL: detokenizes, cleans, removes train/test overlap, optional LaBSE filter |
| [ttt/uzlpc_to_manifest.py](ttt/uzlpc_to_manifest.py) | [blinoff/UzLPC](https://huggingface.co/datasets/blinoff/UzLPC) (en/ru → uz) → JSONL |
| `ttt/third_party/llama.cpp` | git submodule: [llama.cpp](https://github.com/ggml-org/llama.cpp) |

Typical workflow:

```sh
# 1. Build manifests (output defaults to data/external/...; copy or point config.yaml at them)
python ttt/til_to_manifest.py --data-dir data/uz-ru

# 2. Baseline
cd ttt/riva_test
python gpu_clean.py
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
/data/experiments/nemotron-multi/                   # ASR checkpoints, .nemo, TensorBoard logs
/data/experiments/riva/                             # checkpoint-*/, final/, logs/, results/
```
