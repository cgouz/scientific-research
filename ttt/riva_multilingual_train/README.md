# riva_multilingual_train

Fine-tunes `nvidia/Riva-Translate-4B-Instruct-v2` on **Uzbek, Russian and English in every direction**
(`uz-ru`, `ru-uz`, `uz-en`, `en-uz`, `en-ru`, `ru-en`). It works the same way as [../riva_train](../riva_train/README.md),
but more is detected automatically:

| Setting (config.yaml) | `auto` means |
|---|---|
| `data.pairs` | every direction found in the train file (languages in `LANG_NAMES`: uz, ru, en) |
| `data.max_len` | token length that covers 99.5% of a 20k-row sample (multiple of 32, 128..1024) |
| `train.batch_size` | largest per-GPU batch whose full step fits in `memory_fraction` of GPU memory; with several GPUs, the smallest of them |
| `train.precision` | bf16 on Ampere or newer (A100, H100, B200, RTX 30xx/40xx), fp16 on older GPUs (V100, T4) |
| `train.num_workers` | CPU cores / GPUs, at most 8 |
| GPUs | `run.sh` starts `torchrun` on every GPU that `nvidia-smi` shows |

The values that were picked are written to `<output.dir>/config_used.yaml`. Every setting, and how epochs turn into
steps, is explained in [CONFIG.md](CONFIG.md).

## How to run

### 1. Get the code and the environment (once)

```sh
cd /workspace
git clone --recurse-submodules <repo-url> scientific-research    # or, if already cloned: cd scientific-research && git pull
cd /workspace/scientific-research

python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"   # must print True and your GPU count
hf auth login                                                     # only if the model download asks for it
```

In every new shell: `source /workspace/scientific-research/.venv/bin/activate`.

### 2. Check the data

```sh
ls /data/datasets/ttt/multilingual/                               # train.jsonl  dev.jsonl  test.jsonl
wc -l /data/datasets/ttt/multilingual/*.jsonl                     # row counts
python -c "
import json, collections
c = collections.Counter(json.loads(l)['pair'] for l in open('/data/datasets/ttt/multilingual/train.jsonl'))
print(c)"                                                         # rows per direction
```

Other paths? Edit `data.train` / `data.dev` / `data.test` in [config.yaml](config.yaml), or pass `--set data.train=/path/train.jsonl`.

### 3. Free the GPUs

```sh
cd /workspace/scientific-research/ttt/riva_multilingual_train
nvidia-smi
python ../../general/gpu_clean.py              # shows what holds GPU memory
python ../../general/gpu_clean.py --kill       # stops your own processes holding it (asks first)
```

### 4. Quick test (foreground, a few minutes)

Runs the whole pipeline on 20,000 rows, so mistakes show up before the long run:

```sh
python train.py --set data.max_train_rows=20000 train.eval_steps=50 output.dir=/data/experiments/riva_multilingual_test
```

Check in the output:
- `GPU 0: ...`, `precision: bf16 mixed` (or fp16 on old GPUs)
- `directions: en-ru, en-uz, ru-en, ru-uz, uz-en, uz-ru` and the row counts per direction
- `script filter dropped ...`: how many wrong-alphabet rows were removed
- `max_len auto: ...`, then `batch 8: ... ok`, `batch 16: ... ok`, ... and the chosen batch
- `... sentences/step, N steps`, then the loss going down

Stop it with Ctrl+C once you have seen a couple of evals. To test all GPUs too: `torchrun --nproc_per_node=gpu train.py --set ...` (same options).

### 5. Full training (background)

```sh
bash run.sh                                    # all GPUs, keeps running after you log out
```

It prints the pid, the log path and the stop command. Watch it:

```sh
tail -f /data/experiments/riva_multilingual/logs/train.log
grep -E "eval_loss" /data/experiments/riva_multilingual/logs/train.log      # dev loss over time
watch -n 5 nvidia-smi                                                        # GPU use and memory
```

Options are passed to `train.py`, e.g. `bash run.sh --set train.epochs=1 data.max_rows_per_pair=1000000`.

### 6. Stop / resume

```sh
pkill -f "(python|torchrun).* /workspace/scientific-research/ttt/riva_multilingual_train/train.py"   # stop (run.sh prints this)
bash run.sh --resume                           # continue from the latest checkpoint-*/
```

Resume with the same settings: changing the data or batch settings makes the step count no longer match the checkpoint.

### 7. Test the model

When training ends, the best checkpoint (lowest dev loss) is saved to `/data/experiments/riva_multilingual/final/`.
Translate the test set with the base and the fine-tuned model and compare:

```sh
cd ../riva_test
T=/data/datasets/ttt/multilingual/test.jsonl
P="uz-ru ru-uz uz-en en-uz en-ru ru-en"
bash run.sh --test-file $T --pairs $P --out multi_base.json
bash run.sh --model /data/experiments/riva_multilingual/final --test-file $T --pairs $P --out multi_finetuned.json
tail -f /data/experiments/riva/logs/test.log
# results: /data/experiments/riva/results/multi_base.json, multi_finetuned.json
```

`--samples 500` sets rows per direction (default 200).

## Output

Everything goes to `/data/experiments/riva_multilingual/` (`output.dir`):
- `checkpoint-*/`: best + latest checkpoints, used by `--resume`
- `final/`: best model + tokenizer
- `logs/`: `train_<time>.log`, with `train.log` pointing to the latest
- `config_used.yaml`: settings with the auto values filled in
- `train.pid`

## Problems

| Message | Fix |
|---|---|
| `Training is already running (pid ...)` | a run from this folder is alive: stop it (step 6) or wait |
| `No CUDA GPU found` | `nvidia-smi` must work, and torch must be a CUDA build (`pip install torch` for your CUDA from pytorch.org) |
| `Even batch 1 does not fit` | `--set train.grad_checkpointing=true`, or a smaller `data.max_len` (e.g. 256) |
| `CUDA out of memory` during training | `--set train.memory_fraction=0.75` (or a fixed `train.batch_size=16`) |
| loss is `nan` (fp16 GPUs) | `--set train.lr=5.0e-6`, or use a bf16 GPU |
| `--set: unknown setting` | the key must exist in config.yaml, e.g. `train.lr`, not `lr` |
| disk full | each checkpoint is ~50 GB; 2 are kept (`save_total_limit`) |

## Prompt

The prompt is the same as in `riva_test.py` and `riva_train`. The loss is computed only on the translation:
```
<s>System\nYou are an expert at translating text from Uzbek to English.</s>\n
<s>User\nWhat is the English translation of the sentence: {source}</s>\n
<s>Assistant\n{target}</s>
```
