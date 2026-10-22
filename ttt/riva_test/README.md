# riva_test

Checks how a Riva-Translate model translates **Uzbek → Russian** and **Russian → Uzbek**.
Each test row is translated in the direction given by its `pair`. Each result row holds the source, the reference target and the model's translation.

```sh
cd ttt/riva_test
python ../../general/gpu_clean.py                                    # check the GPUs are free
python riva_test.py --test-file /data/datasets/ttt/uz-ru/test.jsonl --out base.json   # foreground
bash run.sh --test-file /data/datasets/ttt/uz-ru/test.jsonl --out base.json           # background (nohup)
bash run.sh --model /data/experiments/riva/final --test-file /data/datasets/ttt/uz-ru/test.jsonl --out finetuned.json
tail -f /data/experiments/riva/logs/test.log
```

A relative `--out` is saved under `/data/experiments/riva/results/`. Options:
- `--samples 500`: rows per direction
- `--pairs ru-uz`: one direction only
- `--batch-size 256`

The test file is JSONL with rows `{"pair": "uz-ru" | "ru-uz", "source": ..., "target": ...}`. All six uz / ru / en
directions work (`--pairs uz-ru ru-uz uz-en en-uz en-ru ru-en`).

## Scores

After translating, it prints corpus BLEU and chrF++ ([sacrebleu](https://github.com/mjpost/sacrebleu), `pip install sacrebleu`)
per direction, plus their unweighted mean, and saves them to `<out>.scores.json`. To compare result files you already have
(no GPU, a few seconds):

```sh
python riva_test.py --score multi_base.json multi_finetuned.json
```
```
         rows |   multi_base    | multi_finetuned |     change
              |   BLEU   chrF++ |   BLEU   chrF++ |   BLEU   chrF++
en-ru     500 |   ...      ...  |   ...      ...  |   +...     +...
...
mean     3000 |   ...      ...  |   ...      ...  |   +...     +...
```
chrF++ is the more reliable of the two for Uzbek (rich morphology: BLEU punishes every wrong word ending as a full miss).
Compare base and fine-tuned on the same test file, `--samples` and `--seed`, so they translate the same sentences.

## Result file (`--out`, default `/data/experiments/riva/results/riva_test.json`)

`.json` gives one JSON list; `.jsonl` gives one row per line:
```json
[
  {"pair": "uz-ru", "source": "Men har kuni ertalab kitob o'qiyman.", "target": "Я каждое утро читаю книгу.", "translation": "<model output>"},
  {"pair": "ru-uz", "source": "Я каждое утро читаю книгу.", "target": "Men har kuni ertalab kitob o'qiyman.", "translation": "<model output>"}
]
```

## GPU memory

`riva_test.py` frees its own CUDA memory when it finishes, even after an error. To see or stop other processes
holding the GPUs, use [general/gpu_clean.py](../../general/README.md#gpu-cleaner-gpu_cleanpy).
