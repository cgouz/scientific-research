# riva_test

Checks how a Riva-Translate model translates **Uzbek → Russian** and **Russian → Uzbek**.
Each test row is translated in the direction given by its `pair`. Each result row holds the source, the reference target and the model's translation.

```sh
cd ttt/riva_test
python gpu_clean.py                                                  # check the GPUs are free
python riva_test.py --test-file /data/datasets/ttt/uz-ru/test.jsonl --out results/base.json
python riva_test.py --test-file /data/datasets/ttt/uz-ru/test.jsonl --samples 500   # 500 per direction
python riva_test.py --test-file /data/datasets/ttt/uz-ru/test.jsonl --pairs ru-uz   # one direction
python riva_test.py --model ../outputs/riva-uz-ru/final --test-file /data/datasets/ttt/uz-ru/test.jsonl --out results/finetuned.json
```

The test file is JSONL with rows `{"pair": "uz-ru" | "ru-uz", "source": ..., "target": ...}`.

## Result file (`--out`)

`.json` gives one JSON list; `.jsonl` gives one row per line:
```json
[
  {"pair": "uz-ru", "source": "Men har kuni ertalab kitob o'qiyman.", "target": "Я каждое утро читаю книгу.", "translation": "<model output>"},
  {"pair": "ru-uz", "source": "Я каждое утро читаю книгу.", "target": "Men har kuni ertalab kitob o'qiyman.", "translation": "<model output>"}
]
```

## GPU cleaner (`gpu_clean.py`)

```sh
python gpu_clean.py                  # GPUs, memory used, which processes hold it
python gpu_clean.py --kill           # stop your processes holding GPU memory (asks first)
python gpu_clean.py --kill --gpu 0 -y
```

It only stops processes you own: SIGTERM first, then SIGKILL after 10 s.
`riva_test.py` frees its own CUDA memory when it finishes, even after an error.
