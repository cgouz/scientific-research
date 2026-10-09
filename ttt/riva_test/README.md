# riva_test

Checks how a Riva-Translate model translates **Uzbek → Russian**.
Each test sentence becomes one row in a JSONL file: source, reference target, and the model's translation.

```sh
cd ttt/riva_test
python gpu_clean.py                                                  # check the GPUs are free
python riva_test.py                                                  # base model, 200 sentences
python riva_test.py --test-file /data/til_uz-ru/test.jsonl --samples 500
python riva_test.py --model ../outputs/riva-uz-ru/final --out results/finetuned.jsonl
```

The test file is JSONL with `{"pair": "uz-ru", "source": <uz>, "target": <ru>}`.

## Result file (`--out`, default `results/riva_uz_ru.jsonl`)

One line per sentence:
```json
{"source": "Men har kuni ertalab kitob o'qiyman.", "target": "Я каждое утро читаю книгу.", "translation": "<model output>"}
```

## GPU cleaner (`gpu_clean.py`)

```sh
python gpu_clean.py                  # GPUs, memory used, which processes hold it
python gpu_clean.py --kill           # stop your processes holding GPU memory (asks first)
python gpu_clean.py --kill --gpu 0 -y
```

It only stops processes you own: SIGTERM first, then SIGKILL after 10 s.
`riva_test.py` frees its own CUDA memory when it finishes, even after an error.
