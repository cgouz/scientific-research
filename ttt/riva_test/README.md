# riva_test

Tests how well a Riva-Translate model translates **uz → ru** and **ru → uz**.
It scores chrF / BLEU and checks the output script. Scores and every translation go to one JSON file.

```sh
cd ttt/riva_test
python gpu_clean.py                                          # check the GPUs are free
python riva_test.py                                          # base model, both directions
python riva_test.py --test-file /data/til_uz-ru/test.jsonl --samples 500
python riva_test.py --model ../outputs/riva-uz-ru/final --out results/finetuned.json
python riva_test.py --directions ru-uz                       # one direction only
```

The test file is JSONL with `{"pair": "uz-ru", "source": <uz>, "target": <ru>}`. Every row is used
for both directions: ru → uz swaps source and target.

## Result file (`--out`, default `results/riva_test.json`)

```json
{
  "model": "...", "test_file": "...", "date": "...",
  "scores": {
    "uz-ru": {"chrf": 0.0, "bleu": 0.0, "target_script_pct": 0.0, "copied_input": 0, "empty": 0, "samples": 200, "seconds": 0.0},
    "ru-uz": {"...": "..."}
  },
  "translations": [{"direction": "uz-ru", "source": "...", "reference": "...", "hypothesis": "...", "origin": "..."}]
}
```

## GPU cleaner (`gpu_clean.py`)

```sh
python gpu_clean.py                  # GPUs, memory used, which processes hold it
python gpu_clean.py --kill           # stop your processes holding GPU memory (asks first)
python gpu_clean.py --kill --gpu 0 -y
```

It only stops processes you own: SIGTERM first, then SIGKILL after 10 s.
`riva_test.py` frees its own CUDA memory when it finishes, even after an error.
