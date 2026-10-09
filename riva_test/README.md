# riva_test

Zero-shot test of `nvidia/Riva-Translate-4B-Instruct-v2` on uz -> ru, with **no training**.
It scores chrF / BLEU and saves every translation to `results/riva_zero_shot/`.

```sh
cd riva_test
python gpu_clean.py                          # check the GPUs are free
python riva_test.py 2>&1 | tee riva_test.log # 200 sentences from test_data/test.jsonl
python riva_test.py --samples 500 --test-file /data/til_uz-ru/test.jsonl
python riva_test.py --model ../ttt/outputs/riva-uz-ru/final   # test a fine-tuned model
```

## GPU cleaner (`gpu_clean.py`)

```sh
python gpu_clean.py                  # GPUs, memory used, which processes hold it
python gpu_clean.py --kill           # stop your processes holding GPU memory (asks first)
python gpu_clean.py --kill --gpu 0 -y
```

It only stops processes you own: SIGTERM first, then SIGKILL after 10 s.
`riva_test.py` frees its own CUDA memory when it finishes, even after an error.
