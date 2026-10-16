# nemotron_test

Checks how a Nemotron ASR model transcribes. It can transcribe single audio files, or measure **WER per language** on a
manifest split. zh/ja/ko get CER instead of WER. By default every language is transcribed with the **`auto` prompt**,
so the model has to detect the language itself, the way it is used in practice.
- `--prompt locale`: use each language's own prompt (`uz-UZ`, `ru-RU`, ...) instead.
- `--prompt both`: run both and report how often the outputs are identical. Close to 100% means the model detects the
  language as well as if you had told it.

```sh
cd stt/nemotron_test
python ../../general/gpu_clean.py                                        # check the GPUs are free
python nemotron_test.py --audio a.wav b.wav                              # transcribe; language detected (--lang uz-UZ forces it)
python nemotron_test.py --bench                                          # dev WER with auto, /data/experiments/nemotron/final.nemo
bash run.sh --bench --split test --prompt both --out finetuned.json       # background (nohup)
bash run.sh --model nvidia/nemotron-3.5-asr-streaming-0.6b --bench --split test --out base.json
tail -f /data/experiments/nemotron/logs/test.log
```

A relative `--out` is saved under `/data/experiments/nemotron/results/`. Options:
- `--model`: a `.nemo` file or a Hugging Face model name (default `/data/experiments/nemotron/final.nemo`)
- `--root /data/datasets/stt --split test`: reads `<root>/<language>/<split>.jsonl`
- `--langs uzbek russian`: only these language folders
- `--n 500`: rows per language (every N-th row, the same rows on every run)
- `--show 3`: reference/hypothesis examples printed per language
- `--batch 16`

The checkpoint is picked on dev, so use `--split test` for a fair score.

Text is lower-cased and whitespace-normalized before scoring. Punctuation is kept.

## Result file (`--out`)

```json
{
  "uzbek":   {"locale": "uz-UZ", "rows": 200, "metric": "WER", "rate_auto": 0.1301, "rate_locale": 0.1234, "agree": 0.91, "rate": 0.1301},
  "korean":  {"locale": "ko-KR", "rows": 200, "metric": "CER", "rate_auto": 0.0712, "rate": 0.0712}
}
```
`rate_<prompt>` is there for each prompt that was run, and `agree` only with `--prompt both`. `rate` is the auto score
when auto was run, and it is the score used for the best/worst/average summary.
