# nemotron_train

Fine-tunes `nvidia/nemotron-3.5-asr-streaming-0.6b` on several languages at once: **English, Uzbek, Russian,
Karakalpak, Korean**. Training runs in bf16, and the best checkpoint is chosen by dev WER.

Before training, the script does the following:
- **Tokenizer check.** If many rows tokenize to `<unk>`, training does not start. With `tokenizer.auto_fix`, it finds a
  replacement for each unknown character, rewrites the manifests and keeps the originals as `<split>.jsonl.bak`.
- **Language mix.** Languages are sampled by temperature, `p ~ (hours / total) ** (1/T)`, so small languages are seen more
  often (`data.temperature`).
- **Language detection (main mode).** 70% of every language (`data.auto_lang_ratio`) is trained with the `auto` prompt,
  so the model learns to find the language itself. The other 30% keep their own locale, so you can still force a
  language with `uz-UZ`, `ru-RU` and so on. `1.0` means auto only, but then forcing is no longer trained and new
  languages never get their own prompt.
- **New languages.** A language the model doesn't know (for example Karakalpak) gets a free prompt id (`data.extra_prompts`).
- **Dev set.** Every N-th row of each language goes into one dev file, up to `data.dev_max_per_lang` rows. The dev rows
  are the same on every run. They are scored with the `auto` prompt (`data.dev_prompt: auto`), so `final.nemo` is the
  checkpoint that detects the language best. Set it to `locale` to score with each language's own prompt, or `both` for
  each row twice.

Batches are sized by seconds of audio, not by number of rows (`train.batch_duration`, Lhotse). If you run out of
memory, lower it. With several GPUs, `bash run.sh` uses all of them. Lightning starts one process per GPU, so run it
with `python`, not `torchrun`.

Data: `/data/datasets/stt/<language>/{train,dev}.jsonl`. These are NeMo manifests with rows like
`{"audio_filepath": ..., "duration": ..., "text": ..., "locale": "uz-UZ"}`. To change paths, languages and settings,
edit [config.yaml](config.yaml).

```sh
cd stt/nemotron_train
python ../../general/gpu_clean.py                                          # check the GPUs are free
python train.py --set data.langs=[uzbek] train.max_steps=500 train.val_every=100   # quick test first (foreground)
bash run.sh                                                                # full training in background (nohup, all GPUs)
bash run.sh --resume                                                       # continue from checkpoints/last.ckpt
tail -f /data/experiments/nemotron/logs/train.log                          # watch progress
tensorboard --logdir /data/experiments/nemotron/tb
pkill -f "python.* $PWD/train.py"                                          # stop (run.sh prints the exact command)
```

Everything goes to `/data/experiments/nemotron/` (`output.dir` in config.yaml):
- `final.nemo`: best model by dev WER
- `last.nemo`: model at the last step (usually worse than `final.nemo`)
- `checkpoints/`: top `save_top_k` by `val_wer`, plus `last.ckpt` used by `--resume`
- `tb/`: TensorBoard logs
- `logs/`: `train_<time>.log`, with `train.log` pointing to the latest
- `results/`: test outputs from nemotron_test
- `dev_combined.jsonl`, `auto_manifests/`, `config_used.yaml`, `train.pid`

To undo the tokenizer's character fix for one language:
```sh
cd /data/datasets/stt/<language> && for f in *.jsonl.bak; do mv "$f" "${f%.bak}"; done
```

Compare with the base model:
```sh
cd ../nemotron_test
bash run.sh --model nvidia/nemotron-3.5-asr-streaming-0.6b --bench --split test --out base.json
bash run.sh --model /data/experiments/nemotron/final.nemo --bench --split test --prompt both --out finetuned.json
# -> /data/experiments/nemotron/results/base.json, finetuned.json
```
