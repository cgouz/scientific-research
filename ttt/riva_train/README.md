# riva_train

Fine-tunes `nvidia/Riva-Translate-4B-Instruct-v2` on **Uzbek → Russian** and **Russian → Uzbek**.
All weights are trained in bf16, and the best checkpoint is chosen by dev loss.

To use the GPU fully, the script finds the largest batch that fits before training starts. It tests with the longest sentences and counts the
optimizer's memory too (`batch_size: auto`, `memory_fraction: 0.85`). Gradient checkpointing is off for speed. To train faster with
several GPUs, run `bash run.sh`, which starts `torchrun` on all of them.

Data: `/data/datasets/ttt/uz-ru/{train,dev,test}.jsonl`. Each row looks like `{"pair": "uz-ru" | "ru-uz", "source": ..., "target": ...}`
and is trained in the direction given by its `pair`. To change paths and settings, edit [config.yaml](config.yaml).

```sh
cd ttt/riva_train
python ../riva_test/gpu_clean.py                                       # check the GPUs are free
python train.py --set data.max_train_rows=20000 train.eval_steps=100   # quick test first (foreground)
bash run.sh                                                            # full training in background (nohup, all GPUs)
bash run.sh --resume                                                   # continue from the last checkpoint
tail -f /data/experiments/riva/logs/train.log                          # watch progress
pkill -f "python.* train.py|torchrun.* train.py"          # stop (all runs)
```

Everything goes to `/data/experiments/riva/` (`output.dir` in config.yaml):
- `checkpoint-*/`: best + latest checkpoints, used by `--resume`
- `final/`: best model + tokenizer
- `logs/`: `train_<time>.log`, with `train.log` pointing to the latest
- `results/`: test outputs from riva_test
- `config_used.yaml`, `train.pid`

Compare with the base model:
```sh
cd ../riva_test
bash run.sh --test-file /data/datasets/ttt/uz-ru/test.jsonl --out base.json
bash run.sh --model /data/experiments/riva/final --test-file /data/datasets/ttt/uz-ru/test.jsonl --out finetuned.json
# -> /data/experiments/riva/results/base.json, finetuned.json
```

The prompt is the same as in `riva_test.py`. The loss is computed only on the translation:
```
<s>System\nYou are an expert at translating text from Uzbek to Russian.</s>\n
<s>User\nWhat is the Russian translation of the sentence: {source}</s>\n
<s>Assistant\n{target}</s>
```
