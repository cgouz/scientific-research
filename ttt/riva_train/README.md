# riva_train

Fine-tunes `nvidia/Riva-Translate-4B-Instruct-v2` on **Uzbek → Russian** and **Russian → Uzbek**.
All weights are trained in bf16, and the best checkpoint is chosen by dev loss.

Data: `/data/datasets/ttt/uz-ru/{train,dev,test}.jsonl`. Each row looks like `{"pair": "uz-ru" | "ru-uz", "source": ..., "target": ...}`
and is trained in the direction given by its `pair`. To change paths and settings, edit [config.yaml](config.yaml).

```sh
cd ttt/riva_train
python ../riva_test/gpu_clean.py                                       # check the GPUs are free
python train.py --set data.max_train_rows=20000 train.eval_steps=100   # quick test first
python train.py                                                        # full training, one GPU
torchrun --nproc_per_node=gpu train.py                                 # all GPUs
python train.py --resume                                               # continue after a crash
```

Output goes to `ttt/outputs/riva-uz-ru/`:
- `final/`: best model + tokenizer
- `checkpoint-*/`: best + latest checkpoints
- `config_used.yaml`

Compare with the base model:
```sh
cd ../riva_test
python riva_test.py --test-file /data/datasets/ttt/uz-ru/test.jsonl --out results/base.json
python riva_test.py --model ../outputs/riva-uz-ru/final --test-file /data/datasets/ttt/uz-ru/test.jsonl --out results/finetuned.json
```

The prompt is the same as in `riva_test.py`. The loss is computed only on the translation:
```
<s>System\nYou are an expert at translating text from Uzbek to Russian.</s>\n
<s>User\nWhat is the Russian translation of the sentence: {source}</s>\n
<s>Assistant\n{target}</s>
```
