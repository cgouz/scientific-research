# scientific-research
scientific-research

# STT

# TTT
```sh
python -m venv --system-site-packages .venv
  
source .venv/bin/activate

pip install -r requirements.txt
pip check
```

## Riva-Translate uz <-> ru
- [ttt/riva_train/](ttt/riva_train/README.md): fine-tune on `/data/datasets/ttt/uz-ru`
- [ttt/riva_test/](ttt/riva_test/README.md): translate the test set with the base or fine-tuned model, and a GPU cleaner
- `ttt/til_to_manifest.py`, `ttt/uzlpc_to_manifest.py`: build train/dev/test JSONL manifests

```sh
cd ttt/riva_train && python train.py
cd ttt/riva_test && python riva_test.py --test-file /data/datasets/ttt/uz-ru/test.jsonl --out results/base.json
```
