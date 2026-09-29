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

```sh
mkdir -p results
python riva_test.py --test-file test_data/test.jsonl 2>&1 | tee results/riva_test.log
```