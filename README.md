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

# Riva zero-shot test
See [ttt/riva_test/README.md](ttt/riva_test/README.md).
```sh
cd ttt/riva_test && python riva_test.py 2>&1 | tee riva_test.log
```