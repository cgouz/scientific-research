# general

Helpers that stt/ and ttt/ share. Scripts in those folders put the repo root on `sys.path` and import
`general.<module>`, so they work from any directory.

| File | What it does |
|---|---|
| [gpu_clean.py](gpu_clean.py) | Shows what holds GPU memory and frees it. Run it from the shell, or call `free_gpu()` from a script |
| [config.py](config.py) | `load_config(path, overrides)`: reads `config.yaml` and applies `--set key=value` overrides |
| [jsonl.py](jsonl.py) | `read_jsonl(path, limit=None)` and `write_jsonl(path, rows)` for manifests |

## GPU cleaner (`gpu_clean.py`)

```sh
python general/gpu_clean.py                  # GPUs, memory used, which processes hold it
python general/gpu_clean.py --kill           # stop your processes holding GPU memory (asks first)
python general/gpu_clean.py --kill --gpu 0 -y
```

It only stops processes you own: SIGTERM first, then SIGKILL after 10 s (`--timeout`).
`riva_test.py` and `nemotron_test.py` call `free_gpu()` when they finish, even after an error.

## Using it in a new script

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (script is in <task>/<folder>/)
from general.config import load_config
from general.jsonl import read_jsonl, write_jsonl
from general.gpu_clean import free_gpu
```

`--set` values are parsed as YAML. Write numbers like `5.0e-6`, because PyYAML reads `5e-6` as a string.
Scripts call `float()` on these values, so both forms work there.
