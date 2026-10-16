"""YAML config with command-line overrides: load_config("config.yaml", ["train.lr=5e-6", "data.langs=[uz,ru]"])."""

import sys
from pathlib import Path

import yaml


def load_config(path, overrides):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for item in overrides:  # --set train.lr=5e-6 data.max_train_rows=1000
        key, sep, val = item.partition("=")
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if not sep or not isinstance(node, dict) or leaf not in node:
            sys.exit(f"--set: unknown setting '{key}' (see config.yaml)")
        node[leaf] = yaml.safe_load(val)
    return cfg
