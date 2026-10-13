"""YAML config loading with dotted command-line overrides."""
import hashlib
import json
from pathlib import Path

import yaml


def load_config(path, overrides=()):
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node[part]
        if leaf not in node:
            raise KeyError(f"unknown config key: {key}")
        node[leaf] = yaml.safe_load(value)
    return cfg


def config_hash(cfg):
    """Short stable hash stored in checkpoints and reports for traceability."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
