"""Configuration loading with safe defaults."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict:
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if path:
        with open(path, encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    # resolve relative paths against the project root
    intel_dir = Path(cfg["intel"]["directory"])
    if not intel_dir.is_absolute():
        cfg["intel"]["directory"] = str(PROJECT_ROOT / intel_dir)
    log_file = Path(cfg["response"]["notify"]["log_file"])
    if not log_file.is_absolute():
        cfg["response"]["notify"]["log_file"] = str(PROJECT_ROOT / log_file)
    return cfg
