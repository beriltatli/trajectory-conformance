"""Configuration and repository paths.

Two environment variables let the whole pipeline run on another dataset without touching the
committed results (the synthetic smoke run, `make smoke`):

- TC_CONFIG: config file to load instead of config.yaml. A config may name a base file under
  `extends:` (relative to itself); its keys are merged into the base, nested mappings key by key.
- TC_OUT: directory that takes the place of data/, results/ and figures/ for every path built
  with repo_path.
"""

import os
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
REDIRECTED = ("data", "results", "figures")


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = Path(path or os.environ.get("TC_CONFIG") or REPO / "config.yaml")
    if not path.is_absolute():
        path = REPO / path
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    base = cfg.pop("extends", None)
    return _merge(load_config(path.parent / base), cfg) if base else cfg


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def repo_path(relative: str) -> Path:
    out = os.environ.get("TC_OUT")
    if out and Path(relative).parts[0] in REDIRECTED:
        root = Path(out) if Path(out).is_absolute() else REPO / out
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return REPO / relative
