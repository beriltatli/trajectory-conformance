from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or REPO / "config.yaml") as fh:
        return yaml.safe_load(fh)


def repo_path(relative: str) -> Path:
    return REPO / relative
