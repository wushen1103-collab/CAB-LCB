from __future__ import annotations
from pathlib import Path
import yaml
from datetime import datetime

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def make_run_dir(runs_dir: str, exp_name: str) -> Path:
    date = datetime.now().strftime("%Y-%m-%d")
    run_dir = Path(runs_dir) / date / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
