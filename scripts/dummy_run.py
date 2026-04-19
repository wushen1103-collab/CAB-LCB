from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime

from src.dti_cp.utils.io import load_yaml, save_yaml, make_run_dir
from src.dti_cp.utils.repro import set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--exp_name", type=str, default="smoke_test")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    run_dir = make_run_dir(cfg["paths"]["runs_dir"], args.exp_name)

    # 1) save full config
    save_yaml(cfg, run_dir / "config.yaml")

    # 2) write a dummy metrics line
    metrics_path = run_dir / "metrics.jsonl"
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "record_type": "smoke_test",
        "dataset": cfg.get("dataset"),
        "split": cfg.get("split"),
        "seed": cfg.get("seed"),
        "note": "dummy run to validate the run-output schema",
    }
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 3) stdout log (minimal)
    log_path = run_dir / "stdout.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{record['time']}] dummy run ok: {run_dir}\n")

    print(f"OK. Wrote run outputs to: {run_dir}")

if __name__ == "__main__":
    main()
