#!/usr/bin/env python3
"""
backfill_missing_report_metrics.py

Given a missing-keys CSV (dataset, split, seed, cp_subdir), re-run a driver to
generate/overwrite conformal_metrics.json for those missing configs.

Default driver matches your repo help output:
  python scripts/run_split_conformal_from_preds.py --run_dir <EXP_DIR> --alpha <ALPHA> --out_subdir <CP_SUBDIR>

Notes:
- This script assumes experiment dirs follow:
    <runs_tag>/deepdta_point_{dataset}_{split}_seed{seed}
- It parses alpha from cp_subdir name (e.g., alpha0p02 -> 0.02).
- Use --force to recompute even if conformal_metrics.json already exists.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_alpha(cp_subdir: str) -> Optional[float]:
    m = re.search(r"alpha(\d+(?:p\d+)?)", str(cp_subdir))
    if not m:
        return None
    s = m.group(1).replace("p", ".")
    try:
        return float(s)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_tag", required=True, type=str)
    ap.add_argument("--missing_csv", required=True, type=str, help="CSV produced by make_three_way_pub_v2 missing_report_metrics_*.csv")
    ap.add_argument("--driver", default="scripts/run_split_conformal_from_preds.py", type=str)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--backup", action="store_true", help="Backup existing conformal_metrics.json before overwriting.")
    args = ap.parse_args()

    runs = Path(args.runs_tag)
    miss = pd.read_csv(args.missing_csv)

    required = ["dataset", "split", "seed", "cp_subdir"]
    for c in required:
        if c not in miss.columns:
            raise KeyError(f"missing_csv must include {required}, got columns={list(miss.columns)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_total = len(miss)
    n_run = 0
    n_skip = 0
    n_fail = 0

    for r in miss.itertuples(index=False):
        dataset = str(getattr(r, "dataset"))
        split = str(getattr(r, "split"))
        seed = int(getattr(r, "seed"))
        cp_subdir = str(getattr(r, "cp_subdir"))

        alpha = parse_alpha(cp_subdir)
        if alpha is None:
            print(f"[SKIP] cannot parse alpha from cp_subdir={cp_subdir}")
            n_skip += 1
            continue

        exp_dir = runs / f"deepdta_point_{dataset}_{split}_seed{seed}"
        if not exp_dir.exists():
            print(f"[FAIL] exp_dir not found: {exp_dir}")
            n_fail += 1
            continue

        out_dir = exp_dir / cp_subdir
        metrics_path = out_dir / "conformal_metrics.json"

        if metrics_path.exists() and (not args.force):
            # If user doesn't force, we still consider it missing because fixed_table didn't collect it.
            # But sometimes it's a naming mismatch in fixed_table; forcing is safer when in doubt.
            print(f"[SKIP] metrics exists (use --force to recompute): {metrics_path}")
            n_skip += 1
            continue

        cmd = [
            str(Path().resolve() / ".venv/bin/python") if (Path(".venv/bin/python").exists()) else "python",
            args.driver,
            "--run_dir",
            str(exp_dir),
            "--alpha",
            str(alpha),
            "--out_subdir",
            cp_subdir,
        ]

        # Backup if requested
        if args.backup and metrics_path.exists():
            bak = metrics_path.with_suffix(f".json.bak_{ts}")
            shutil.copy2(metrics_path, bak)
            print(f"[INFO] backup: {bak}")

        print("[RUN]", " ".join(cmd))
        if not args.write:
            continue

        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print("[FAIL] returncode=", p.returncode)
            print(p.stdout)
            print(p.stderr)
            n_fail += 1
            continue

        n_run += 1

    print("\n=== Summary ===")
    print("total_missing_rows:", n_total)
    print("ran_driver:", n_run)
    print("skipped:", n_skip)
    print("failed:", n_fail)
    if not args.write:
        print("[DRYRUN] pass --write to execute.")


if __name__ == "__main__":
    main()
