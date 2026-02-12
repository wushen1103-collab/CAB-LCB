#!/usr/bin/env python3
"""
Rebuild:
  - conformal_autosel_all.fixed_eval.csv
  - conformal_autosel_all.fixed_test.csv
  - conformal_autosel_all.fixed_eval_test.csv
  - three_way_main_table_*.csv and audit_*.csv

This assumes:
- conformal_metrics.json already contains real coverage_eval/avg_width_eval and coverage_test/avg_width_test.

It simply calls the existing scripts in the correct order.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


def _run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_tag", required=True)
    ap.add_argument("--tables_dir", required=True)
    ap.add_argument("--target_coverage", type=float, default=0.9)
    ap.add_argument("--alpha_fixed", type=float, default=0.1)
    ap.add_argument("--baseline_cp_subdir", required=True)
    ap.add_argument("--suffix", default="pub_v2", help="Suffix for three-way outputs (for your script naming)")
    args = ap.parse_args()

    py = sys.executable
    tables_dir = Path(args.tables_dir).expanduser().resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)

    fixed_eval = tables_dir / "conformal_autosel_all.fixed_eval.csv"
    fixed_test = tables_dir / "conformal_autosel_all.fixed_test.csv"
    fixed_eval_test = tables_dir / "conformal_autosel_all.fixed_eval_test.csv"

    # 1) fixed eval
    _run(
        [
            py,
            "scripts/make_conformal_autosel_all_fixed.py",
            "--runs_tag",
            args.runs_tag,
            "--report_split",
            "eval",
            "--out",
            str(fixed_eval),
        ]
    )

    # 2) fixed test
    _run(
        [
            py,
            "scripts/make_conformal_autosel_all_fixed.py",
            "--runs_tag",
            args.runs_tag,
            "--report_split",
            "test",
            "--out",
            str(fixed_test),
        ]
    )

    # 3) merge
    _run(
        [
            py,
            "scripts/merge_fixed_eval_test.py",
            "--eval",
            str(fixed_eval),
            "--test",
            str(fixed_test),
            "--out",
            str(fixed_eval_test),
        ]
    )

    # 4) three-way table
    _run(
        [
            py,
            "scripts/make_three_way_pub_v2.py",
            "--dir",
            str(tables_dir),
            "--fixed_table",
            str(fixed_eval_test),
            "--target_coverage",
            str(args.target_coverage),
            "--alpha_fixed",
            str(args.alpha_fixed),
            "--selection_split",
            "eval",
            "--report_split",
            "test",
            "--baseline_cp_subdir",
            args.baseline_cp_subdir,
        ]
    )


if __name__ == "__main__":
    main()
