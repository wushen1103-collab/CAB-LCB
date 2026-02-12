#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def parse_exp_dir_name(exp_name: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    # Expected: deepdta_point_{dataset}_{split}_seed{seed}
    if not exp_name.startswith("deepdta_point_") or "_seed" not in exp_name:
        return None, None, None
    body = exp_name[len("deepdta_point_") :]
    left, right = body.rsplit("_seed", 1)
    try:
        seed = int(right)
    except Exception:
        seed = None
    toks = left.split("_")
    if len(toks) < 2:
        return None, None, seed
    dataset = toks[0]
    split = "_".join(toks[1:])
    return dataset, split, seed


def pick_metrics(cm: Dict[str, Any], report_split: str) -> Tuple[float, float, float, float, float]:
    # Return: coverage_mean, coverage_std, width_mean, width_std, n
    if report_split not in ("eval", "test"):
        raise ValueError("report_split must be eval|test")

    if report_split == "eval":
        cov = safe_float(cm.get("coverage_eval", cm.get("coverage_mean", cm.get("coverage"))))
        wid = safe_float(cm.get("avg_width_eval", cm.get("width_mean", cm.get("avg_width"))))
    else:
        cov = safe_float(cm.get("coverage_test", cm.get("coverage_eval", cm.get("coverage_mean", cm.get("coverage")))))
        wid = safe_float(cm.get("avg_width_test", cm.get("avg_width_eval", cm.get("width_mean", cm.get("avg_width")))))

    cov_std = safe_float(cm.get("coverage_std"))
    wid_std = safe_float(cm.get("width_std"))

    if report_split == "eval":
        n = safe_float(cm.get("n_eval_used_eval", cm.get("n_eval_used", cm.get("n_test", cm.get("n_eval_entities", cm.get("n"))))))
    else:
        n = safe_float(cm.get("n_test_used", cm.get("n_test", cm.get("n_eval_used", cm.get("n_eval_entities", cm.get("n"))))))

    return cov, cov_std, wid, wid_std, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report_split", choices=["eval", "test"], required=True)
    ap.add_argument("--split", choices=["eval", "test"], help="Alias for --report_split (backward compatibility).")
    ap.add_argument("--metrics_name", default="conformal_metrics.json")
    args = ap.parse_args()

    report_split = args.report_split
    if args.split is not None:
        report_split = args.split

    runs_tag = Path(args.runs_tag)
    out_path = Path(args.out)

    metric_files = sorted(runs_tag.rglob(args.metrics_name))
    print(f"conformal_metrics.json found: {len(metric_files)}")

    rows = []
    exp_dirs = sorted([p for p in runs_tag.iterdir() if p.is_dir() and p.name.startswith("deepdta_point_")])
    print(f"exp_dirs matched: {len(exp_dirs)}")

    exp_name_set = set([p.name for p in exp_dirs])

    for mp in metric_files:
        # Expect .../{exp_dir}/{cp_subdir}/conformal_metrics.json
        if mp.parent is None or mp.parent.parent is None:
            continue
        exp_dir = mp.parent.parent
        if exp_dir.name not in exp_name_set:
            continue

        cm = json.loads(mp.read_text(encoding="utf-8", errors="ignore") or "{}")

        dataset, split, seed = parse_exp_dir_name(exp_dir.name)
        cp_subdir = mp.parent.name
        alpha = safe_float(cm.get("alpha"))

        cov_mean, cov_std, wid_mean, wid_std, n = pick_metrics(cm, report_split)

        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "seed": seed,
                "cp_subdir": cp_subdir,
                "alpha": alpha,
                "coverage_mean": cov_mean,
                "coverage_std": cov_std,
                "width_mean": wid_mean,
                "width_std": wid_std,
                "n": n,
                "eval_split": report_split,
                "run_date": runs_tag.name,
                "exp_name": exp_dir.name,
                "cp_dir": str(mp.parent),
            }
        )

    df = pd.DataFrame(rows)
    print(f"rows collected: {len(df)}")
    if len(df) == 0:
        df.to_csv(out_path, index=False)
        raise SystemExit("[FATAL] No rows collected. This usually means split metrics are missing in conformal_metrics.json.")

    # Basic sanity
    miss_cov = int(df["coverage_mean"].isna().sum())
    miss_wid = int(df["width_mean"].isna().sum())
    print(f"non-null coverage_mean: {len(df) - miss_cov}")
    print(f"non-null width_mean: {len(df) - miss_wid}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("[OK] wrote:", str(out_path))
    print("eval_split counts:")
    print(df["eval_split"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
