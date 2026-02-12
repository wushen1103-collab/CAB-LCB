#!/usr/bin/env python3
"""
Batch runner for Weighted Split Conformal (WSC) baseline (feature set B).

Feature set B:
  - drug_idx, target_idx (categorical)
  - len(smiles), len(sequence) (numeric)

This script pairs point runs with calproxy runs and generates:
  - per-run interval file: pred_intervals_test_wsc_B.csv.gz
  - summary CSV: results/shift_aware_summary_wsc_B_alpha{alpha}_{out_tag}.csv

Expected directory naming patterns (examples):
  DeepDTA points:
    deepdta_point_{dataset}_{split}_seed{seed}/preds_test.csv.gz
  DeepDTA calproxy:
    deepdta_point_{dataset}_{split}__calcp_seed{seed}/preds_test.csv.gz

  GraphDTA points:
    graphdta_point_gat_gcn_{dataset}_{split}_seed{seed}/preds_test.csv.gz
  GraphDTA calproxy:
    graphdta_point_gat_gcn_{dataset}_{split}_seed{seed}/preds_test.csv.gz   (in a separate calcp_root)

If your naming differs, adjust the regex below.
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path
import pandas as pd

KEY_PATTERNS = [
    re.compile(r"deepdta_point_(davis|kiba)_(random|cold_drug|cold_target|cold_pair)_seed(\d+)$"),
    re.compile(r"deepdta_point_(davis|kiba)_(random|cold_drug|cold_target|cold_pair)__calcp_seed(\d+)$"),
    re.compile(r"graphdta_point_gat_gcn_(davis|kiba)_(random|cold_drug|cold_target|cold_pair)_seed(\d+)$"),
]

def parse_key(run_dir_name: str):
    for pat in KEY_PATTERNS:
        m = pat.match(run_dir_name)
        if m:
            dataset, split, seed = m.group(1), m.group(2), int(m.group(3))
            return dataset, split, seed
    return None

def collect_preds(root: Path):
    paths = list(root.rglob("preds_test.csv.gz"))
    m = {}
    bad = []
    for p in paths:
        run_dir = p.parent
        key = parse_key(run_dir.name)
        if key is None:
            bad.append(str(run_dir))
            continue
        # If duplicates exist, keep the first one found.
        m.setdefault(key, p)
    return m, bad, len(paths)

def eval_intervals(intervals_gz: Path):
    df = pd.read_csv(intervals_gz)
    if "y" not in df.columns:
        raise ValueError(f"Missing y in {intervals_gz}")
    if "pi_lo" not in df.columns or "pi_hi" not in df.columns:
        raise ValueError(f"Missing pi_lo/pi_hi in {intervals_gz}")
    covered = ((df["y"] >= df["pi_lo"]) & (df["y"] <= df["pi_hi"])).astype(int)
    width = (df["pi_hi"] - df["pi_lo"]).astype(float)
    cov = float(covered.mean())
    mw = float(width.mean())
    meet = float(cov >= 0.90)
    return cov, mw, meet, int(len(df))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points_root", required=True, help="Root directory containing point runs.")
    ap.add_argument("--calcp_root", required=True, help="Root directory containing calproxy runs.")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out_tag", required=True, help="Tag for naming the summary file.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    points_root = Path(args.points_root)
    calcp_root = Path(args.calcp_root)

    points_map, points_bad, points_total = collect_preds(points_root)
    calcp_map, calcp_bad, calcp_total = collect_preds(calcp_root)

    keys = sorted(set(points_map.keys()) & set(calcp_map.keys()))
    print(f"[scan] points_total={points_total} calcp_total={calcp_total} paired_jobs={len(keys)}")
    print(f"[scan] bad_points={len(points_bad)} bad_calcp={len(calcp_bad)}")

    if args.dry_run:
        for k in keys[:8]:
            print("[pair]", k, points_map[k].parent.name, "<->", calcp_map[k].parent.name)
        return

    out_rows = []
    for dataset, split, seed in keys:
        points = points_map[(dataset, split, seed)]
        calcp = calcp_map[(dataset, split, seed)]
        out_intervals = points.parent / "pred_intervals_test_wsc_B.csv.gz"
        if out_intervals.exists() and (not args.overwrite):
            cov, mw, meet, n = eval_intervals(out_intervals)
            out_rows.append({
                "dataset": dataset, "split": split, "seed": seed,
                "run_dir": str(points.parent),
                "coverage": cov, "mean_width": mw, "meet_90": meet, "n": n,
            })
            print("[skip]", (dataset, split, seed), "exists")
            continue

        cmd = [
            sys.executable, "scripts/weighted_conformal_baseline.py",
            "--calproxy", str(calcp),
            "--test", str(points),
            "--alpha", str(args.alpha),
            "--scale_mode", "auto",
            "--w_clip_min", "0.05",
            "--w_clip_max", "20.0",
            "--C", "1.0",
            "--seed", str(seed),
            "--out", str(out_intervals),
        ]
        print("[wsc_B]", (dataset, split, seed), points.parent.name)
        subprocess.check_call(cmd)

        cov, mw, meet, n = eval_intervals(out_intervals)
        out_rows.append({
            "dataset": dataset, "split": split, "seed": seed,
            "run_dir": str(points.parent),
            "coverage": cov, "mean_width": mw, "meet_90": meet, "n": n,
        })

    out_df = pd.DataFrame(out_rows).sort_values(["dataset","split","seed"])
    out_path = Path("results") / f"shift_aware_summary_wsc_B_alpha{args.alpha}_{args.out_tag}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print("[saved]", out_path, "rows=", len(out_df))

if __name__ == "__main__":
    main()
