#!/usr/bin/env python3
"""
Aggregate per-seed WSC summary into a Table2-add CSV (mean±std, meet-rate).

Input summary columns:
  dataset, split, seed, coverage, mean_width, meet_90, n

Output columns:
  dataset, split, Method, Coverage, Width(pKd), Meet-rate, n_test, n_seeds
"""
from __future__ import annotations
import argparse
import pandas as pd

def fmt_pm(m: float, s: float, nd: int = 3) -> str:
    return f"{m:.{nd}{s:.{nd}f}"f}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--method_name", default="WSC-B")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    g = df.groupby(["dataset","split"], as_index=False).agg(
        coverage_mean=("coverage","mean"),
        coverage_std=("coverage","std"),
        width_mean=("mean_width","mean"),
        width_std=("mean_width","std"),
        meet_rate=("meet_90","mean"),
        n_test=("n","first"),
        n_seeds=("seed","nunique"),
    )

    out = pd.DataFrame({
        "dataset": g["dataset"],
        "split": g["split"],
        "Method": args.method_name,
        "Coverage": [fmt_pm(m,s,3) for m,s in zip(g.coverage_mean, g.coverage_std)],
        "Width(pKd)": [fmt_pm(m,s,3) for m,s in zip(g.width_mean, g.width_std)],
        "Meet-rate": [f"{x*100:.1f}%" for x in g.meet_rate],
        "n_test": g["n_test"],
        "n_seeds": g["n_seeds"],
    }).sort_values(["dataset","split"])

    out.to_csv(args.out_csv, index=False)
    print("[saved]", args.out_csv)
    print(out)

if __name__ == "__main__":
    main()
