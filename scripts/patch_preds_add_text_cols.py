#!/usr/bin/env python3
"""
Patch DeepDTA exported preds files to include 'smiles' and 'sequence' columns,
so existing CP hyperparam selection scripts can run with group_by=target/pair.

This script DOES NOT retrain models. It only joins (pair_idx -> smiles/sequence) from pairs.csv.gz.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def load_pairs_map(pairs_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pairs_path, compression="gzip")
    # Ensure required columns exist
    for c in ["smiles", "sequence"]:
        if c not in df.columns:
            raise RuntimeError(f"pairs file missing column: {c} ({pairs_path})")
    return df[["smiles", "sequence"]]


def patch_one(pred_path: Path, pairs_text: pd.DataFrame) -> None:
    df = pd.read_csv(pred_path, compression="gzip")
    if "pair_idx" not in df.columns:
        raise RuntimeError(f"Missing pair_idx in {pred_path}")

    # If already present, skip
    if "smiles" in df.columns and "sequence" in df.columns:
        return

    # Join by pair_idx (which is the row index into pairs.csv.gz)
    text = pairs_text.iloc[df["pair_idx"].astype(int).to_numpy()].reset_index(drop=True)
    if "smiles" not in df.columns:
        df["smiles"] = text["smiles"].astype(str).to_numpy()
    if "sequence" not in df.columns:
        df["sequence"] = text["sequence"].astype(str).to_numpy()

    df.to_csv(pred_path, index=False, compression="gzip")


def infer_dataset_from_run(run_dir: Path) -> str:
    # expects run_dir name contains _davis_ or _kiba_
    name = run_dir.name.lower()
    if "_davis_" in name:
        return "davis"
    if "_kiba_" in name:
        return "kiba"
    # fallback: search parent dirs
    name2 = str(run_dir).lower()
    if "_davis_" in name2:
        return "davis"
    if "_kiba_" in name2:
        return "kiba"
    raise RuntimeError(f"Cannot infer dataset from run_dir: {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True, help="e.g. runs/2025-12-23_deepdta_points")
    ap.add_argument("--pattern", default="deepdta_point_*", help="glob pattern under runs_root")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    run_dirs = sorted(runs_root.glob(args.pattern))
    if not run_dirs:
        raise RuntimeError(f"No run dirs matched: {runs_root}/{args.pattern}")

    # Cache pairs_text per dataset
    cache = {}

    patched = 0
    for rd in run_dirs:
        ds = infer_dataset_from_run(rd)
        if ds not in cache:
            pairs_path = Path("data/processed") / ds / "pairs.csv.gz"
            cache[ds] = load_pairs_map(pairs_path)

        for fn in ["preds_cal.csv.gz", "preds_test.csv.gz"]:
            p = rd / fn
            if not p.exists():
                raise RuntimeError(f"Missing {p}")
            patch_one(p, cache[ds])
        patched += 1

    print(f"[DONE] patched {patched} run dirs under {runs_root}")


if __name__ == "__main__":
    main()
