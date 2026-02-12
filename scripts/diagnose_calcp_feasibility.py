#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd


CP_RE = re.compile(
    r"cp_local_autosel_target_k(?P<k>\d+)_m(?P<m>\d+)_gamma(?P<gamma>(?:\d+)(?:p\d+)?)_alpha(?P<alpha>(?:\d+)(?:p\d+)?)"
)

def _p_to_float(s: str) -> float:
    return float(s.replace("p", "."))

def _find_col(df: pd.DataFrame, candidates: list[str], df_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"[{df_name}] cannot find any of columns: {candidates}")

def _ensure_alpha(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "alpha" in df.columns:
        df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce").round(4)
        return df

    cp_col = _find_col(df, ["cp_subdir", "selected_cp_subdir", "cp_subdir_sel", "cp_subdir_selected"], "calcp_all")
    cp = df[cp_col].astype(str).str.replace(r"_calcp$", "", regex=True)
    ext = cp.str.extract(CP_RE)
    if ext["alpha"].isna().all():
        raise RuntimeError("cannot parse alpha from cp_subdir")
    df["alpha"] = ext["alpha"].map(_p_to_float).round(4)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2025-12-20")
    ap.add_argument("--target_coverage", type=float, default=0.9)
    args = ap.parse_args()

    root = Path("results/tables") / args.date
    f_calcp = root / "conformal_autosel_calcp_all.csv"
    if not f_calcp.exists():
        raise FileNotFoundError(f"missing file: {f_calcp}")

    df = pd.read_csv(f_calcp)

    # Minimal required columns for feasibility.
    for c in ["dataset", "split", "seed", "coverage"]:
        if c not in df.columns:
            raise RuntimeError(f"missing column: {c}")

    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")

    df = _ensure_alpha(df)

    cov_pivot = df.pivot_table(
        index=["dataset", "split", "seed"],
        columns="alpha",
        values="coverage",
        aggfunc="mean",
    ).reset_index()

    alpha_cols = [c for c in cov_pivot.columns if isinstance(c, float)]
    if not alpha_cols:
        raise RuntimeError("no alpha columns found after pivot; check alpha parsing")

    cov_pivot["best_calcp_coverage"] = cov_pivot[alpha_cols].max(axis=1)
    cov_pivot["feasible_on_calcp"] = cov_pivot["best_calcp_coverage"] >= args.target_coverage

    out = root / "calcp_feasibility_by_seed.csv"
    cov_pivot.to_csv(out, index=False)

    grp = (
        cov_pivot.groupby(["dataset", "split"])["feasible_on_calcp"]
        .mean()
        .reset_index(name="feasible_rate")
        .sort_values(["feasible_rate", "dataset", "split"])
    )

    print("[OK] wrote:", out)
    print("\n=== feasible_rate on calcp (by dataset/split) ===")
    print(grp.to_string(index=False))

    worst = grp.iloc[0]
    w_ds, w_sp = worst["dataset"], worst["split"]
    print(f"\n=== worst group details: {w_ds} / {w_sp} ===")
    show = cov_pivot[(cov_pivot["dataset"] == w_ds) & (cov_pivot["split"] == w_sp)].sort_values("seed")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
