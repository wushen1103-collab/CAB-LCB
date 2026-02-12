#!/usr/bin/env python3
"""
Audit constrained autosel artifacts (robust to column variations).

Expected under results/tables/{date}/:
- constrained_autosel_selection_by_calcp.csv
- conformal_autosel_calcp_all.csv
- constrained_autosel_eval_selected.csv

Writes:
- constrained_autosel_audit_manifest.csv
- constrained_autosel_audit_summary.csv
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np


DATASET_RE = re.compile(r"_(davis|kiba)_(random|cold_drug|cold_target|cold_pair)_seed(\d+)")
CP_RE = re.compile(
    r"cp_local_autosel_(?:drug|target)_k(?P<k>\d+)_m(?P<m>\d+)_gamma(?P<gamma>(?:\d+(?:p\d+)?|\d+\.\d+))_alpha(?P<alpha>(?:\d+(?:p\d+)?|\d+\.\d+))"
)

def _to_float(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("p", ".")
    return float(s)

def _mode(series: pd.Series) -> float:
    vc = series.value_counts()
    return float(vc.index[0]) if len(vc) else np.nan

def find_col(df: pd.DataFrame, candidates: list[str], df_name: str, required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError(f"[{df_name}] cannot find any of columns: {candidates}\nAvailable: {list(df.columns)}")
    return None

def ensure_meta(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    df = df.copy()

    if all(c in df.columns for c in ["dataset", "split", "seed"]) and df["dataset"].notna().any():
        df["seed"] = df["seed"].astype(float).astype(int)
        return df

    exp_col = find_col(df, ["exp_dir", "run_dir", "run", "experiment_dir"], df_name, required=True)
    ds, sp, sd = [], [], []
    for v in df[exp_col].astype(str).tolist():
        m = DATASET_RE.search(v)
        if not m:
            ds.append(np.nan); sp.append(np.nan); sd.append(np.nan)
        else:
            ds.append(m.group(1)); sp.append(m.group(2)); sd.append(int(m.group(3)))

    df["dataset"] = df.get("dataset", pd.Series([np.nan] * len(df))).fillna(pd.Series(ds))
    df["split"]   = df.get("split",   pd.Series([np.nan] * len(df))).fillna(pd.Series(sp))
    df["seed"]    = df.get("seed",    pd.Series([np.nan] * len(df))).fillna(pd.Series(sd))

    if df[["dataset", "split", "seed"]].isna().any().any():
        bad = df[df["dataset"].isna() | df["split"].isna() | df["seed"].isna()][[exp_col]].head(20)
        raise RuntimeError(f"[{df_name}] cannot parse meta from {exp_col} for some rows. Sample:\n{bad.to_string(index=False)}")

    df["seed"] = df["seed"].astype(int)
    return df

def ensure_params(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    df = df.copy()

    # Normalize alpha
    alpha_col = find_col(df, ["alpha", "alpha_sel", "alpha_selected", "selected_alpha"], df_name, required=False)
    if alpha_col is not None and alpha_col != "alpha":
        df["alpha"] = df[alpha_col].map(_to_float)
    elif alpha_col == "alpha":
        df["alpha"] = df["alpha"].map(_to_float)

    # If k/m/gamma already exist, normalize types
    if all(c in df.columns for c in ["k", "m", "gamma"]) and df[["k", "m", "gamma"]].notna().any().any():
        df["k"] = df["k"].astype(int)
        df["m"] = df["m"].astype(int)
        df["gamma"] = df["gamma"].map(_to_float)
        df["alpha"] = df["alpha"].map(_to_float)
        return df

    # Otherwise parse from cp_subdir-like column
    cp_col = find_col(
        df,
        ["cp_subdir", "selected_cp_subdir", "cp_subdir_sel", "cp_subdir_selected", "chosen_cp_subdir",
         "cp_base_sel", "cp_base", "cp_sel"],
        df_name,
        required=False,
    )
    if cp_col is None:
        raise RuntimeError(
            f"[{df_name}] missing params and cannot find cp_subdir-like column to parse from.\nAvailable: {list(df.columns)}"
        )

    ks, ms, gs, als = [], [], [], []
    for v in df[cp_col].astype(str).tolist():
        m = CP_RE.search(v)
        if not m:
            ks.append(np.nan); ms.append(np.nan); gs.append(np.nan); als.append(np.nan)
        else:
            ks.append(int(m.group("k")))
            ms.append(int(m.group("m")))
            gs.append(_to_float(m.group("gamma")))
            als.append(_to_float(m.group("alpha")))

    df["k"] = ks
    df["m"] = ms
    df["gamma"] = gs
    df["alpha"] = df.get("alpha", pd.Series([np.nan] * len(df))).fillna(pd.Series(als))

    if df[["alpha", "k", "m", "gamma"]].isna().any().any():
        bad = df[df[["alpha", "k", "m", "gamma"]].isna().any(axis=1)][[cp_col]].head(20)
        raise RuntimeError(f"[{df_name}] cannot parse alpha/k/m/gamma from {cp_col} for some rows. Sample:\n{bad.to_string(index=False)}")

    df["k"] = df["k"].astype(int)
    df["m"] = df["m"].astype(int)
    df["gamma"] = df["gamma"].astype(float)
    df["alpha"] = df["alpha"].astype(float)
    return df

def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[{name}] not found: {path}")
    df = pd.read_csv(path)
    dups = df.columns[df.columns.duplicated()].tolist()
    if dups:
        raise RuntimeError(f"[{name}] has duplicated column names: {dups}\nColumns: {list(df.columns)}")
    return df

def select_cols_with_optional(df: pd.DataFrame, required: list[str], optional: list[str]) -> pd.DataFrame:
    cols = []
    for c in required:
        if c not in df.columns:
            raise KeyError(f"required column '{c}' not found. Available: {list(df.columns)}")
        cols.append(c)
    for c in optional:
        if c in df.columns:
            cols.append(c)
    out = df[cols].copy()
    # Fill missing optional columns with NA if not present (for consistent downstream)
    for c in optional:
        if c not in out.columns:
            out[c] = np.nan
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--target_coverage", type=float, default=0.9)
    args = ap.parse_args()

    base = Path("results") / "tables" / args.date
    base.mkdir(parents=True, exist_ok=True)

    sel_path = base / "constrained_autosel_selection_by_calcp.csv"
    calcp_all_path = base / "conformal_autosel_calcp_all.csv"
    eval_sel_path = base / "constrained_autosel_eval_selected.csv"

    sel_df = ensure_params(ensure_meta(load_csv(sel_path, "selection_by_calcp"), "selection_by_calcp"), "selection_by_calcp")
    calcp_df = ensure_params(ensure_meta(load_csv(calcp_all_path, "calcp_all"), "calcp_all"), "calcp_all")
    eval_df = ensure_params(ensure_meta(load_csv(eval_sel_path, "eval_selected"), "eval_selected"), "eval_selected")

    # Selection file might not have k/m/gamma explicitly; ensure_params already handled parsing.
    sel_keep = ["dataset", "split", "seed", "alpha", "k", "m", "gamma"]
    sel_optional = ["coverage_calcp", "width_calcp", "met_target"]
    sel_df = select_cols_with_optional(sel_df, sel_keep, sel_optional).drop_duplicates()

    if sel_df.duplicated(subset=["dataset", "split", "seed"]).any():
        bad = sel_df[sel_df.duplicated(subset=["dataset", "split", "seed"], keep=False)].head(50)
        raise RuntimeError(f"[selection_by_calcp] duplicate keys (dataset,split,seed). Sample:\n{bad.to_string(index=False)}")

    keys = ["dataset", "split", "seed", "alpha", "k", "m", "gamma"]

    # calcp_all should have coverage/avg_width; n_eval_used is optional
    calcp_small = select_cols_with_optional(
        calcp_df,
        required=["dataset","split","seed","alpha","k","m","gamma","coverage","avg_width"],
        optional=["n_eval_used"],
    ).rename(columns={"coverage": "coverage_calcp_all", "avg_width": "width_calcp_all", "n_eval_used": "n_eval_used_calcp"})

    # eval_selected should have coverage/avg_width; n_eval_used is optional
    eval_small = select_cols_with_optional(
        eval_df,
        required=["dataset","split","seed","alpha","k","m","gamma","coverage","avg_width"],
        optional=["n_eval_used"],
    ).rename(columns={"coverage": "coverage_eval", "avg_width": "width_eval", "n_eval_used": "n_eval_used_eval"})

    j1 = sel_df.merge(calcp_small, on=keys, how="left", validate="one_to_one")
    miss_calcp = int(j1["coverage_calcp_all"].isna().sum())
    if miss_calcp:
        bad = j1[j1["coverage_calcp_all"].isna()][keys].head(30)
        raise RuntimeError(f"[AUDIT] missing calcp_all rows for {miss_calcp} selected configs. Sample:\n{bad.to_string(index=False)}")

    final = j1.merge(eval_small, on=keys, how="left", validate="one_to_one")
    miss_eval = int(final["coverage_eval"].isna().sum())
    if miss_eval:
        bad = final[final["coverage_eval"].isna()][keys].head(30)
        raise RuntimeError(f"[AUDIT] missing eval_selected rows for {miss_eval} selected configs. Sample:\n{bad.to_string(index=False)}")

    final["feasible_on_calcp"] = final["coverage_calcp_all"] >= float(args.target_coverage)

    out_manifest = base / "constrained_autosel_audit_manifest.csv"
    final.to_csv(out_manifest, index=False)

    summary = final.groupby(["dataset","split"]).agg(
        n_seeds=("seed","nunique"),
        met_target_rate=("feasible_on_calcp","mean"),
        alpha_sel_mode=("alpha", _mode),
        alpha_sel_mean=("alpha","mean"),
        coverage_eval_mean=("coverage_eval","mean"),
        coverage_eval_std=("coverage_eval","std"),
        width_eval_mean=("width_eval","mean"),
        width_eval_std=("width_eval","std"),
        coverage_calcp_mean=("coverage_calcp_all","mean"),
        coverage_calcp_std=("coverage_calcp_all","std"),
    ).reset_index()

    out_summary = base / "constrained_autosel_audit_summary.csv"
    summary.to_csv(out_summary, index=False)

    print(f"[OK] wrote: {out_manifest}")
    print(f"[OK] wrote: {out_summary}")
    print("\n=== summary (per dataset/split) ===")
    print(summary.sort_values(["dataset","split"]).to_string(index=False))

    bad = final[~final["feasible_on_calcp"]].sort_values(["dataset","split","seed"])
    if len(bad) > 0:
        cols = ["dataset","split","seed","alpha","k","m","gamma","coverage_calcp_all","coverage_eval","width_eval"]
        print("\n=== infeasible selected seeds on calcp (coverage_calcp < target) ===")
        print(bad[cols].to_string(index=False))
    else:
        print("\n[OK] all selected seeds are feasible on calcp.")

if __name__ == "__main__":
    main()
