#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _find_col(df: pd.DataFrame, candidates, df_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"[{df_name}] cannot find any of columns: {candidates}. Available: {list(df.columns)}")


def _parse_float_token(tok: str) -> float:
    # Supports patterns like: "0", "0p05", "0.05", "1p0"
    tok = str(tok)
    tok = tok.replace("p", ".")
    return float(tok)


def _ensure_params_from_cp(df: pd.DataFrame, cp_col: str, df_name: str) -> pd.DataFrame:
    """
    Parse k/m/gamma/alpha from cp_subdir-like names:
      cp_local_autosel_target_k240_m50_gamma0_alpha0p05
      cp_local_autosel_target_k60_m50_gamma0p2_alpha0p1  (rare)
    """
    out = df.copy()
    cp = out[cp_col].astype(str)

    # Strip trailing "_calcp" to get base name for parsing.
    out["cp_base"] = cp.str.replace(r"_calcp$", "", regex=True)

    # Regex: be tolerant to gamma token variants (0 / 0p05 / 0.05 / 0p0 etc.)
    pat = re.compile(
        r"cp_local_autosel_(?:drug|target)_(?:aacomp|tfidf)?"
        r".*?_k(?P<k>\d+)_m(?P<m>\d+)_gamma(?P<gamma>[0-9p\.]+)_alpha(?P<alpha>[0-9p\.]+)$"
    )

    k_list, m_list, g_list, a_list = [], [], [], []
    for s in out["cp_base"].tolist():
        m = pat.search(s)
        if not m:
            k_list.append(np.nan)
            m_list.append(np.nan)
            g_list.append(np.nan)
            a_list.append(np.nan)
            continue
        k_list.append(int(m.group("k")))
        m_list.append(int(m.group("m")))
        g_list.append(_parse_float_token(m.group("gamma")))
        a_list.append(_parse_float_token(m.group("alpha")))

    out["k"] = out.get("k", pd.Series(k_list)).fillna(pd.Series(k_list))
    out["m"] = out.get("m", pd.Series(m_list)).fillna(pd.Series(m_list))
    out["gamma"] = out.get("gamma", pd.Series(g_list)).fillna(pd.Series(g_list))

    # Some CSVs already have alpha column; only fill when missing.
    if "alpha" not in out.columns:
        out["alpha"] = a_list
    else:
        out["alpha"] = out["alpha"].fillna(pd.Series(a_list))

    missing_params = out[["k", "m", "gamma", "alpha"]].isna().any(axis=1).sum()
    if missing_params > 0:
        # Do not fail hard; just warn because some rows may be non-autosel.
        print(f"[WARN] [{df_name}] {int(missing_params)} rows missing parsed params from {cp_col}.")
    return out


def _load_table(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[{name}] missing file: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {name}: {path} rows={len(df)} cols={len(df.columns)}")
    return df


def _normalize_core_cols(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    out = df.copy()

    dataset_col = _find_col(out, ["dataset"], df_name)
    split_col = _find_col(out, ["split"], df_name)
    seed_col = _find_col(out, ["seed"], df_name)

    alpha_col = out.columns.intersection(["alpha", "alpha_sel"]).tolist()
    if len(alpha_col) == 0:
        # allow alpha embedded in cp_subdir
        alpha_col = ["alpha"]
        out["alpha"] = np.nan
    else:
        alpha_col = alpha_col[0]

    cp_col = out.columns.intersection(["cp_subdir", "cp_base_sel", "cp_subdir_sel", "selected_cp_subdir", "cp_subdir_selected"]).tolist()
    if len(cp_col) == 0:
        # If no cp_subdir column exists, create a placeholder.
        cp_col = ["cp_subdir"]
        out["cp_subdir"] = ""
    else:
        cp_col = cp_col[0]

    # Coverage + width columns are messy across files.
    cov_col = None
    for c in ["coverage", "coverage_eval", "coverage_mean", "coverage_calcp", "coverage_eval_mean"]:
        if c in out.columns:
            cov_col = c
            break
    if cov_col is None:
        raise RuntimeError(f"[{df_name}] cannot find a coverage column. Available: {list(out.columns)}")

    wid_col = None
    for c in ["avg_width", "avg_width_eval", "width", "width_eval", "width_mean", "width_calcp", "avg_width_mean"]:
        if c in out.columns:
            wid_col = c
            break
    if wid_col is None:
        raise RuntimeError(f"[{df_name}] cannot find a width column. Available: {list(out.columns)}")

    # Optional n_eval_used (calcp metrics json has it; eval tables may not)
    n_eval_col = None
    for c in ["n_eval_used", "n_eval_used_mean"]:
        if c in out.columns:
            n_eval_col = c
            break

    out = out.rename(
        columns={
            dataset_col: "dataset",
            split_col: "split",
            seed_col: "seed",
            alpha_col: "alpha",
            cp_col: "cp_subdir",
            cov_col: "coverage",
            wid_col: "avg_width",
        }
    )
    if n_eval_col is not None and n_eval_col != "n_eval_used":
        out = out.rename(columns={n_eval_col: "n_eval_used"})
    if "n_eval_used" not in out.columns:
        out["n_eval_used"] = np.nan

    # Make types consistent
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    out["alpha"] = pd.to_numeric(out["alpha"], errors="coerce")
    out["coverage"] = pd.to_numeric(out["coverage"], errors="coerce")
    out["avg_width"] = pd.to_numeric(out["avg_width"], errors="coerce")
    out["cp_subdir"] = out["cp_subdir"].astype(str)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--target_coverage", type=float, default=0.9)
    args = ap.parse_args()

    date = args.date
    base_dir = Path("results") / "tables" / date

    calcp_path = base_dir / "conformal_autosel_calcp_all.csv"
    eval_path = base_dir / "conformal_autosel_all.fixed.csv"

    calcp_raw = _load_table(calcp_path, "autosel_calcp_all")
    eval_raw = _load_table(eval_path, "autosel_eval_all_fixed")

    calcp = _normalize_core_cols(calcp_raw, "autosel_calcp_all")
    eval_df = _normalize_core_cols(eval_raw, "autosel_eval_all_fixed")

    # calcp rows should correspond to cp_subdir ending with "_calcp"
    if not calcp["cp_subdir"].str.contains("_calcp", regex=False).any():
        print("[WARN] calcp table has no '_calcp' in cp_subdir; continuing anyway.")

    # Parse params from cp_subdir for both tables.
    calcp = _ensure_params_from_cp(calcp, "cp_subdir", "autosel_calcp_all")
    eval_df = _ensure_params_from_cp(eval_df, "cp_subdir", "autosel_eval_all_fixed")

    # Build merge keys: (dataset, split, seed, alpha, cp_base)
    key = ["dataset", "split", "seed", "alpha", "cp_base"]

    calcp_small = calcp[key + ["k", "m", "gamma", "coverage", "avg_width", "n_eval_used"]].rename(
        columns={"coverage": "coverage_calcp", "avg_width": "width_calcp", "n_eval_used": "n_eval_used_calcp"}
    )

    eval_small = eval_df[key + ["coverage", "avg_width"]].rename(
        columns={"coverage": "coverage_eval", "avg_width": "width_eval"}
    )

    joined = calcp_small.merge(eval_small, on=key, how="left")

    # Global sanity: are n_eval_used consistent within each dataset/split?
    n_eval_stats = (
        calcp_small.groupby(["dataset", "split"])["n_eval_used_calcp"]
        .agg(["count", "nunique", "min", "max"])
        .reset_index()
    )
    bad_n = n_eval_stats[n_eval_stats["nunique"] > 1]
    print("\n=== calcp n_eval_used consistency check ===")
    print(n_eval_stats.to_string(index=False))
    if len(bad_n) > 0:
        print("\n[WARN] Some dataset/split have varying n_eval_used across rows (may be OK, but verify):")
        print(bad_n.to_string(index=False))

    # Focus on the requested group
    ds, sp, sd = args.dataset, args.split, args.seed
    g = joined[(joined["dataset"] == ds) & (joined["split"] == sp) & (joined["seed"] == sd)].copy()

    if len(g) == 0:
        raise RuntimeError(f"No rows found for dataset={ds} split={sp} seed={sd}. Check inputs and CSVs.")

    g = g.sort_values(["alpha", "k", "m", "gamma"]).reset_index(drop=True)

    # Feasibility within the candidate set: does any alpha meet target on calcp?
    g["meets_target_calcp"] = g["coverage_calcp"] >= args.target_coverage
    best_row = g.sort_values(["coverage_calcp", "width_calcp"], ascending=[False, True]).head(1)

    print(f"\n=== candidate pool on calcp for {ds}/{sp}/seed{sd} (autosel-chosen configs per alpha) ===")
    cols_show = ["alpha", "k", "m", "gamma", "coverage_calcp", "width_calcp", "n_eval_used_calcp", "coverage_eval", "width_eval", "meets_target_calcp", "cp_base"]
    print(g[cols_show].to_string(index=False))

    print("\n=== feasibility summary (within candidate set) ===")
    print("target_coverage =", args.target_coverage)
    print("any_meets_target_calcp =", bool(g["meets_target_calcp"].any()))
    print("\n=== best-on-calcp row (max coverage_calcp, tie-break min width_calcp) ===")
    print(best_row[cols_show].to_string(index=False))

    # Outlier analysis: compare this seed to other seeds for the same dataset/split by alpha.
    print(f"\n=== compare seed{sd} vs other seeds on calcp (same dataset/split) ===")
    calcp_group = calcp_small[(calcp_small["dataset"] == ds) & (calcp_small["split"] == sp)].copy()
    # Keep only rows that correspond to autosel cp_base per alpha (should be one per seed per alpha)
    calcp_group = calcp_group.drop_duplicates(subset=["dataset", "split", "seed", "alpha", "cp_base"])

    def _rank_stats(df_alpha: pd.DataFrame) -> pd.DataFrame:
        df_alpha = df_alpha.dropna(subset=["coverage_calcp"]).copy()
        df_alpha["rank_desc"] = df_alpha["coverage_calcp"].rank(method="min", ascending=False)
        df_alpha["zscore"] = (df_alpha["coverage_calcp"] - df_alpha["coverage_calcp"].mean()) / (df_alpha["coverage_calcp"].std(ddof=0) + 1e-12)
        return df_alpha

    out_rows = []
    for a, df_a in calcp_group.groupby("alpha"):
        df_a = _rank_stats(df_a)
        me = df_a[df_a["seed"] == sd]
        if len(me) == 0:
            continue
        out_rows.append({
            "alpha": float(a),
            "seed": sd,
            "coverage_calcp": float(me["coverage_calcp"].iloc[0]),
            "rank_desc": float(me["rank_desc"].iloc[0]),
            "zscore": float(me["zscore"].iloc[0]),
            "min_cov": float(df_a["coverage_calcp"].min()),
            "mean_cov": float(df_a["coverage_calcp"].mean()),
            "max_cov": float(df_a["coverage_calcp"].max()),
            "n_seeds": int(df_a["seed"].nunique()),
        })
    out_df = pd.DataFrame(out_rows).sort_values("alpha")
    print(out_df.to_string(index=False))

    # Save a joined diagnostic CSV for later paper writing.
    out_path = base_dir / f"diagnose_{ds}_{sp}_seed{sd}_calcp_vs_eval.csv"
    g.to_csv(out_path, index=False)
    print("\n[OK] wrote:", out_path)


if __name__ == "__main__":
    main()
