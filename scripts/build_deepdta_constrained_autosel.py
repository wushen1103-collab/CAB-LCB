#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build constrained autosel tables for DeepDTA.

# Key guarantees (fixes the earlier metric-definition mismatch):
- Selection is done on CALCP metrics (feasibility / LCB / mean coverage).
- Reported eval metrics MUST come from POINTS (eval_split=eval).
- No CALCP fallback unless you explicitly pass --allow_calcp_fallback.
- Output eval table uses coverage_eval / avg_width_eval columns (explicit naming).

Outputs under results/tables/<date>/:
- conformal_autosel_all.fixed.csv
- conformal_autosel_calcp_all.csv
- constrained_autosel_selection_by_calcp.csv
- constrained_autosel_eval_selected.csv
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd


# ----------------------------
# utils
# ----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[{name}] Missing: {path}")
    return pd.read_csv(path)

def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[OK] wrote: {path}")

def _norm_split(s: pd.Series) -> pd.Series:
    # normalize split label across tables (strip calcp suffix)
    return s.astype(str).str.replace("__calcp", "", regex=False)

def _require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"[{name}] missing columns: {miss}\nAvailable: {list(df.columns)}")


# ----------------------------
# collectors
# ----------------------------

def collect_from_runs(runs_dir: Path) -> pd.DataFrame:
    """
    Collect all cp*/conformal_metrics.json under each experiment dir.

    Expected dir:
      runs_dir/deepdta_point_*/cp*/conformal_metrics.json
      runs_dir/deepdta_calcp_*/cp*/conformal_metrics.json  (or similar)

    We intentionally do NOT depend on your collect_conformal_tables.py execution permission.
    """
    import json

    rows = []
    exp_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    for exp in exp_dirs:
        for cm_path in exp.glob("cp*/conformal_metrics.json"):
            try:
                with open(cm_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
            except Exception:
                continue

            # we accept either naming; adapt to your metrics json structure
            # Common keys we want:
            # - dataset, split, seed, eval_split
            # - cp_subdir (directory name)
            # - coverage, avg_width
            dataset = m.get("dataset")
            split = m.get("split")
            seed = m.get("seed")
            eval_split = m.get("eval_split", "eval")

            # metrics
            coverage = m.get("coverage", m.get("coverage_mean"))
            avg_width = m.get("avg_width", m.get("width_mean"))

            # fallback: some json store in nested dict
            if coverage is None and isinstance(m.get("metrics"), dict):
                coverage = m["metrics"].get("coverage")
            if avg_width is None and isinstance(m.get("metrics"), dict):
                avg_width = m["metrics"].get("avg_width")

            cp_subdir = cm_path.parent.name

            if dataset is None or split is None or seed is None:
                # last resort: parse from path name like deepdta_point_<dataset>_<split>_seedX
                # only if your exp names follow that convention
                name = exp.name
                # deepdta_point_davis_cold_drug_seed0
                parts = name.split("_")
                if "seed" in name:
                    try:
                        seed = int(parts[-1].replace("seed", ""))
                        dataset = parts[2]
                        split = "_".join(parts[3:-1])
                    except Exception:
                        pass

            if dataset is None or split is None or seed is None:
                continue

            rows.append({
                "dataset": str(dataset),
                "split": str(split),
                "seed": int(seed),
                "eval_split": str(eval_split),
                "cp_subdir": str(cp_subdir),
                "coverage": coverage,
                "avg_width": avg_width,
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"No conformal_metrics.json found under {runs_dir}")

    # clean / types
    df["split"] = _norm_split(df["split"])
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
    df["avg_width"] = pd.to_numeric(df["avg_width"], errors="coerce")
    return df


# ----------------------------
# selection logic (CALCP)
# ----------------------------

def compute_lcb(n: float, coverage_mean: float, coverage_std: float, delta: float) -> float:
    # simple gaussian-ish LCB used in your earlier code; if you use Beta-Binomial LCB, replace here.
    if np.isnan(coverage_std):
        coverage_std = 0.0
    return float(coverage_mean - delta * coverage_std)

def select_by_calcp(calcp_df: pd.DataFrame,
                    target_coverage: float,
                    selection_rule: str,
                    lcb_delta: float) -> pd.DataFrame:
    """
    Input calcp_df must have:
      dataset, split, seed, cp_subdir, alpha, coverage_mean, coverage_std, width_mean, n
    Returns one row per (dataset, split, seed): selected config.
    """
    _require_cols(
        calcp_df,
        ["dataset", "split", "seed", "cp_subdir", "alpha", "coverage_mean", "width_mean"],
        "calcp_all"
    )
    if "coverage_std" not in calcp_df.columns:
        calcp_df["coverage_std"] = 0.0
    if "n" not in calcp_df.columns:
        calcp_df["n"] = np.nan

    df = calcp_df.copy()
    df["met_target"] = df["coverage_mean"] >= target_coverage
    df["coverage_lcb"] = df.apply(
        lambda r: compute_lcb(r["n"], r["coverage_mean"], r["coverage_std"], lcb_delta),
        axis=1
    )
    df["feasible_on_calcp"] = df["coverage_mean"] >= target_coverage

    gcols = ["dataset", "split", "seed"]
    out = []
    for key, g in df.groupby(gcols, sort=False):
        # Only choose among feasible if possible; else choose best (rule-dependent)
        feasible = g[g["feasible_on_calcp"]].copy()
        pool = feasible if len(feasible) > 0 else g.copy()

        if selection_rule == "lcb":
            # choose max LCB, tie-break by smaller width
            pool = pool.sort_values(["coverage_lcb", "width_mean"], ascending=[False, True])
        elif selection_rule == "mean":
            pool = pool.sort_values(["coverage_mean", "width_mean"], ascending=[False, True])
        else:
            raise ValueError(f"Unknown selection_rule: {selection_rule}")

        sel = pool.iloc[0].copy()
        out.append(sel)

    sel_df = pd.DataFrame(out).reset_index(drop=True)
    sel_df["alpha_sel"] = sel_df["alpha"]
    sel_df["selection_rule"] = selection_rule
    sel_df["lcb_delta"] = float(lcb_delta)
    sel_df["target_coverage"] = float(target_coverage)
    return sel_df


# ----------------------------
# build eval_selected (POINTS strict)
# ----------------------------

def build_eval_selected(sel_df: pd.DataFrame,
                        points_df: pd.DataFrame,
                        allow_calcp_fallback: bool,
                        calcp_df_for_fallback: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Join selection with POINTS eval metrics:
      (dataset, split, seed, cp_subdir) -> coverage_eval, avg_width_eval

    If missing and allow_calcp_fallback, fallback to calcp metrics; else raise.
    """
    _require_cols(sel_df, ["dataset", "split", "seed", "cp_subdir", "alpha_sel"], "selection_by_calcp")

    # take only eval rows from points (your paper table wants eval)
    pts = points_df.copy()
    _require_cols(pts, ["dataset", "split", "seed", "cp_subdir", "coverage", "avg_width", "eval_split"], "points_all")

    pts["split"] = _norm_split(pts["split"])
    pts = pts[pts["eval_split"].astype(str) == "eval"].copy()

    key = ["dataset", "split", "seed", "cp_subdir"]
    pts_key = pts[key + ["coverage", "avg_width"]].drop_duplicates()

    merged = sel_df.merge(pts_key, on=key, how="left", indicator=True)
    merged.rename(columns={"coverage": "coverage_eval", "avg_width": "avg_width_eval"}, inplace=True)

    missing = merged[merged["_merge"] != "both"].copy()
    merged.drop(columns=["_merge"], inplace=True)

    if len(missing) > 0:
        if not allow_calcp_fallback:
            msg = missing[key + ["alpha_sel"]].head(50).to_string(index=False)
            raise RuntimeError(
                f"STRICT POINTS mode: {len(missing)} selected rows not found in POINTS eval table.\n"
                f"Examples (first 50):\n{msg}\n"
                f"Fix: ensure those cp_subdir have POINTS eval metrics collected into points table."
            )

        if calcp_df_for_fallback is None:
            raise RuntimeError("allow_calcp_fallback=True but calcp_df_for_fallback is None")

        # fallback fill
        cal = calcp_df_for_fallback.copy()
        cal["split"] = _norm_split(cal["split"])
        _require_cols(cal, key + ["coverage_mean", "width_mean"], "calcp_all_for_fallback")
        cal_key = cal[key + ["coverage_mean", "width_mean"]].drop_duplicates()
        merged2 = merged.merge(cal_key, on=key, how="left")
        fb_mask = merged2["coverage_eval"].isna() | merged2["avg_width_eval"].isna()
        merged2.loc[fb_mask, "coverage_eval"] = merged2.loc[fb_mask, "coverage_mean"]
        merged2.loc[fb_mask, "avg_width_eval"] = merged2.loc[fb_mask, "width_mean"]
        merged2.drop(columns=["coverage_mean", "width_mean"], inplace=True)
        merged2["eval_source"] = np.where(fb_mask, "calcp_fallback", "points")
        merged = merged2
    else:
        merged["eval_source"] = "points"

    return merged


# ----------------------------
# main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--runs_tag_points", required=True)
    ap.add_argument("--runs_tag_calcp", required=True)
    ap.add_argument("--target_coverage", type=float, required=True)
    ap.add_argument("--candidate_alphas", type=str, required=True)
    ap.add_argument("--alpha_fixed", type=float, default=0.1)
    ap.add_argument("--selection_rule", type=str, default="lcb", choices=["lcb", "mean"])
    ap.add_argument("--lcb_delta", type=float, default=0.10)
    ap.add_argument("--allow_calcp_fallback", action="store_true",
                    help="If set, allow eval_selected to fill missing POINTS rows with CALCP metrics. "
                         "Default is strict (no fallback).")
    args = ap.parse_args()

    out_dir = Path("results/tables") / args.date
    _ensure_dir(out_dir)

    # 1) collect POINTS table
    points_runs = Path(args.runs_tag_points)
    pts_all = collect_from_runs(points_runs)
    # NOTE: keep full table for debug
    pts_path = out_dir / "conformal_points_all.csv"
    _write_csv(pts_all, pts_path)

    # 2) collect CALCP table
    calcp_runs = Path(args.runs_tag_calcp)
    calcp_raw = collect_from_runs(calcp_runs)
    calcp_path_raw = out_dir / "conformal_autosel_calcp_all.raw.csv"
    _write_csv(calcp_raw, calcp_path_raw)

    # candidate alphas filter (if your cp_subdir encodes alpha, you may already have alpha column elsewhere;
    # here we parse alpha from cp_subdir suffix "alpha0p05" style if needed.)
    def parse_alpha(cp_subdir: str) -> Optional[float]:
        # cp_local_..._alpha0p05
        if "alpha" not in cp_subdir:
            return None
        s = cp_subdir.split("alpha")[-1]
        # 0p05 / 0p1 etc
        s = s.replace("p", ".")
        try:
            return float(s)
        except Exception:
            return None

    calcp = calcp_raw.copy()
    if "alpha" not in calcp.columns:
        calcp["alpha"] = calcp["cp_subdir"].astype(str).map(parse_alpha)
    # create mean/std columns expected by selection (if your json has only scalar coverage)
    if "coverage_mean" not in calcp.columns:
        calcp["coverage_mean"] = calcp["coverage"]
    if "width_mean" not in calcp.columns:
        calcp["width_mean"] = calcp["avg_width"]
    if "coverage_std" not in calcp.columns:
        calcp["coverage_std"] = 0.0
    if "n" not in calcp.columns:
        calcp["n"] = np.nan

    cand = [float(x) for x in args.candidate_alphas.split(",")]
    calcp = calcp[calcp["alpha"].isin(cand)].copy()

    # write fixed autosel csv (what make_three_way_table.py expects in your pipeline)
    fixed_path = out_dir / "conformal_autosel_all.fixed.csv"
    # for compatibility: include required cols if absent
    fixed = calcp.copy()
    # keep columns minimal but stable
    fixed_cols = ["dataset", "split", "seed", "cp_subdir", "alpha", "coverage_mean", "coverage_std", "width_mean", "n", "eval_split"]
    for c in fixed_cols:
        if c not in fixed.columns:
            fixed[c] = np.nan
    fixed = fixed[fixed_cols]
    _write_csv(fixed, fixed_path)

    # write calcp_all (compat)
    calcp_all_path = out_dir / "conformal_autosel_calcp_all.csv"
    _write_csv(calcp, calcp_all_path)

    # 3) select by CALCP
    sel = select_by_calcp(calcp_df=calcp,
                          target_coverage=args.target_coverage,
                          selection_rule=args.selection_rule,
                          lcb_delta=args.lcb_delta)
    sel_path = out_dir / "constrained_autosel_selection_by_calcp.csv"
    _write_csv(sel, sel_path)

    # 4) build eval_selected STRICT on POINTS (default)
    eval_sel = build_eval_selected(sel_df=sel,
                                   points_df=pts_all,
                                   allow_calcp_fallback=args.allow_calcp_fallback,
                                   calcp_df_for_fallback=calcp)

    eval_path = out_dir / "constrained_autosel_eval_selected.csv"
    _write_csv(eval_sel, eval_path)

    # sanity
    vc = eval_sel["eval_source"].value_counts(dropna=False)
    print("[INFO] eval_source counts:\n", vc.to_string())
    if not args.allow_calcp_fallback:
        assert (eval_sel["eval_source"].astype(str) == "points").all()

    print("[OK] done.")

if __name__ == "__main__":
    main()
