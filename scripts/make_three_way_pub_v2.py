#!/usr/bin/env python3
"""
make_three_way_pub_v2.py

Build a publication-friendly "three-way" comparison table:

1) baseline_fixed(alpha=alpha_fixed)
   - A fixed, user-specified cp_subdir (baseline_cp_subdir), reported on report_split.

2) search_autosel(alpha=alpha_fixed)
   - On selection_split, choose the best autosel candidate among alpha==alpha_fixed
     by minimizing width (tie-break by higher coverage).
   - Report the chosen candidate on report_split.

3) final_constrained_autosel
   - On selection_split, choose the best autosel candidate using a proxy coverage
     signal (prefers *_calcp / *_lcb if available), subject to proxy >= target_coverage.
     If none feasible, fall back to max proxy coverage (tie-break by min width).
   - Report the chosen candidate on report_split.

4) search_autosel(best_feasible)
   - On selection_split, choose the best autosel candidate among ALL alphas
     that meets coverage >= target_coverage (true selection coverage).
     If none feasible, fall back to max coverage (tie-break by min width).
   - Report the chosen candidate on report_split.

Key design choice:
- best_feasible now ALWAYS returns a choice per seed (via fallback),
  so its n_seeds should match other schemes (typically 5), avoiding
  "subset-of-seeds" comparisons in the main table.

Inputs:
- --fixed_table: a CSV that contains BOTH eval_split==selection_split and eval_split==report_split
  rows (e.g., merged fixed_eval + fixed_test), produced by make_conformal_autosel_all_fixed.py
  and merge_fixed_eval_test.py.

Outputs:
- three_way_main_table_alpha{alpha_tag}_{report_split}_pub_v2.csv
- audit_three_way_choices_{selection_split}_to_{report_split}_pub_v2.csv
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Utilities
# -------------------------

def alpha_to_tag(alpha: float) -> str:
    """Convert 0.1 -> '0p1', 0.02 -> '0p02'."""
    s = f"{alpha:.10f}".rstrip("0").rstrip(".")
    s = s.replace(".", "p")
    # Ensure something like '0p0' does not happen for alpha=0
    return s if "p" in s else f"{s}p0"


def safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def float_close(a: float, b: float, tol: float = 1e-12) -> bool:
    if np.isnan(a) or np.isnan(b):
        return False
    return abs(a - b) <= tol


def parse_alpha_from_cp_subdir(cp_subdir: str) -> float:
    """
    Parse alpha from strings like:
      '..._alpha0p1'
      '..._alpha0p05'
      '..._alpha0p02'
    Returns NaN if not found.
    """
    if not isinstance(cp_subdir, str):
        return float("nan")
    m = re.search(r"alpha(\d+)(?:p(\d+))?", cp_subdir)
    if not m:
        return float("nan")
    a_int = m.group(1)
    a_frac = m.group(2) or "0"
    # alpha0p05 -> 0.05, alpha0p1 -> 0.1
    try:
        return float(f"{a_int}.{a_frac}")
    except Exception:
        return float("nan")


def to_report_cp_subdir(cp_subdir: str) -> str:
    """
    Map a selection-time cp_subdir to the corresponding report-time cp_subdir.

    Some pipelines suffix selection-only markers, e.g.:
      - '_calcp' (CalCP / proxy split)
      - '_lcbXX' (proxy lower confidence bound)
      - '_evalOnCalCP', '_safe'

    Report-time directories usually omit these markers. We strip them for report lookup.
    """
    if not isinstance(cp_subdir, str):
        return str(cp_subdir)

    s = cp_subdir
    if s.endswith("_calcp"):
        s = s[:-len("_calcp")]
    s = re.sub(r"_lcb\d+$", "", s)
    s = s.replace("_evalOnCalCP", "")
    s = s.replace("_safe", "")
    return s



def detect_metric_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detect (coverage_col, width_col) used for reporting.

    We prefer the standardized columns produced by make_conformal_autosel_all_fixed.py:
      - coverage_mean
      - width_mean

    Falls back to common alternatives if needed.
    """
    cov_candidates = [
        "coverage_mean",
        "coverage",
        "coverage_eval",
        "coverage_test",
        "coverage_split",
    ]
    width_candidates = [
        "width_mean",
        "avg_width_mean",
        "avg_width",
        "width",
        "avg_width_eval",
        "avg_width_test",
    ]
    cov_col = next((c for c in cov_candidates if c in df.columns), None)
    width_col = next((c for c in width_candidates if c in df.columns), None)
    if cov_col is None or width_col is None:
        raise RuntimeError(
            f"Cannot detect metric columns. Need one of {cov_candidates} and one of {width_candidates}. "
            f"Found columns: {list(df.columns)[:50]} ..."
        )
    return cov_col, width_col


def detect_proxy_cov_col(df: pd.DataFrame) -> Optional[str]:
    """
    Detect a proxy coverage column for constrained selection.

    Preference order:
      - LCB-like calcp coverage
      - calcp coverage
      - (None) -> caller should fall back to selection coverage_mean
    """
    candidates = [
        # LCB variants (most preferred)
        "coverage_calcp_lcb",
        "coverage_calcp_lcb_mean",
        "coverage_lcb_calcp",
        "coverage_calcp_lcb10",
        "coverage_calcp_lcb_10",
        "coverage_calcp_all_lcb",
        "coverage_lcb",
        "coverage_lcb_mean",
        # Non-LCB calcp coverage
        "coverage_calcp",
        "coverage_calcp_mean",
        "coverage_calcp_all",
        "coverage_calcp_all_mean",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def met_target_rate_from_coverage(cov: pd.Series, target: float) -> float:
    """Fraction of seeds with coverage >= target. NaN counts as not meeting target."""
    if cov is None or len(cov) == 0:
        return float("nan")
    ok = cov.fillna(-np.inf) >= target
    return float(ok.mean())


def mode_or_min(values: pd.Series) -> float:
    """Mode of a numeric series; if multi-modal, return the smallest value."""
    v = values.dropna()
    if len(v) == 0:
        return float("nan")
    m = v.mode()
    if len(m) == 0:
        return float("nan")
    return float(m.min())


# -------------------------
# Selection logic
# -------------------------

@dataclass
class Choice:
    cp_subdir: str
    alpha: float
    # selection metrics
    sel_cov: float
    sel_width: float
    sel_proxy_cov: float
    # feasibility flags
    feasible: bool
    used_fallback: bool


def _prep_candidates(
    df_sel: pd.DataFrame,
    cov_col: str,
    width_col: str,
    alpha_col: str,
    *,
    restrict_autosel: bool = True,
) -> pd.DataFrame:
    """
    Prepare candidates on selection split.

    If restrict_autosel=True, prefer cp_subdir containing 'autosel_target'.
    If that yields no rows, fall back to all rows.
    """
    cand = df_sel.copy()
    cand = cand.dropna(subset=[cov_col, width_col])

    if restrict_autosel and "cp_subdir" in cand.columns:
        mask = cand["cp_subdir"].astype(str).str.contains("autosel_target", case=False, na=False)
        autosel = cand[mask]
        if len(autosel) > 0:
            cand = autosel

    # Ensure alpha exists
    if alpha_col not in cand.columns:
        cand[alpha_col] = cand["cp_subdir"].astype(str).map(parse_alpha_from_cp_subdir)

    return cand


def choose_search_autosel_fixed_alpha(
    df_sel: pd.DataFrame,
    cov_col: str,
    width_col: str,
    alpha_col: str,
    alpha_fixed: float,
) -> Optional[Choice]:
    """
    On selection split, among autosel candidates with alpha==alpha_fixed,
    pick the one with min width (tie-break by max coverage).

    If no candidate with alpha_fixed exists, return None.
    """
    cand = _prep_candidates(df_sel, cov_col, width_col, alpha_col, restrict_autosel=True)
    if len(cand) == 0:
        return None

    # Filter by alpha_fixed (tolerant compare by rounding to 1e-12).
    cand = cand[cand[alpha_col].apply(lambda a: float_close(safe_float(a), alpha_fixed))]
    if len(cand) == 0:
        return None

    cand = cand.sort_values([width_col, cov_col], ascending=[True, False])
    row = cand.iloc[0]
    return Choice(
        cp_subdir=str(row["cp_subdir"]),
        alpha=float(row[alpha_col]),
        sel_cov=float(row[cov_col]),
        sel_width=float(row[width_col]),
        sel_proxy_cov=float(row[cov_col]),
        feasible=False,
        used_fallback=False,
    )


def choose_best_feasible_with_fallback(
    df_sel: pd.DataFrame,
    cov_col: str,
    width_col: str,
    alpha_col: str,
    target_coverage: float,
) -> Optional[Choice]:
    """
    best_feasible:
      - If any candidate has selection coverage >= target, pick the feasible candidate
        with minimum width (tie-break by higher coverage).
      - Else fall back to the candidate with maximum coverage (tie-break by minimum width).

    This guarantees one choice per seed (as long as there is at least one candidate row).
    """
    cand = _prep_candidates(df_sel, cov_col, width_col, alpha_col, restrict_autosel=True)
    if len(cand) == 0:
        return None

    feasible = cand[cand[cov_col] >= target_coverage]
    if len(feasible) > 0:
        feasible = feasible.sort_values([width_col, cov_col], ascending=[True, False])
        row = feasible.iloc[0]
        return Choice(
            cp_subdir=str(row["cp_subdir"]),
            alpha=float(row[alpha_col]),
            sel_cov=float(row[cov_col]),
            sel_width=float(row[width_col]),
            sel_proxy_cov=float(row[cov_col]),
            feasible=True,
            used_fallback=False,
        )

    # Fallback: max coverage, tie-break by min width
    cand = cand.sort_values([cov_col, width_col], ascending=[False, True])
    row = cand.iloc[0]
    return Choice(
        cp_subdir=str(row["cp_subdir"]),
        alpha=float(row[alpha_col]),
        sel_cov=float(row[cov_col]),
        sel_width=float(row[width_col]),
        sel_proxy_cov=float(row[cov_col]),
        feasible=False,
        used_fallback=True,
    )


def choose_constrained_with_proxy(
    df_sel: pd.DataFrame,
    cov_col: str,
    width_col: str,
    alpha_col: str,
    target_coverage: float,
    proxy_cov_col: Optional[str],
) -> Optional[Choice]:
    """
    constrained selection:
      - Use proxy_cov_col if provided; otherwise fall back to selection coverage (cov_col).
      - If any candidate satisfies proxy_cov >= target, pick minimum width (tie-break by higher proxy_cov).
      - Else fall back to maximum proxy_cov (tie-break by minimum width).

    Note: This is designed to be robust to different fixed_table schemas.
    """
    cand = _prep_candidates(df_sel, cov_col, width_col, alpha_col, restrict_autosel=True)
    if len(cand) == 0:
        return None

    proxy = proxy_cov_col if (proxy_cov_col is not None and proxy_cov_col in cand.columns) else cov_col
    cand = cand.dropna(subset=[proxy])

    feasible = cand[cand[proxy] >= target_coverage]
    if len(feasible) > 0:
        feasible = feasible.sort_values([width_col, proxy], ascending=[True, False])
        row = feasible.iloc[0]
        return Choice(
            cp_subdir=str(row["cp_subdir"]),
            alpha=float(row[alpha_col]),
            sel_cov=float(row[cov_col]),
            sel_width=float(row[width_col]),
            sel_proxy_cov=float(row[proxy]),
            feasible=True,
            used_fallback=False,
        )

    cand = cand.sort_values([proxy, width_col], ascending=[False, True])
    row = cand.iloc[0]
    return Choice(
        cp_subdir=str(row["cp_subdir"]),
        alpha=float(row[alpha_col]),
        sel_cov=float(row[cov_col]),
        sel_width=float(row[width_col]),
        sel_proxy_cov=float(row[proxy]),
        feasible=False,
        used_fallback=True,
    )


# -------------------------
# Summaries
# -------------------------

def summarize_per_group(df: pd.DataFrame, target_coverage: float) -> pd.Series:
    """
    Summarize per (scheme, dataset, split) across seeds.
    """
    cov = df["report_coverage"]
    wid = df["report_width"]
    alpha = df["alpha_sel"]

    out = {
        "n_seeds": int(df["seed"].nunique()),
        "coverage_mean": float(cov.mean()) if len(cov) else float("nan"),
        "coverage_std": float(cov.std()) if len(cov) else float("nan"),
        "width_mean": float(wid.mean()) if len(wid) else float("nan"),
        "width_std": float(wid.std()) if len(wid) else float("nan"),
        "alpha_sel_mode": mode_or_min(alpha),
        "alpha_sel_mean": float(alpha.mean()) if len(alpha) else float("nan"),
        "met_target_rate": met_target_rate_from_coverage(cov, target_coverage),
    }

    # Nominal target is (1 - alpha). Report mean nominal target to align with variable-alpha schemes.
    nominal_target_mean = float((1.0 - alpha).mean()) if len(alpha) else float("nan")
    out["nominal_target_mean"] = nominal_target_mean

    # Helpful deltas
    out["coverage_gap_to_target"] = out["coverage_mean"] - target_coverage if not np.isnan(out["coverage_mean"]) else float("nan")
    out["coverage_gap_to_nominal"] = out["coverage_mean"] - nominal_target_mean if not np.isnan(out["coverage_mean"]) else float("nan")

    return pd.Series(out)


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Output directory (tables will be written here).")
    ap.add_argument(
        "--fixed_table",
        required=True,
        help="Merged fixed table containing both eval_split==selection_split and eval_split==report_split rows.",
    )
    ap.add_argument("--target_coverage", type=float, required=True)
    ap.add_argument("--alpha_fixed", type=float, required=True)
    ap.add_argument(
        "--selection_split",
        default="eval",
        help="Value of the eval_split column used for selection (e.g., eval, test, calcp).",
    )
    ap.add_argument(
        "--report_split",
        default="test",
        help="Value of the eval_split column used for reporting (e.g., test, eval).",
    )
    ap.add_argument("--baseline_cp_subdir", required=True)
    ap.add_argument(
        "--allow_report_fallback",
        action="store_true",
        help="If report metrics are missing for a chosen cp_subdir, fall back to selection metrics.",
    )
    args = ap.parse_args()

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_path = Path(args.fixed_table)
    df = pd.read_csv(fixed_path)

    required_cols = ["dataset", "split", "seed", "cp_subdir", "eval_split"]
    available_splits = set(df["eval_split"].astype(str).unique())
    if args.selection_split not in available_splits:
        raise ValueError(
            f"selection_split='{args.selection_split}' not found in eval_split values: {sorted(available_splits)}"
        )
    if args.report_split not in available_splits:
        raise ValueError(
            f"report_split='{args.report_split}' not found in eval_split values: {sorted(available_splits)}"
        )
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"fixed_table is missing required columns: {missing}. Found: {list(df.columns)[:60]} ...")

    cov_col, width_col = detect_metric_cols(df)

    # Ensure alpha column exists in df (best effort).
    if "alpha" not in df.columns:
        df["alpha"] = df["cp_subdir"].astype(str).map(parse_alpha_from_cp_subdir)
    alpha_col = "alpha"

    # Split into selection and report views.
    sel_df = df[df["eval_split"] == args.selection_split].copy()
    rep_df = df[df["eval_split"] == args.report_split].copy()

    print(f"[INFO] selection_split={args.selection_split} rows={len(sel_df)}")
    print(f"[INFO] report_split={args.report_split} rows={len(rep_df)}")
    print(f"[INFO] baseline_cp_subdir = {args.baseline_cp_subdir}")

    proxy_cov_col = detect_proxy_cov_col(sel_df)

    # Index report metrics for fast lookup
    rep_idx_cols = ["dataset", "split", "seed", "cp_subdir"]
    rep_map = rep_df.set_index(rep_idx_cols)

    sel_map = sel_df.set_index(rep_idx_cols)  # for fallback

    # Build per-seed choices across schemes
    choices_rows: List[Dict] = []

    group_keys = ["dataset", "split", "seed"]
    for (dataset, split, seed), g_sel in sel_df.groupby(group_keys, sort=True):
        # --- baseline_fixed ---
        base_cp = args.baseline_cp_subdir
        base_alpha = args.alpha_fixed
        base_row = {
            "scheme": f"baseline_fixed(alpha={args.alpha_fixed})",
            "dataset": dataset,
            "split": split,
            "seed": int(seed),
            "cp_subdir": base_cp,
            "alpha_sel": float(base_alpha),
            "selection_cov": float("nan"),
            "selection_width": float("nan"),
            "selection_proxy_cov": float("nan"),
            "selection_feasible": False,
            "selection_used_fallback": False,
        }

        key_sel = (dataset, split, seed, base_cp)
        base_cp_report = to_report_cp_subdir(base_cp)
        base_row["report_cp_subdir"] = base_cp_report
        key = (dataset, split, seed, base_cp_report)
        if key in rep_map.index:
            r = rep_map.loc[key]
            base_row["report_coverage"] = safe_float(r.get(cov_col))
            base_row["report_width"] = safe_float(r.get(width_col))
            base_row["report_used_fallback"] = False
        elif args.allow_report_fallback and key_sel in sel_map.index:
            r = sel_map.loc[key]
            base_row["report_coverage"] = safe_float(r.get(cov_col))
            base_row["report_width"] = safe_float(r.get(width_col))
            base_row["report_used_fallback"] = True
        else:
            base_row["report_coverage"] = float("nan")
            base_row["report_width"] = float("nan")
            base_row["report_used_fallback"] = False

        choices_rows.append(base_row)

        # --- search_autosel(alpha_fixed) ---
        ch_search = choose_search_autosel_fixed_alpha(g_sel, cov_col, width_col, alpha_col, args.alpha_fixed)
        if ch_search is not None:
            row = {
                "scheme": f"search_autosel(alpha={args.alpha_fixed})",
                "dataset": dataset,
                "split": split,
                "seed": int(seed),
                "cp_subdir": ch_search.cp_subdir,
                "alpha_sel": float(ch_search.alpha),
                "selection_cov": float(ch_search.sel_cov),
                "selection_width": float(ch_search.sel_width),
                "selection_proxy_cov": float(ch_search.sel_proxy_cov),
                "selection_feasible": bool(ch_search.feasible),
                "selection_used_fallback": bool(ch_search.used_fallback),
            }
            key2_sel = (dataset, split, seed, ch_search.cp_subdir)
            cp2_report = to_report_cp_subdir(ch_search.cp_subdir)
            row["report_cp_subdir"] = cp2_report
            key2 = (dataset, split, seed, cp2_report)
            if key2 in rep_map.index:
                r = rep_map.loc[key2]
                row["report_coverage"] = safe_float(r.get(cov_col))
                row["report_width"] = safe_float(r.get(width_col))
                row["report_used_fallback"] = False
            elif args.allow_report_fallback and key2_sel in sel_map.index:
                r = sel_map.loc[key2]
                row["report_coverage"] = safe_float(r.get(cov_col))
                row["report_width"] = safe_float(r.get(width_col))
                row["report_used_fallback"] = True
            else:
                row["report_coverage"] = float("nan")
                row["report_width"] = float("nan")
                row["report_used_fallback"] = False
            choices_rows.append(row)

        # --- final_constrained_autosel ---
        ch_con = choose_constrained_with_proxy(g_sel, cov_col, width_col, alpha_col, args.target_coverage, proxy_cov_col)
        if ch_con is not None:
            row = {
                "scheme": "final_constrained_autosel",
                "dataset": dataset,
                "split": split,
                "seed": int(seed),
                "cp_subdir": ch_con.cp_subdir,
                "alpha_sel": float(ch_con.alpha),
                "selection_cov": float(ch_con.sel_cov),
                "selection_width": float(ch_con.sel_width),
                "selection_proxy_cov": float(ch_con.sel_proxy_cov),
                "selection_feasible": bool(ch_con.feasible),
                "selection_used_fallback": bool(ch_con.used_fallback),
            }
            key3_sel = (dataset, split, seed, ch_con.cp_subdir)
            cp3_report = to_report_cp_subdir(ch_con.cp_subdir)
            row["report_cp_subdir"] = cp3_report
            key3 = (dataset, split, seed, cp3_report)
            if key3 in rep_map.index:
                r = rep_map.loc[key3]
                row["report_coverage"] = safe_float(r.get(cov_col))
                row["report_width"] = safe_float(r.get(width_col))
                row["report_used_fallback"] = False
            elif args.allow_report_fallback and key3_sel in sel_map.index:
                r = sel_map.loc[key3]
                row["report_coverage"] = safe_float(r.get(cov_col))
                row["report_width"] = safe_float(r.get(width_col))
                row["report_used_fallback"] = True
            else:
                row["report_coverage"] = float("nan")
                row["report_width"] = float("nan")
                row["report_used_fallback"] = False
            choices_rows.append(row)

        # --- search_autosel(best_feasible) with fallback (guarantee a choice) ---
        ch_bf = choose_best_feasible_with_fallback(g_sel, cov_col, width_col, alpha_col, args.target_coverage)
        if ch_bf is not None:
            row = {
                "scheme": "search_autosel(best_feasible)",
                "dataset": dataset,
                "split": split,
                "seed": int(seed),
                "cp_subdir": ch_bf.cp_subdir,
                "alpha_sel": float(ch_bf.alpha),
                "selection_cov": float(ch_bf.sel_cov),
                "selection_width": float(ch_bf.sel_width),
                "selection_proxy_cov": float(ch_bf.sel_proxy_cov),
                "selection_feasible": bool(ch_bf.feasible),
                "selection_used_fallback": bool(ch_bf.used_fallback),
            }
            key4_sel = (dataset, split, seed, ch_bf.cp_subdir)
            cp4_report = to_report_cp_subdir(ch_bf.cp_subdir)
            row["report_cp_subdir"] = cp4_report
            key4 = (dataset, split, seed, cp4_report)
            if key4 in rep_map.index:
                r = rep_map.loc[key4]
                row["report_coverage"] = safe_float(r.get(cov_col))
                row["report_width"] = safe_float(r.get(width_col))
                row["report_used_fallback"] = False
            elif args.allow_report_fallback and key4_sel in sel_map.index:
                r = sel_map.loc[key4]
                row["report_coverage"] = safe_float(r.get(cov_col))
                row["report_width"] = safe_float(r.get(width_col))
                row["report_used_fallback"] = True
            else:
                row["report_coverage"] = float("nan")
                row["report_width"] = float("nan")
                row["report_used_fallback"] = False
            choices_rows.append(row)

    choices = pd.DataFrame(choices_rows)

    # Write audit file (per-seed choices)
    audit_name = f"audit_three_way_choices_{args.selection_split}_to_{args.report_split}_pub_v2.csv"
    audit_path = out_dir / audit_name
    choices.to_csv(audit_path, index=False)
    print(f"[OK] wrote audit: {audit_path}")

    # If report metrics are missing and fallback is NOT allowed, warn with a CSV.
    missing_report = choices[choices["report_coverage"].isna() | choices["report_width"].isna()].copy()
    if len(missing_report) > 0 and not args.allow_report_fallback:
        miss_path = out_dir / "missing_report_metrics.csv"
        cols = ["scheme", "dataset", "split", "seed", "cp_subdir"]
        missing_report[cols].drop_duplicates().to_csv(miss_path, index=False)
        raise RuntimeError(
            f"Missing report metrics for {len(missing_report)} rows. "
            f"Wrote unique missing keys to: {miss_path}. "
            f"Either backfill those report metrics or pass --allow_report_fallback."
        )

    # Drop rows that still have missing report metrics (should not happen if allow_report_fallback and selection exists).
    choices_clean = choices.dropna(subset=["report_coverage", "report_width"]).copy()

    # Summarize into main table
    main = (
        choices_clean
        .groupby(["scheme", "dataset", "split"], sort=True)
        .apply(lambda d: summarize_per_group(d, args.target_coverage))
        .reset_index()
    )

    alpha_tag = alpha_to_tag(args.alpha_fixed)
    main_name = f"three_way_main_table_alpha{alpha_tag}_{args.report_split}_pub_v2.csv"
    main_path = out_dir / main_name
    main.to_csv(main_path, index=False)
    print(f"[OK] wrote main:  {main_path}")

    # Console preview (similar to your logs)
    print("\nMain table preview:\n")
    with pd.option_context("display.max_rows", 60, "display.width", 200):
        print(main.to_string(index=False))


if __name__ == "__main__":
    main()
