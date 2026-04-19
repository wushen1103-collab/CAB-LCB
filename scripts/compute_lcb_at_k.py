#!/usr/bin/env python3
"""
Compute screening utility curves (Hits@k, LCB@k, MinHits@k) from per-test interval artifacts.

Key fix vs v1:
- cp_subdir is looked up PER (dataset, split, seed, scheme) from the audit CSV
  (because CAS-LCB cp_subdir varies by setting).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np

KEY_PATTERNS = [
    re.compile(r"deepdta_point_(davis|kiba)_(random|cold_drug|cold_target|cold_pair)_seed(\d+)$"),
    re.compile(r"deepdta_point_(davis|kiba)_(random|cold_drug|cold_target|cold_pair)__calcp_seed(\d+)$"),
    re.compile(r"graphdta_point_gat_gcn_(davis|kiba)_(random|cold_drug|cold_target|cold_pair)_seed(\d+)$"),
]

SCHEME_TO_METHOD = {
    "baseline_fixed(alpha=0.1)": "Fixed",
    "search_autosel(alpha=0.1)": "NaiveAutoSel",
    "final_constrained_autosel": "CAS-LCB",
    "final_constrained_autosel_bonf": "CAS-LCB-Bonferroni",
}

def parse_key(run_dir_name: str):
    s = run_dir_name.lower()
    for pat in KEY_PATTERNS:
        m = pat.match(s)
        if m:
            return (m.group(1), m.group(2), int(m.group(3)))
    return None

def find_point_run_dirs(runs_root: Path) -> list[Path]:
    return sorted({p.parent for p in runs_root.rglob("preds_test.csv.gz")})

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    for k in candidates:
        k2 = k.lower()
        for cl, orig in cols.items():
            if k2 in cl:
                return orig
    raise KeyError(f"Cannot find any of {candidates} in columns={list(df.columns)}")

def clopper_pearson_lower(x: int, n: int, delta: float) -> float:
    if n <= 0:
        return float("nan")
    if x <= 0:
        return 0.0
    if x >= n:
        return 1.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(delta, x, n - x + 1))
    except Exception:
        # Wilson fallback (approx)
        import math
        try:
            from scipy.stats import norm
            z = float(norm.ppf(1 - delta))
        except Exception:
            z = 1.6448536269514722 if abs(delta-0.05) < 1e-9 else 1.2815515655446004
        phat = x / n
        denom = 1 + z*z/n
        center = (phat + z*z/(2*n)) / denom
        half = z * math.sqrt((phat*(1-phat) + z*z/(4*n))/n) / denom
        return float(max(0.0, center - half))

def load_intervals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    y_col  = pick_col(df, ["y_true", "label", "y", "affinity"])   # prefer y_true if exists
    lo_col = pick_col(df, ["pi_lo", "lower", "lo"])
    hi_col = pick_col(df, ["pi_hi", "upper", "hi"])
    out = pd.DataFrame({
        "y":     pd.to_numeric(df[y_col], errors="coerce"),
        "pi_lo": pd.to_numeric(df[lo_col], errors="coerce"),
        "pi_hi": pd.to_numeric(df[hi_col], errors="coerce"),
    }).dropna()
    return out

def compute_curves(df: pd.DataFrame, q: float, ks: list[int], delta: float) -> pd.DataFrame:
    y = df["y"].to_numpy()
    score = df["pi_lo"].to_numpy()
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    # hit = top-q quantile by y (scale-robust for Davis vs KIBA)
    thr = np.quantile(y, 1 - q)
    hit = (y >= thr).astype(int)

    order = np.argsort(-score, kind="mergesort")
    hit_sorted = hit[order]

    rows=[]
    for k in ks:
        if k <= 0 or k > n:
            continue
        x = int(hit_sorted[:k].sum())
        lcb = clopper_pearson_lower(x, k, delta)
        rows.append({
            "k": int(k),
            "hits": x,
            "hits_at_k": x / k,
            "lcb_at_k": lcb,
            "min_hits_at_k": lcb * k,
        })
    return pd.DataFrame(rows)

def build_audit_map(audit_csv: Path) -> dict[tuple[str,str,int,str], str]:
    """
    return {(dataset, split, seed, scheme): cp_subdir_str}
    Prefer report_cp_subdir if present, else cp_subdir
    """
    audit = pd.read_csv(audit_csv)
    cp_col = "report_cp_subdir" if "report_cp_subdir" in audit.columns else "cp_subdir"
    m = {}
    for _, r in audit.iterrows():
        ds = str(r["dataset"]).lower()
        sp = str(r["split"]).lower()
        sd = int(r["seed"])
        sc = str(r["scheme"])
        cp = str(r[cp_col])
        m[(ds, sp, sd, sc)] = cp
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=".")
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--audit_graphdta", type=str, required=True)
    ap.add_argument("--audit_deepdta", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--q_list", type=float, nargs="*", default=[0.01, 0.05, 0.10])
    ap.add_argument("--delta_list", type=float, nargs="*", default=[0.05])
    ap.add_argument("--k_list", type=int, nargs="*", default=[10, 20, 50, 100, 200, 500, 1000])
    ap.add_argument("--include_tc_sc", action="store_true")
    ap.add_argument("--include_wsc_b", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    runs_root = repo / args.runs_dir

    map_g = build_audit_map(repo / args.audit_graphdta)
    map_d = build_audit_map(repo / args.audit_deepdta)

    run_dirs = find_point_run_dirs(runs_root)
    rows_out=[]

    for run_dir in run_dirs:
        name = run_dir.name.lower()
        model = "graphdta" if name.startswith("graphdta_point") else ("deepdta" if name.startswith("deepdta_point") else None)
        if model is None:
            continue
        key = parse_key(run_dir.name)
        if key is None:
            continue
        dataset, split, seed = key
        audit_map = map_g if model=="graphdta" else map_d

        # 3 CP-based methods from audit
        for scheme, method_name in SCHEME_TO_METHOD.items():
            cp_subdir = audit_map.get((dataset, split, seed, scheme), None)
            if not cp_subdir:
                continue
            intervals_path = run_dir / cp_subdir / "pred_intervals_test.csv.gz"
            if not intervals_path.exists():
                continue
            df = load_intervals(intervals_path)
            for q in args.q_list:
                for delta in args.delta_list:
                    curves = compute_curves(df, q=q, ks=args.k_list, delta=delta)
                    for _, r in curves.iterrows():
                        rows_out.append({
                            "model": model,
                            "dataset": dataset,
                            "split": split,
                            "seed": seed,
                            "method": method_name,
                            "q": q,
                            "delta": delta,
                            **r.to_dict()
                        })

        # shift-aware methods at run root (if present)
        if args.include_tc_sc:
            p = run_dir / "pred_intervals_test_tc_sc.csv.gz"
            if p.exists():
                df = load_intervals(p)
                for q in args.q_list:
                    for delta in args.delta_list:
                        curves = compute_curves(df, q=q, ks=args.k_list, delta=delta)
                        for _, r in curves.iterrows():
                            rows_out.append({
                                "model": model, "dataset": dataset, "split": split, "seed": seed,
                                "method": "TC-SC", "q": q, "delta": delta, **r.to_dict()
                            })

        if args.include_wsc_b:
            p = run_dir / "pred_intervals_test_wsc_B.csv.gz"
            if p.exists():
                df = load_intervals(p)
                for q in args.q_list:
                    for delta in args.delta_list:
                        curves = compute_curves(df, q=q, ks=args.k_list, delta=delta)
                        for _, r in curves.iterrows():
                            rows_out.append({
                                "model": model, "dataset": dataset, "split": split, "seed": seed,
                                "method": "WSC-B", "q": q, "delta": delta, **r.to_dict()
                            })

    out_df = pd.DataFrame(rows_out)
    out_path = repo / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[OK] wrote {len(out_df)} rows -> {out_path}")

if __name__ == "__main__":
    main()
