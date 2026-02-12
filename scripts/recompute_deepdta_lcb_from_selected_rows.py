#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

def clopper_pearson_lcb(x: int, n: int, delta: float) -> float:
    # lower bound for Binomial proportion with one-sided (delta) using Beta inverse CDF
    if n <= 0:
        return 0.0
    if x <= 0:
        return 0.0
    if x >= n:
        return 1.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(delta, x, n - x + 1))
    except Exception as e:
        raise RuntimeError("Need scipy for Clopper-Pearson beta ppf (install scipy).") from e

def stable_topk(score: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), int(len(score)))
    return np.argsort(-score, kind="mergesort")[:k]

def hit_mask_topq(y_true: np.ndarray, q: float) -> np.ndarray:
    n = int(len(y_true))
    m = max(1, int(math.ceil(float(q) * n)))
    idx = np.argsort(-y_true, kind="mergesort")[:m]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask

def read_intervals(p: Path):
    df = pd.read_csv(p)
    # robust column names
    if "pi_lo" not in df.columns:
        raise RuntimeError(f"{p} missing pi_lo; cols={list(df.columns)}")
    if "y_true" in df.columns:
        y = df["y_true"].to_numpy(np.float64)
    elif "y" in df.columns:
        y = df["y"].to_numpy(np.float64)
    else:
        raise RuntimeError(f"{p} missing y_true/y; cols={list(df.columns)}")
    s = df["pi_lo"].to_numpy(np.float64)
    return y, s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcb_in", default="results/tables/lcb_at_k_results_with_bonf.csv")
    ap.add_argument("--selected_rows_csv", required=True)
    ap.add_argument("--out_csv", default="results/tables/lcb_at_k_results_with_bonf.csv")
    ap.add_argument("--strict", action="store_true", help="error if any interval file missing")
    args = ap.parse_args()

    lcb = pd.read_csv(args.lcb_in)
    sel = pd.read_csv(args.selected_rows_csv)

    # build mapping: (method, dataset, split, seed) -> (run_dir, cp_subdir)
    sel = sel[(sel["model"] == "deepdta")].copy()
    need_cols = {"method","dataset","split","seed","run_dir","cp_subdir"}
    miss = need_cols - set(sel.columns)
    if miss:
        raise RuntimeError(f"selected_rows_csv missing cols: {miss}")

    key2path = {}
    for _, r in sel.iterrows():
        k = (str(r["method"]), str(r["dataset"]), str(r["split"]), int(r["seed"]))
        key2path[k] = (Path(str(r["run_dir"])), str(r["cp_subdir"]))

    deep = lcb[lcb["model"] == "deepdta"].copy()
    other = lcb[lcb["model"] != "deepdta"].copy()

    # cache intervals per (method,dataset,split,seed)
    cache = {}
    missing_files = []
    for (method, dataset, split, seed), _ in deep.groupby(["method","dataset","split","seed"]):
        kk = (str(method), str(dataset), str(split), int(seed))
        if kk not in key2path:
            raise RuntimeError(f"Mapping missing for {kk} in {args.selected_rows_csv}")
        run_dir, cp_subdir = key2path[kk]
        p = run_dir / cp_subdir / "pred_intervals_test.csv.gz"
        if not p.exists():
            missing_files.append(str(p))
            if args.strict:
                raise FileNotFoundError(str(p))
            continue
        cache[kk] = read_intervals(p)

    if missing_files:
        print("[WARN] missing interval files:", len(missing_files))
        print("\n".join(missing_files[:5]))

    out_rows = []
    for _, r in deep.iterrows():
        method = str(r["method"]); dataset = str(r["dataset"]); split = str(r["split"]); seed = int(r["seed"])
        q = float(r["q"]); delta = float(r["delta"]); k = int(float(r["k"]))

        kk = (method, dataset, split, seed)
        if kk not in cache:
            # keep original row if missing
            out_rows.append(dict(r))
            continue

        y_true, score = cache[kk]
        hm = hit_mask_topq(y_true, q)
        sel_idx = stable_topk(score, k)
        hits = int(hm[sel_idx].sum())
        k_eff = int(min(k, len(y_true)))
        hits_at_k = hits / float(k_eff) if k_eff > 0 else 0.0
        lcb_at_k = clopper_pearson_lcb(hits, k_eff, delta)
        min_hits = float(k_eff) * lcb_at_k

        rr = dict(r)
        rr["hits"] = float(hits)
        rr["hits_at_k"] = float(hits_at_k)
        rr["lcb_at_k"] = float(lcb_at_k)
        rr["min_hits_at_k"] = float(min_hits)
        out_rows.append(rr)

    out = pd.concat([other, pd.DataFrame(out_rows)], ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} (deepdta recomputed from pi_lo; others preserved)")

if __name__ == "__main__":
    main()
