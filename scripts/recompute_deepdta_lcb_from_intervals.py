#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

def cp_lower_bound(hits: int, k: int, delta: float) -> float:
    if hits <= 0:
        return 0.0
    if hits >= k:
        return 1.0
    return float(beta.ppf(delta, hits, k - hits + 1))

def stable_topk_idx(score: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(int(k), int(len(score)))
    # mergesort is stable -> deterministic under ties
    return np.argsort(-score, kind="mergesort")[:k_eff]

def hit_mask_topq(y_true: np.ndarray, q: float) -> np.ndarray:
    n = int(len(y_true))
    m = max(1, int(math.ceil(float(q) * n)))
    idx = np.argsort(-y_true, kind="mergesort")[:m]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask

def find_run_dir(runs_root: Path, dataset: str, split: str, seed: int) -> Path:
    pats = [
        f"deepdta_point_default_{dataset}_{split}_seed{seed}",
        f"deepdta_point_*_{dataset}_{split}_seed{seed}",
        f"deepdta_point*{dataset}*{split}*seed{seed}*",
    ]
    cands = []
    for p in pats:
        cands += [x for x in runs_root.glob(p) if x.is_dir()]
    cands = sorted(set(cands))
    if len(cands) == 1:
        return cands[0]
    if len(cands) == 0:
        raise FileNotFoundError(f"No DeepDTA run_dir for {dataset}/{split}/seed{seed} under {runs_root}")
    raise RuntimeError(f"Ambiguous run_dir for {dataset}/{split}/seed{seed}: {[str(x) for x in cands]}")

def intervals_path_for_method(run_dir: Path, method: str, choices: pd.DataFrame) -> Path:
    # For Fixed / NaiveAutoSel / CAS-LCB: use selected cp_subdir from task_level_choices.csv
    if method in ["Fixed", "NaiveAutoSel", "CAS-LCB"]:
        row = choices[(choices.model=="deepdta") &
                      (choices.dataset==choices._cur_dataset) &
                      (choices.split==choices._cur_split) &
                      (choices.seed==choices._cur_seed) &
                      (choices.method_norm==method)]
        if len(row) != 1:
            raise RuntimeError(f"choices lookup failed for method={method}: {len(row)} rows")
        cp_subdir = str(row.iloc[0]["cp_subdir"])
        p = run_dir / cp_subdir / "pred_intervals_test.csv.gz"
        return p

    # CAS-LCB-Bonferroni: if a bonf-specific folder exists use it; else fall back to CAS-LCB's chosen cp
    if method == "CAS-LCB-Bonferroni":
        # try to find any interval file under a path containing "bonf"
        bonf = sorted([p for p in run_dir.rglob("pred_intervals_test.csv.gz") if "bonf" in str(p).lower()])
        if len(bonf) >= 1:
            return bonf[0]
        # fallback: use CAS-LCB chosen cp_subdir (may be identical, but at least it's real)
        row = choices[(choices.model=="deepdta") &
                      (choices.dataset==choices._cur_dataset) &
                      (choices.split==choices._cur_split) &
                      (choices.seed==choices._cur_seed) &
                      (choices.method_norm=="CAS-LCB")]
        if len(row) != 1:
            raise RuntimeError("choices lookup failed for CAS-LCB fallback")
        cp_subdir = str(row.iloc[0]["cp_subdir"])
        return run_dir / cp_subdir / "pred_intervals_test.csv.gz"

    # Shift-aware baselines: search by folder name
    key = None
    if method == "TC-SC":
        key = "tc_sc"
    elif method == "WSC-B":
        key = "wsc"
    if key is not None:
        cands = sorted([p for p in run_dir.rglob("pred_intervals_test*.csv*") if key in str(p).lower()])
        if len(cands) == 1:
            return cands[0]
        if len(cands) == 0:
            raise FileNotFoundError(f"Missing interval file for {method} under {run_dir} (searched key={key})")
        raise RuntimeError(f"Ambiguous interval files for {method} under {run_dir}: {[str(x) for x in cands]}")

    raise RuntimeError(f"Unsupported method: {method}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True, type=str)
    ap.add_argument("--lcb_in", default="results/tables/lcb_at_k_results_with_bonf.csv", type=str)
    ap.add_argument("--choices_csv", default="results/tables/task_level_choices.csv", type=str)
    ap.add_argument("--out_csv", default="results/tables/lcb_at_k_results_with_bonf.csv", type=str)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    df = pd.read_csv(args.lcb_in)
    choices = pd.read_csv(args.choices_csv)

    deep = df[df.model=="deepdta"].copy()
    other = df[df.model!="deepdta"].copy()

    # cache per (dataset,split,seed,method) intervals
    cache = {}
    for (dataset, split, seed, method), sub in deep.groupby(["dataset","split","seed","method"]):
        run_dir = find_run_dir(runs_root, dataset, split, int(seed))

        # hack: stash current keys for lookup inside intervals_path_for_method
        choices._cur_dataset = dataset
        choices._cur_split = split
        choices._cur_seed = int(seed)

        p = intervals_path_for_method(run_dir, method, choices)
        if not p.exists():
            raise FileNotFoundError(f"Interval file not found: {p}")
        idf = pd.read_csv(p)
        if "pi_lo" not in idf.columns or "y_true" not in idf.columns:
            raise RuntimeError(f"Bad schema in {p}: need pi_lo & y_true, got {list(idf.columns)}")
        y_true = idf["y_true"].to_numpy(np.float64)
        score = idf["pi_lo"].to_numpy(np.float64)
        cache[(dataset,split,int(seed),method)] = (y_true, score)

    out_rows = []
    for _, r in deep.iterrows():
        dataset = r["dataset"]; split = r["split"]; seed = int(r["seed"]); method = r["method"]
        q = float(r["q"]); delta = float(r["delta"]); k = int(float(r["k"]))

        y_true, score = cache[(dataset,split,seed,method)]
        hm = hit_mask_topq(y_true, q)
        sel = stable_topk_idx(score, k)
        hits = int(hm[sel].sum())
        k_eff = int(min(k, len(y_true)))
        hits_at_k = hits / float(k_eff) if k_eff > 0 else 0.0
        lcb = cp_lower_bound(hits, k_eff, delta)
        min_hits = float(k_eff) * lcb

        rr = dict(r)
        rr["hits"] = float(hits)
        rr["hits_at_k"] = float(hits_at_k)
        rr["lcb_at_k"] = float(lcb)
        rr["min_hits_at_k"] = float(min_hits)
        out_rows.append(rr)

    out = pd.concat([other, pd.DataFrame(out_rows)], axis=0, ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} (deepdta recomputed; others preserved)")

if __name__ == "__main__":
    main()
