#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, re
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.stats import beta
except Exception as e:
    raise RuntimeError("scipy is required (for beta.ppf). Install scipy in this env.") from e

def cp_lcb(x:int, n:int, delta:float)->float:
    if n <= 0: return 0.0
    if x <= 0: return 0.0
    if x >= n: return 1.0
    return float(beta.ppf(delta, x, n-x+1))

def stable_topk(score: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), int(len(score)))
    return np.argsort(-score, kind="mergesort")[:k]

def hit_mask_topq(y_true: np.ndarray, q: float) -> np.ndarray:
    n = int(len(y_true))
    m = max(1, int(math.ceil(float(q) * n)))
    idx = np.argsort(-y_true, kind="mergesort")[:m]
    mask = np.zeros(n, dtype=bool); mask[idx] = True
    return mask

def read_intervals(p: Path):
    df = pd.read_csv(p)
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

def find_run_dir(runs_root: Path, dataset: str, split: str, seed: int) -> Path:
    pats = [f"*{dataset}*{split}*seed{seed}*"]
    cands=[]
    for p in pats:
        cands += [x for x in runs_root.glob(p) if x.is_dir()]
    cands = sorted(set(cands))
    pref = [x for x in cands if x.name.startswith("deepdta_point")]
    if len(pref)==1: return pref[0]
    if len(cands)==1: return cands[0]
    if len(cands)==0:
        raise FileNotFoundError(f"No run_dir for {dataset}/{split}/seed{seed} under {runs_root}")
    raise RuntimeError(f"Ambiguous run_dir for {dataset}/{split}/seed{seed}: {[x.name for x in cands[:10]]}...")

def parse_alpha(name: str) -> float|None:
    # alpha0p1 -> 0.1 ; alpha0p02 -> 0.02
    m = re.search(r"alpha(\d+p\d+)", name)
    if not m:
        return None
    return float(m.group(1).replace("p","."))

def parse_k(name: str) -> int|None:
    m = re.search(r"_k(\d+)", name)
    return int(m.group(1)) if m else None

def choice_cp_from_any_choices(dataset: str, split: str, seed: int, method: str) -> str|None:
    # search any task_level_choices.csv under results/
    files = sorted(Path("results").rglob("task_level_choices.csv"))
    need = {"model","dataset","split","seed","method_norm","cp_subdir"}
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if not need.issubset(df.columns):
            continue
        sub = df[
            (df["model"]=="deepdta") &
            (df["dataset"].astype(str)==dataset) &
            (df["split"].astype(str)==split) &
            (df["seed"].astype(int)==int(seed)) &
            (df["method_norm"].astype(str)==method)
        ]
        if len(sub)==1:
            return str(sub.iloc[0]["cp_subdir"])
    return None

def pick_interval_file(run_dir: Path, dataset: str, split: str, seed: int, method: str) -> Path:
    m = method.strip()
    mlow = m.lower()

    # Shift-aware baselines are run-level files (confirmed in your runs)
    if mlow in ["tc-sc","tc_sc","tcsc"]:
        p = run_dir / "pred_intervals_test_tc_sc.csv.gz"
        if not p.exists(): raise FileNotFoundError(str(p))
        return p
    if mlow in ["wsc-b","wsc_b","wsc"]:
        p = run_dir / "pred_intervals_test_wsc_B.csv.gz"
        if not p.exists(): raise FileNotFoundError(str(p))
        return p

    cp_dirs = sorted([d for d in run_dir.glob("cp_*") if d.is_dir() and (d/"pred_intervals_test.csv.gz").exists()])
    if not cp_dirs:
        raise FileNotFoundError(f"No cp_* with pred_intervals_test.csv.gz under {run_dir}")
    names = [d.name for d in cp_dirs]
    low = [n.lower() for n in names]

    def resolve_ambiguous(cand: list[str]) -> str:
        # 1) choices wins (if any)
        cp = choice_cp_from_any_choices(dataset, split, seed, m)
        if cp is not None and cp in cand:
            return cp

        # 2) deterministic heuristics per method
        if m == "NaiveAutoSel":
            # prefer alpha=0.1 then larger k
            scored=[]
            for c in cand:
                a = parse_alpha(c) or -1.0
                k = parse_k(c) or -1
                sc = 0
                if abs(a-0.1) < 1e-9: sc += 1000
                sc += k
                scored.append((sc, c))
            scored.sort(reverse=True)
            return scored[0][1]

        if m == "CAS-LCB":
            # prefer aacomp if present
            pref = [c for c in cand if "aacomp" in c.lower()]
            if len(pref)==1:
                return pref[0]
            # else prefer larger k
            scored=[]
            for c in cand:
                k = parse_k(c) or -1
                scored.append((k, c))
            scored.sort(reverse=True)
            return scored[0][1]

        # fallback: pick lexicographically stable
        return sorted(cand)[0]

    if m == "CAS-LCB-Bonferroni":
        cand = [names[i] for i,s in enumerate(low) if "bonf" in s]
        if len(cand)==0:
            raise RuntimeError(f"Missing bonf cp_subdir under {run_dir}; have={names}")
        pick = cand[0] if len(cand)==1 else resolve_ambiguous(cand)
        return run_dir / pick / "pred_intervals_test.csv.gz"

    if m == "CAS-LCB":
        cand = [names[i] for i,s in enumerate(low) if "constrained_lcb" in s]
        if len(cand)==0:
            cand = [names[i] for i,s in enumerate(low) if ("constrained" in s and "lcb" in s and "bonf" not in s)]
        if len(cand)==0:
            raise RuntimeError(f"Missing constrained_lcb cp_subdir under {run_dir}; have={names}")
        pick = cand[0] if len(cand)==1 else resolve_ambiguous(cand)
        return run_dir / pick / "pred_intervals_test.csv.gz"

    if m == "NaiveAutoSel":
        cand = [names[i] for i,s in enumerate(low) if "autosel" in s]
        if len(cand)==0:
            raise RuntimeError(f"Missing autosel cp_subdir under {run_dir}; have={names}")
        pick = cand[0] if len(cand)==1 else resolve_ambiguous(cand)
        return run_dir / pick / "pred_intervals_test.csv.gz"

    if m == "Fixed":
        cand=[]
        for n,s in zip(names, low):
            if not s.startswith("cp_local_"):
                continue
            if ("autosel" in s) or ("constrained" in s) or ("lcb" in s) or ("bonf" in s):
                continue
            cand.append(n)
        if len(cand)==0:
            raise RuntimeError(f"Missing Fixed baseline cp_local_* under {run_dir}; have={names}")
        # prefer target_tfidf if multiple
        if len(cand)>1:
            pref = [c for c in cand if "target_tfidf" in c.lower()]
            if len(pref)==1:
                cand = pref
        pick = cand[0] if len(cand)==1 else resolve_ambiguous(cand)
        return run_dir / pick / "pred_intervals_test.csv.gz"

    raise RuntimeError(f"Unsupported method={method}. Expected Fixed/NaiveAutoSel/CAS-LCB/CAS-LCB-Bonferroni/TC-SC/WSC-B")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--lcb_in", default="results/tables/lcb_at_k_results_with_bonf.csv")
    ap.add_argument("--out_csv", default="results/tables/lcb_at_k_results_with_bonf.csv")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    df = pd.read_csv(args.lcb_in)
    deep = df[df["model"]=="deepdta"].copy()
    other = df[df["model"]!="deepdta"].copy()

    cache={}
    for (dataset, split, seed, method), _ in deep.groupby(["dataset","split","seed","method"]):
        dataset=str(dataset); split=str(split); seed=int(seed); method=str(method)
        run_dir = find_run_dir(runs_root, dataset, split, seed)
        p = pick_interval_file(run_dir, dataset, split, seed, method)
        cache[(dataset,split,seed,method)] = read_intervals(p)

    out_rows=[]
    for _, r in deep.iterrows():
        dataset=str(r["dataset"]); split=str(r["split"]); seed=int(r["seed"]); method=str(r["method"])
        q=float(r["q"]); delta=float(r["delta"]); k=int(float(r["k"]))

        y_true, score = cache[(dataset,split,seed,method)]
        hm = hit_mask_topq(y_true, q)
        sel_idx = stable_topk(score, k)

        hits = int(hm[sel_idx].sum())
        k_eff = int(min(k, len(y_true)))

        rr = dict(r)
        rr["hits"] = float(hits)
        rr["hits_at_k"] = float(hits/float(k_eff) if k_eff>0 else 0.0)
        rr["lcb_at_k"] = float(cp_lcb(hits, k_eff, delta))
        rr["min_hits_at_k"] = float(k_eff) * rr["lcb_at_k"]
        out_rows.append(rr)

    out = pd.concat([other, pd.DataFrame(out_rows)], ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print("[OK] wrote", args.out_csv, "(deepdta recomputed; others preserved)")

if __name__=="__main__":
    main()
