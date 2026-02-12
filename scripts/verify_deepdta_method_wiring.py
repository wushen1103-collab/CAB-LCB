#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify DeepDTA method wiring:
- method -> correct interval file (cp_* subdir or run-level file)
- fingerprints show methods are not all pointing to the same artifact
- sampled recomputation of hits/lcb/minhits matches the table

Usage:
  python scripts/verify_deepdta_method_wiring.py \
    --runs_root runs/2025-12-23_deepdta_points \
    --lcb_csv results/tables/lcb_at_k_results_with_bonf.csv \
    --sample_rows 40

Optional:
  --choices_glob "results/**/task_level_choices.csv"
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

try:
    from scipy.stats import beta
except Exception as e:
    raise RuntimeError("scipy is required (for beta.ppf). Install scipy in this env.") from e


METHODS_CANON = ["Fixed", "NaiveAutoSel", "CAS-LCB", "CAS-LCB-Bonf", "TC-SC", "WSC-B"]


def cp_lcb(x: int, n: int, delta: float) -> float:
    # Clopper-Pearson lower bound for Bernoulli rate
    if n <= 0:
        return 0.0
    if x <= 0:
        return 0.0
    if x >= n:
        return 1.0
    return float(beta.ppf(delta, x, n - x + 1))


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


def parse_alpha(name: str) -> Optional[float]:
    # alpha0p1 -> 0.1 ; alpha0p02 -> 0.02
    m = re.search(r"alpha(\d+p\d+)", name)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


def parse_k(name: str) -> Optional[int]:
    m = re.search(r"_k(\d+)", name)
    return int(m.group(1)) if m else None


def fast_fingerprint(path: Path, head_bytes: int = 1_000_000) -> Tuple[int, str]:
    """
    Fast fingerprint: sha1(first 1MB + last 1MB + size).
    Works even for huge files; good enough to detect reuse.
    """
    size = path.stat().st_size
    h = hashlib.sha1()
    with path.open("rb") as f:
        head = f.read(min(head_bytes, size))
        h.update(head)
        if size > head_bytes:
            try:
                f.seek(max(0, size - head_bytes))
                tail = f.read(head_bytes)
                h.update(tail)
            except Exception:
                pass
    h.update(str(size).encode("utf-8"))
    return size, h.hexdigest()


def read_intervals_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads an interval CSV/CSV.GZ; expects pi_lo and y_true/y.
    """
    df = pd.read_csv(path)
    if "pi_lo" not in df.columns:
        raise RuntimeError(f"{path} missing pi_lo; cols={list(df.columns)}")
    if "y_true" in df.columns:
        y = df["y_true"].to_numpy(np.float64)
    elif "y" in df.columns:
        y = df["y"].to_numpy(np.float64)
    else:
        raise RuntimeError(f"{path} missing y_true/y; cols={list(df.columns)}")
    s = df["pi_lo"].to_numpy(np.float64)
    return y, s


def find_run_dir(runs_root: Path, dataset: str, split: str, seed: int) -> Path:
    pats = [f"*{dataset}*{split}*seed{seed}*"]
    cands = []
    for p in pats:
        cands += [x for x in runs_root.glob(p) if x.is_dir()]
    cands = sorted(set(cands))
    pref = [x for x in cands if x.name.startswith("deepdta_point")]
    if len(pref) == 1:
        return pref[0]
    if len(cands) == 1:
        return cands[0]
    if len(cands) == 0:
        raise FileNotFoundError(f"No run_dir for {dataset}/{split}/seed{seed} under {runs_root}")
    raise RuntimeError(f"Ambiguous run_dir for {dataset}/{split}/seed{seed}: {[x.name for x in cands[:10]]}...")


def load_choices(choices_glob: str) -> Optional[pd.DataFrame]:
    files = sorted(Path(".").glob(choices_glob))
    if not files:
        return None
    dfs = []
    for fp in files:
        try:
            d = pd.read_csv(fp)
        except Exception:
            continue
        need = {"model", "dataset", "split", "seed", "method_norm", "cp_subdir"}
        if need.issubset(d.columns):
            d["_choices_file"] = str(fp)
            dfs.append(d)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def choice_cp_subdir(choices: Optional[pd.DataFrame], dataset: str, split: str, seed: int, method: str) -> Optional[str]:
    if choices is None:
        return None
    sub = choices[
        (choices["model"] == "deepdta") &
        (choices["dataset"].astype(str) == str(dataset)) &
        (choices["split"].astype(str) == str(split)) &
        (choices["seed"].astype(int) == int(seed)) &
        (choices["method_norm"].astype(str) == str(method))
    ]
    if len(sub) == 1:
        return str(sub.iloc[0]["cp_subdir"])
    return None


def pick_interval_file(run_dir: Path, dataset: str, split: str, seed: int, method: str,
                       choices: Optional[pd.DataFrame]) -> Path:
    """
    Deterministic mapping consistent with your fixed wiring.
    If choices csv has a cp_subdir for this method, it wins (when applicable).
    """
    m = method.strip()
    mlow = m.lower()

    # Shift-aware baselines are run-level files
    if mlow in ["tc-sc", "tc_sc", "tcsc"]:
        p = run_dir / "pred_intervals_test_tc_sc.csv.gz"
        if not p.exists():
            raise FileNotFoundError(str(p))
        return p
    if mlow in ["wsc-b", "wsc_b", "wsc"]:
        p = run_dir / "pred_intervals_test_wsc_B.csv.gz"
        if not p.exists():
            raise FileNotFoundError(str(p))
        return p

    # cp_* subdirs
    cp_dirs = sorted([d for d in run_dir.glob("cp_*") if d.is_dir() and (d / "pred_intervals_test.csv.gz").exists()])
    if not cp_dirs:
        raise FileNotFoundError(f"No cp_* with pred_intervals_test.csv.gz under {run_dir}")
    names = [d.name for d in cp_dirs]
    low = [n.lower() for n in names]

    def resolve_ambiguous(cand: List[str]) -> str:
        # 1) choices wins if exists
        cp = choice_cp_subdir(choices, dataset, split, seed, m)
        if cp is not None and cp in cand:
            return cp

        # 2) heuristics per method
        if m == "NaiveAutoSel":
            scored = []
            for c in cand:
                a = parse_alpha(c) or -1.0
                k = parse_k(c) or -1
                sc = 0
                if abs(a - 0.1) < 1e-9:
                    sc += 1000
                sc += k
                scored.append((sc, c))
            scored.sort(reverse=True)
            return scored[0][1]

        if m == "CAS-LCB":
            pref = [c for c in cand if "aacomp" in c.lower()]
            if len(pref) == 1:
                return pref[0]
            scored = []
            for c in cand:
                k = parse_k(c) or -1
                scored.append((k, c))
            scored.sort(reverse=True)
            return scored[0][1]

        # fallback: stable
        return sorted(cand)[0]

    if m == "CAS-LCB-Bonf":
        cand = [names[i] for i, s in enumerate(low) if "bonf" in s]
        if not cand:
            raise RuntimeError(f"Missing bonf cp_subdir under {run_dir}; have={names}")
        pick = cand[0] if len(cand) == 1 else resolve_ambiguous(cand)
        return run_dir / pick / "pred_intervals_test.csv.gz"

    if m == "CAS-LCB":
        cand = [names[i] for i, s in enumerate(low) if "constrained_lcb" in s]
        if not cand:
            cand = [names[i] for i, s in enumerate(low) if ("constrained" in s and "lcb" in s and "bonf" not in s)]
        if not cand:
            raise RuntimeError(f"Missing constrained_lcb cp_subdir under {run_dir}; have={names}")
        pick = cand[0] if len(cand) == 1 else resolve_ambiguous(cand)
        return run_dir / pick / "pred_intervals_test.csv.gz"

    if m == "NaiveAutoSel":
        cand = [names[i] for i, s in enumerate(low) if "autosel" in s]
        if not cand:
            raise RuntimeError(f"Missing autosel cp_subdir under {run_dir}; have={names}")
        pick = cand[0] if len(cand) == 1 else resolve_ambiguous(cand)
        return run_dir / pick / "pred_intervals_test.csv.gz"

    if m == "Fixed":
        cand = []
        for n, s in zip(names, low):
            if not s.startswith("cp_local_"):
                continue
            if ("autosel" in s) or ("constrained" in s) or ("lcb" in s) or ("bonf" in s):
                continue
            cand.append(n)
        if not cand:
            raise RuntimeError(f"Missing Fixed baseline cp_local_* under {run_dir}; have={names}")
        if len(cand) > 1:
            pref = [c for c in cand if "target_tfidf" in c.lower()]
            if len(pref) == 1:
                cand = pref
        pick = cand[0] if len(cand) == 1 else resolve_ambiguous(cand)
        return run_dir / pick / "pred_intervals_test.csv.gz"

    raise RuntimeError(f"Unsupported method={method}. Expected {METHODS_CANON}")


def recompute_one_from_intervals(interval_path: Path, q: float, delta: float, k: int) -> Dict[str, float]:
    y_true, score = read_intervals_csv(interval_path)
    hm = hit_mask_topq(y_true, q)
    sel = stable_topk(score, k)
    hits = int(hm[sel].sum())
    k_eff = int(min(k, len(y_true)))
    lcb = cp_lcb(hits, k_eff, delta)
    return {
        "hits": float(hits),
        "hits_at_k": float(hits / k_eff if k_eff > 0 else 0.0),
        "lcb_at_k": float(lcb),
        "min_hits_at_k": float(k_eff * lcb),
        "k_eff": float(k_eff),
        "n_test": float(len(y_true)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True, type=str)
    ap.add_argument("--lcb_csv", required=True, type=str)
    ap.add_argument("--out_csv", default="verify_deepdta_method_wiring_report.csv", type=str)
    ap.add_argument("--choices_glob", default="results/**/task_level_choices.csv", type=str)
    ap.add_argument("--sample_rows", default=30, type=int, help="How many DeepDTA rows to sample for recomputation.")
    ap.add_argument("--seed", default=123, type=int)
    ap.add_argument("--tol", default=1e-9, type=float, help="Tolerance for float comparisons.")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    lcb = pd.read_csv(args.lcb_csv)

    deep = lcb[lcb["model"] == "deepdta"].copy()
    if deep.empty:
        raise RuntimeError("No deepdta rows found in lcb_csv")

    # Load optional choices
    choices = load_choices(args.choices_glob)

    # Build mapping for unique (dataset, split, seed, method)
    uniq = deep[["dataset", "split", "seed", "method"]].drop_duplicates()
    rows = []
    bad_missing = 0

    for _, r in uniq.iterrows():
        dataset = str(r["dataset"])
        split = str(r["split"])
        seed = int(r["seed"])
        method = str(r["method"])

        run_dir = find_run_dir(runs_root, dataset, split, seed)
        interval_path = pick_interval_file(run_dir, dataset, split, seed, method, choices=choices)

        exists = interval_path.exists()
        size = None
        sha1 = None
        if exists:
            size, sha1 = fast_fingerprint(interval_path)
        else:
            bad_missing += 1

        # choices consistency if applicable
        chosen_cp = None
        choices_cp = choice_cp_subdir(choices, dataset, split, seed, method) if choices is not None else None
        if "cp_" in interval_path.parts[-2] if len(interval_path.parts) >= 2 else False:
            chosen_cp = interval_path.parent.name
        elif interval_path.name.startswith("pred_intervals_test"):
            chosen_cp = interval_path.name

        rows.append({
            "dataset": dataset,
            "split": split,
            "seed": seed,
            "method": method,
            "run_dir": str(run_dir),
            "interval_path": str(interval_path),
            "exists": bool(exists),
            "size_bytes": int(size) if size is not None else np.nan,
            "sha1_fast": sha1 if sha1 is not None else "",
            "choices_cp_subdir": choices_cp if choices_cp is not None else "",
            "picked_component": chosen_cp if chosen_cp is not None else "",
            "choices_match": (choices_cp == chosen_cp) if (choices_cp and chosen_cp and ("cp_" in chosen_cp)) else np.nan,
        })

    rep = pd.DataFrame(rows)

    # 1) method reuse statistics within each run (dataset/split/seed)
    def uniq_hash_count(g: pd.DataFrame) -> int:
        return int(g["sha1_fast"].replace("", np.nan).dropna().nunique())

    reuse = rep.groupby(["dataset", "split", "seed"]).apply(
        lambda g: pd.Series({
            "n_methods": int(len(g)),
            "n_unique_fingerprints": uniq_hash_count(g),
            "all_same_fingerprint": bool(uniq_hash_count(g) == 1 and len(g) > 1),
        })
    ).reset_index()

    # 2) pairwise equality count (how often two methods share same fingerprint)
    pair_stats = []
    for a in METHODS_CANON:
        for b in METHODS_CANON:
            if a >= b:
                continue
            ra = rep[rep["method"] == a][["dataset", "split", "seed", "sha1_fast"]].rename(columns={"sha1_fast": "ha"})
            rb = rep[rep["method"] == b][["dataset", "split", "seed", "sha1_fast"]].rename(columns={"sha1_fast": "hb"})
            m = ra.merge(rb, on=["dataset", "split", "seed"], how="inner")
            if m.empty:
                continue
            eq = (m["ha"] == m["hb"]) & (m["ha"] != "")
            pair_stats.append({
                "method_a": a,
                "method_b": b,
                "n_runs_compared": int(len(m)),
                "n_equal_fingerprint": int(eq.sum()),
                "equal_ratio": float(eq.mean()),
            })
    pair_df = pd.DataFrame(pair_stats).sort_values(["equal_ratio", "n_equal_fingerprint"], ascending=False)

    # 3) sampled recomputation check against table rows
    random.seed(args.seed)
    sample_n = min(args.sample_rows, len(deep))
    samp = deep.sample(sample_n, random_state=args.seed).copy()

    mismatches = []
    for _, r in samp.iterrows():
        dataset = str(r["dataset"])
        split = str(r["split"])
        seed = int(r["seed"])
        method = str(r["method"])
        q = float(r["q"])
        delta = float(r["delta"])
        k = int(float(r["k"]))

        run_dir = find_run_dir(runs_root, dataset, split, seed)
        interval_path = pick_interval_file(run_dir, dataset, split, seed, method, choices=choices)
        got = recompute_one_from_intervals(interval_path, q=q, delta=delta, k=k)

        # compare
        for col in ["hits", "lcb_at_k", "min_hits_at_k"]:
            v_table = float(r[col])
            v_got = float(got[col])
            if abs(v_table - v_got) > args.tol:
                mismatches.append({
                    "dataset": dataset, "split": split, "seed": seed, "method": method,
                    "q": q, "delta": delta, "k": k,
                    "field": col,
                    "table": v_table,
                    "recomputed": v_got,
                    "abs_err": abs(v_table - v_got),
                    "interval_path": str(interval_path),
                })

    mism_df = pd.DataFrame(mismatches)

    # Print summary to stdout
    print("\n[VERIFY] mapping rows:", len(rep))
    print("[VERIFY] missing interval files:", int((~rep["exists"]).sum()))
    print("[VERIFY] runs with all methods same fingerprint:", int(reuse["all_same_fingerprint"].sum()),
          "/", int(len(reuse)))

    # Highlight suspicious runs (all same fingerprint) if any
    sus = reuse[reuse["all_same_fingerprint"]].head(10)
    if len(sus):
        print("\n[SUSPICIOUS] example runs where all methods share the same fingerprint (showing up to 10):")
        print(sus.to_string(index=False))

    # Show top pairwise equalities
    if not pair_df.empty:
        print("\n[PAIRWISE] top equal-fingerprint pairs (first 10):")
        print(pair_df.head(10).to_string(index=False))

    # Recompute mismatch report
    if len(mism_df):
        print("\n[RECOMPUTE CHECK] mismatches found:", len(mism_df))
        print(mism_df.sort_values("abs_err", ascending=False).head(20).to_string(index=False))
    else:
        print("\n[RECOMPUTE CHECK] OK: sampled rows match recomputation within tol =", args.tol)

    # Save reports
    out_base = Path(args.out_csv)
    rep.to_csv(out_base, index=False)
    reuse.to_csv(out_base.with_name(out_base.stem + "_reuse_by_run.csv"), index=False)
    pair_df.to_csv(out_base.with_name(out_base.stem + "_pairwise_equal_fingerprint.csv"), index=False)
    mism_df.to_csv(out_base.with_name(out_base.stem + "_recompute_mismatches.csv"), index=False)

    print("\n[WROTE]")
    print("  mapping:", out_base)
    print("  reuse_by_run:", out_base.with_name(out_base.stem + "_reuse_by_run.csv"))
    print("  pairwise_equal_fingerprint:", out_base.with_name(out_base.stem + "_pairwise_equal_fingerprint.csv"))
    print("  recompute_mismatches:", out_base.with_name(out_base.stem + "_recompute_mismatches.csv"))

    # exit status: fail if missing files or mismatches exist
    if int((~rep["exists"]).sum()) > 0 or len(mism_df) > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
