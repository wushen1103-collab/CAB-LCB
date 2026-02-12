#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd


def _find_best_interval_file(cp_dir: Path, split: str) -> Optional[Path]:
    """
    Try hard to find a per-split interval artifact under cp_dir.
    We prefer files that contain both split token and interval-like tokens.
    """
    patterns = [
        f"**/*{split}*interval*.csv",
        f"**/*{split}*interval*.tsv",
        f"**/*{split}*pi*.csv",
        f"**/*{split}*pi*.tsv",
        f"**/*{split}*pred*interval*.csv",
        f"**/*{split}*pred*interval*.tsv",
        f"**/*{split}*pred*.npz",
        f"**/*{split}*interval*.npz",
    ]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend([p for p in cp_dir.glob(pat) if p.is_file()])

    if not candidates:
        # Fallback: any file containing split token and likely lower/upper columns
        loose = [p for p in cp_dir.glob(f"**/*{split}*.csv") if p.is_file()]
        loose += [p for p in cp_dir.glob(f"**/*{split}*.tsv") if p.is_file()]
        candidates.extend(loose)

    if not candidates:
        return None

    def score(p: Path) -> int:
        name = p.name.lower()
        s = 0
        if "interval" in name:
            s += 5
        if "pi" in name:
            s += 3
        if "pred" in name:
            s += 2
        if name.endswith(".npz"):
            s += 1
        # Prefer files closer to cp_dir (shallower)
        try:
            rel = p.relative_to(cp_dir)
            s -= len(rel.parts)
        except Exception:
            pass
        return s

    candidates = sorted(set(candidates), key=score, reverse=True)
    return candidates[0]


def _pick_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Identify y_true, lower, upper column names robustly.
    """
    cols = {c.lower(): c for c in df.columns}

    y_candidates = [
        "y_true", "y", "label", "target", "affinity", "gt", "truth"
    ]
    lo_candidates = [
        "lower", "lo", "l", "y_lower", "pi_lower", "interval_lower", "pred_lower"
    ]
    hi_candidates = [
        "upper", "hi", "u", "y_upper", "pi_upper", "interval_upper", "pred_upper"
    ]

    def find_one(cands: List[str]) -> Optional[str]:
        for k in cands:
            if k in cols:
                return cols[k]
        # second pass: substring match
        for k in cands:
            for cl in cols:
                if k in cl:
                    return cols[cl]
        return None

    y_col = find_one(y_candidates)
    lo_col = find_one(lo_candidates)
    hi_col = find_one(hi_candidates)

    if not (y_col and lo_col and hi_col):
        raise KeyError(
            f"Could not infer columns. Have={list(df.columns)}; "
            f"need y_true/lower/upper (or equivalents)."
        )
    return y_col, lo_col, hi_col


def _load_intervals(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (y_true, lower, upper) arrays.
    """
    if path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=True)
        keys = {k.lower(): k for k in z.files}

        def get_any(cands: List[str]) -> Optional[np.ndarray]:
            for c in cands:
                if c in keys:
                    return z[keys[c]]
            for c in cands:
                for kk in keys:
                    if c in kk:
                        return z[keys[kk]]
            return None

        y = get_any(["y_true", "y", "label", "target", "truth"])
        lo = get_any(["lower", "lo", "y_lower", "pi_lower", "interval_lower"])
        hi = get_any(["upper", "hi", "y_upper", "pi_upper", "interval_upper"])

        if y is None or lo is None or hi is None:
            raise KeyError(f"NPZ missing keys. Have={z.files}")
        return np.asarray(y).astype(float), np.asarray(lo).astype(float), np.asarray(hi).astype(float)

    # CSV/TSV
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    y_col, lo_col, hi_col = _pick_cols(df)
    y = df[y_col].astype(float).to_numpy()
    lo = df[lo_col].astype(float).to_numpy()
    hi = df[hi_col].astype(float).to_numpy()
    return y, lo, hi


def _coverage_width(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[float, float]:
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if m.sum() == 0:
        return float("nan"), float("nan")
    y2, lo2, hi2 = y[m], lo[m], hi[m]
    cov = float(((y2 >= lo2) & (y2 <= hi2)).mean())
    wid = float((hi2 - lo2).mean())
    return cov, wid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_tag", required=True, help="e.g. runs/2025-12-23_deepdta_points/")
    ap.add_argument("--split", default="test", help="split token to search, default=test")
    ap.add_argument("--metrics_name", default="conformal_metrics.json")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    runs_tag = Path(args.runs_tag)
    if not runs_tag.exists():
        raise SystemExit(f"[FATAL] runs_tag not found: {runs_tag}")

    metrics_files = list(runs_tag.rglob(args.metrics_name))
    print(f"[INFO] metrics files found: {len(metrics_files)}")

    patched = 0
    missing = 0
    failures = 0

    for mf in metrics_files:
        cp_dir = mf.parent
        try:
            d: Dict[str, Any] = json.loads(mf.read_text(encoding="utf-8"))
        except Exception as e:
            failures += 1
            print(f"[WARN] cannot read json: {mf} ({e})")
            continue

        # Skip if already has test metrics
        if any(k in d for k in ["coverage_test", "avg_width_test"]):
            continue

        interval_file = _find_best_interval_file(cp_dir, args.split)
        if interval_file is None:
            missing += 1
            continue

        try:
            y, lo, hi = _load_intervals(interval_file)
            cov, wid = _coverage_width(y, lo, hi)
        except Exception as e:
            failures += 1
            print(f"[WARN] failed on {interval_file}: {e}")
            continue

        if not np.isfinite(cov) or not np.isfinite(wid):
            failures += 1
            print(f"[WARN] non-finite metrics from {interval_file}")
            continue

        d["coverage_test"] = cov
        d["avg_width_test"] = wid
        d["n_test_used"] = int(np.isfinite(y).sum())

        if not args.dry_run:
            mf.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        patched += 1

    print(f"[INFO] patched: {patched}")
    print(f"[INFO] missing interval artifacts: {missing}")
    print(f"[INFO] failures: {failures}")

    if patched == 0:
        raise SystemExit(
            "[FATAL] No test metrics were backfilled.\n"
            "This almost certainly means no per-test interval/prediction artifacts were saved.\n"
            "You must re-run the evaluation/export step to generate test intervals."
        )


if __name__ == "__main__":
    main()
