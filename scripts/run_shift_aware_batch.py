from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd
import subprocess
import sys

# Matches BOTH:
# deepdta_point_kiba_random_seed2
# deepdta_point_kiba_random__calcp_seed2
PAT = re.compile(
    r"deepdta_point_(?P<dataset>davis|kiba)_(?P<split>random|cold_drug|cold_target|cold_pair)(?:__calcp)?_seed(?P<seed>\d+)$"
)

def find_run_dirs(root: Path):
    return sorted({p.parent for p in root.rglob("preds_test.csv.gz")})

def extract_key(run_dir: Path):
    name = run_dir.name.lower()
    m = PAT.match(name)
    if not m:
        return None
    return (m.group("dataset"), m.group("split"), int(m.group("seed")))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--points_root", type=str, default="2025-12-23_deepdta_points")
    ap.add_argument("--calcp_root", type=str, default="2025-12-23_deepdta_calcp_points")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    repo = Path(".").resolve()
    runs = repo / args.runs_dir
    points_top = runs / args.points_root
    calcp_top  = runs / args.calcp_root

    points_runs = find_run_dirs(points_top)
    calcp_runs  = find_run_dirs(calcp_top)

    # Build calcp map by key
    calcp_map = {}
    calcp_bad = 0
    for rd in calcp_runs:
        k = extract_key(rd)
        if k is None:
            calcp_bad += 1
            continue
        calcp_map[k] = rd

    jobs = []
    points_bad = 0
    points_missing_cal = 0
    points_missing_pair = 0

    for pr in points_runs:
        k = extract_key(pr)
        if k is None:
            points_bad += 1
            continue

        preds_test = pr / "preds_test.csv.gz"
        preds_cal  = pr / "preds_cal.csv.gz"
        if not preds_cal.exists():
            points_missing_cal += 1
            continue

        cr = calcp_map.get(k)
        if cr is None:
            points_missing_pair += 1
            continue

        proxy = cr / "preds_test.csv.gz"
        jobs.append((k, pr, preds_test, proxy))

    print(f"[scan] points_runs={len(points_runs)} calcp_runs={len(calcp_runs)} paired_jobs={len(jobs)}")
    print(f"[scan] points_bad={points_bad} calcp_bad={calcp_bad} missing_preds_cal={points_missing_cal} missing_pair={points_missing_pair}")

    if args.dry_run:
        for k, pr, preds_test, proxy in jobs[:10]:
            print("[pair]", k)
            print("  point:", pr)
            print("   test:", preds_test)
            print("  proxy:", proxy)
        return

    out_rows = []
    for k, pr, preds_test, proxy in jobs:
        out_tc = pr / "pred_intervals_test_tc_sc.csv.gz"
        cmd = [
            sys.executable, "scripts/shift_aware_baselines.py",
            "--method", "tc_sc",
            "--target_proxy", str(proxy),
            "--test", str(preds_test),
            "--alpha", str(args.alpha),
            "--out", str(out_tc),
        ]
        print("[tc_sc]", k, pr.name)
        subprocess.check_call(cmd)

        df = pd.read_csv(out_tc)
        if "covered" in df.columns:
            out_rows.append({
                "dataset": k[0], "split": k[1], "seed": k[2],
                "run_dir": str(pr),
                "coverage": float(df["covered"].mean()),
                "mean_width": float((df["pi_hi"] - df["pi_lo"]).mean()),
                "meet_90": float(df["covered"].mean() >= (1.0-args.alpha)),
                "n": int(len(df)),
            })

    if out_rows:
        out = pd.DataFrame(out_rows).sort_values(["dataset","split","seed"])
        out_path = Path("results") / f"shift_aware_summary_tc_sc_alpha{args.alpha}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print("[saved]", out_path, "rows=", len(out))

if __name__ == "__main__":
    main()
