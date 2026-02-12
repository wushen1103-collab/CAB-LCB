from __future__ import annotations
import argparse
from pathlib import Path
import re
import subprocess
import sys
import pandas as pd

PAT = re.compile(r"(davis|kiba)_(random|cold_drug|cold_target|cold_pair)_seed(\d+)$")

def find_run_dirs(root: Path):
    return sorted({p.parent for p in root.rglob("preds_test.csv.gz")})

def extract_key(run_dir: Path):
    name = run_dir.name.lower()
    m = PAT.search(name)
    if not m:
        return None
    return (m.group(1), m.group(2), int(m.group(3)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points_root", type=str, default="runs/2025-12-20")
    ap.add_argument("--calcp_root", type=str, default="runs/2025-12-20_calcp")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out_csv", type=str, default="results/shift_aware_summary_tc_sc_alpha0.1_graphdta.csv")
    args = ap.parse_args()

    points_top = Path(args.points_root)
    calcp_top = Path(args.calcp_root)

    if not points_top.exists():
        raise SystemExit(f"Missing points_root: {points_top}")
    if not calcp_top.exists():
        raise SystemExit(f"Missing calcp_root: {calcp_top}")

    points_runs = find_run_dirs(points_top)
    calcp_runs = find_run_dirs(calcp_top)

    calcp_map = {}
    bad_calcp = 0
    for rd in calcp_runs:
        k = extract_key(rd)
        if k is None:
            bad_calcp += 1
            continue
        calcp_map[k] = rd

    jobs = []
    bad_points = 0
    missing_pair = 0
    for pr in points_runs:
        k = extract_key(pr)
        if k is None:
            bad_points += 1
            continue
        cr = calcp_map.get(k)
        if cr is None:
            missing_pair += 1
            continue
        jobs.append((k, pr, cr))

    print(f"[scan] points_runs={len(points_runs)} calcp_runs={len(calcp_runs)} paired_jobs={len(jobs)}")
    print(f"[scan] bad_points={bad_points} bad_calcp={bad_calcp} missing_pair={missing_pair}")

    if args.dry_run:
        for k, pr, cr in jobs[:10]:
            print("[pair]", k, pr.name, "<->", cr.name)
        return

    rows = []
    for k, pr, cr in jobs:
        test_path = pr / "preds_test.csv.gz"
        proxy_path = cr / "preds_test.csv.gz"
        out_path = pr / "pred_intervals_test_tc_sc.csv.gz"

        cmd = [
            sys.executable, "scripts/shift_aware_baselines.py",
            "--method", "tc_sc",
            "--target_proxy", str(proxy_path),
            "--test", str(test_path),
            "--alpha", str(args.alpha),
            "--out", str(out_path),
        ]
        print("[tc_sc]", k, pr.name)
        subprocess.check_call(cmd)

        df = pd.read_csv(out_path)
        if "covered" in df.columns:
            rows.append({
                "dataset": k[0],
                "split": k[1],
                "seed": k[2],
                "run_dir": str(pr),
                "coverage": float(df["covered"].mean()),
                "mean_width": float(df["width"].mean()),
                "meet_90": float(df["covered"].mean() >= (1.0 - args.alpha)),
                "n": int(len(df)),
            })

    out = pd.DataFrame(rows).sort_values(["dataset", "split", "seed"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("[saved]", args.out_csv, "rows=", len(out))

if __name__ == "__main__":
    main()
