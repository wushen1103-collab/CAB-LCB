from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd
import subprocess
import sys

SPLITS = ["random","cold_drug","cold_target","cold_pair"]
DATASETS = ["davis","kiba"]

def find_run_dirs(root: Path):
    return sorted({p.parent for p in root.rglob("preds_test.csv.gz")})

def extract_key(run_dir: Path):
    s = run_dir.name.lower()
    # seed
    m = re.search(r"seed(\d+)", s)
    if not m: return None
    seed = int(m.group(1))
    # split
    split = None
    for sp in SPLITS:
        if sp in s:
            split = sp; break
    if split is None: return None
    # dataset
    dataset = None
    for ds in DATASETS:
        if ds in s:
            dataset = ds; break
    if dataset is None: return None
    return (dataset, split, seed)

def pick_points_and_calcp(runs_dir: Path, model_hint: str):
    """
    Auto-pick candidate points_root and calcp_root directories by name.
    model_hint: 'graphdta' or 'deepdta' etc.
    """
    dirs = [p for p in runs_dir.iterdir() if p.is_dir() and "points" in p.name]
    # prioritize names that contain model_hint and calcp
    points = [d for d in dirs if model_hint in d.name.lower() and "calcp" not in d.name.lower()]
    calcp  = [d for d in dirs if model_hint in d.name.lower() and "calcp" in d.name.lower()]
    # fallback: if model_hint not in name, just use any points/calcp_points
    if not points:
        points = [d for d in dirs if "calcp" not in d.name.lower()]
    if not calcp:
        calcp  = [d for d in dirs if "calcp" in d.name.lower()]
    # pick the most recent (by mtime) to avoid old experiments
    points = sorted(points, key=lambda p: p.stat().st_mtime, reverse=True)
    calcp  = sorted(calcp,  key=lambda p: p.stat().st_mtime, reverse=True)
    if not points or not calcp:
        raise SystemExit("Could not auto-pick points/calcp_points. Please pass --points_root and --calcp_root explicitly.")
    return points[0], calcp[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--model_hint", type=str, default="graphdta",
                    help="Substring to identify model dirs. e.g., graphdta / deepdta")
    ap.add_argument("--points_root", type=str, default=None)
    ap.add_argument("--calcp_root", type=str, default=None)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out_tag", type=str, default=None,
                    help="Suffix for summary filename (e.g., graphdta_pkd)")
    args = ap.parse_args()

    repo = Path(".").resolve()
    runs = repo / args.runs_dir

    if args.points_root and args.calcp_root:
        points_top = runs / args.points_root
        calcp_top  = runs / args.calcp_root
    else:
        points_top, calcp_top = pick_points_and_calcp(runs, args.model_hint)

    if not points_top.exists(): raise SystemExit(f"Missing points dir: {points_top}")
    if not calcp_top.exists():  raise SystemExit(f"Missing calcp dir: {calcp_top}")

    points_runs = find_run_dirs(points_top)
    calcp_runs  = find_run_dirs(calcp_top)

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

    print("[picked] points_root =", points_top.name)
    print("[picked] calcp_root  =", calcp_top.name)
    print(f"[scan] points_runs={len(points_runs)} calcp_runs={len(calcp_runs)} paired_jobs={len(jobs)}")
    print(f"[scan] points_bad={points_bad} calcp_bad={calcp_bad} missing_preds_cal={points_missing_cal} missing_pair={points_missing_pair}")

    if args.dry_run:
        for k, pr, preds_test, proxy in jobs[:8]:
            print("[pair]", k, pr.name)
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

    tag = args.out_tag or args.model_hint
    out = pd.DataFrame(out_rows).sort_values(["dataset","split","seed"])
    out_path = Path("results") / f"shift_aware_summary_tc_sc_alpha{args.alpha}_{tag}.csv"
    out.to_csv(out_path, index=False)
    print("[saved]", out_path, "rows=", len(out))

if __name__ == "__main__":
    main()
