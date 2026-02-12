#!/usr/bin/env python3
"""
Backfill per-sample prediction intervals for existing local-CP outputs.

It scans:
  <runs_root>/<run_dir>/cp_local_*/config.json
and re-runs scripts/run_local_conformal_from_preds.py with the exact args
stored in config.json, so that pred_intervals_test.csv.gz is emitted.

Usage:
  python scripts/backfill_local_pred_intervals.py \
    --runs_root runs/2025-12-23_deepdta_points \
    --pattern 'deepdta_point_*'
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

def load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True, type=str)
    ap.add_argument("--pattern", default="*", type=str)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--python_bin", default=sys.executable, type=str)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(str(runs_root))

    run_dirs = sorted([p for p in runs_root.glob(args.pattern) if p.is_dir()])
    if not run_dirs:
        print(f"[WARN] No run dirs matched: {runs_root}/{args.pattern}")
        return

    total = 0
    done = 0
    skipped = 0
    failed = 0

    for run_dir in run_dirs:
        cp_dirs = sorted([p for p in run_dir.glob("cp_local_*") if p.is_dir()])
        if not cp_dirs:
            continue

        for cp_dir in cp_dirs:
            total += 1
            out_file = cp_dir / "pred_intervals_test.csv.gz"
            if out_file.exists() and (not args.overwrite):
                skipped += 1
                continue

            cfg_path = cp_dir / "config.json"
            if not cfg_path.exists():
                print(f"[WARN] Missing config.json: {cfg_path}")
                failed += 1
                continue

            cfg = load_json(cfg_path)
            a = cfg.get("args", {})
            # run_local_conformal_from_preds.py expects --run_dir and takes the rest from args
            cmd = [args.python_bin, "scripts/run_local_conformal_from_preds.py", "--run_dir", str(run_dir)]

            def add_flag(k, v):
                if v is None:
                    return
                if isinstance(v, bool):
                    if v:
                        cmd.append(f"--{k}")
                    return
                cmd.extend([f"--{k}", str(v)])

            # Only pass flags that exist in the script.
            allow = {
                "alpha","group_by","k_neighbors","min_cal_samples",
                "drug_repr","target_repr",
                "tfidf_max_features","tfidf_ngram_max","tfidf_char_ngram",
                "knn_metric","dist_norm","distance_inflate_gamma","pca_dim",
                "out_subdir","label_transform","cal_idx_npy","eval_idx_npy",
            }
            for k, v in a.items():
                if k in allow:
                    add_flag(k, v)

            # force output subdir to the existing cp_local_* so we backfill in place
            cmd.extend(["--out_subdir", cp_dir.name])

            print("[RUN]", " ".join(cmd))
            if args.dry_run:
                continue

            try:
                subprocess.run(cmd, check=True)
                if out_file.exists():
                    done += 1
                else:
                    print(f"[WARN] Finished but still missing: {out_file}")
                    failed += 1
            except subprocess.CalledProcessError as e:
                print(f"[ERR] {cp_dir} -> {e}")
                failed += 1

    print("\n[SUMMARY]")
    print(f"total_candidates={total}")
    print(f"done={done} skipped={skipped} failed={failed}")

if __name__ == "__main__":
    main()
