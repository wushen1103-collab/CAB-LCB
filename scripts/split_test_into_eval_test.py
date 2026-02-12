#!/usr/bin/env python3
"""
Split preds_test.csv.gz into two disjoint files:
  - preds_eval.csv.gz (for selection / dev)
  - preds_test.csv.gz (for final report)

This is a pragmatic fix when the pipeline only produced preds_test.
We keep a backup of the original file as preds_test_full.csv.gz.

NOTE:
- This changes what you call "test" (it becomes a held-out subset of the original test).
- For paper-grade setups, the better option is to generate a true validation split.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SplitManifest:
    created_at_utc: str
    script: str
    runs_tag: str
    exp_dir: str
    source_file: str
    backup_file: str
    eval_file: str
    test_file: str
    seed: int
    eval_frac: float
    n_total: int
    n_eval: int
    n_test: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _safe_backup(src: Path, dst: Path, force: bool) -> None:
    if dst.exists():
        if force:
            dst.unlink()
        else:
            raise RuntimeError(f"Backup already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def split_one_exp(
    exp_dir: Path,
    eval_frac: float,
    seed: int,
    write: bool,
    force: bool,
    backup: bool,
) -> tuple[bool, str]:
    preds_test = exp_dir / "preds_test.csv.gz"
    preds_eval = exp_dir / "preds_eval.csv.gz"
    preds_test_full = exp_dir / "preds_test_full.csv.gz"
    manifest_path = exp_dir / "eval_test_split_manifest.json"

    if not preds_test.exists():
        return False, f"[SKIP] missing preds_test.csv.gz: {exp_dir}"

    if preds_eval.exists() and not force:
        return False, f"[SKIP] preds_eval.csv.gz already exists (use --force to overwrite): {exp_dir}"

    if not (0.0 < eval_frac < 1.0):
        raise ValueError(f"--eval_frac must be in (0,1), got {eval_frac}")

    # If we already created a full backup before, use it as the source to avoid compounding splits.
    src = preds_test_full if preds_test_full.exists() else preds_test

    df = pd.read_csv(src, compression="gzip")
    n = len(df)
    if n < 10:
        return False, f"[SKIP] too few rows ({n}) in {src}"

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_eval = int(round(eval_frac * n))
    n_eval = max(1, min(n - 1, n_eval))  # keep both non-empty
    eval_idx = idx[:n_eval]
    test_idx = idx[n_eval:]

    df_eval = df.iloc[eval_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    msg = (
        f"[PLAN] {exp_dir.name}: n_total={n}, n_eval={len(df_eval)}, n_test={len(df_test)}, "
        f"src={src.name} -> preds_eval.csv.gz + preds_test.csv.gz"
    )
    if not write:
        return True, msg

    # Backup original test once
    if backup and not preds_test_full.exists():
        _safe_backup(preds_test, preds_test_full, force=force)

    # Write new files
    df_eval.to_csv(preds_eval, index=False, compression="gzip")
    df_test.to_csv(preds_test, index=False, compression="gzip")

    manifest = SplitManifest(
        created_at_utc=_utc_now(),
        script="scripts/split_test_into_eval_test.py",
        runs_tag=str(exp_dir.parent),
        exp_dir=str(exp_dir),
        source_file=str(src),
        backup_file=str(preds_test_full),
        eval_file=str(preds_eval),
        test_file=str(preds_test),
        seed=seed,
        eval_frac=eval_frac,
        n_total=n,
        n_eval=len(df_eval),
        n_test=len(df_test),
    )
    _write_json(manifest_path, asdict(manifest))

    return True, f"[OK] {msg}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_tag", required=True, help="e.g. runs/2025-12-23_deepdta_points/")
    ap.add_argument("--glob", default="deepdta_point_*", help="Experiment dir glob under runs_tag")
    ap.add_argument("--eval_frac", type=float, default=0.2, help="Fraction of rows moved into preds_eval")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for splitting")
    ap.add_argument("--write", action="store_true", help="Actually write files (otherwise dry-run)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing preds_eval/test")
    ap.add_argument("--no_backup", action="store_true", help="Do not create preds_test_full.csv.gz backup")
    args = ap.parse_args()

    runs_tag = Path(args.runs_tag).expanduser().resolve()
    if not runs_tag.exists():
        raise FileNotFoundError(runs_tag)

    exp_dirs = sorted([p for p in runs_tag.glob(args.glob) if p.is_dir()])
    if not exp_dirs:
        raise RuntimeError(f"No exp dirs matched under {runs_tag} with glob={args.glob}")

    ok = 0
    changed = 0
    for exp in exp_dirs:
        did, msg = split_one_exp(
            exp_dir=exp,
            eval_frac=args.eval_frac,
            seed=args.seed,
            write=args.write,
            force=args.force,
            backup=(not args.no_backup),
        )
        print(msg)
        ok += 1
        if did and args.write:
            changed += 1

    print("\n=== Summary ===")
    print(f"exp_dirs_scanned: {ok}")
    print(f"exp_dirs_written: {changed}")
    if not args.write:
        print("[DRYRUN] Re-run with --write to apply changes.")


if __name__ == "__main__":
    main()
