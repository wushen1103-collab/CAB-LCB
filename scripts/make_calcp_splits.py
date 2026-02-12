#!/usr/bin/env python3
"""
Create synthetic split directories '*__calcp' for constrained alpha selection.

For each (dataset, split, seed):
- Load original idx_cal.npy and idx_test.npy
- Partition original test into two parts using a deterministic RNG(seed):
    - idx_calcp_test: fraction `test_holdout_frac` of original test (this will become idx_test.npy of split__calcp)
    - idx_calcp_cal: remaining original test (added into idx_cal.npy of split__calcp)
- New split__calcp has:
    - idx_train.npy: copied from original (unchanged)
    - idx_cal.npy: concat(original idx_cal, idx_calcp_cal)
    - idx_test.npy: idx_calcp_test
This enables selecting alpha on calcp without touching final eval/test of original split.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def load_idx(p: Path) -> np.ndarray:
    if not p.exists():
        raise FileNotFoundError(str(p))
    return np.asarray(np.load(p), dtype=np.int64)


def save_idx(p: Path, arr: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, np.asarray(arr, dtype=np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["davis", "kiba"])
    ap.add_argument("--split", required=True, choices=["random", "cold_drug", "cold_target", "cold_pair"])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--test_holdout_frac", type=float, default=0.5,
                    help="Fraction of original test to be used as calcp test (idx_test of split__calcp).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    ds = args.dataset
    sp = args.split
    frac = float(args.test_holdout_frac)

    base = Path("data/processed") / ds / "splits"
    src_split = base / sp
    dst_split = base / f"{sp}__calcp"

    for seed in seeds:
        src = src_split / f"seed_{seed}"
        dst = dst_split / f"seed_{seed}"

        idx_train = load_idx(src / "idx_train.npy")
        idx_cal = load_idx(src / "idx_cal.npy")
        idx_test = load_idx(src / "idx_test.npy")

        # Deterministic partition of original test
        rng = np.random.RandomState(seed + 12345)  # offset to avoid collisions with other uses
        perm = idx_test.copy()
        rng.shuffle(perm)

        n_hold = int(round(len(perm) * frac))
        n_hold = max(1, min(n_hold, len(perm) - 1))  # ensure both sides non-empty
        idx_calcp_test = perm[:n_hold]
        idx_calcp_cal = perm[n_hold:]

        new_cal = np.concatenate([idx_cal, idx_calcp_cal], axis=0)

        # Basic sanity checks
        if len(np.unique(idx_train)) != len(idx_train):
            raise RuntimeError(f"Duplicate in idx_train for {ds}/{sp}/seed{seed}")
        if len(np.unique(new_cal)) != len(new_cal):
            raise RuntimeError(f"Duplicate in new idx_cal for {ds}/{sp}/seed{seed}")
        if len(np.unique(idx_calcp_test)) != len(idx_calcp_test):
            raise RuntimeError(f"Duplicate in new idx_test for {ds}/{sp}/seed{seed}")

        # Ensure disjointness between new cal and new test
        inter = np.intersect1d(new_cal, idx_calcp_test)
        if inter.size > 0:
            raise RuntimeError(f"New cal and new test overlap for {ds}/{sp}/seed{seed}: {inter.size}")

        # Write outputs
        if dst.exists() and not args.overwrite:
            # If files exist, skip (idempotent)
            if (dst / "idx_train.npy").exists() and (dst / "idx_cal.npy").exists() and (dst / "idx_test.npy").exists():
                continue

        save_idx(dst / "idx_train.npy", idx_train)
        save_idx(dst / "idx_cal.npy", new_cal)
        save_idx(dst / "idx_test.npy", idx_calcp_test)

        print(f"[OK] wrote split__calcp: {ds}/{sp} seed{seed} -> {dst}")

    print("[DONE]")


if __name__ == "__main__":
    main()
