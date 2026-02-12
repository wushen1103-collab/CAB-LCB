#!/usr/bin/env python3
"""
Rebuild correct cold_pair splits.

Definition (pair cold-start):
- Test contains (drug, target) pairs not seen in training pairs (always true if indices disjoint).
- BUT drugs and targets in test must both appear in the training set (not cold_drug / not cold_target).

We keep split sizes consistent with the existing random split for each dataset/seed:
- n_train = len(random/idx_train)
- n_cal   = len(random/idx_cal)
- n_test  = len(random/idx_test)

We do NOT try to match the exact train/cal of random; we only match sizes and enforce cold_pair constraints.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_idx(p: Path) -> np.ndarray:
    return np.asarray(np.load(p), dtype=np.int64)


def save_idx(p: Path, arr: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, np.asarray(arr, dtype=np.int64))


def build_cold_pair(
    drugs: np.ndarray,
    targets: np.ndarray,
    n_train: int,
    n_cal: int,
    n_test: int,
    seed: int,
    max_tries: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(drugs)
    rng = np.random.RandomState(seed)

    all_idx = np.arange(N, dtype=np.int64)

    for t in range(max_tries):
        perm = rng.permutation(all_idx)

        idx_train = perm[:n_train]
        rest = perm[n_train:]

        train_drugs = set(drugs[idx_train].tolist())
        train_targets = set(targets[idx_train].tolist())

        eligible = rest[
            np.isin(drugs[rest], list(train_drugs)) & np.isin(targets[rest], list(train_targets))
        ]

        if len(eligible) < n_test:
            continue

        rng.shuffle(eligible)
        idx_test = eligible[:n_test]

        # Calibration from remaining indices (excluding idx_test)
        mask = np.ones(len(rest), dtype=bool)
        test_set = set(idx_test.tolist())
        for i, ridx in enumerate(rest):
            if int(ridx) in test_set:
                mask[i] = False
        rest2 = rest[mask]

        if len(rest2) < n_cal:
            continue

        idx_cal = rest2[:n_cal]

        # Final sanity: disjoint
        if len(set(idx_train) & set(idx_cal)) > 0:
            continue
        if len(set(idx_train) & set(idx_test)) > 0:
            continue
        if len(set(idx_cal) & set(idx_test)) > 0:
            continue

        # Ensure not cold_drug / not cold_target w.r.t TRAIN ONLY
        if not set(drugs[idx_test].tolist()).issubset(train_drugs):
            continue
        if not set(targets[idx_test].tolist()).issubset(train_targets):
            continue

        return idx_train, idx_cal, idx_test

    raise RuntimeError("Failed to construct cold_pair split with constraints. Try increasing max_tries or adjust sizes.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["davis", "kiba"])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    ds = args.dataset

    pairs_path = Path("data/processed") / ds / "pairs.csv.gz"
    df = pd.read_csv(pairs_path, compression="gzip")
    if "drug_idx" not in df.columns or "target_idx" not in df.columns:
        raise RuntimeError(f"pairs.csv.gz missing drug_idx/target_idx: {pairs_path}")

    drugs = df["drug_idx"].to_numpy(dtype=np.int64)
    targets = df["target_idx"].to_numpy(dtype=np.int64)

    base = Path("data/processed") / ds / "splits"
    random_dir = base / "random"
    out_dir = base / "cold_pair"

    for seed in seeds:
        ref = random_dir / f"seed_{seed}"
        n_train = len(load_idx(ref / "idx_train.npy"))
        n_cal = len(load_idx(ref / "idx_cal.npy"))
        n_test = len(load_idx(ref / "idx_test.npy"))

        dst = out_dir / f"seed_{seed}"
        if dst.exists() and not args.overwrite:
            if (dst / "idx_train.npy").exists() and (dst / "idx_cal.npy").exists() and (dst / "idx_test.npy").exists():
                print(f"[SKIP] exists: {dst}")
                continue

        idx_train, idx_cal, idx_test = build_cold_pair(
            drugs=drugs, targets=targets,
            n_train=n_train, n_cal=n_cal, n_test=n_test,
            seed=seed
        )

        save_idx(dst / "idx_train.npy", idx_train)
        save_idx(dst / "idx_cal.npy", idx_cal)
        save_idx(dst / "idx_test.npy", idx_test)

        print(f"[OK] wrote {ds}/cold_pair seed{seed}: n_train={n_train} n_cal={n_cal} n_test={n_test}")

    print("[DONE]")


if __name__ == "__main__":
    main()
