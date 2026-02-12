#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path

import numpy as np


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_idx(path: Path) -> np.ndarray:
    return np.load(str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/processed")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--splits", type=str, default="random,cold_pair,cold_drug,cold_target")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    dataset = args.dataset
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    rows = []
    for split in splits:
        for seed in seeds:
            d = data_root / dataset / "splits" / split / f"seed_{seed}"
            if not d.exists():
                print(f"[MISSING] {d}")
                continue
            p_tr = d / "idx_train.npy"
            p_cal = d / "idx_cal.npy"
            p_te = d / "idx_test.npy"
            if not (p_tr.exists() and p_cal.exists() and p_te.exists()):
                print(f"[MISSING_FILES] {d}")
                continue

            tr = load_idx(p_tr)
            cal = load_idx(p_cal)
            te = load_idx(p_te)

            s_tr, s_cal, s_te = set(tr.tolist()), set(cal.tolist()), set(te.tolist())
            ov_tr_cal = len(s_tr & s_cal)
            ov_tr_te = len(s_tr & s_te)
            ov_cal_te = len(s_cal & s_te)

            rows.append(
                (
                    dataset,
                    split,
                    seed,
                    len(tr),
                    len(cal),
                    len(te),
                    ov_tr_cal,
                    ov_tr_te,
                    ov_cal_te,
                    md5_file(p_te),
                )
            )

    print("dataset,split,seed,n_train,n_cal,n_test,overlap_tr_cal,overlap_tr_te,overlap_cal_te,md5_test")
    for r in rows:
        print(",".join(map(str, r)))

    # detect identical test splits
    key_to_runs = {}
    for (ds, sp, sd, *_rest, md5t) in rows:
        key_to_runs.setdefault(md5t, []).append((sp, sd))

    dup = {k: v for k, v in key_to_runs.items() if len(v) > 1}
    if dup:
        print("\n[WARN] Found identical idx_test.npy across different (split,seed):")
        for k, v in dup.items():
            v2 = ", ".join([f"{sp}:seed{sd}" for sp, sd in sorted(v)])
            print(f"  md5={k} -> {v2}")
    else:
        print("\n[OK] No identical idx_test.npy across checked runs.")


if __name__ == "__main__":
    main()
