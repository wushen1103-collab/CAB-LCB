from __future__ import annotations

import argparse
import json

from src.dti_cp.data.dataset import PairDataset
from src.dti_cp.data.splits import SplitConfig, make_split, save_split_to_processed
from src.dti_cp.utils.io import make_run_dir, save_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["davis", "kiba"])
    ap.add_argument("--split", type=str, required=True, choices=["random", "cold_drug", "cold_target", "cold_pair"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cal_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--exp_name", type=str, default=None)
    args = ap.parse_args()

    exp_name = args.exp_name or f"day3_split_{args.dataset}_{args.split}_seed{args.seed}"
    run_dir = make_run_dir("runs", exp_name)

    cfg = SplitConfig(split=args.split, seed=args.seed, cal_frac=args.cal_frac, test_frac=args.test_frac)
    save_yaml(
        {
            "dataset": args.dataset,
            "split": args.split,
            "seed": args.seed,
            "cal_frac": args.cal_frac,
            "test_frac": args.test_frac,
            "data_root": args.data_root,
        },
        run_dir / "config.yaml",
    )

    ds = PairDataset(root=args.data_root, dataset=args.dataset)
    df = ds.pairs

    idx_train, idx_cal, idx_test, meta = make_split(df, cfg)
    out_dir = save_split_to_processed(args.data_root, args.dataset, cfg, idx_train, idx_cal, idx_test, meta)

    (run_dir / "stdout.log").write_text(
        f"Saved split indices to: {out_dir}\nMeta:\n{json.dumps(meta, indent=2)}\n",
        encoding="utf-8",
    )

    print(f"Saved to: {out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
