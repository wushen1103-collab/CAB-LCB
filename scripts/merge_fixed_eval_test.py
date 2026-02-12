#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p_eval = Path(args.eval)
    p_test = Path(args.test)
    p_out = Path(args.out)

    df1 = pd.read_csv(p_eval)
    df2 = pd.read_csv(p_test)

    if df1.shape[0] == 0 or df2.shape[0] == 0:
        raise SystemExit("[FATAL] One of the inputs is empty; cannot merge.")

    # Align columns
    cols = sorted(set(df1.columns).union(set(df2.columns)))
    df1 = df1.reindex(columns=cols)
    df2 = df2.reindex(columns=cols)

    df = pd.concat([df1, df2], ignore_index=True)

    # De-dup if needed
    key = ["run_date", "exp_name", "cp_subdir", "alpha", "eval_split"]
    if all(c in df.columns for c in key):
        df = df.drop_duplicates(subset=key, keep="last")

    p_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p_out, index=False)
    print("[OK] wrote:", str(p_out))
    if "eval_split" in df.columns:
        print(df["eval_split"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
