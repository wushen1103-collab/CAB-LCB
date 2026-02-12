import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def conformal_qhat(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n < 2:
        raise ValueError("Calibration set too small.")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(scores, k - 1)[k - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Run directory containing preds_cal.csv.gz and preds_test.csv.gz")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out_subdir", type=str, default=None, help="Optional subdir under run_dir to store outputs")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    cal_path = run_dir / "preds_cal.csv.gz"
    test_path = run_dir / "preds_test.csv.gz"
    if not cal_path.exists():
        raise FileNotFoundError(f"missing: {cal_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"missing: {test_path}")

    cal = pd.read_csv(cal_path, compression="gzip")
    test = pd.read_csv(test_path, compression="gzip")

    for col in ["y", "y_pred"]:
        if col not in cal.columns or col not in test.columns:
            raise ValueError(f"missing required column '{col}' in preds files")

    cal_scores = np.abs(cal["y"].to_numpy(dtype=np.float64) - cal["y_pred"].to_numpy(dtype=np.float64))
    qhat = conformal_qhat(cal_scores, args.alpha)

    y_test = test["y"].to_numpy(dtype=np.float64)
    p_test = test["y_pred"].to_numpy(dtype=np.float64)
    lo = p_test - qhat
    hi = p_test + qhat

    covered = (y_test >= lo) & (y_test <= hi)
    coverage = float(np.mean(covered))
    avg_width = float(np.mean(hi - lo))

    metrics = {
        "alpha": float(args.alpha),
        "qhat": float(qhat),
        "coverage_test": coverage,
        "avg_width_test": avg_width,
        "n_cal": int(len(cal)),
        "n_test": int(len(test)),
    }

    out_dir = run_dir if args.out_subdir is None else (run_dir / args.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "conformal_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    out = test.copy()
    out["pi_lo"] = lo
    out["pi_hi"] = hi
    out["covered"] = covered.astype(int)
    out.to_csv(out_dir / "pred_intervals_test.csv.gz", index=False, compression="gzip")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
