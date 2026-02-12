import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_day", type=str, required=True, help="e.g., 2025-12-17")
    ap.add_argument("--results_dir", type=str, default="results/tables")
    ap.add_argument("--alpha_subdir", type=str, default="cp_split_alpha0.1")
    args = ap.parse_args()

    runs_root = Path("runs") / args.runs_day
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in sorted(runs_root.glob("graphdta_point_gat_gcn_*_seed42")):
        m_path = run_dir / "metrics.json"
        c_path = run_dir / args.alpha_subdir / "conformal_metrics.json"
        if not m_path.exists() or not c_path.exists():
            continue

        m = load_json(m_path)
        c = load_json(c_path)

        row = {
            "dataset": m.get("dataset"),
            "split": m.get("split"),
            "seed": m.get("seed"),
            "model": m.get("model"),
            "graphdta_commit": m.get("graphdta_commit"),
            "rmse_test": m.get("rmse_test"),
            "cindex_test": m.get("cindex_test"),
            "alpha": c.get("alpha"),
            "qhat": c.get("qhat"),
            "coverage_test": c.get("coverage_test"),
            "avg_width_test": c.get("avg_width_test"),
            "n_train": m.get("n_train"),
            "n_cal": c.get("n_cal"),
            "n_test": c.get("n_test"),
            "run_dir": str(run_dir),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset", "split"]).reset_index(drop=True)

    out_csv = out_dir / f"graphdta_gat_gcn_seed42_alpha0.1_summary_{args.runs_day}.csv"
    df.to_csv(out_csv, index=False)
    print(str(out_csv))


if __name__ == "__main__":
    main()
