from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def read_yaml(p: Path) -> Dict[str, Any]:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def get_model_name(run_dir: Path) -> str:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return "unknown"
    cfg = read_yaml(cfg_path)
    if not isinstance(cfg, dict):
        return "unknown"
    model = cfg.get("model", {})
    if isinstance(model, dict) and "name" in model:
        return str(model["name"])
    return "unknown"


def make_row(metrics_path: Path) -> Dict[str, Any]:
    run_dir = metrics_path.parent
    m = read_json(metrics_path)
    return {
        "run_dir": str(run_dir),
        "exp_name": run_dir.name,
        "date_dir": run_dir.parent.name,
        "mtime": metrics_path.stat().st_mtime,
        "dataset": m.get("dataset"),
        "split": m.get("split"),
        "seed": m.get("seed"),
        "model": get_model_name(run_dir),
        "rmse_cal": m.get("rmse_cal"),
        "rmse_test": m.get("rmse_test"),
        "cindex_cal": m.get("cindex_cal"),
        "cindex_test": m.get("cindex_test"),
        "n_train": m.get("n_train"),
        "n_cal": m.get("n_cal"),
        "n_test": m.get("n_test"),
    }


def dedup_latest(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    df2 = df.sort_values(keys + ["mtime"]).reset_index(drop=True)
    df2 = df2.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    return df2


def escape_latex(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def fmt_num(x: Any, float_fmt: str) -> str:
    if x is None:
        return "-"
    try:
        if pd.isna(x):
            return "-"
    except Exception:
        pass
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    if isinstance(x, (float,)) or isinstance(x, (int,)):
        try:
            return float_fmt % float(x)
        except Exception:
            return str(x)
    return escape_latex(str(x))


def df_to_latex_simple(df: pd.DataFrame, float_fmt: str) -> str:
    cols = list(df.columns)
    col_spec = "l" * len(cols)
    lines: List[str] = []
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\hline")
    header = " & ".join(escape_latex(str(c)) for c in cols) + r" \\"
    lines.append(header)
    lines.append(r"\hline")
    for _, row in df.iterrows():
        vals = [fmt_num(row[c], float_fmt) for c in cols]
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--metrics_filename", type=str, default="metrics.json")
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--latex_float_format", type=str, default="%.6f")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    results_dir = Path(args.results_dir)
    metrics_dir = results_dir / "metrics"
    tables_dir = results_dir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    metric_paths = sorted(runs_dir.rglob(args.metrics_filename))
    if not metric_paths:
        raise SystemExit(f"No '{args.metrics_filename}' found under: {runs_dir}")

    rows = [make_row(p) for p in metric_paths]
    df = pd.DataFrame(rows).sort_values(["dataset", "split", "model", "seed", "mtime"]).reset_index(drop=True)

    out_all = metrics_dir / "point_metrics_runs.csv"
    df.to_csv(out_all, index=False)

    dedup_keys = ["dataset", "split", "model", "seed"]
    df_dedup = dedup_latest(df, dedup_keys)
    out_dedup = metrics_dir / "point_metrics_runs_dedup.csv"
    df_dedup.to_csv(out_dedup, index=False)

    numeric_cols = ["rmse_cal", "rmse_test", "cindex_cal", "cindex_test"]
    agg = (
        df_dedup.groupby(["dataset", "split", "model"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            **{f"{c}_mean": (c, "mean") for c in numeric_cols},
            **{f"{c}_std": (c, "std") for c in numeric_cols},
            n_train=("n_train", "mean"),
            n_cal=("n_cal", "mean"),
            n_test=("n_test", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "split", "model"])
        .reset_index(drop=True)
    )

    out_summary = metrics_dir / "point_metrics_summary.csv"
    agg.to_csv(out_summary, index=False)

    view = agg[
        [
            "dataset",
            "split",
            "model",
            "n_seeds",
            "rmse_test_mean",
            "rmse_test_std",
            "cindex_test_mean",
            "cindex_test_std",
        ]
    ].copy()

    latex_path = tables_dir / "table_point_metrics_test.tex"
    latex_str = df_to_latex_simple(view, args.latex_float_format)
    latex_path.write_text(latex_str, encoding="utf-8")

    print(f"Wrote CSV (all runs):      {out_all}")
    print(f"Wrote CSV (dedup latest):  {out_dedup}")
    print(f"Wrote CSV (summary):       {out_summary}")
    print(f"Wrote LaTeX table:         {latex_path}")
    print()
    print(view.to_string(index=False))


if __name__ == "__main__":
    main()
