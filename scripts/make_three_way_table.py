#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def must_have(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{name}] missing columns: {missing}\nAvailable: {list(df.columns)}")


def ensure_int_seed(df: pd.DataFrame) -> pd.DataFrame:
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64").astype(int)
    return df


def norm_split(df: pd.DataFrame) -> pd.DataFrame:
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.replace("__calcp", "", regex=False)
    return df


def coerce_metric_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Accept either (coverage, avg_width) or (coverage_mean, width_mean).
    if "coverage" not in df.columns and "coverage_mean" in df.columns:
        df["coverage"] = df["coverage_mean"]
    if "avg_width" not in df.columns and "width_mean" in df.columns:
        df["avg_width"] = df["width_mean"]
    return df


def to_bool_series(s: pd.Series) -> pd.Series:
    # Accept bool dtype directly; otherwise parse common truthy strings.
    if getattr(s, "dtype", None) == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def fmt_alpha(alpha: float) -> str:
    # 0.10 -> 0p10
    return f"{alpha:.2f}".replace(".", "p")


def assert_n_seeds_is(df_sum: pd.DataFrame, expected: int, scheme: str) -> None:
    if df_sum.empty:
        return
    bad = df_sum[df_sum["n_seeds"] != expected][["dataset", "split", "n_seeds"]]
    if not bad.empty:
        raise RuntimeError(
            f"[{scheme}] unexpected n_seeds (expected {expected}).\n"
            f"{bad.to_string(index=False)}"
        )


def summarize_seed_rows(
    df_seed: pd.DataFrame,
    target_coverage: float,
    scheme_name: str,
    with_alpha: bool,
) -> pd.DataFrame:
    # df_seed must contain one row per (dataset, split, seed)
    if df_seed.empty:
        return pd.DataFrame(
            columns=[
                "scheme",
                "dataset",
                "split",
                "n_seeds",
                "coverage_mean",
                "coverage_std",
                "coverage_gap_to_target",
                "width_mean",
                "width_std",
                "alpha_sel_mode",
                "alpha_sel_mean",
                "nominal_target_mean",
                "coverage_gap_to_nominal",
                "met_target_rate",
            ]
        )

    df_seed = df_seed.copy()
    df_seed = coerce_metric_cols(df_seed)
    must_have(df_seed, ["dataset", "split", "seed", "coverage", "avg_width"], scheme_name)

    grp = df_seed.groupby(["dataset", "split"], as_index=False)
    out = grp.agg(
        n_seeds=("seed", "nunique"),
        coverage_mean=("coverage", "mean"),
        coverage_std=("coverage", "std"),
        width_mean=("avg_width", "mean"),
        width_std=("avg_width", "std"),
    )
    out["scheme"] = scheme_name
    out["coverage_gap_to_target"] = out["coverage_mean"] - float(target_coverage)

    # Alpha-driven columns (publication-facing): derive from eval_selected alpha.
    if with_alpha and "alpha" in df_seed.columns:
        alpha_grp = df_seed.groupby(["dataset", "split"])["alpha"]
        alpha_mean = alpha_grp.mean().reset_index(name="alpha_sel_mean")

        # Mode with deterministic tie-break (smallest value).
        alpha_mode = (
            alpha_grp.apply(lambda x: float(pd.Series(x).mode().sort_values().iloc[0]))
            .reset_index(name="alpha_sel_mode")
        )

        out = out.merge(alpha_mode, on=["dataset", "split"], how="left")
        out = out.merge(alpha_mean, on=["dataset", "split"], how="left")

        out["nominal_target_mean"] = 1.0 - out["alpha_sel_mean"]
        out["coverage_gap_to_nominal"] = out["coverage_mean"] - out["nominal_target_mean"]
    else:
        out["alpha_sel_mode"] = np.nan
        out["alpha_sel_mean"] = np.nan
        out["nominal_target_mean"] = np.nan
        out["coverage_gap_to_nominal"] = np.nan

    # met_target_rate must use eval-side definition if present (publication-facing).
    if "met_target" in df_seed.columns:
        mt = df_seed[["dataset", "split", "seed", "met_target"]].copy()
        mt["met_target"] = to_bool_series(mt["met_target"])
        mt_rate = mt.groupby(["dataset", "split"])["met_target"].mean().reset_index(name="met_target_rate")
        out = out.merge(mt_rate, on=["dataset", "split"], how="left")
    else:
        out["met_target_rate"] = np.nan

    out = out[
        [
            "scheme",
            "dataset",
            "split",
            "n_seeds",
            "coverage_mean",
            "coverage_std",
            "coverage_gap_to_target",
            "width_mean",
            "width_std",
            "alpha_sel_mode",
            "alpha_sel_mean",
            "nominal_target_mean",
            "coverage_gap_to_nominal",
            "met_target_rate",
        ]
    ]
    return out.sort_values(["dataset", "split", "scheme"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--target_coverage", type=float, required=True)
    ap.add_argument("--alpha_fixed", type=float, required=True)
    ap.add_argument("--runs_tag", required=True, help="Kept for CLI compatibility.")
    ap.add_argument("--baseline_cp_subdir", required=True)
    args = ap.parse_args()

    out_dir = Path("results/tables") / args.date
    if not out_dir.exists():
        raise FileNotFoundError(f"Missing out_dir: {out_dir}")

    eval_sel_path = out_dir / "constrained_autosel_eval_selected.csv"
    if not eval_sel_path.exists():
        raise FileNotFoundError(f"Missing {eval_sel_path}")

    # Prefer the fixed autosel table if present.
    autosel_all_path = out_dir / "conformal_autosel_all.fixed.csv"
    if not autosel_all_path.exists():
        autosel_all_path = out_dir / "conformal_autosel_all.csv"
    if not autosel_all_path.exists():
        raise FileNotFoundError(f"Missing {out_dir}/conformal_autosel_all(.fixed).csv")

    eval_sel = pd.read_csv(eval_sel_path)
    autosel_all = pd.read_csv(autosel_all_path)

    for df in (eval_sel, autosel_all):
        ensure_int_seed(df)
        norm_split(df)

    alpha_fixed = float(args.alpha_fixed)

    # ---- baseline_fixed(alpha=alpha_fixed) ----
    baseline_seed = autosel_all.copy()
    must_have(baseline_seed, ["cp_subdir", "alpha"], "autosel_all")
    baseline_seed = baseline_seed[baseline_seed["cp_subdir"].astype(str) == str(args.baseline_cp_subdir)]
    baseline_seed = baseline_seed[pd.to_numeric(baseline_seed["alpha"], errors="coerce") == alpha_fixed]
    baseline_seed = coerce_metric_cols(baseline_seed)
    baseline_seed = baseline_seed.dropna(subset=["coverage", "avg_width"])
    baseline_seed = baseline_seed.sort_values(["dataset", "split", "seed"]).drop_duplicates(
        subset=["dataset", "split", "seed"], keep="first"
    )
    baseline_sum = summarize_seed_rows(
        baseline_seed,
        target_coverage=args.target_coverage,
        scheme_name=f"baseline_fixed(alpha={alpha_fixed})",
        with_alpha=False,
    )
    assert_n_seeds_is(baseline_sum, 5, "baseline_fixed")

    # ---- search_autosel(alpha=alpha_fixed): choose best autosel candidate per seed by min width ----
    cand = autosel_all.copy()
    cand = cand[cand["cp_subdir"].astype(str).str.startswith("cp_local_autosel_", na=False)]
    cand = cand[pd.to_numeric(cand["alpha"], errors="coerce") == alpha_fixed]
    cand = coerce_metric_cols(cand)
    cand = cand.dropna(subset=["coverage", "avg_width"])

    # Select per (dataset, split, seed): min width, tie-break max coverage.
    cand = cand.sort_values(
        ["dataset", "split", "seed", "avg_width", "coverage"],
        ascending=[True, True, True, True, False],
    )
    search_seed = cand.drop_duplicates(subset=["dataset", "split", "seed"], keep="first")
    search_sum = summarize_seed_rows(
        search_seed,
        target_coverage=args.target_coverage,
        scheme_name=f"search_autosel(alpha={alpha_fixed})",
        with_alpha=False,
    )
    assert_n_seeds_is(search_sum, 5, "search_autosel")

    # ---- final_constrained_autosel: publication-facing met_target comes from eval_selected ----
    final_seed = eval_sel.copy()
    final_seed = coerce_metric_cols(final_seed)
    must_have(final_seed, ["dataset", "split", "seed", "alpha"], "eval_selected")
    final_seed = final_seed.dropna(subset=["coverage", "avg_width"])
    final_seed = final_seed.sort_values(["dataset", "split", "seed"]).drop_duplicates(
        subset=["dataset", "split", "seed"], keep="first"
    )
    final_sum = summarize_seed_rows(
        final_seed,
        target_coverage=args.target_coverage,
        scheme_name="final_constrained_autosel",
        with_alpha=True,
    )
    assert_n_seeds_is(final_sum, 5, "final_constrained_autosel")

    out = pd.concat([baseline_sum, final_sum, search_sum], ignore_index=True)

    out_path = out_dir / f"three_way_main_table_alpha{fmt_alpha(alpha_fixed)}.csv"
    out.to_csv(out_path, index=False)

    print(f"[INFO] baseline_cp_subdir = {args.baseline_cp_subdir}")
    print("\n=== three-way table (publication definition) ===")
    print(out.to_string(index=False))
    print(f"\n[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
