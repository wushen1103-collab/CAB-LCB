#!/usr/bin/env python3
import argparse
import json
import math
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def fmt_token(x: float) -> str:
    s = f"{x:g}"
    return s.replace(".", "p")


def clopper_pearson_lcb(p_hat: float, n: int, delta: float) -> float:
    if n <= 0:
        return float("nan")
    k = int(round(float(p_hat) * n))
    k = max(0, min(n, k))
    if k == 0:
        return 0.0
    try:
        from scipy.stats import beta  # type: ignore

        return float(beta.ppf(delta, k, n - k + 1))
    except Exception:
        # Wilson score one-sided lower bound (fallback)
        z = _norm_ppf(1.0 - delta)
        ph = k / n
        denom = 1.0 + z * z / n
        center = (ph + z * z / (2.0 * n)) / denom
        half = (z / denom) * math.sqrt((ph * (1.0 - ph) / n) + (z * z / (4.0 * n * n)))
        return max(0.0, center - half)


def _norm_ppf(q: float) -> float:
    # Approximation to inverse CDF for standard normal.
    # Peter J. Acklam's approximation.
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1)")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if q < plow:
        r = math.sqrt(-2 * math.log(q))
        return (
            (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5])
            / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
        )
    if q > phigh:
        r = math.sqrt(-2 * math.log(1 - q))
        return -(
            (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5])
            / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
        )
    r = q - 0.5
    s = r * r
    return (
        (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r
        / (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1)
    )


@dataclass(frozen=True)
class Selection:
    dataset: str
    split: str
    seed: int
    alpha: float
    k: int
    m: int
    gamma: float
    exp_dir_calcp: str
    cp_subdir_calcp: str
    coverage_calcp: float
    width_calcp: float
    n_eval_used: int
    lcb_calcp: float
    met_target_lcb: bool


def parse_group_and_repr(cp_subdir: str) -> Tuple[str, str]:
    # Expected examples:
    #   cp_local_autosel_target_k240_m50_gamma0_alpha0p05_calcp
    #   cp_local_autosel_target_tfidf_k60_m200_gamma0p0_alpha0p1_calcp
    s = cp_subdir
    group_by = "target" if "target" in s else "drug"
    if "target_tfidf" in s:
        return ("target", "tfidf")
    if "target_aacomp" in s:
        return ("target", "aacomp")
    if "drug_morgan" in s:
        return ("drug", "morgan")
    if group_by == "target":
        return ("target", "aacomp")
    return ("drug", "morgan")


def run_local_cp(run_dir: str, out_subdir: str, alpha: float, k: int, m: int, gamma: float, group_by: str, repr_name: str, overwrite: bool) -> None:
    cmd = [
        "python",
        "scripts/run_local_conformal_from_preds.py",
        "--run_dir",
        run_dir,
        "--alpha",
        str(alpha),
        "--k_neighbors",
        str(int(k)),
        "--min_cal_samples",
        str(int(m)),
        "--distance_inflate_gamma",
        str(float(gamma)),
        "--group_by",
        group_by,
        "--out_subdir",
        out_subdir,
    ]
    if group_by == "target":
        cmd += ["--target_repr", repr_name]
    else:
        cmd += ["--drug_repr", repr_name]
    if overwrite:
        cmd += ["--overwrite"]

    env = dict(**os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_metrics_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_date", type=str, required=True)
    ap.add_argument("--out_date", type=str, required=True)
    ap.add_argument("--runs_tag_points", type=str, required=True)
    ap.add_argument("--target_coverage", type=float, required=True)
    ap.add_argument("--candidate_alphas", type=str, required=True)
    ap.add_argument("--lcb_delta", type=float, default=0.05)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    in_dir = Path("results/tables") / args.in_date
    out_dir = Path("results/tables") / args.out_date
    out_dir.mkdir(parents=True, exist_ok=True)

    calcp_csv = in_dir / "conformal_autosel_calcp_all.csv"
    fixed_csv = in_dir / "conformal_autosel_all.fixed.csv"
    if not calcp_csv.exists():
        raise FileNotFoundError(f"Missing {calcp_csv}")
    if not fixed_csv.exists():
        raise FileNotFoundError(f"Missing {fixed_csv}")

    cand_alphas = [float(x) for x in args.candidate_alphas.split(",") if x.strip()]
    df = pd.read_csv(calcp_csv)
    df = df[df["alpha"].astype(float).isin(cand_alphas)].copy()
    if df.empty:
        raise RuntimeError("No rows left after filtering candidate_alphas.")

    df["n_eval_used"] = df["n_eval_used"].astype(int)
    df["coverage"] = df["coverage"].astype(float)
    df["avg_width"] = df["avg_width"].astype(float)
    df["lcb_calcp"] = [
        clopper_pearson_lcb(p, int(n), args.lcb_delta) for p, n in zip(df["coverage"].values, df["n_eval_used"].values)
    ]
    df["met_target_lcb"] = df["lcb_calcp"] >= float(args.target_coverage)

    selections = []
    for (dataset, split, seed), g in df.groupby(["dataset", "split", "seed"], sort=False):
        g2 = g.copy()
        feasible = g2[g2["met_target_lcb"]]
        if len(feasible) > 0:
            best = feasible.sort_values(["avg_width", "alpha", "k", "m", "gamma"]).iloc[0]
            met = True
        else:
            best = g2.sort_values(["lcb_calcp", "avg_width"], ascending=[False, True]).iloc[0]
            met = False
        selections.append(
            Selection(
                dataset=str(best["dataset"]),
                split=str(best["split"]),
                seed=int(best["seed"]),
                alpha=float(best["alpha"]),
                k=int(best["k"]),
                m=int(best["m"]),
                gamma=float(best["gamma"]),
                exp_dir_calcp=str(best["exp_dir"]),
                cp_subdir_calcp=str(best["cp_subdir"]),
                coverage_calcp=float(best["coverage"]),
                width_calcp=float(best["avg_width"]),
                n_eval_used=int(best["n_eval_used"]),
                lcb_calcp=float(best["lcb_calcp"]),
                met_target_lcb=bool(met),
            )
        )

    sel_rows = []
    eval_rows = []

    runs_tag_points = Path(args.runs_tag_points)

    def map_points_run_dir(exp_dir_calcp: str) -> Path:
        name = Path(exp_dir_calcp).name.replace("__calcp", "")
        return runs_tag_points / name

    # Run eval (build constrained dirs) if requested
    futures = {}
    if not args.skip_eval:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
            for s in selections:
                run_dir = map_points_run_dir(s.exp_dir_calcp)
                if not run_dir.exists():
                    raise FileNotFoundError(f"Missing points run_dir: {run_dir}")

                group_by, repr_name = parse_group_and_repr(s.cp_subdir_calcp)
                out_subdir = (
                    f"cp_local_constrained_lcb_{group_by}_{repr_name}"
                    f"_k{s.k}_m{s.m}_gamma{fmt_token(s.gamma)}_alpha{fmt_token(s.alpha)}"
                )
                fut = ex.submit(
                    _run_one,
                    str(run_dir),
                    out_subdir,
                    s.alpha,
                    s.k,
                    s.m,
                    s.gamma,
                    group_by,
                    repr_name,
                    bool(args.overwrite),
                )
                futures[fut] = (s, str(run_dir), out_subdir)

            for fut in as_completed(list(futures.keys())):
                fut.result()

    for s in selections:
        run_dir = map_points_run_dir(s.exp_dir_calcp)
        cp_base_sel = s.cp_subdir_calcp.replace("_calcp", "")
        group_by, repr_name = parse_group_and_repr(s.cp_subdir_calcp)
        out_subdir = (
            f"cp_local_constrained_lcb_{group_by}_{repr_name}"
            f"_k{s.k}_m{s.m}_gamma{fmt_token(s.gamma)}_alpha{fmt_token(s.alpha)}"
        )

        sel_rows.append(
            {
                "dataset": s.dataset,
                "split": s.split,
                "seed": s.seed,
                "run_id": str(run_dir),
                "cp_subdir_calcp_sel": s.cp_subdir_calcp,
                "cp_base_sel": cp_base_sel,
                "alpha_sel": s.alpha,
                "k_sel": s.k,
                "m_sel": s.m,
                "gamma_sel": s.gamma,
                "coverage_calcp": s.coverage_calcp,
                "width_calcp": s.width_calcp,
                "lcb_calcp": s.lcb_calcp,
                "met_target": bool(s.met_target_lcb),
                "n_eval_used_calcp": s.n_eval_used,
                "lcb_delta": float(args.lcb_delta),
            }
        )

        coverage_eval = float("nan")
        width_eval = float("nan")
        if not args.skip_eval:
            metrics_path = Path(run_dir) / out_subdir / "conformal_metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics: {metrics_path}")
            mj = read_metrics_json(metrics_path)
            coverage_eval = float(mj.get("coverage", mj.get("coverage_mean", float("nan"))))
            width_eval = float(mj.get("avg_width", mj.get("width_mean", float("nan"))))

        eval_rows.append(
            {
                "dataset": s.dataset,
                "split": s.split,
                "seed": s.seed,
                "run_id": str(run_dir),
                "cp_subdir_calcp_sel": s.cp_subdir_calcp,
                "cp_base_sel": cp_base_sel,
                "alpha_sel": s.alpha,
                "k_sel": s.k,
                "m_sel": s.m,
                "gamma_sel": s.gamma,
                "coverage_calcp": s.coverage_calcp,
                "width_calcp": s.width_calcp,
                "coverage_eval": coverage_eval,
                "width_eval": width_eval,
                "coverage": coverage_eval,
                "avg_width": width_eval,
                "met_target": bool(s.met_target_lcb),
                "lcb_calcp": s.lcb_calcp,
                "lcb_delta": float(args.lcb_delta),
            }
        )

    pd.DataFrame(sel_rows).to_csv(out_dir / "constrained_autosel_selection_by_calcp.csv", index=False)
    pd.DataFrame(eval_rows).to_csv(out_dir / "constrained_autosel_eval_selected.csv", index=False)

    # Copy fixed autosel file so make_three_way_table.py can run on out_date
    shutil.copy2(fixed_csv, out_dir / "conformal_autosel_all.fixed.csv")

    print(f"[OK] wrote: {out_dir/'constrained_autosel_selection_by_calcp.csv'}")
    print(f"[OK] wrote: {out_dir/'constrained_autosel_eval_selected.csv'}")
    print(f"[OK] copied: {out_dir/'conformal_autosel_all.fixed.csv'}")


def _run_one(run_dir: str, out_subdir: str, alpha: float, k: int, m: int, gamma: float, group_by: str, repr_name: str, overwrite: bool) -> None:
    # Keep this helper at top-level for ProcessPoolExecutor pickling.
    cmd = [
        "python",
        "scripts/run_local_conformal_from_preds.py",
        "--run_dir",
        run_dir,
        "--alpha",
        str(alpha),
        "--k_neighbors",
        str(int(k)),
        "--min_cal_samples",
        str(int(m)),
        "--distance_inflate_gamma",
        str(float(gamma)),
        "--group_by",
        group_by,
        "--out_subdir",
        out_subdir,
    ]
    if group_by == "target":
        cmd += ["--target_repr", repr_name]
    else:
        cmd += ["--drug_repr", repr_name]
    if overwrite:
        cmd += ["--overwrite"]

    env = dict(**os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    import os

    main()
