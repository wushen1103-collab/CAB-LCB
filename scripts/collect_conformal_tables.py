import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd


# --- Metadata resolution ---
# Some runs may have stale or patched fields in metrics.json (e.g., split="cold_drug__calcp")
# which breaks downstream table joins. For paper-grade reporting, we prefer dataset/split/seed
# parsed from the experiment directory name whenever available.
EXP_NAME_RE = re.compile(r"_(?P<dataset>davis|kiba)_(?P<split>.+)_seed(?P<seed>\d+)$")


def parse_exp_name(exp_name: str):
    """Parse dataset/split/seed from an experiment directory name.

    Returns (dataset, split, seed) or (None, None, None) if parsing fails.
    """
    m = EXP_NAME_RE.search(exp_name)
    if not m:
        return None, None, None
    return m.group("dataset"), m.group("split"), int(m.group("seed"))


def resolve_metadata(exp_name: str, exp_metrics: dict):
    """Resolve dataset/split/seed with exp_dir name as the source of truth.

    If metrics.json disagrees with exp_name-derived values, exp_name wins to keep joins stable.
    """
    ds_n, sp_n, sd_n = parse_exp_name(exp_name)
    ds_m = exp_metrics.get("dataset")
    sp_m = exp_metrics.get("split")
    sd_m = exp_metrics.get("seed")

    ds = ds_n or ds_m
    sp = sp_n or sp_m
    try:
        sd = int(sd_m) if sd_m is not None else sd_n
    except Exception:
        sd = sd_n

    if ds_n is not None and ds_m is not None and ds_m != ds_n:
        ds = ds_n
    if sp_n is not None and sp_m is not None and sp_m != sp_n:
        sp = sp_n
    if sd_n is not None and sd is not None and sd != sd_n:
        sd = sd_n

    return ds, sp, sd



def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def normalize_subdir(subdir: str) -> str:
    s = subdir
    s = s.replace("_safe", "")
    s = s.replace("_evalOnCalCP", "")
    s = re.sub(r"_gamma[0-9a-zp\.]+", "_gamma*", s)
    s = re.sub(r"_K\d+", "_K*", s)
    s = re.sub(r"_k\d+", "_k*", s)
    s = re.sub(r"_m\d+", "_m*", s)
    return s


def method_class(subdir: str) -> str:
    if subdir.startswith("cp_split_"):
        return "split"
    if subdir.startswith("cp_cluster_"):
        return "cluster"
    if subdir.startswith("cp_knn_"):
        return "knn"
    if subdir.startswith("cp_local_"):
        return "local"
    return "other"


def extract_tags(subdir: str) -> Dict[str, Any]:
    tags: Dict[str, Any] = {}
    tags["method"] = method_class(subdir)

    m = re.search(r"cp_(cluster|knn|local)_(drug|target)", subdir)
    if m:
        tags["group_by"] = m.group(2)

    m = re.search(r"_K(\d+)", subdir)
    if m:
        tags["n_clusters"] = int(m.group(1))

    m = re.search(r"_k(\d+)", subdir)
    if m:
        tags["k_neighbors"] = int(m.group(1))

    m = re.search(r"_m(\d+)", subdir)
    if m:
        tags["min_cal_samples"] = int(m.group(1))

    m = re.search(r"_gamma([0-9a-zp\.]+)", subdir)
    if m:
        raw = m.group(1).replace("p", ".")
        try:
            tags["distance_inflate_gamma"] = float(raw)
        except Exception:
            tags["distance_inflate_gamma"] = raw

    if "tfidf" in subdir:
        tags["target_repr"] = "tfidf"
    if "aacomp" in subdir:
        tags["target_repr"] = "aacomp"

    return tags


def pick_coverage_width(cm: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
    if "coverage_eval" in cm or "avg_width_eval" in cm:
        return safe_float(cm.get("coverage_eval")), safe_float(cm.get("avg_width_eval")), "eval"
    return safe_float(cm.get("coverage_test")), safe_float(cm.get("avg_width_test")), "test"


def iter_experiment_dirs(runs_dir: Path) -> List[Path]:
    exp_dirs: List[Path] = []
    if not runs_dir.exists():
        return exp_dirs
    for p in runs_dir.glob("*/*"):
        if p.is_dir():
            exp_dirs.append(p)
    return exp_dirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--out_dir", type=str, default="results/tables")
    ap.add_argument("--target_coverage", type=float, default=0.9)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for exp_dir in iter_experiment_dirs(runs_dir):
        run_date = exp_dir.parent.name
        exp_name = exp_dir.name

        exp_metrics_path = exp_dir / "metrics.json"
        exp_metrics: Dict[str, Any] = {}
        if exp_metrics_path.exists():
            try:
                exp_metrics = load_json(exp_metrics_path)
            except Exception:
                exp_metrics = {}

        dataset, split, seed = resolve_metadata(exp_name, exp_metrics)

        base_info: Dict[str, Any] = {
            "run_date": run_date,
            "exp_name": exp_name,
            "exp_dir": str(exp_dir),
            "dataset": dataset,
            "split": split,
            "seed": seed,
            "model": exp_metrics.get("model"),
            "graphdta_commit": exp_metrics.get("graphdta_commit"),
            "rmse_test_point": exp_metrics.get("rmse_test"),
            "cindex_test_point": exp_metrics.get("cindex_test"),
            "rmse_cal_point": exp_metrics.get("rmse_cal"),
            "cindex_cal_point": exp_metrics.get("cindex_cal"),
            "n_train": exp_metrics.get("n_train"),
            "n_cal": exp_metrics.get("n_cal"),
            "n_test": exp_metrics.get("n_test"),
        }

        for cm_path in exp_dir.glob("cp*/conformal_metrics.json"):
            subdir = cm_path.parent.name
            try:
                cm = load_json(cm_path)
            except Exception:
                continue

            cov, wid, eval_split = pick_coverage_width(cm)

            r: Dict[str, Any] = dict(base_info)
            r.update({
                "cp_subdir": subdir,
                "cp_dir": str(cm_path.parent),
                "alpha": cm.get("alpha"),
                "eval_split": eval_split,
                "coverage": cov,
                "avg_width": wid,
                "global_qhat": cm.get("global_qhat", cm.get("qhat")),
            })
            r.update(extract_tags(subdir))
            r["normalized_family"] = normalize_subdir(subdir)

            for k, v in cm.items():
                if k in r:
                    continue
                if isinstance(v, (int, float, str, bool)) or v is None:
                    r[f"cm_{k}"] = v

            rows.append(r)

    if not rows:
        raise RuntimeError(f"No conformal_metrics.json found under {runs_dir}")

    df = pd.DataFrame(rows)

    all_csv = out_dir / "conformal_all.csv"
    df.to_csv(all_csv, index=False)

    df_eval = df[df["eval_split"] == "eval"].copy()
    if df_eval.empty:
        selected_df = pd.DataFrame()
    else:
        key_cols = ["dataset", "split", "seed", "model", "method", "group_by", "target_repr"]
        for c in key_cols:
            if c not in df_eval.columns:
                df_eval[c] = None

        def sort_key(frame: pd.DataFrame) -> pd.DataFrame:
            frame = frame.copy()
            frame["cov_ok"] = (frame["coverage"] >= float(args.target_coverage)).astype(int)
            frame["gamma_val"] = frame.get("distance_inflate_gamma")
            frame["gamma_val"] = frame["gamma_val"].apply(lambda x: float(x) if (x is not None and not pd.isna(x)) else 0.0)
            frame["k_val"] = frame.get("k_neighbors")
            frame["k_val"] = frame["k_val"].apply(lambda x: int(x) if (x is not None and not pd.isna(x)) else 10**9)
            frame["K_val"] = frame.get("n_clusters")
            frame["K_val"] = frame["K_val"].apply(lambda x: int(x) if (x is not None and not pd.isna(x)) else 10**9)
            frame["width_val"] = frame["avg_width"].apply(lambda x: float(x) if x is not None else 1e30)
            frame["cov_val"] = frame["coverage"].apply(lambda x: float(x) if x is not None else -1.0)

            frame = frame.sort_values(
                by=["cov_ok", "width_val", "gamma_val", "k_val", "K_val", "cov_val"],
                ascending=[False, True, True, True, True, False],
                kind="mergesort",
            )
            return frame

        selected_rows: List[pd.Series] = []
        for _, g in df_eval.groupby(key_cols, dropna=False):
            gg = sort_key(g)
            selected_rows.append(gg.iloc[0])

        selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)

        df_test = df[df["eval_split"] == "test"].copy()
        test_map = {}
        for _, r in df_test.iterrows():
            test_map[(r["exp_dir"], r["cp_subdir"])] = r

        test_covs = []
        test_widths = []
        test_subdirs = []

        for _, r in selected_df.iterrows():
            exp_dir = r["exp_dir"]
            subdir_eval = r["cp_subdir"]
            subdir_test_guess = subdir_eval.replace("_evalOnCalCP", "")

            rr = test_map.get((exp_dir, subdir_test_guess))
            if rr is None:
                rr = test_map.get((exp_dir, subdir_eval))

            if rr is None:
                test_covs.append(None)
                test_widths.append(None)
                test_subdirs.append(None)
            else:
                test_covs.append(rr.get("coverage"))
                test_widths.append(rr.get("avg_width"))
                test_subdirs.append(rr.get("cp_subdir"))

        selected_df["matched_test_cp_subdir"] = test_subdirs
        selected_df["coverage_test_matched"] = test_covs
        selected_df["avg_width_test_matched"] = test_widths

    sel_csv = out_dir / "conformal_selected_by_calcp.csv"
    selected_df.to_csv(sel_csv, index=False)

    print(f"[OK] Wrote: {all_csv}")
    print(f"[OK] Wrote: {sel_csv}")
    if selected_df.empty:
        print("[WARN] No eval-based rows found (coverage_eval/avg_width_eval).")
        print("       You need evalOnCalCP runs to enable strict cal_cp selection.")


if __name__ == "__main__":
    main()
