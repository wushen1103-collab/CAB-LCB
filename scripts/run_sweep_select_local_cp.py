#!/usr/bin/env python3
"""
Run calibration-only hyperparam selection for many run directories in parallel.

Notes:
- This is primarily CPU work. GPU ids are used only to assign CUDA_VISIBLE_DEVICES
  for process isolation; they do not accelerate the computations.
- We force OMP/MKL thread counts to 1 per process to avoid oversubscription.

Example:
  PY=/home/test/miniconda3/envs/dti_cp/bin/python
  $PY scripts/run_sweep_select_local_cp.py \
    --gpus 0,1,2,3,4,5,6,7 \
    --runs_root runs/2025-12-20 \
    --max_parallel 8 \
    --alphas 0.05,0.1,0.2 \
    --group_by target \
    --tfidf_ngram_min 3 --tfidf_ngram_max 5 --tfidf_max_features 8000 \
    --k_list 30,60,120,240 \
    --m_list 50,100,200,400 \
    --gamma_list 0.0,0.05,0.1 \
    --cal_select_frac 0.5 \
    --out_prefix cp_local_autosel \
    --overwrite
"""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


def parse_gpus(s: str) -> List[str]:
    s = s.strip()
    if s == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip() != ""]


def find_run_dirs(runs_root: Path) -> List[Path]:
    run_dirs: List[Path] = []
    for p in runs_root.rglob("*"):
        if not p.is_dir():
            continue
        if (p / "preds_cal.csv.gz").exists() and (p / "preds_test.csv.gz").exists():
            run_dirs.append(p)
    run_dirs = sorted(run_dirs)
    return run_dirs


def run_one(py: str, run_dir: Path, gpu_id: str, base_env: Dict[str, str], args: argparse.Namespace) -> Tuple[Path, int]:
    cmd = [
        py,
        str(Path("scripts") / "select_local_cp_hparams.py"),
        "--run_dir",
        str(run_dir),
        "--alphas",
        args.alphas,
        "--group_by",
        args.group_by,
        "--drug_repr",
        args.drug_repr,
        "--target_repr",
        args.target_repr,
        "--tfidf_ngram_min",
        str(args.tfidf_ngram_min),
        "--tfidf_ngram_max",
        str(args.tfidf_ngram_max),
        "--tfidf_max_features",
        str(args.tfidf_max_features),
        "--morgan_radius",
        str(args.morgan_radius),
        "--morgan_nbits",
        str(args.morgan_nbits),
        "--k_list",
        args.k_list,
        "--m_list",
        args.m_list,
        "--gamma_list",
        args.gamma_list,
        "--cal_select_frac",
        str(args.cal_select_frac),
        "--select_seed",
        str(args.select_seed),
        "--knn_metric",
        args.knn_metric,
        "--dist_norm",
        args.dist_norm,
        "--out_prefix",
        args.out_prefix,
    ]
    if args.overwrite:
        cmd.append("--overwrite")

    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    # Avoid CPU thread oversubscription
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"select_{run_dir.name}.log"

    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(Path.cwd()))
    return run_dir, int(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel runner for select_local_cp_hparams.py")
    ap.add_argument("--py", default="", type=str, help="Python executable path. If empty, uses sys.executable.")
    ap.add_argument("--gpus", default="", type=str, help="Comma-separated GPU ids (used only for CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--runs_root", required=True, type=str)
    ap.add_argument("--max_parallel", default=4, type=int)
    ap.add_argument("--log_dir", default="runs/_sweep_logs_select", type=str)

    ap.add_argument("--alphas", default="0.1", type=str)
    ap.add_argument("--group_by", default="target", type=str, choices=["target", "drug", "pair"])
    ap.add_argument("--drug_repr", default="morgan", type=str, choices=["morgan"])
    ap.add_argument("--target_repr", default="tfidf", type=str, choices=["tfidf"])

    ap.add_argument("--tfidf_ngram_min", default=3, type=int)
    ap.add_argument("--tfidf_ngram_max", default=5, type=int)
    ap.add_argument("--tfidf_max_features", default=8000, type=int)

    ap.add_argument("--morgan_radius", default=2, type=int)
    ap.add_argument("--morgan_nbits", default=2048, type=int)

    ap.add_argument("--k_list", default="30,60,120,240", type=str)
    ap.add_argument("--m_list", default="50,100,200,400", type=str)
    ap.add_argument("--gamma_list", default="0.0,0.05,0.1", type=str)

    ap.add_argument("--cal_select_frac", default=0.5, type=float)
    ap.add_argument("--select_seed", default=-1, type=int)

    ap.add_argument("--knn_metric", default="cosine", type=str)
    ap.add_argument("--dist_norm", default="median", type=str, choices=["median", "none"])

    ap.add_argument("--out_prefix", default="cp_local_autosel", type=str)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    py = args.py.strip()
    if py == "":
        py = os.environ.get("PY", "")
    if py == "":
        raise RuntimeError("Python executable not provided. Set --py or export PY.")

    runs_root = Path(args.runs_root).resolve()
    run_dirs = find_run_dirs(runs_root)
    if len(run_dirs) == 0:
        raise RuntimeError(f"No run dirs found under {runs_root}")

    gpu_ids = parse_gpus(args.gpus)
    if len(gpu_ids) == 0:
        gpu_ids = ["-1"]

    base_env = dict(os.environ)

    futures = []
    ok = 0
    bad = 0

    with ThreadPoolExecutor(max_workers=int(args.max_parallel)) as ex:
        for i, rd in enumerate(run_dirs):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            futures.append(ex.submit(run_one, py, rd, gpu_id, base_env, args))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="jobs done"):
            rd, rc = fut.result()
            if rc == 0:
                ok += 1
            else:
                bad += 1

    print(f"[DONE] ok={ok} bad={bad} total={len(run_dirs)}")
    print(f"[LOGS] {Path(args.log_dir).resolve()}")


if __name__ == "__main__":
    main()
