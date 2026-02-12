#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_range(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(part))
    return out


@dataclass
class Job:
    dataset: str
    split: str
    seed: int
    model: str
    gpu: Optional[str]
    cmd: List[str]
    log_path: Path
    returncode: Optional[int] = None


def build_cmd(args, train_py: Path, dataset: str, split: str, seed: int) -> List[str]:
    cmd = [
        sys.executable,
        str(train_py),
        "--dataset",
        dataset,
        "--split",
        split,
        "--seed",
        str(seed),
        "--model",
        args.model,
        "--lr",
        str(args.lr),
        "--batch_size",
        str(args.batch_size),
        "--max_epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--val_frac",
        str(args.val_frac),
        "--num_workers",
        str(args.num_workers),
    ]
    if args.out_root:
        exp_name = f"graphdta_point_{args.model}_{dataset}_{split}_seed{seed}"
        out_subdir = Path(args.out_root) / datetime.now().strftime("%Y-%m-%d") / exp_name
        cmd += ["--out_subdir", str(out_subdir)]
    return cmd


def tail_text(path: Path, n: int = 60) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep training jobs for GraphDTA point models")
    ap.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids. Example: 0,1,2,3")
    ap.add_argument("--datasets", type=str, required=True, help="Comma-separated datasets. Example: davis,kiba")
    ap.add_argument("--splits", type=str, required=True, help="Comma-separated splits. Example: random,cold_drug,cold_target,cold_pair")
    ap.add_argument("--seeds", type=str, required=True, help="Seed range. Example: 0-4 or 0,1,2")
    ap.add_argument("--model", type=str, default="gat_gcn")
    ap.add_argument("--max_parallel", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val_frac", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_root", type=str, default="", help="Optional root dir for outputs (relative to repo root).")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--show_failures", type=int, default=3)
    args = ap.parse_args()

    root = repo_root()
    train_py = root / "scripts" / "train_graphdta_point.py"
    if not train_py.exists():
        raise FileNotFoundError(f"Missing train script: {train_py}")

    datasets = parse_csv_list(args.datasets)
    splits = parse_csv_list(args.splits)
    seeds = parse_int_range(args.seeds)
    gpus = parse_csv_list(args.gpus) if args.gpus else []

    if args.out_root:
        out_root = Path(args.out_root)
        if not out_root.is_absolute():
            args.out_root = str(root / out_root)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = root / "runs" / "_sweep_logs" / stamp
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Job] = []
    for ds in datasets:
        for sp in splits:
            for sd in seeds:
                cmd = build_cmd(args, train_py, ds, sp, sd)
                log_path = log_dir / f"train_{args.model}_{ds}_{sp}_seed{sd}.log"
                jobs.append(Job(dataset=ds, split=sp, seed=sd, model=args.model, gpu=None, cmd=cmd, log_path=log_path))

    if args.dry_run:
        for j in jobs:
            print(" ".join(j.cmd))
        print(f"[DRY_RUN] jobs={len(jobs)} log_dir={log_dir}")
        return

    max_parallel = int(args.max_parallel)
    if max_parallel < 1:
        max_parallel = 1

    if gpus:
        max_parallel = min(max_parallel, len(gpus))

    pending = jobs[:]
    running: List[Tuple[Job, subprocess.Popen, object]] = []
    ok: List[Job] = []
    bad: List[Job] = []

    gpu_pool: List[Optional[str]] = gpus[:] if gpus else [None] * max_parallel

    pbar = tqdm(total=len(jobs), desc="jobs done")

    def start_one(job: Job, gpu: Optional[str]) -> Tuple[subprocess.Popen, object]:
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        log_f = open(job.log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            job.cmd,
            cwd=str(root),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
        )
        job.gpu = gpu
        return proc, log_f

    while pending or running:
        while pending and gpu_pool and len(running) < max_parallel:
            gpu = gpu_pool.pop(0)
            job = pending.pop(0)
            proc, log_f = start_one(job, gpu)
            running.append((job, proc, log_f))

        time.sleep(0.2)

        still_running: List[Tuple[Job, subprocess.Popen, object]] = []
        for job, proc, log_f in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((job, proc, log_f))
                continue
            job.returncode = int(ret)
            try:
                log_f.close()
            except Exception:
                pass
            gpu_pool.append(job.gpu)
            if job.returncode == 0:
                ok.append(job)
            else:
                bad.append(job)
            pbar.update(1)
        running = still_running

    pbar.close()

    print(f"[DONE] ok={len(ok)} bad={len(bad)} total={len(jobs)}")
    print(f"[LOGS] {log_dir}")

    if bad and int(args.show_failures) > 0:
        n = min(int(args.show_failures), len(bad))
        print(f"[FAILURES] showing {n}/{len(bad)}")
        for j in bad[:n]:
            print("-" * 80)
            print(f"job: dataset={j.dataset} split={j.split} seed={j.seed} model={j.model} gpu={j.gpu} rc={j.returncode}")
            print(f"log: {j.log_path}")
            print("cmd:", " ".join(j.cmd))
            print("[log tail]")
            print(tail_text(j.log_path, n=80))


if __name__ == "__main__":
    main()
