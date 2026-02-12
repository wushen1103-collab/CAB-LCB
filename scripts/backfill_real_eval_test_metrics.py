#!/usr/bin/env python3
"""
Recompute and backfill *real* eval/test split metrics into each conformal_metrics.json.

This script exists because some older workflows only had preds_test.csv.gz and used
eval-as-test aliasing. For paper-grade reporting, selection (eval) and reporting (test)
must be disjoint.

Important:
- There are two different conformal drivers in this repo:
  (A) split/global conformal: scripts/run_split_conformal_from_preds.py
      -> writes coverage_test / avg_width_test / n_test
  (B) local conformal: scripts/run_local_conformal_from_preds.py
      -> writes coverage_eval / avg_width_eval / n_eval_used (and local CP diagnostics)

If you run the split/global driver on a local-CP subdir (cp_local*), you will overwrite
local-CP metrics and different cp_subdirs can become identical on test. This script
avoids that by auto-selecting the correct driver per cp_subdir.

How it works (per exp_dir, per cp_subdir):
  1) Temporarily swap exp_dir/preds_test.csv.gz -> use exp_dir/preds_eval.csv.gz
     Run the appropriate driver and read metrics => eval metrics
  2) Restore original preds_test.csv.gz and run again => test metrics
  3) Patch the FINAL conformal_metrics.json (after step 2) to include:
       coverage_eval / avg_width_eval (from step 1)
       coverage_test / avg_width_test (from step 2)
       n_eval_used_eval (eval split size)
       n_test_used      (test split size)

Notes:
- Parallelism is only across exp_dirs (safe). Do NOT parallelize within an exp_dir.
- Dry-run is side-effect free (no swapping, no driver runs).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_LOCAL_CP_RE = re.compile(
    r"^cp_local(?:_autosel)?_"
    r"(?P<group_by>drug|target)"
    r"(?:_(?P<repr>tfidf|aacomp|morgan))?"
    r"_k(?P<k>\d+)_m(?P<m>\d+)"
    r"_gamma(?P<gamma>\d+(?:p\d+)?)_alpha(?P<alpha>\d+(?:p\d+)?)$"
)


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8", errors="ignore") or "{}")


def _write_json_atomic(p: Path, obj: Dict[str, Any]) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _swap_preds_test_for_eval(exp_dir: Path) -> tuple[Path, Path]:
    preds_test = exp_dir / "preds_test.csv.gz"
    preds_eval = exp_dir / "preds_eval.csv.gz"
    if not preds_eval.exists():
        raise FileNotFoundError(preds_eval)
    if not preds_test.exists():
        raise FileNotFoundError(preds_test)

    bak = exp_dir / f"preds_test.__orig_for_eval__.{os.getpid()}.csv.gz"
    os.replace(preds_test, bak)
    _link_or_copy(preds_eval, preds_test)
    return bak, preds_test


def _restore_preds_test(exp_dir: Path, bak_path: Path) -> None:
    preds_test = exp_dir / "preds_test.csv.gz"
    if preds_test.exists() or preds_test.is_symlink():
        preds_test.unlink()
    os.replace(bak_path, preds_test)


def _parse_alpha_from_name(name: str) -> Optional[float]:
    m = re.search(r"alpha(\d+)(?:p(\d+))?", name)
    if not m:
        return None
    left = m.group(1)
    right = m.group(2) or "0"
    return float(f"{left}.{right}")


def _is_local_cp_subdir(cp_subdir: str, exp_dir: Path) -> bool:
    if cp_subdir.startswith("cp_local"):
        return True
    cfg = exp_dir / cp_subdir / "config.json"
    if cfg.exists():
        try:
            j = _read_json(cfg)
            a = j.get("args") or {}
            return "group_by" in a and ("k_neighbors" in a or "min_cal_samples" in a)
        except Exception:
            return cp_subdir.startswith("cp_local")
    return False


def _load_local_args(exp_dir: Path, cp_subdir: str) -> Dict[str, Any]:
    """
    Prefer config.json if available (authoritative). Fallback to parsing cp_subdir name.
    """
    cfg = exp_dir / cp_subdir / "config.json"
    if cfg.exists():
        j = _read_json(cfg)
        a = j.get("args") or {}
        keep = {
            "group_by",
            "k_neighbors",
            "min_cal_samples",
            "drug_repr",
            "target_repr",
            "tfidf_ngram_min",
            "tfidf_ngram_max",
            "tfidf_max_features",
            "knn_metric",
            "dist_norm",
            "distance_inflate_gamma",
            "pca_dim",
        }
        return {k: a[k] for k in keep if k in a}

    m = _LOCAL_CP_RE.match(cp_subdir)
    if not m:
        return {}

    gd = m.groupdict()
    group_by = gd["group_by"]
    repr_name = gd.get("repr")

    if group_by == "target":
        target_repr = repr_name if repr_name in ("tfidf", "aacomp") else "aacomp"
        drug_repr = "morgan"
    else:
        drug_repr = "morgan"
        target_repr = "aacomp"

    def tag_to_float(s: str) -> float:
        return float(s.replace("p", ".")) if "p" in s else float(s)

    return {
        "group_by": group_by,
        "k_neighbors": int(gd["k"]),
        "min_cal_samples": int(gd["m"]),
        "drug_repr": drug_repr,
        "target_repr": target_repr,
        "distance_inflate_gamma": tag_to_float(gd["gamma"]),
    }


def _build_cmd_split(python_bin: str, driver: Path, exp_dir: Path, alpha: float, cp_subdir: str) -> list[str]:
    return [
        python_bin,
        str(driver),
        "--run_dir",
        str(exp_dir),
        "--alpha",
        str(alpha),
        "--out_subdir",
        cp_subdir,
    ]


def _build_cmd_local(
    python_bin: str,
    local_driver: Path,
    exp_dir: Path,
    alpha: float,
    cp_subdir: str,
    local_args: Dict[str, Any],
) -> list[str]:
    if "group_by" not in local_args:
        raise ValueError(f"Missing local_args.group_by for cp_subdir={cp_subdir}")

    cmd = [
        python_bin,
        str(local_driver),
        "--run_dir",
        str(exp_dir),
        "--alpha",
        str(alpha),
        "--group_by",
        str(local_args["group_by"]),
        "--out_subdir",
        cp_subdir,
    ]

    if "k_neighbors" in local_args:
        cmd += ["--k_neighbors", str(int(local_args["k_neighbors"]))]
    if "min_cal_samples" in local_args:
        cmd += ["--min_cal_samples", str(int(local_args["min_cal_samples"]))]

    if "drug_repr" in local_args:
        cmd += ["--drug_repr", str(local_args["drug_repr"])]
    if "target_repr" in local_args:
        cmd += ["--target_repr", str(local_args["target_repr"])]

    if "distance_inflate_gamma" in local_args:
        cmd += ["--distance_inflate_gamma", str(float(local_args["distance_inflate_gamma"]))]

    if "knn_metric" in local_args:
        cmd += ["--knn_metric", str(local_args["knn_metric"])]
    if "dist_norm" in local_args:
        cmd += ["--dist_norm", str(local_args["dist_norm"])]
    if "pca_dim" in local_args and int(local_args["pca_dim"]) > 0:
        cmd += ["--pca_dim", str(int(local_args["pca_dim"]))]

    for k in ("tfidf_ngram_min", "tfidf_ngram_max", "tfidf_max_features"):
        if k in local_args:
            cmd += [f"--{k}", str(int(local_args[k]))]

    return cmd


def _extract_cov_width_n(cm: Dict[str, Any]) -> Tuple[float, float, Optional[int]]:
    cov = None
    for k in ("coverage_test", "coverage_eval", "coverage_mean", "coverage"):
        if k in cm:
            cov = float(cm[k])
            break

    wid = None
    for k in ("avg_width_test", "avg_width_eval", "width_mean", "avg_width"):
        if k in cm:
            wid = float(cm[k])
            break

    n = None
    for k in ("n_test", "n_eval_used", "n_eval_entities", "n"):
        if k in cm:
            try:
                n = int(cm[k])
            except Exception:
                try:
                    n = int(float(cm[k]))
                except Exception:
                    n = None
            break

    if cov is None or wid is None:
        raise KeyError(f"Cannot extract cov/width from keys: {sorted(cm.keys())[:30]} ...")

    return float(cov), float(wid), n


def process_one_exp_dir(
    exp_dir: Path,
    driver: Path,
    local_driver: Path,
    python_bin: str,
    backup_metrics: bool,
    force: bool,
    write: bool,
    mode: str,
    skip_missing_eval: bool,
) -> Dict[str, int]:
    metrics_paths = sorted(exp_dir.glob("**/conformal_metrics.json"))
    metrics_paths = [p for p in metrics_paths if p.parent.parent == exp_dir]

    out = {"exp_dirs": 1, "cp_subdirs": 0, "patched": 0, "skipped": 0, "failed": 0}

    if not (exp_dir / "preds_test.csv.gz").exists():
        return out

    for mp in metrics_paths:
        try:
            cp_subdir = mp.parent.name
            out["cp_subdirs"] += 1

            cm0 = _read_json(mp)
            alpha = cm0.get("alpha")
            alpha = float(alpha) if alpha is not None else None
            if alpha is None or (isinstance(alpha, float) and (alpha != alpha)):
                alpha = _parse_alpha_from_name(cp_subdir)
            if alpha is None:
                raise ValueError(f"Cannot determine alpha for cp_subdir={cp_subdir}")

            is_local = _is_local_cp_subdir(cp_subdir, exp_dir)
            if mode == "split":
                is_local = False
            elif mode == "local":
                is_local = True

            local_args: Dict[str, Any] = {}
            if is_local:
                local_args = _load_local_args(exp_dir, cp_subdir)

            if not write:
                # Dry-run: do not swap files or run drivers; just count cp_subdirs.
                out["skipped"] += 1
                continue

            if backup_metrics:
                bak = mp.with_suffix(".json.bak_before_backfill")
                if bak.exists() and not force:
                    pass
                else:
                    shutil.copy2(mp, bak)

            # (1) Eval pass
            cov_eval = wid_eval = None
            n_eval = None
            bak_preds = None
            try:
                bak_preds, _ = _swap_preds_test_for_eval(exp_dir)
            except FileNotFoundError:
                if skip_missing_eval:
                    out["skipped"] += 1
                    continue
                raise

            try:
                if is_local:
                    cmd = _build_cmd_local(python_bin, local_driver, exp_dir, float(alpha), cp_subdir, local_args)
                else:
                    cmd = _build_cmd_split(python_bin, driver, exp_dir, float(alpha), cp_subdir)

                _run(cmd)
                m_eval = _read_json(mp)
                cov_eval, wid_eval, n_eval = _extract_cov_width_n(m_eval)
            finally:
                if bak_preds is not None:
                    _restore_preds_test(exp_dir, bak_preds)

            # (2) Test pass
            if is_local:
                cmd2 = _build_cmd_local(python_bin, local_driver, exp_dir, float(alpha), cp_subdir, local_args)
            else:
                cmd2 = _build_cmd_split(python_bin, driver, exp_dir, float(alpha), cp_subdir)

            _run(cmd2)

            m_test = _read_json(mp)
            cov_test, wid_test, n_test = _extract_cov_width_n(m_test)

            m_test["coverage_eval"] = float(cov_eval)
            m_test["avg_width_eval"] = float(wid_eval)
            m_test["coverage_test"] = float(cov_test)
            m_test["avg_width_test"] = float(wid_test)

            if n_eval is not None:
                m_test["n_eval_used_eval"] = int(n_eval)
            if n_test is not None:
                m_test["n_test_used"] = int(n_test)

            m_test["split_metrics_note"] = (
                "Backfilled real eval/test metrics by running the correct driver per cp_subdir "
                f"(mode={mode}, local={is_local})."
            )

            _write_json_atomic(mp, m_test)
            out["patched"] += 1

        except Exception:
            out["failed"] += 1
            continue

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_tag", required=True, help="e.g. runs/2025-12-23_deepdta_points/")
    ap.add_argument("--glob", default="deepdta_point_*", help="Experiment dir glob under runs_tag")

    ap.add_argument("--driver", default="scripts/run_split_conformal_from_preds.py", help="Split/global driver path")
    ap.add_argument("--local_driver", default="scripts/run_local_conformal_from_preds.py", help="Local-CP driver path")
    ap.add_argument("--mode", choices=["auto", "split", "local"], default="auto",
                    help="auto: cp_local* -> local driver; else split driver")

    ap.add_argument("--python", default=None, help="Python executable (default: python in PATH)")
    ap.add_argument("--jobs", type=int, default=1, help="Parallelism across exp_dirs (safe)")

    ap.add_argument("--write", action="store_true", help="Actually run and patch (otherwise dry-run)")
    ap.add_argument("--backup", action="store_true", help="Backup conformal_metrics.json before patching")
    ap.add_argument("--force", action="store_true", help="Overwrite backups if needed")
    ap.add_argument("--skip_missing_eval", action="store_true",
                    help="Skip exp_dirs without preds_eval.csv.gz instead of failing")

    args = ap.parse_args()

    runs_tag = Path(args.runs_tag).expanduser().resolve()
    driver = Path(args.driver).expanduser().resolve()
    local_driver = Path(args.local_driver).expanduser().resolve()
    python_bin = args.python or shutil.which("python") or "python"

    if not runs_tag.exists():
        raise FileNotFoundError(runs_tag)
    if not driver.exists():
        raise FileNotFoundError(driver)
    if not local_driver.exists():
        raise FileNotFoundError(local_driver)

    exp_dirs = sorted([p for p in runs_tag.glob(args.glob) if p.is_dir()])
    if not exp_dirs:
        raise RuntimeError(f"No exp dirs under {runs_tag} matched glob={args.glob}")

    totals = {"exp_dirs": 0, "cp_subdirs": 0, "patched": 0, "skipped": 0, "failed": 0}

    if args.jobs <= 1:
        for exp_dir in exp_dirs:
            r = process_one_exp_dir(
                exp_dir=exp_dir,
                driver=driver,
                local_driver=local_driver,
                python_bin=python_bin,
                backup_metrics=args.backup,
                force=args.force,
                write=args.write,
                mode=args.mode,
                skip_missing_eval=args.skip_missing_eval,
            )
            for k in totals:
                totals[k] += r.get(k, 0)
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futs = [
                ex.submit(
                    process_one_exp_dir,
                    exp_dir,
                    driver,
                    local_driver,
                    python_bin,
                    args.backup,
                    args.force,
                    args.write,
                    args.mode,
                    args.skip_missing_eval,
                )
                for exp_dir in exp_dirs
            ]
            for fut in as_completed(futs):
                r = fut.result()
                for k in totals:
                    totals[k] += r.get(k, 0)

    print("\n=== Summary ===")
    for k, v in totals.items():
        print(f"{k}: {v}")
    if not args.write:
        print("[DRYRUN] Re-run with --write to execute.")


if __name__ == "__main__":
    main()
