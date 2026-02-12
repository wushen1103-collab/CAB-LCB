#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd


Y_CANDIDATES = ["y", "y_true", "label", "target", "affinity", "true", "gt"]
LO_CANDIDATES = ["lower", "lo", "l", "y_lower", "pred_lo", "pi_lower", "interval_lo"]
HI_CANDIDATES = ["upper", "hi", "u", "y_upper", "pred_hi", "pi_upper", "interval_hi"]


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _safe_json_load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON: {path} ({e})")


def _safe_json_dump(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)


def _find_best_artifact(cp_dir: Path, split: str, max_depth: int = 2) -> Optional[Path]:
    """
    Heuristically search for a per-example interval artifact for a given split.
    Supports csv/tsv/npz (parquet optional).
    """
    exts = {".csv", ".tsv", ".npz", ".parquet"}
    base_parts = len(cp_dir.parts)

    candidates: List[Path] = []
    for p in cp_dir.rglob("*"):
        if p.is_dir():
            continue
        if (len(p.parts) - base_parts) > max_depth:
            continue
        if p.suffix.lower() not in exts:
            continue
        name = p.name.lower()
        if split.lower() not in name:
            continue
        # Prefer files that look like intervals/prediction sets
        if any(k in name for k in ["interval", "conformal", "pred", "pi", "set"]):
            candidates.append(p)

    if not candidates:
        return None

    # Rank: contains 'interval'/'conformal' first, then by file size
    def score(p: Path) -> Tuple[int, int]:
        name = p.name.lower()
        kw = int(any(k in name for k in ["interval", "conformal", "pi"]))
        try:
            sz = p.stat().st_size
        except Exception:
            sz = 0
        return (kw, sz)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def _extract_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = {c.lower(): c for c in df.columns}

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    ycol = pick([c.lower() for c in Y_CANDIDATES])
    lcol = pick([c.lower() for c in LO_CANDIDATES])
    ucol = pick([c.lower() for c in HI_CANDIDATES])

    if ycol is None or lcol is None or ucol is None:
        raise ValueError(f"Could not find (y, lo, hi) columns in df. Columns={list(df.columns)}")

    y = df[ycol].to_numpy(dtype=float, copy=False)
    lo = df[lcol].to_numpy(dtype=float, copy=False)
    hi = df[ucol].to_numpy(dtype=float, copy=False)
    return y, lo, hi


def _extract_from_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(str(path), allow_pickle=True)
    keys = {k.lower(): k for k in z.files}

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c.lower() in keys:
                return keys[c.lower()]
        return None

    yk = pick(Y_CANDIDATES)
    lk = pick(LO_CANDIDATES)
    uk = pick(HI_CANDIDATES)

    if yk is None or lk is None or uk is None:
        raise ValueError(f"Could not find (y, lo, hi) arrays in npz. Keys={z.files}")

    y = np.asarray(z[yk], dtype=float)
    lo = np.asarray(z[lk], dtype=float)
    hi = np.asarray(z[uk], dtype=float)
    return y, lo, hi


def _load_intervals(artifact: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if artifact.suffix.lower() == ".npz":
        return _extract_from_npz(artifact)

    if artifact.suffix.lower() in [".csv", ".tsv"]:
        sep = "\t" if artifact.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(artifact, sep=sep)
        return _extract_from_df(df)

    if artifact.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(artifact)
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet (need pyarrow/fastparquet?): {artifact} ({e})")
        return _extract_from_df(df)

    raise RuntimeError(f"Unsupported artifact type: {artifact}")


def _compute_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[float, float, int]:
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    y2, lo2, hi2 = y[m], lo[m], hi[m]
    if y2.size == 0:
        raise ValueError("No finite rows found to compute metrics.")
    cov = float(np.mean((y2 >= lo2) & (y2 <= hi2)))
    width = float(np.mean(hi2 - lo2))
    return cov, width, int(y2.size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_tag", required=True, help="e.g., runs/2025-12-23_deepdta_points/")
    ap.add_argument("--split", required=True, choices=["eval", "test"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_depth", type=int, default=2)
    args = ap.parse_args()

    runs_tag = Path(args.runs_tag)
    if not runs_tag.exists():
        print(f"[FATAL] runs_tag not found: {runs_tag}", file=sys.stderr)
        return 2

    metric_files = sorted(runs_tag.rglob("conformal_metrics.json"))
    print(f"[INFO] metrics files found: {len(metric_files)}")

    cov_key = f"coverage_{args.split}"
    wid_key = f"avg_width_{args.split}"
    n_key = f"n_{args.split}_used"

    updated = 0
    skipped = 0
    missing_artifact: List[str] = []
    failed: List[str] = []

    for mf in metric_files:
        cp_dir = mf.parent
        try:
            d = _safe_json_load(mf)
            if (cov_key in d and wid_key in d) and (not args.overwrite):
                skipped += 1
                continue

            art = _find_best_artifact(cp_dir, args.split, max_depth=args.max_depth)
            if art is None:
                missing_artifact.append(str(cp_dir))
                continue

            y, lo, hi = _load_intervals(art)
            cov, wid, n = _compute_metrics(y, lo, hi)

            d[cov_key] = cov
            d[wid_key] = wid
            d[n_key] = n

            _atomic_write_text(mf, _safe_json_dump(d) + "\n")
            updated += 1

        except Exception as e:
            failed.append(f"{cp_dir}: {e}")

    print(f"[INFO] updated: {updated}, skipped: {skipped}, missing_artifact: {len(missing_artifact)}, failed: {len(failed)}")

    if missing_artifact:
        print("\n[WARN] No per-example artifact found for these cp_dirs (need test interval/pred files):")
        for s in missing_artifact[:30]:
            print("  " + s)
        if len(missing_artifact) > 30:
            print(f"  ... (+{len(missing_artifact)-30} more)")

    if failed:
        print("\n[WARN] Failures (first 20):")
        for s in failed[:20]:
            print("  " + s)

    # Hard fail if nothing was updated and test keys are still absent everywhere.
    if args.split == "test" and updated == 0:
        print("[FATAL] No test metrics were backfilled. This usually means you never saved any test interval/pred artifacts.", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
