#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


EXP_RE = re.compile(r"deepdta_(?:point|calcp)_(?P<dataset>[^_]+)_(?P<split>.+)_seed(?P<seed>\d+)$")


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def parse_exp_dir_name(name: str) -> Optional[Tuple[str, str, int]]:
    m = EXP_RE.match(name)
    if not m:
        return None
    return m.group("dataset"), m.group("split"), int(m.group("seed"))


def read_metrics(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def get_split_metrics(cm: Dict[str, Any], split: str) -> Tuple[float, float]:
    if split == "eval":
        cov = cm.get("coverage_eval", cm.get("coverage", None))
        w = cm.get("avg_width_eval", cm.get("avg_width", None))
        return safe_float(cov), safe_float(w)
    cov = cm.get("coverage_test", None)
    w = cm.get("avg_width_test", None)
    return safe_float(cov), safe_float(w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_tag", type=str, required=True)
    ap.add_argument("--split", type=str, required=True, choices=["eval", "test"])
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    runs_tag = Path(args.runs_tag)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_files = sorted(runs_tag.rglob("conformal_metrics.json"))
    print(f"[INFO] metrics files found: {len(metric_files)}")

    rows = []
    for mp in metric_files:
        try:
            exp_dir = mp.parent.parent
            parsed = parse_exp_dir_name(exp_dir.name)
            if parsed is None:
                continue
            dataset, dsplit, seed = parsed
            cp_subdir = mp.parent.name
            cm = read_metrics(mp)

            cov, w = get_split_metrics(cm, args.split)
            if pd.isna(cov) or pd.isna(w):
                continue

            rows.append(
                dict(
                    dataset=dataset,
                    split=dsplit,
                    seed=seed,
                    cp_subdir=cp_subdir,
                    alpha=safe_float(cm.get("alpha", None)),
                    coverage=cov,
                    avg_width=w,
                    eval_split=args.split,
                )
            )
        except Exception:
            continue

    cols = ["dataset", "split", "seed", "cp_subdir", "alpha", "coverage", "avg_width", "eval_split"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_path, index=False)
    print(f"[OK] wrote: {out_path}")
    print(f"[INFO] rows collected: {len(df)}")

    if len(df) == 0:
        raise SystemExit(
            f"[FATAL] No rows collected for split='{args.split}'. "
            "This means conformal_metrics.json does not contain the requested split keys."
        )


if __name__ == "__main__":
    main()
