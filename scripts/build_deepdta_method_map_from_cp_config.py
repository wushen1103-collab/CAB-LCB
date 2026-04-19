#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import pandas as pd

def load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_run_name(run_dir: Path):
    # Typical: deepdta_point_default_<dataset>_<split>_seed<seed>
    name = run_dir.name
    # dataset/split may contain underscores; do a robust parse
    m = re.search(r"_seed(\d+)$", name)
    seed = int(m.group(1)) if m else None
    body = name[:m.start()] if m else name
    # take last two tokens as dataset, split (best effort)
    toks = body.split("_")
    # find dataset and split near the end
    # ... _<dataset>_<split>
    dataset = toks[-2] if len(toks) >= 2 else None
    split = toks[-1] if len(toks) >= 1 else None
    return dataset, split, seed

def infer_method_from_cpdir(cp_subdir: str) -> str:
    # This is a best-effort label; you can adjust if your naming differs.
    s = cp_subdir.lower()
    if "fixed" in s:
        return "Fixed"
    if "naive" in s or "autosel" in s:
        return "NaiveAutoSel"
    if "cas" in s and "bonf" in s:
        return "CAS-LCB-Bonferroni"
    if "cas" in s or "lcb" in s:
        return "CAS-LCB"
    if "tc_sc" in s or "tcsc" in s:
        return "TC-SC"
    if "wsc" in s:
        return "WSC-B"
    # fallback: unknown; keep cp_subdir as method-ish
    return "UNKNOWN"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--out_csv", default="results/tables/deepdta_method_map_from_cp_config.csv")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    rows = []
    for run_dir in sorted([p for p in runs_root.glob("deepdta_point_*") if p.is_dir()]):
        dataset, split, seed = parse_run_name(run_dir)
        for cp_dir in sorted([p for p in run_dir.glob("cp_*") if p.is_dir()]):
            cfg = cp_dir / "config.json"
            if not cfg.exists():
                continue
            j = load_json(cfg)
            a = j.get("args", {})
            alpha = a.get("alpha", None)
            group_by = a.get("group_by", None)
            # method name from folder by default; can override later
            method_guess = infer_method_from_cpdir(cp_dir.name)
            rows.append({
                "model": "deepdta",
                "dataset": dataset,
                "split": split,
                "seed": seed,
                "run_dir": str(run_dir),
                "cp_subdir": cp_dir.name,
                "method_guess": method_guess,
                "alpha": alpha,
                "group_by": group_by,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} rows={len(df)}")
    # quick health stats
    print(df["method_guess"].value_counts().to_string())

if __name__ == "__main__":
    main()
