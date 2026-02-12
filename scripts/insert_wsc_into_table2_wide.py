#!/usr/bin/env python3
"""
Insert WSC-B rows into an existing KDD-ready wide Table2 CSV.

This script assumes:
  - base table already contains Fixed / Naive / TC-SC / CAS-LCB rows.
  - we insert WSC-B right after TC-SC within each Backbone block.

Inputs:
  --base_table: wide CSV (Backbone, Method, and 8 dataset-split columns)
  --deep_wsc:   Table2-add CSV for DeepDTA WSC-B (dataset, split, Coverage, Width(pKd), Meet-rate)
  --graph_wsc:  Table2-add CSV for GraphDTA WSC-B
Outputs:
  --out_csv: updated wide CSV
  --out_tex: optional LaTeX table
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd

PM_PAT = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*±\s*([0-9]*\.?[0-9]+)\s*$")

SPLIT_TO_COL_SUFFIX = {
    "random": "random",
    "cold_drug": "cold-drug",
    "cold_target": "cold-target",
    "cold_pair": "cold-pair",
}

def parse_pm(s: str) -> tuple[float, float]:
    m = PM_PAT.match(str(s))
    if not m:
        raise ValueError(f"Cannot parse mean±std from: {s!r}")
    return float(m.group(1)), float(m.group(2))

def fmt_cell(cov_pm: str, wid_pm: str, meet: str) -> str:
    cov_m, cov_s = parse_pm(cov_pm)
    wid_m, wid_s = parse_pm(wid_pm)
    cov = f"{cov_m:.3f}±{cov_s:.3f}"
    wid = f"{wid_m:.2f}±{wid_s:.2f}"
    meet_pct = int(round(float(str(meet).strip().replace("%",""))))
    return f"{cov} | {wid} | {meet_pct}%"

def build_row(backbone: str, method: str, add_df: pd.DataFrame, columns: list[str]) -> dict:
    row = {c:"" for c in columns}
    row["Backbone"] = backbone
    row["Method"] = method
    for _, r in add_df.iterrows():
        dataset = str(r["dataset"]).strip().title()
        split = str(r["split"]).strip()
        col = f"{dataset} {SPLIT_TO_COL_SUFFIX[split]}"
        row[col] = fmt_cell(r["Coverage"], r["Width(pKd)"], r["Meet-rate"])
    return row

def insert_after_method(df: pd.DataFrame, backbone: str, method_after: str, new_row: dict) -> pd.DataFrame:
    sub = df[df["Backbone"] == backbone]
    if sub.empty:
        raise ValueError(f"Backbone not found: {backbone}")
    idxs = sub.index.tolist()
    after_idx = None
    for i in idxs:
        if str(df.loc[i,"Method"]).strip() == method_after:
            after_idx = i
            break
    if after_idx is None:
        # Fallback: insert after Naive AutoSel if TC-SC not found
        for i in idxs:
            if "Naive AutoSel" in str(df.loc[i,"Method"]):
                after_idx = i
                break
        if after_idx is None:
            after_idx = idxs[-1]

    existing = sub[sub["Method"].astype(str).str.strip() == str(new_row["Method"]).strip()]
    if not existing.empty:
        df.loc[existing.index[0]] = pd.Series(new_row)
        return df

    top = df.loc[:after_idx].copy()
    bot = df.loc[after_idx+1:].copy()
    new = pd.DataFrame([new_row], columns=df.columns)
    out = pd.concat([top, new, bot], ignore_index=True)
    return out

def df_to_latex(df: pd.DataFrame, out_path: Path) -> None:
    def esc(s: str) -> str:
        s = str(s)
        for k,v in {"&":r"\&","%":r"\%","_":r"\_","#":r"\#","{":r"\{","}":r"\}"}.items():
            s = s.replace(k,v)
        return s
    def cell(cell: str) -> str:
        cell = str(cell)
        if cell.strip()=="" or cell.lower()=="nan":
            return ""
        cell = cell.replace("±", r"$\pm$")
        cell = cell.replace(" | ", r" \textbar{} ")
        cell = cell.replace("α", r"$\alpha$")
        return esc(cell)
    d = df.copy().applymap(cell)
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\begin{tabular}{ll" + "c"*8 + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(d.columns) + r" \\")
    lines.append(r"\midrule")
    for _, r in d.iterrows():
        lines.append(" & ".join(r.tolist()) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Main test performance at nominal 90\%: coverage $\mid$ width (pKd) $\mid$ meet-rate.}")
    lines.append(r"\label{tab:main_table2}")
    lines.append(r"\end{table*}")
    out_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_table", required=True)
    ap.add_argument("--deep_wsc", required=True)
    ap.add_argument("--graph_wsc", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_tex", default=None)
    args = ap.parse_args()

    base = pd.read_csv(args.base_table)
    deep_add = pd.read_csv(args.deep_wsc)
    graph_add = pd.read_csv(args.graph_wsc)

    deep_row = build_row("DeepDTA", "WSC-B", deep_add, list(base.columns))
    graph_row = build_row("GraphDTA(pKd)", "WSC-B", graph_add, list(base.columns))

    out = base.copy()
    out = insert_after_method(out, "DeepDTA", "TC-SC", deep_row)
    out = insert_after_method(out, "GraphDTA(pKd)", "TC-SC", graph_row)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    if args.out_tex:
        df_to_latex(out, Path(args.out_tex))

    print("[saved]", args.out_csv)
    if args.out_tex:
        print("[saved]", args.out_tex)

if __name__ == "__main__":
    main()
