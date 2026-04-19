#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEEPDTA_VERIFY="${1:-legacy/artifacts/verify_deepdta_method_wiring_report.csv}"
OUT_CSV="${2:-results/tables/lcb_at_k_smoke.csv}"

if [[ ! -f "$DEEPDTA_VERIFY" ]]; then
  echo "[ERR] missing audit report CSV: $DEEPDTA_VERIFY" >&2
  echo "[HINT] pass a valid path as arg1, e.g. legacy/artifacts/verify_deepdta_method_wiring_report.csv" >&2
  exit 1
fi

Q_LIST="${Q_LIST:-0.05}"
DELTA_LIST="${DELTA_LIST:-0.05}"
K_LIST="${K_LIST:-10 100 1000}"

mkdir -p results/tables

# 1) build audit_deepdta_for_lcb.csv (streaming, low-mem)
DEEPDTA_VERIFY="$DEEPDTA_VERIFY" python - <<'PY'
import csv
from pathlib import Path

import os

src = Path(os.environ["DEEPDTA_VERIFY"])
dst = Path("audit_deepdta_for_lcb.csv")

method2scheme = {
    "Fixed": "baseline_fixed(alpha=0.1)",
    "NaiveAutoSel": "search_autosel(alpha=0.1)",
    "CAS-LCB": "final_constrained_autosel",
    "CAS-LCB-Bonferroni": "final_constrained_autosel_bonf",
}

with src.open("r", newline="") as fin, dst.open("w", newline="") as fout:
    r = csv.DictReader(fin)
    w = csv.DictWriter(fout, fieldnames=["dataset","split","seed","scheme","report_cp_subdir"])
    w.writeheader()
    n=0
    for row in r:
        m = row.get("method","")
        if m not in method2scheme:
            continue
        w.writerow({
            "dataset": str(row["dataset"]).lower(),
            "split": str(row["split"]).lower(),
            "seed": int(row["seed"]),
            "scheme": method2scheme[m],
            "report_cp_subdir": row.get("picked_component","") or "",
        })
        n += 1
print(f"[OK] wrote {n} rows -> {dst}")
PY

# 2) empty graphdta audit (so compute script won't crash)
python - <<'PY'
import csv
from pathlib import Path
dst = Path("audit_graphdta_for_lcb.csv")
with dst.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset","split","seed","scheme","report_cp_subdir"])
print(f"[OK] wrote empty -> {dst}")
PY

# 3) run LCB@k
python scripts/compute_lcb_at_k.py \
  --audit_deepdta audit_deepdta_for_lcb.csv \
  --audit_graphdta audit_graphdta_for_lcb.csv \
  --out "$OUT_CSV" \
  --q_list $Q_LIST \
  --delta_list $DELTA_LIST \
  --k_list $K_LIST

echo "[DONE] $OUT_CSV"
