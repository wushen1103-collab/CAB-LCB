#!/usr/bin/env bash
set -euo pipefail

# Lightweight CAS-LCB release pack:
# - no datasets, runs, logs, mirrored backbones, or local scratch space
# - keep the paper-facing code paths only

NAME="${1:-cas-lcb-min}"
HASH="$(git rev-parse --short HEAD 2>/dev/null || echo nohash)"
OUTDIR="dist"
OUT="${OUTDIR}/${NAME}-${HASH}.tar.gz"

mkdir -p "${OUTDIR}"

# White-list only (must be tracked by git)
INCLUDE=(
  README.md
  .gitignore
  configs
  src
  scripts/build_constrained_autosel_lcb.py
  scripts/build_deepdta_constrained_autosel.py
  scripts/compute_lcb_at_k.py
  scripts/make_calcp_splits.py
  scripts/make_splits.py
  scripts/prepare_dataset.py
  scripts/reproduce_lcb_at_k.sh
  scripts/run_cluster_conformal_from_preds.py
  scripts/run_knn_conformal_from_preds.py
  scripts/run_local_conformal_from_preds.py
  scripts/run_split_conformal_from_preds.py
  scripts/split_test_into_eval_test.py
  scripts/train_deepdta_point.py
  scripts/train_graphdta_point.py
  tools/make_release.sh
)

# Filter only existing paths (avoid pathspec errors)
EXIST=()
for p in "${INCLUDE[@]}"; do
  if git ls-files --error-unmatch "$p" >/dev/null 2>&1; then
    EXIST+=("$p")
  else
    echo "[WARN] not tracked or missing: $p"
  fi
done

if [[ "${#EXIST[@]}" -eq 0 ]]; then
  echo "[ERR] nothing to include (none of the whitelist paths are tracked)."
  exit 1
fi

git archive --format=tar.gz --prefix="${NAME}/" -o "${OUT}" HEAD -- "${EXIST[@]}"

echo "[OK] wrote: ${OUT}"
echo "[INFO] included tracked paths:"
printf '  - %s\n' "${EXIST[@]}"
