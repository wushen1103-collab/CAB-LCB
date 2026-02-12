#!/usr/bin/env bash
set -euo pipefail

# Ultra-minimal reviewer/open-source pack:
# - NO runs/ data/ results/ logs/ legacy/ tables/ weights
# - Only core library + configs + minimal entry scripts + README

NAME="${1:-cas-lcb-min}"
HASH="$(git rev-parse --short HEAD 2>/dev/null || echo nohash)"
OUTDIR="dist"
OUT="${OUTDIR}/${NAME}-${HASH}.tar.gz"

mkdir -p "${OUTDIR}"

# White-list only (must be tracked by git)
INCLUDE=(
  README.md
  configs
  src
  scripts/compute_lcb_at_k.py
  scripts/reproduce_lcb_at_k.sh
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
