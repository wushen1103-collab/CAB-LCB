#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-$(which python)}
DATE_TAG=${DATE_TAG:-$(date +%F)_deepdta_calcp_points}

GPUS=(1 2 4 5)

MAX_EPOCHS=${MAX_EPOCHS:-100}
PATIENCE=${PATIENCE:-15}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-1e-3}
VAL_FRAC=${VAL_FRAC:-0.1}
NUM_WORKERS=${NUM_WORKERS:-4}
USE_AMP=${USE_AMP:-1}

run_one () {
  local gpu=$1
  local ds=$2
  local sp=$3
  local sd=$4

  local calcp_sp="${sp}__calcp"
  local out_dir="runs/${DATE_TAG}/deepdta_point_${ds}_${calcp_sp}_seed${sd}"

  echo "[GPU${gpu}] ${ds} ${calcp_sp} seed${sd} -> ${out_dir}"

  if [[ "${USE_AMP}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES=${gpu} ${PY} scripts/train_deepdta_point.py \
      --dataset ${ds} --split ${calcp_sp} --seed ${sd} \
      --max_epochs ${MAX_EPOCHS} --patience ${PATIENCE} \
      --batch_size ${BATCH_SIZE} --lr ${LR} \
      --val_frac ${VAL_FRAC} --num_workers ${NUM_WORKERS} \
      --device cuda --amp \
      --out_subdir ${out_dir}
  else
    CUDA_VISIBLE_DEVICES=${gpu} ${PY} scripts/train_deepdta_point.py \
      --dataset ${ds} --split ${calcp_sp} --seed ${sd} \
      --max_epochs ${MAX_EPOCHS} --patience ${PATIENCE} \
      --batch_size ${BATCH_SIZE} --lr ${LR} \
      --val_frac ${VAL_FRAC} --num_workers ${NUM_WORKERS} \
      --device cuda \
      --out_subdir ${out_dir}
  fi
}

pids=()
gpu_i=0

for ds in davis kiba; do
  for sp in random cold_drug cold_target cold_pair; do
    for sd in 0 1 2 3 4; do
      gpu=${GPUS[$gpu_i]}
      run_one "${gpu}" "${ds}" "${sp}" "${sd}" &

      pids+=($!)
      gpu_i=$(( (gpu_i + 1) % ${#GPUS[@]} ))

      if (( ${#pids[@]} >= ${#GPUS[@]} )); then
        wait "${pids[@]}"
        pids=()
      fi
    done
  done
done

if (( ${#pids[@]} > 0 )); then
  wait "${pids[@]}"
fi

echo "[DONE] All DeepDTA calcp point runs finished under runs/${DATE_TAG}/"
