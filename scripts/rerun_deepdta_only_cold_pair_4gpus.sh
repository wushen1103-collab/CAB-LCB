#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-$(which python)}
GPUS=(1 2 4 5)
USE_AMP=${USE_AMP:-1}

MAX_EPOCHS=${MAX_EPOCHS:-100}
PATIENCE=${PATIENCE:-15}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-1e-3}
VAL_FRAC=${VAL_FRAC:-0.1}
NUM_WORKERS=${NUM_WORKERS:-4}

DATE_TAG_POINTS=${DATE_TAG_POINTS:-2025-12-23_deepdta_points}
DATE_TAG_CALCP=${DATE_TAG_CALCP:-2025-12-23_deepdta_calcp_points}

run_one () {
  local gpu=$1
  local ds=$2
  local split=$3
  local seed=$4
  local out_dir=$5

  echo "[GPU${gpu}] ${ds} ${split} seed${seed} -> ${out_dir}"

  if [[ "${USE_AMP}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES=${gpu} ${PY} scripts/train_deepdta_point.py \
      --dataset ${ds} --split ${split} --seed ${seed} \
      --max_epochs ${MAX_EPOCHS} --patience ${PATIENCE} \
      --batch_size ${BATCH_SIZE} --lr ${LR} \
      --val_frac ${VAL_FRAC} --num_workers ${NUM_WORKERS} \
      --device cuda --amp \
      --out_subdir ${out_dir}
  else
    CUDA_VISIBLE_DEVICES=${gpu} ${PY} scripts/train_deepdta_point.py \
      --dataset ${ds} --split ${split} --seed ${seed} \
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
  for seed in 0 1 2 3 4; do
    gpu=${GPUS[$gpu_i]}
    out1="runs/${DATE_TAG_POINTS}/deepdta_point_${ds}_cold_pair_seed${seed}"
    out2="runs/${DATE_TAG_CALCP}/deepdta_point_${ds}_cold_pair__calcp_seed${seed}"

    run_one "${gpu}" "${ds}" "cold_pair" "${seed}" "${out1}" &
    pids+=($!)
    gpu_i=$(( (gpu_i + 1) % ${#GPUS[@]} ))
    if (( ${#pids[@]} >= ${#GPUS[@]} )); then wait "${pids[@]}"; pids=(); fi

    gpu=${GPUS[$gpu_i]}
    run_one "${gpu}" "${ds}" "cold_pair__calcp" "${seed}" "${out2}" &
    pids+=($!)
    gpu_i=$(( (gpu_i + 1) % ${#GPUS[@]} ))
    if (( ${#pids[@]} >= ${#GPUS[@]} )); then wait "${pids[@]}"; pids=(); fi
  done
done

if (( ${#pids[@]} > 0 )); then wait "${pids[@]}"; fi
echo "[DONE] Re-ran DeepDTA for cold_pair and cold_pair__calcp."
