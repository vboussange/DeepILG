#!/usr/bin/env bash
# Run the complete benchmark: folds, inputs, ResistanceGA, learned models,
# evaluation, report, and figures. Rerunning skips completed steps.
#
#   GPUS=0,1,2 GA_WORKERS=4 scripts/run_benchmark.sh [configs/benchmark.toml]
#
# Learned fits share at most three GPUs, one sequential queue per device. The
# ResistanceGA fits, one per fold and region, run concurrently on GA_WORKERS
# CPU workers each. If any step fails, every remaining job is stopped.
set -euo pipefail
cd "$(dirname "$0")/.."

# Run in a process group of our own, so that a failure can stop every job.
[[ $(ps -o pgid= -p $$ | tr -d ' ') == "$$" ]] || exec setsid "$0" "$@"
trap 'status=$?; trap - EXIT; ((status == 0)) || kill 0' EXIT

CONFIG=${1:-configs/benchmark.toml}
IFS=, read -r -a GPU_IDS <<< "${GPUS:-0,1,2}"
GPU_IDS=("${GPU_IDS[@]:0:3}")
GA_WORKERS=${GA_WORKERS:-4}
export JAX_ENABLE_X64=true
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_IDS[*]}")
export CUDA_VISIBLE_DEVICES

uv sync --locked --extra gpu --quiet
deepilg() { uv run --no-sync deepilg "$@"; }
python_lines() { uv run --no-sync python -c "$1" "$CONFIG"; }

mapfile -t SETTINGS < <(python_lines '
import sys
from deepilg.config import load_config
config = load_config(sys.argv[1])
print(config.project.output_dir)
print(config.data.folds)')
OUT=${SETTINGS[0]}
[[ -f ${SETTINGS[1]} ]] || deepilg folds --config "$CONFIG"
mapfile -t FOLD_IDS < <(python_lines '
import sys
from deepilg.config import load_config
from deepilg.splits import load_fold_assignments
print(*load_fold_assignments(load_config(sys.argv[1]).data.folds).fold_indices, sep="\n")')
mapfile -t REGIONS < <(python_lines '
import sys
from deepilg.config import load_config
from deepilg.data import discover_fst_regions
print(*discover_fst_regions(load_config(sys.argv[1]).data.fst_dir), sep="\n")')

deepilg validate --config "$CONFIG"
deepilg download --config "$CONFIG"
deepilg preprocess --config "$CONFIG"

# ResistanceGA: one CPU fit and its test predictions per fold and region.
ga_fit() {
  local fold=$1 region=$2 fit_dir
  fit_dir=$OUT/resistancega/fits/fold_$(printf %02d "$fold")/$region
  # The wrapper writes fit_timing.json only after a successful fit.
  [[ -f $fit_dir/fit_timing.json ]] || deepilg resistancega --config "$CONFIG" --fold "$fold" \
    --region "$region" --mode fit --workers "$GA_WORKERS"
  [[ -f $fit_dir/test_predictions.csv ]] || deepilg resistancega --config "$CONFIG" \
    --fold "$fold" --region "$region" --mode predict --split test
}
ga_pids=()
for fold in "${FOLD_IDS[@]}"; do
  # The export writes the last region's test pairs last.
  [[ -f $OUT/resistancega/inputs/fold_$(printf %02d "$fold")/${REGIONS[-1]}/test_pairs.csv ]] \
    || deepilg resistancega-export --config "$CONFIG" --fold "$fold"
  for region in "${REGIONS[@]}"; do
    (
      export JAX_PLATFORMS=cpu
      ga_fit "$fold" "$region"
    ) &
    ga_pids+=($!)
  done
done

# Learned models: "model pooling fold [region]" fits, dealt round-robin to the GPU queues.
train_fit() {
  local model=$1 pooling=$2 fold=$3 region=${4:-} fit_dir resume=()
  fit_dir=$OUT/$model/$pooling/fold_$(printf %02d "$fold")${region:+/$region}
  local args=(--config "$CONFIG" --model "$model" --pooling "$pooling" --fold "$fold")
  [[ -z $region ]] || args+=(--region "$region")
  if [[ $model == reference ]]; then
    [[ -f $fit_dir/training_state.json ]] || deepilg train "${args[@]}"
    return
  fi
  # Resuming a finished fit returns without further training.
  [[ ! -f $fit_dir/training_state.json ]] || resume=(--resume)
  deepilg train "${args[@]}" "${resume[@]}"
}
evaluate_fold() {
  local model=$1 pooling=$2 fold=$3
  [[ -f $OUT/evaluation/$model/$pooling/fold_$(printf %02d "$fold")/metrics.json ]] \
    || deepilg evaluate --config "$CONFIG" --model "$model" --pooling "$pooling" --fold "$fold"
}
run_queues() {
  local step=$1 queue pid
  shift
  local -a items=("$@") pids=()
  for queue in "${!GPU_IDS[@]}"; do
    (
      export CUDA_VISIBLE_DEVICES=${GPU_IDS[$queue]}
      for ((i = queue; i < ${#items[@]}; i += ${#GPU_IDS[@]})); do
        read -r -a item <<< "${items[$i]}"
        "$step" "${item[@]}"
      done
    ) &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

fits=()
evaluations=()
for fold in "${FOLD_IDS[@]}"; do
  for treatment in "reference complete" "resnet9 complete" "loglinear complete" \
    "unet complete" "resnet9 none" "loglinear none"; do
    evaluations+=("$treatment $fold")
    if [[ $treatment == *" none" ]]; then
      for region in "${REGIONS[@]}"; do fits+=("$treatment $fold $region"); done
    else
      fits+=("$treatment $fold")
    fi
  done
done
run_queues train_fit "${fits[@]}"
run_queues evaluate_fold "${evaluations[@]}"

for pid in "${ga_pids[@]}"; do wait "$pid"; done
for fold in "${FOLD_IDS[@]}"; do evaluate_fold resistancega none "$fold"; done

deepilg figures --config "$CONFIG"
CUDA_VISIBLE_DEVICES=${GPU_IDS[0]} deepilg figures --config "$CONFIG" --kind resistance
