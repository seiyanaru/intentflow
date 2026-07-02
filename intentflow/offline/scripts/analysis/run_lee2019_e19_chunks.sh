#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="intentflow/offline/results/research_outputs/260629_lee2019_source_size_ablation_e19_primary_chunks"
mkdir -p "$OUT_ROOT/logs"

export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR=/tmp/matplotlib
export MNE_DATA=/home/islabshi/workspace-local2/mne_data
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_chunk() {
  local label="$1"
  local eval_subjects="$2"
  local out_dir="$OUT_ROOT/$label"
  mkdir -p "$out_dir"
  conda run --no-capture-output -n intentflow python -u \
    intentflow/offline/scripts/analysis/lee2019_source_size_ablation_e19.py \
    --subjects 1-54 \
    --eval-subjects "$eval_subjects" \
    --per-class 8 16 24 36 50 \
    --repeats 4 \
    --source-only-fractions 0.25 \
    --longitudinal-fractions 0.10 \
    --output-dir "$out_dir" \
    > "$OUT_ROOT/logs/${label}.log" 2>&1
}

run_chunk chunk_01_09 1-9 &
run_chunk chunk_10_18 10-18 &
run_chunk chunk_19_27 19-27 &
run_chunk chunk_28_36 28-36 &
run_chunk chunk_37_45 37-45 &
run_chunk chunk_46_54 46-54 &

wait
