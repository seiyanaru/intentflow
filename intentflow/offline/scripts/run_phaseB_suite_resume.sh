#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OFFLINE_DIR="${PROJECT_ROOT}/intentflow/offline"

CONDA_SH="${CONDA_SH:-/home/islab-shi/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-intentflow}"

TRAIN_CONFIG="${1:-configs/tcformer_aug_shinv/tcformer_aug_shinv.yaml}"
EVAL_CONFIG="${2:-configs/tcformer_otta/tcformer_otta_bs1.yaml}"
GPU_ID="${3:-0}"
SEED="${4:-0}"
SUITE_TAG="${5:-20260409}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${OFFLINE_DIR}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export MNE_DONTWRITE_HOME="${MNE_DONTWRITE_HOME:-true}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

chmod +x ./scripts/run_phaseB_condition.sh

SUITE_ROOT="results/phaseB_suite_${SUITE_TAG}_seed${SEED}"
mkdir -p "${SUITE_ROOT}"

echo "[PhaseB-resume] train config: ${TRAIN_CONFIG}"
echo "[PhaseB-resume] eval config:  ${EVAL_CONFIG}"
echo "[PhaseB-resume] gpu_id:       ${GPU_ID}"
echo "[PhaseB-resume] seed:         ${SEED}"
echo "[PhaseB-resume] suite root:   ${SUITE_ROOT}"

is_complete() {
  local root="$1"
  [[ -f "${root}/source_only/results.txt" && -f "${root}/hybrid_eval/results.txt" ]]
}

build_train_overrides() {
  local session_shift_aug="$1"
  local gain_std="$2"
  local lambda_shinv="$3"
  local py_session_shift_aug="False"
  if [[ "${session_shift_aug}" == "true" ]]; then
    py_session_shift_aug="True"
  fi
  python - <<PY
import json
print(json.dumps({
    "session_shift_aug": ${py_session_shift_aug},
    "gain_jitter_std": float("${gain_std}"),
    "lambda_aug": 0.5,
    "lambda_shallow_inv": float("${lambda_shinv}")
}))
PY
}

build_hybrid_overrides() {
  python - <<'PY'
import json
print(json.dumps({
    "enable_otta": True,
    "adapt_mode": "bn_stat_clean",
    "bn_momentum": 0.01,
    "bn_update_target": "shallow_mean_deep_both",
    "pmax_threshold": 0.7,
    "sal_threshold": 0.5
}))
PY
}

missing_subject_csv() {
  local source_dir="$1"
  local missing=()
  for sid in 1 2 3 4 5 6 7 8 9; do
    if [[ ! -f "${source_dir}/checkpoints/subject_${sid}_model.ckpt" ]]; then
      missing+=("${sid}")
    fi
  done
  (IFS=,; echo "${missing[*]}")
}

make_subset_config() {
  local base_config="$1"
  local output_config="$2"
  local subjects_csv="$3"
  python - "${base_config}" "${output_config}" "${subjects_csv}" <<'PY'
import sys
import yaml

base_config, output_config, subjects_csv = sys.argv[1:4]
with open(base_config, "r") as f:
    cfg = yaml.safe_load(f)
cfg["subject_ids"] = [int(x) for x in subjects_csv.split(",") if x]
with open(output_config, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

resume_partial_variant() {
  local variant="$1"
  local gain_std="$2"
  local lambda_shinv="$3"
  local result_root="${SUITE_ROOT}/${variant}"
  local source_dir="${result_root}/source_only"
  local hybrid_dir="${result_root}/hybrid_eval"
  local session_shift_aug="false"
  if [[ "${variant}" != "plain" ]]; then
    session_shift_aug="true"
  fi

  local train_overrides
  train_overrides="$(build_train_overrides "${session_shift_aug}" "${gain_std}" "${lambda_shinv}")"
  local hybrid_overrides
  hybrid_overrides="$(build_hybrid_overrides)"

  mkdir -p "${source_dir}" "${hybrid_dir}"

  local missing_csv
  missing_csv="$(missing_subject_csv "${source_dir}")"
  if [[ -n "${missing_csv}" ]]; then
    echo "[PhaseB-resume] missing source_only subjects for ${variant}: ${missing_csv}"
    local subset_cfg
    subset_cfg="$(mktemp /tmp/phaseb_resume_subset_XXXX.yaml)"
    make_subset_config "${TRAIN_CONFIG}" "${subset_cfg}" "${missing_csv}"
    python train_pipeline.py \
      --model tcformer_aug_shinv \
      --dataset bcic2a \
      --seed "${SEED}" \
      --gpu_id "${GPU_ID}" \
      --config "${subset_cfg}" \
      --results_dir "${source_dir}" \
      --model_kwargs "${train_overrides}"
    rm -f "${subset_cfg}"
  else
    echo "[PhaseB-resume] all source_only checkpoints already exist for ${variant}"
  fi

  echo "[PhaseB-resume] rebuilding all-9 source_only summary for ${variant}"
  python train_pipeline.py \
    --model tcformer_aug_shinv \
    --dataset bcic2a \
    --seed "${SEED}" \
    --gpu_id "${GPU_ID}" \
    --config "${TRAIN_CONFIG}" \
    --checkpoint_dir "${source_dir}" \
    --results_dir "${source_dir}" \
    --model_kwargs "${train_overrides}"

  echo "[PhaseB-resume] rebuilding all-9 hybrid_eval for ${variant}"
  python train_pipeline.py \
    --model tcformer_otta \
    --dataset bcic2a \
    --seed "${SEED}" \
    --gpu_id "${GPU_ID}" \
    --config "${EVAL_CONFIG}" \
    --checkpoint_dir "${source_dir}" \
    --results_dir "${hybrid_dir}" \
    --model_kwargs "${hybrid_overrides}"
}

run_one() {
  local variant="$1"
  local gain_std="$2"
  local lambda_shinv="$3"
  local result_root="${SUITE_ROOT}/${variant}"

  if is_complete "${result_root}"; then
    echo "[PhaseB-resume] skip completed: ${variant}"
    return 0
  fi

  echo
  echo "============================================================"
  echo "[PhaseB-resume] Resuming ${variant}"
  echo "  gain_std=${gain_std}"
  echo "  lambda_shinv=${lambda_shinv}"
  echo "  result_root=${result_root}"
  echo "============================================================"
  if [[ -d "${result_root}" ]]; then
    resume_partial_variant "${variant}" "${gain_std}" "${lambda_shinv}"
  else
    ./scripts/run_phaseB_condition.sh \
      "${TRAIN_CONFIG}" \
      "${EVAL_CONFIG}" \
      "${GPU_ID}" \
      "${SEED}" \
      "${variant}" \
      "${gain_std}" \
      "${lambda_shinv}" \
      "${result_root}"
  fi
}

run_one plain 0.0 0.0
run_one aug_only_0025 0.025 0.0
run_one aug_only_005 0.05 0.0
run_one aug_shinv_0025 0.025 0.1
run_one aug_shinv_005 0.05 0.1

echo
echo "[PhaseB-resume] all remaining conditions completed"
echo "[PhaseB-resume] results: ${SUITE_ROOT}"
