#!/usr/bin/env bash
set -euo pipefail

OFFLINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$OFFLINE_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export MNE_DONTWRITE_HOME="${MNE_DONTWRITE_HOME:-true}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

TRAIN_CONFIG="${1:-configs/tcformer_aug_shinv/tcformer_aug_shinv.yaml}"
EVAL_CONFIG="${2:-configs/tcformer_otta/tcformer_otta_bs1.yaml}"
GPU_ID="${3:-0}"
SEED="${4:-0}"
VARIANT="${5:-aug_only_005}"
GAIN_STD="${6:-0.05}"
LAMBDA_SHINV="${7:-0.0}"
RESULT_ROOT="${8:-results/phaseB_condition_${VARIANT}_$(date +%Y%m%d_%H%M%S)}"
SHINV_MODE="${9:-mean_logvar}"
LAMBDA_AUG="${10:-0.5}"

# Enforce GPU-by-default execution policy.
if [[ "${GPU_ID}" != "-1" ]]; then
  python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("[PhaseB] ERROR: gpu_id>=0 requested but CUDA is unavailable.", file=sys.stderr)
    sys.exit(1)
PY
fi

case "$VARIANT" in
  plain|strong_plain)
    SESSION_SHIFT_AUG="false"
    ;;
  aug_only_*|aug_shinv_*|strong_*)
    SESSION_SHIFT_AUG="true"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT" >&2
    exit 1
    ;;
esac

if [[ "$SESSION_SHIFT_AUG" == "true" ]]; then
  PY_SESSION_SHIFT_AUG="True"
else
  PY_SESSION_SHIFT_AUG="False"
fi

TRAIN_DIR="${RESULT_ROOT}/source_only"
HYBRID_DIR="${RESULT_ROOT}/hybrid_eval"

TRAIN_OVERRIDES=$(python - <<PY
import json
print(json.dumps({
    "session_shift_aug": ${PY_SESSION_SHIFT_AUG},
    "gain_jitter_std": float("${GAIN_STD}"),
    "lambda_aug": float("${LAMBDA_AUG}"),
    "lambda_shallow_inv": float("${LAMBDA_SHINV}"),
    "shallow_inv_mode": "${SHINV_MODE}"
}))
PY
)

HYBRID_OVERRIDES=$(python - <<'PY'
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
)

echo "[PhaseB] train config: ${TRAIN_CONFIG}"
echo "[PhaseB] eval config:  ${EVAL_CONFIG}"
echo "[PhaseB] variant:      ${VARIANT}"
echo "[PhaseB] results root: ${RESULT_ROOT}"
echo "[PhaseB] shinv_mode:   ${SHINV_MODE}"
echo "[PhaseB] lambda_aug:   ${LAMBDA_AUG}"

python train_pipeline.py \
  --model tcformer_aug_shinv \
  --dataset bcic2a \
  --seed "${SEED}" \
  --gpu_id "${GPU_ID}" \
  --config "${TRAIN_CONFIG}" \
  --results_dir "${TRAIN_DIR}" \
  --model_kwargs "${TRAIN_OVERRIDES}"

python train_pipeline.py \
  --model tcformer_otta \
  --dataset bcic2a \
  --seed "${SEED}" \
  --gpu_id "${GPU_ID}" \
  --config "${EVAL_CONFIG}" \
  --checkpoint_dir "${TRAIN_DIR}" \
  --results_dir "${HYBRID_DIR}" \
  --model_kwargs "${HYBRID_OVERRIDES}"

echo "[PhaseB] completed: ${RESULT_ROOT}"
