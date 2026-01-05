#!/bin/bash
set -euo pipefail

# Default dataset
DATASET="bcic2a"
GPU_ID=0
MODE="single"   # single | sweep

# --- Sweep defaults (override via env if you want) ---
# Target knobs for fixing clip saturation + protecting stable subjects:
# - base_lr: reduces adaptation aggressiveness
# - ttt_loss_scale: reduces inner-loop gradient magnitude (avoid clip_ratio=1.0)
# - ttt_reg_lambda: anchoring strength (prevent drift)
# - entropy_threshold/alpha_max/lr_scale_max: dead-zone + cap for reactive gating
SWEEP_BASE_LR_LIST=(${SWEEP_BASE_LR_LIST:-0.05 0.1 0.2})
SWEEP_TTT_LOSS_SCALE_LIST=(${SWEEP_TTT_LOSS_SCALE_LIST:-0.05 0.1 0.2})
SWEEP_TTT_REG_LAMBDA_LIST=(${SWEEP_TTT_REG_LAMBDA_LIST:-0.02 0.05 0.1})
SWEEP_ENTROPY_THRESHOLD_LIST=(${SWEEP_ENTROPY_THRESHOLD_LIST:-0.85 0.95 1.05})
SWEEP_ALPHA_MAX_LIST=(${SWEEP_ALPHA_MAX_LIST:-0.3 0.5})
SWEEP_LR_SCALE_MAX_LIST=(${SWEEP_LR_SCALE_MAX_LIST:-0.3 0.5})

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --gpu_id) GPU_ID="$2"; shift ;;
        --sweep) MODE="sweep" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set paths
PROJECT_ROOT="/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline"
cd "$PROJECT_ROOT" || exit 1

PYTHON_EXEC="/home/islab-shi/anaconda3/envs/intentflow/bin/python"

# Generate Timestamp and Results Directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_DIR="$PROJECT_ROOT/results/paper_experiments/${DATASET}/${TIMESTAMP}"
DATA_DIR="${EXP_DIR}/data"
FIG_DIR="${EXP_DIR}/figures"

mkdir -p "$DATA_DIR"
mkdir -p "$FIG_DIR"

echo ">>> Starting Paper Experiments for Dataset: $DATASET"
echo ">>> Results will be saved to: $EXP_DIR"

# Flags based on dataset
EXTRA_FLAGS=""
if [ "$DATASET" == "bcic2b" ]; then
    EXTRA_FLAGS="--no_interaug"
fi

# Helper: create JSON overrides safely (avoid bash escaping hell)
make_overrides_json () {
  local base_lr="$1"
  local loss_scale="$2"
  local reg_lambda="$3"
  local entropy_thr="$4"
  local alpha_max="$5"
  local lr_scale_max="$6"

  $PYTHON_EXEC - <<PY
import json
overrides = {
  "ttt_config": {
    "base_lr": float("$base_lr"),
    "ttt_loss_scale": float("$loss_scale"),
    "ttt_reg_lambda": float("$reg_lambda"),
    "ttt_anchor_scale_mode": "none",
  },
  "gating_mode": "entropy",
  "entropy_gating_in_train": False,
  "entropy_threshold": float("$entropy_thr"),
  "alpha_max": float("$alpha_max"),
  "lr_scale_max": float("$lr_scale_max"),
}
print(json.dumps(overrides))
PY
}

# Helper: compute average acc from final_acc json (prints float)
avg_acc_from_json () {
  local json_path="$1"
  $PYTHON_EXEC - <<PY
import json, sys
p = "$json_path"
with open(p, "r") as f:
    d = json.load(f)
print(d.get("Average", "nan"))
PY
}

run_single () {
  echo ">>> Running TCFormer_Hybrid (single config)..."
  $PYTHON_EXEC train_pipeline.py --model tcformer_hybrid --dataset "$DATASET" --gpu_id "$GPU_ID" $EXTRA_FLAGS --results_dir "$DATA_DIR"
}

run_sweep () {
  local RUNS_DIR="${EXP_DIR}/runs"
  mkdir -p "$RUNS_DIR"

  local summary_csv="${EXP_DIR}/sweep_summary.csv"
  echo "tag,avg_acc,base_lr,ttt_loss_scale,ttt_reg_lambda,entropy_threshold,alpha_max,lr_scale_max" > "$summary_csv"

  echo ">>> Running parameter sweep..."
  echo ">>> Summary will be written to: $summary_csv"

  for base_lr in "${SWEEP_BASE_LR_LIST[@]}"; do
    for loss_scale in "${SWEEP_TTT_LOSS_SCALE_LIST[@]}"; do
      for reg_lambda in "${SWEEP_TTT_REG_LAMBDA_LIST[@]}"; do
        for entropy_thr in "${SWEEP_ENTROPY_THRESHOLD_LIST[@]}"; do
          for alpha_max in "${SWEEP_ALPHA_MAX_LIST[@]}"; do
            for lr_scale_max in "${SWEEP_LR_SCALE_MAX_LIST[@]}"; do
              tag="blr${base_lr}_ls${loss_scale}_reg${reg_lambda}_thr${entropy_thr}_am${alpha_max}_lrm${lr_scale_max}"
              run_dir="${RUNS_DIR}/${tag}"
              run_data="${run_dir}/data"
              run_fig="${run_dir}/figures"
              mkdir -p "$run_data" "$run_fig"

              overrides_json="$(make_overrides_json "$base_lr" "$loss_scale" "$reg_lambda" "$entropy_thr" "$alpha_max" "$lr_scale_max")"

              echo ">>> [SWEEP] $tag"
              $PYTHON_EXEC train_pipeline.py \
                --model tcformer_hybrid \
                --dataset "$DATASET" \
                --gpu_id "$GPU_ID" \
                $EXTRA_FLAGS \
                --results_dir "$run_data" \
                --model_kwargs "$overrides_json"

              # Generate figures per run (optional but handy for quick inspection)
              if ls "$run_data"/final_acc_*.json >/dev/null 2>&1; then
                $PYTHON_EXEC scripts/viz_paper_figs.py --data_dir "$run_data" --output_dir "$run_fig" || true
              fi

              avg_acc="$(avg_acc_from_json "${run_data}/final_acc_TCFormer_Hybrid.json")"
              echo "${tag},${avg_acc},${base_lr},${loss_scale},${reg_lambda},${entropy_thr},${alpha_max},${lr_scale_max}" >> "$summary_csv"
            done
          done
        done
      done
    done
  done
  echo ">>> Sweep complete."
}

if [ "$MODE" == "sweep" ]; then
  run_sweep
else
  run_single
fi

# TCFormer_TTT is intentionally skipped for now (focus on Hybrid)

# Base TCFormer is skipped (requested)

# Generate Figures
if ls "$DATA_DIR"/final_acc_*.json >/dev/null 2>&1; then
    echo ">>> Generating Figures..."
    $PYTHON_EXEC scripts/viz_paper_figs.py --data_dir "$DATA_DIR" --output_dir "$FIG_DIR"
else
    echo ">>> No result files found in $DATA_DIR; skipping figure generation."
fi

echo ">>> All Done! Check $FIG_DIR for results."

