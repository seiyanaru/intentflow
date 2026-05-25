#!/usr/bin/env bash
# 72h Deferred-Commit Replay OTTA sweep.
#
# Research logic:
#   The sweep is organized around the DC-Replay claim, not around random
#   hyperparameter fishing.
#
#   Core ablations:
#     - correction_static_only: L1 only, no target memory, no commit.
#     - correction_memory_no_commit: L1 + L2, no model-state commit.
#     - random_sparse_commit: tests whether fewer commits alone explain safety.
#     - commit_no_replay_gate: tests whether replay evidence is necessary.
#     - dc_replay_gated: proposed method.
#
#   Sensitivity:
#     - correction strength, memory admission threshold, replay weighting,
#       strict sim_score, min-memory for commit, and operator responsibility
#       (bias-only vs prototype-only).
#
# Usage:
#   PYTHON_BIN=/home/islabshi/anaconda3/envs/intentflow/bin/python \
#     bash scripts/run_dc_replay_72h_sweep.sh [GPU_ID]
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

GPU_ID="${1:-2}"
PYTHON_BIN="${PYTHON_BIN:-/home/islabshi/anaconda3/envs/intentflow/bin/python}"

PLAN_C_DIR="${PLAN_C_DIR:-results/c_aug_true_9subj_20260506_004923}"
PLAN_B_DIR="${PLAN_B_DIR:-results/b_5seed_4subj_20260506_005153}"

DC_CFG="configs/tcformer_deferred_commit_replay_otta/tcformer_deferred_commit_replay_otta_s2_smoke.yaml"
REPLAY_CFG="configs/tcformer_replay_safe_otta/tcformer_replay_safe_otta_s2_smoke.yaml"

SUBJECTS_C=(1 2 3 4 5 6 7 8 9)
SUBJECTS_B=(2 4 6 7)
SEEDS_B=(1 2 3)

DC_VARIANTS=(
  "dc_no_correction_memory_no_commit|{\"dc_enable_correction\": false, \"dc_enable_memory_update\": true, \"dc_commit_mode\": \"none\"}"
  "dc_correction_static_only|{\"dc_correction_mode\": \"static_prior\", \"dc_enable_memory_update\": false, \"dc_commit_mode\": \"none\"}"
  "dc_correction_memory_no_commit|{\"dc_correction_mode\": \"memory_prior\", \"dc_enable_memory_update\": true, \"dc_commit_mode\": \"none\"}"
  "dc_random_sparse_p05|{\"dc_commit_mode\": \"random_sparse\", \"dc_random_commit_prob\": 0.05}"
  "dc_random_sparse_p10|{\"dc_commit_mode\": \"random_sparse\", \"dc_random_commit_prob\": 0.10}"
  "dc_commit_no_replay_gate|{\"dc_commit_mode\": \"no_replay_gate\"}"
  "dc_replay_gated|{\"dc_commit_mode\": \"replay_gated\"}"
  "dc_replay_gated_strict_5e5|{\"dc_commit_mode\": \"replay_gated\", \"sim_score_tolerance\": 0.00005}"
  "dc_replay_gated_h6|{\"dc_commit_mode\": \"replay_gated\", \"replay_weight_mode\": \"pmax_class_inv\"}"
  "dc_replay_bias_only|{\"dc_commit_mode\": \"replay_gated\", \"dc_allowed_commit_operators\": [\"logit_bias_update\"]}"
  "dc_replay_proto_only|{\"dc_commit_mode\": \"replay_gated\", \"dc_allowed_commit_operators\": [\"prototype_update\"]}"
  "dc_memory_gate_lo45|{\"dc_commit_mode\": \"replay_gated\", \"dc_memory_admission_threshold\": 0.45}"
  "dc_memory_gate_hi65|{\"dc_commit_mode\": \"replay_gated\", \"dc_memory_admission_threshold\": 0.65}"
  "dc_corr_alpha_01|{\"dc_commit_mode\": \"replay_gated\", \"dc_prior_correction_strength\": 0.10}"
  "dc_corr_alpha_04|{\"dc_commit_mode\": \"replay_gated\", \"dc_prior_correction_strength\": 0.40}"
  "dc_commit_minmem_8|{\"dc_commit_mode\": \"replay_gated\", \"dc_min_memory_for_commit\": 8}"
  "dc_commit_minmem_24|{\"dc_commit_mode\": \"replay_gated\", \"dc_min_memory_for_commit\": 24}"
)

SUMMARY_VARIANTS=(
  source_only
  policy_safe_no_shallow
  replay_safe_uniform
  dc_no_correction_memory_no_commit
  dc_correction_static_only
  dc_correction_memory_no_commit
  dc_random_sparse_p05
  dc_random_sparse_p10
  dc_commit_no_replay_gate
  dc_replay_gated
  dc_replay_gated_strict_5e5
  dc_replay_gated_h6
  dc_replay_bias_only
  dc_replay_proto_only
  dc_memory_gate_lo45
  dc_memory_gate_hi65
  dc_corr_alpha_01
  dc_corr_alpha_04
  dc_commit_minmem_8
  dc_commit_minmem_24
)

# Pre-registered extended grid for the unattended 72h run.
# This grid is intentionally structured around the DC-Replay logic:
#   alpha: how strong L1 prior correction is
#   mem: how selective L2 memory admission is
#   minmem: how much evidence is required before L3 can commit
#   tol: how strict replay evidence must be
ALPHA_GRID=("0.00|00" "0.10|01" "0.20|02" "0.40|04" "0.80|08")
MEM_GATE_GRID=("0.45|45" "0.55|55" "0.65|65")
MINMEM_GRID=(8 16 24)
SIM_TOL_GRID=("0.0|0" "0.00005|5e5" "0.00010|1e4")

for alpha_item in "${ALPHA_GRID[@]}"; do
    alpha_val="${alpha_item%%|*}"
    alpha_tag="${alpha_item##*|}"
    for mem_item in "${MEM_GATE_GRID[@]}"; do
        mem_val="${mem_item%%|*}"
        mem_tag="${mem_item##*|}"
        for minmem in "${MINMEM_GRID[@]}"; do
            for tol_item in "${SIM_TOL_GRID[@]}"; do
                tol_val="${tol_item%%|*}"
                tol_tag="${tol_item##*|}"
                variant="dc_grid_a${alpha_tag}_m${mem_tag}_n${minmem}_tol${tol_tag}"
                overrides="{\"dc_commit_mode\": \"replay_gated\", \"dc_prior_correction_strength\": ${alpha_val}, \"dc_memory_admission_threshold\": ${mem_val}, \"dc_min_memory_for_commit\": ${minmem}, \"sim_score_tolerance\": ${tol_val}}"
                DC_VARIANTS+=("${variant}|${overrides}")
                SUMMARY_VARIANTS+=("${variant}")
            done
        done
    done
done

echo "================================================="
echo "Deferred-Commit Replay OTTA 72h sweep"
echo "GPU: ${GPU_ID}"
echo "Python: ${PYTHON_BIN}"
echo "Plan C: ${PLAN_C_DIR}"
echo "Plan B: ${PLAN_B_DIR}"
echo "Started: $(date)"
echo "================================================="

run_eval() {
    local plan="$1"
    local sid="$2"
    local seed="$3"
    local variant="$4"
    local model_name="$5"
    local cfg="$6"
    local overrides="$7"
    local ckpt=""
    local out=""
    local log=""
    local interaug_flag=""
    local result_dir=""

    if [[ "${plan}" == "c" ]]; then
        result_dir="${PLAN_C_DIR}"
        ckpt="${PLAN_C_DIR}/sources/s${sid}"
        out="${PLAN_C_DIR}/eval/s${sid}/${variant}"
        log="${PLAN_C_DIR}/eval/s${sid}_${variant}.log"
        interaug_flag="--interaug"
    else
        result_dir="${PLAN_B_DIR}"
        ckpt="${PLAN_B_DIR}/sources/s${sid}_seed${seed}"
        out="${PLAN_B_DIR}/eval/s${sid}_seed${seed}/${variant}"
        log="${PLAN_B_DIR}/eval/s${sid}_seed${seed}_${variant}.log"
        interaug_flag="--no_interaug"
    fi

    if [[ ! -f "${ckpt}/checkpoints/subject_${sid}_model.ckpt" ]]; then
        echo "  [${plan^^} missing ckpt] s${sid}${seed:+ seed${seed}}: ${ckpt}" >&2
        return 1
    fi
    if [[ -f "${out}/results.txt" ]]; then
        echo "  [${plan^^} skip] s${sid}${seed:+ seed${seed}} ${variant}"
        return 0
    fi

    mkdir -p "${out}" "$(dirname "${log}")"
    echo "  [${plan^^} eval] s${sid}${seed:+ seed${seed}} ${variant}"
    "${PYTHON_BIN}" train_pipeline.py \
        --model "${model_name}" --dataset bcic2a \
        --seed "${seed:-0}" --gpu_id "${GPU_ID}" ${interaug_flag} \
        --config "${cfg}" --subject_ids "${sid}" \
        --checkpoint_dir "${ckpt}" --results_dir "${out}" \
        --model_kwargs "${overrides}" \
        > "${log}" 2>&1

    if [[ ! -f "${out}/results.txt" ]]; then
        echo "  [${plan^^} failed-no-results] ${out}" >&2
        return 1
    fi
    touch "${out}/.dc_replay_eval_done"
}

run_variant_c() {
    local variant="$1"
    local model_name="$2"
    local cfg="$3"
    local overrides="$4"
    for sid in "${SUBJECTS_C[@]}"; do
        if ! run_eval "c" "${sid}" "" "${variant}" "${model_name}" "${cfg}" "${overrides}"; then
            echo "  [C continue-after-failure] s${sid} ${variant}" >&2
        fi
    done
}

run_variant_b() {
    local variant="$1"
    local model_name="$2"
    local cfg="$3"
    local overrides="$4"
    for sid in "${SUBJECTS_B[@]}"; do
        for seed in "${SEEDS_B[@]}"; do
            if ! run_eval "b" "${sid}" "${seed}" "${variant}" "${model_name}" "${cfg}" "${overrides}"; then
                echo "  [B continue-after-failure] s${sid} seed${seed} ${variant}" >&2
            fi
        done
    done
}

run_variant_both() {
    local variant="$1"
    local model_name="$2"
    local cfg="$3"
    local overrides="$4"
    echo
    echo "=== Variant: ${variant} ==="
    run_variant_c "${variant}" "${model_name}" "${cfg}" "${overrides}"
    run_variant_b "${variant}" "${model_name}" "${cfg}" "${overrides}"
}

aggregate_reports() {
    echo
    echo "=== Aggregate DC-Replay variants ==="
    "${PYTHON_BIN}" scripts/analysis/aggregate_replay_variants.py \
        --plan c --result-dir "${PLAN_C_DIR}" \
        --variants "${SUMMARY_VARIANTS[@]}" \
        --out-prefix dc_replay_72h_summary || true
    "${PYTHON_BIN}" scripts/analysis/aggregate_replay_variants.py \
        --plan b --result-dir "${PLAN_B_DIR}" \
        --variants "${SUMMARY_VARIANTS[@]}" \
        --out-prefix dc_replay_72h_summary || true
}

trap aggregate_reports EXIT

echo
echo "=== Ensure replay baseline is present where missing ==="
run_variant_c "replay_safe_uniform" "tcformer_replay_safe_otta" "${REPLAY_CFG}" '{"enable_otta": true}' || true
run_variant_b "replay_safe_uniform" "tcformer_replay_safe_otta" "${REPLAY_CFG}" '{"enable_otta": true}' || true

echo
echo "=== DC-Replay core + sensitivity sweep ==="
for item in "${DC_VARIANTS[@]}"; do
    variant="${item%%|*}"
    overrides="${item#*|}"
    run_variant_both "${variant}" "tcformer_deferred_commit_replay_otta" "${DC_CFG}" "${overrides}"
done

echo
echo "Finished: $(date)"
echo "Done."
