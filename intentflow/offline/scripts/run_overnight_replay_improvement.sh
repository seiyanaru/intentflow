#!/usr/bin/env bash
# Overnight replay OTTA improvement sweep.
#
# Goal:
#   Improve accuracy without weakening the current safety story. The sweep tests
#   three logically distinct directions:
#   1) sim_score_tolerance grid: can a stricter replay gate remove noisy commits?
#   2) operator-profile constraints: can we keep gains while removing risky BN paths?
#   3) best-safe + constraints: can sim_score rank candidates only after the riskiest
#      operator is removed?
#
# This script is designed to be launched once and left unattended.
#
# Usage:
#   PYTHON_BIN=/home/islabshi/anaconda3/envs/intentflow/bin/python \
#     bash scripts/run_overnight_replay_improvement.sh [GPU_ID]
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

GPU_ID="${1:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PLAN_C_DIR="results/c_aug_true_9subj_20260506_004923"
PLAN_B_DIR="results/b_5seed_4subj_20260506_005153"

REPLAY_CFG="configs/tcformer_replay_safe_otta/tcformer_replay_safe_otta_s2_smoke.yaml"
H6_CFG="configs/tcformer_replay_safe_otta/tcformer_replay_safe_otta_h6_weighted.yaml"

NO_SHALLOW='{"enable_otta": true, "allowed_operators": ["prototype_update","logit_bias_update","deep_BN_update","hybrid_BN_update","no_update","abstain"]}'
PROTO_BIAS_ONLY='{"enable_otta": true, "allowed_operators": ["prototype_update","logit_bias_update","no_update","abstain"]}'
DEEP_PROTO_BIAS='{"enable_otta": true, "allowed_operators": ["prototype_update","logit_bias_update","deep_BN_update","no_update","abstain"]}'
BEST_NO_SHALLOW='{"enable_otta": true, "select_best_candidate": true, "allowed_operators": ["prototype_update","logit_bias_update","deep_BN_update","hybrid_BN_update","no_update","abstain"]}'

SUBJECTS_C=(1 2 3 4 5 6 7 8 9)
SUBJECTS_B=(2 4 6 7)
SEEDS_B=(1 2 3)

echo "================================================="
echo "Overnight replay improvement sweep"
echo "GPU: ${GPU_ID}"
echo "Python: ${PYTHON_BIN}"
echo "Plan C: ${PLAN_C_DIR}"
echo "Plan B: ${PLAN_B_DIR}"
echo "Started: $(date)"
echo "================================================="

eval_c() {
    local sid="$1"
    local variant="$2"
    local cfg="$3"
    local overrides="$4"
    local ckpt="${PLAN_C_DIR}/sources/s${sid}"
    local out="${PLAN_C_DIR}/eval/s${sid}/${variant}"
    local log="${PLAN_C_DIR}/eval/s${sid}_${variant}.log"

    if [[ ! -f "${ckpt}/checkpoints/subject_${sid}_model.ckpt" ]]; then
        echo "  [C missing ckpt] s${sid}" >&2
        return 1
    fi
    if [[ -f "${out}/results.txt" ]]; then
        echo "  [C skip] s${sid} ${variant}"
        return 0
    fi
    mkdir -p "${out}"
    echo "  [C eval] s${sid} ${variant}"
    "${PYTHON_BIN}" train_pipeline.py \
        --model tcformer_replay_safe_otta --dataset bcic2a \
        --seed 0 --gpu_id "${GPU_ID}" --interaug \
        --config "${cfg}" --subject_ids "${sid}" \
        --checkpoint_dir "${ckpt}" --results_dir "${out}" \
        --model_kwargs "${overrides}" \
        > "${log}" 2>&1
}

eval_b() {
    local sid="$1"
    local seed="$2"
    local variant="$3"
    local cfg="$4"
    local overrides="$5"
    local ckpt="${PLAN_B_DIR}/sources/s${sid}_seed${seed}"
    local out="${PLAN_B_DIR}/eval/s${sid}_seed${seed}/${variant}"
    local log="${PLAN_B_DIR}/eval/s${sid}_seed${seed}_${variant}.log"

    if [[ ! -f "${ckpt}/checkpoints/subject_${sid}_model.ckpt" ]]; then
        echo "  [B missing ckpt] s${sid} seed${seed}" >&2
        return 1
    fi
    if [[ -f "${out}/results.txt" ]]; then
        echo "  [B skip] s${sid} seed${seed} ${variant}"
        return 0
    fi
    mkdir -p "${out}"
    echo "  [B eval] s${sid} seed${seed} ${variant}"
    "${PYTHON_BIN}" train_pipeline.py \
        --model tcformer_replay_safe_otta --dataset bcic2a \
        --seed "${seed}" --gpu_id "${GPU_ID}" --no_interaug \
        --config "${cfg}" --subject_ids "${sid}" \
        --checkpoint_dir "${ckpt}" --results_dir "${out}" \
        --model_kwargs "${overrides}" \
        > "${log}" 2>&1
}

run_variant_c() {
    local variant="$1"
    local cfg="$2"
    local overrides="$3"
    for sid in "${SUBJECTS_C[@]}"; do
        eval_c "${sid}" "${variant}" "${cfg}" "${overrides}"
    done
}

run_variant_b() {
    local variant="$1"
    local cfg="$2"
    local overrides="$3"
    for sid in "${SUBJECTS_B[@]}"; do
        for seed in "${SEEDS_B[@]}"; do
            eval_b "${sid}" "${seed}" "${variant}" "${cfg}" "${overrides}"
        done
    done
}

run_variant_both() {
    local variant="$1"
    local cfg="$2"
    local overrides="$3"
    echo
    echo "=== Variant: ${variant} ==="
    run_variant_c "${variant}" "${cfg}" "${overrides}"
    run_variant_b "${variant}" "${cfg}" "${overrides}"
}

run_variant_both "replay_uniform_no_shallow" "${REPLAY_CFG}" "${NO_SHALLOW}"
run_variant_both "replay_h6_no_shallow" "${H6_CFG}" "${NO_SHALLOW}"

run_variant_both "replay_uniform_tol_5e5" "${REPLAY_CFG}" '{"enable_otta": true, "sim_score_tolerance": 0.00005}'
run_variant_both "replay_uniform_tol_1e4" "${REPLAY_CFG}" '{"enable_otta": true, "sim_score_tolerance": 0.00010}'
run_variant_both "replay_uniform_tol_2e4" "${REPLAY_CFG}" '{"enable_otta": true, "sim_score_tolerance": 0.00020}'
run_variant_both "replay_h6_tol_5e5" "${H6_CFG}" '{"enable_otta": true, "sim_score_tolerance": 0.00005}'
run_variant_both "replay_h6_tol_1e4" "${H6_CFG}" '{"enable_otta": true, "sim_score_tolerance": 0.00010}'
run_variant_both "replay_h6_tol_2e4" "${H6_CFG}" '{"enable_otta": true, "sim_score_tolerance": 0.00020}'

run_variant_both "replay_proto_bias_only" "${REPLAY_CFG}" "${PROTO_BIAS_ONLY}"
run_variant_both "replay_h6_proto_bias_only" "${H6_CFG}" "${PROTO_BIAS_ONLY}"
run_variant_both "replay_deep_proto_bias_only" "${REPLAY_CFG}" "${DEEP_PROTO_BIAS}"
run_variant_both "replay_h6_deep_proto_bias_only" "${H6_CFG}" "${DEEP_PROTO_BIAS}"

run_variant_both "replay_best_uniform_no_shallow" "${REPLAY_CFG}" "${BEST_NO_SHALLOW}"
run_variant_both "replay_best_h6_no_shallow" "${H6_CFG}" "${BEST_NO_SHALLOW}"

echo
echo "=== Aggregate all available replay variants ==="
"${PYTHON_BIN}" scripts/analysis/aggregate_replay_variants.py \
    --plan c --result-dir "${PLAN_C_DIR}" --out-prefix overnight_replay_summary
"${PYTHON_BIN}" scripts/analysis/aggregate_replay_variants.py \
    --plan b --result-dir "${PLAN_B_DIR}" --out-prefix overnight_replay_summary

echo
echo "=== Refresh P1 reports for broad comparison ==="
"${PYTHON_BIN}" scripts/analysis/analyze_replay_gate_p1.py \
    --result-dir "${PLAN_C_DIR}" \
    --replay-variants replay_safe_uniform replay_h6_weighted replay_uniform_tol_1e4 replay_h6_tol_1e4 replay_deep_proto_bias_only replay_h6_deep_proto_bias_only \
    --out-prefix p1_overnight_replay_comparison || true
"${PYTHON_BIN}" scripts/analysis/analyze_replay_gate_p1.py \
    --result-dir "${PLAN_B_DIR}" \
    --replay-variants replay_safe_uniform replay_h6_weighted replay_uniform_tol_1e4 replay_h6_tol_1e4 replay_deep_proto_bias_only replay_h6_deep_proto_bias_only \
    --out-prefix p1_overnight_replay_comparison || true

echo
echo "Finished: $(date)"
echo "Done."
