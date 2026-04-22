#!/bin/bash
# =============================================================
# Dual-momentum sweep for hybrid BN adaptation
# =============================================================
#
# Goal:
#   Separate deep mean and deep var momentum while keeping
#   shallow var frozen and shallow mean fixed at 0.01.
#
# Design:
#   target                = shallow_mean_deep_both
#   shallow_mean_momentum = 0.01 (fixed)
#   deep_mean_momentum    = {0.0125, 0.015, 0.0175}
#   deep_var_momentum     = {0.003, 0.005, 0.0075, 0.01}
#
# References:
#   - source_only
#   - hybrid shared mom=0.01 (current best)
#   - mean_only shared mom=0.0125 (safe comparator)
#
# Usage:
#   ./scripts/run_hybrid_dual_momentum_sweep.sh \
#       [SOURCE_CKPT_DIR] [GPU_ID] [SEEDS_CSV] [HOURS] [OUTPUT_DIR]
# =============================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

# MNE / Matplotlib try to write under $HOME by default, which may be read-only
# in non-interactive execution environments.
export MNE_DONTWRITE_HOME="${MNE_DONTWRITE_HOME:-true}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-intentflow}"
mkdir -p "${MPLCONFIGDIR}"

SOURCE_DIR="${1:-results/update_op_v2_20260401_125734/source_model}"
GPU_ID="${2:-0}"
SEEDS_CSV="${3:-0}"
HOURS="${4:-3}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="${5:-results/hybrid_dual_momentum_sweep_${TIMESTAMP}}"
CONFIG="configs/tcformer_otta/tcformer_otta_bs1.yaml"

SHALLOW_MEAN_MOM="0.01"
DEEP_MEAN_MOMS=("0.0125" "0.015" "0.0175")
DEEP_VAR_MOMS=("0.003" "0.005" "0.0075" "0.01")

MIN_REMAINING_SEC="${MIN_REMAINING_SEC:-900}"
read -r -a SEEDS <<< "$(echo "${SEEDS_CSV}" | tr ',' ' ')"

DEADLINE_EPOCH=$(python3 - "${HOURS}" <<'PYEOF'
import sys, time
hours = float(sys.argv[1])
print(int(time.time() + hours * 3600))
PYEOF
)

run_logged() {
    local label="$1"
    local log_path="$2"
    shift 2

    if "$@" 2>&1 | tee "${log_path}"; then
        echo "=== ${label} DONE: $(date) ==="
        return 0
    else
        local cmd_status=${PIPESTATUS[0]:-1}
        local tee_status=${PIPESTATUS[1]:-0}
        echo "=== ${label} FAILED (cmd=${cmd_status}, tee=${tee_status}): $(date) ==="
        return "${cmd_status}"
    fi
}

has_time_left() {
    local now
    now=$(date +%s)
    (( now + MIN_REMAINING_SEC <= DEADLINE_EPOCH ))
}

time_left_human() {
    python3 - "${DEADLINE_EPOCH}" <<'PYEOF'
import sys, time
remain = max(0, int(sys.argv[1]) - int(time.time()))
h = remain // 3600
m = (remain % 3600) // 60
s = remain % 60
print(f"{h:02d}:{m:02d}:{s:02d}")
PYEOF
}

echo "================================================="
echo "Hybrid dual-momentum sweep"
echo "Source: ${SOURCE_DIR}"
echo "Output: ${BASE_DIR}"
echo "GPU: ${GPU_ID}"
echo "Seeds: ${SEEDS_CSV}"
echo "shallow_mean_momentum: ${SHALLOW_MEAN_MOM}"
echo "deep_mean_momenta: ${DEEP_MEAN_MOMS[*]}"
echo "deep_var_momenta: ${DEEP_VAR_MOMS[*]}"
echo "Budget: ${HOURS}h (guard=${MIN_REMAINING_SEC}s)"
echo "Start: $(date)"
echo "================================================="

CKPT_COUNT=$(ls "${SOURCE_DIR}/checkpoints/"*.ckpt 2>/dev/null | wc -l)
if [ "${CKPT_COUNT}" -lt 9 ]; then
    echo "[FATAL] Need 9 checkpoints in ${SOURCE_DIR}/checkpoints/, found ${CKPT_COUNT}"
    exit 1
fi

mkdir -p "${BASE_DIR}"

echo ""
echo "=== [reference] source_only (all-9) ==="
run_logged "source_only" "${BASE_DIR}/source_only.log" \
    timeout 1200 python3 train_pipeline.py \
        --model tcformer_otta --dataset bcic2a \
        --seed 0 \
        --gpu_id "${GPU_ID}" --config "${CONFIG}" \
        --results_dir "${BASE_DIR}/source_only" \
        --checkpoint_dir "${SOURCE_DIR}" \
        --model_kwargs '{"adapt_mode": "source_only", "enable_otta": false}'

STOP_EARLY=0

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "#################################################"
    echo "Seed ${SEED}  |  remaining budget: $(time_left_human)"
    echo "#################################################"

    SEED_DIR="${BASE_DIR}/seed${SEED}"
    mkdir -p "${SEED_DIR}"

    if ! has_time_left; then
        echo "[TIME] Remaining budget is below ${MIN_REMAINING_SEC}s before references."
        STOP_EARLY=1
        break
    fi

    run_logged "ref_hybrid_shared_mom0.01" "${SEED_DIR}/ref_hybrid_shared_mom0.01.log" \
        timeout 1800 python3 train_pipeline.py \
            --model tcformer_otta --dataset bcic2a \
            --seed "${SEED}" \
            --gpu_id "${GPU_ID}" --config "${CONFIG}" \
            --results_dir "${SEED_DIR}/ref_hybrid_shared_mom0.01" \
            --checkpoint_dir "${SOURCE_DIR}" \
            --model_kwargs '{"adapt_mode": "bn_stat_clean", "bn_momentum": 0.01, "bn_update_target": "shallow_mean_deep_both"}'

    if ! has_time_left; then
        echo "[TIME] Remaining budget is below ${MIN_REMAINING_SEC}s before mean-only reference."
        STOP_EARLY=1
        break
    fi

    run_logged "ref_mean_only_shared_mom0.0125" "${SEED_DIR}/ref_mean_only_shared_mom0.0125.log" \
        timeout 1800 python3 train_pipeline.py \
            --model tcformer_otta --dataset bcic2a \
            --seed "${SEED}" \
            --gpu_id "${GPU_ID}" --config "${CONFIG}" \
            --results_dir "${SEED_DIR}/ref_mean_only_shared_mom0.0125" \
            --checkpoint_dir "${SOURCE_DIR}" \
            --model_kwargs '{"adapt_mode": "bn_stat_clean", "bn_momentum": 0.0125, "bn_update_target": "shallow_mean_deep_mean_only"}'

    for DM in "${DEEP_MEAN_MOMS[@]}"; do
        for DV in "${DEEP_VAR_MOMS[@]}"; do
            if ! has_time_left; then
                echo ""
                echo "[TIME] Remaining budget is below ${MIN_REMAINING_SEC}s. Stopping before launching a new run."
                STOP_EARLY=1
                break 2
            fi

            TAG="dual_dm${DM}_dv${DV}"
            echo ""
            echo "=== seed=${SEED} | ${TAG} | remaining $(time_left_human) ==="
            run_logged "${TAG}" "${SEED_DIR}/${TAG}.log" \
                timeout 1800 python3 train_pipeline.py \
                    --model tcformer_otta --dataset bcic2a \
                    --seed "${SEED}" \
                    --gpu_id "${GPU_ID}" --config "${CONFIG}" \
                    --results_dir "${SEED_DIR}/${TAG}" \
                    --checkpoint_dir "${SOURCE_DIR}" \
                    --model_kwargs "{\"adapt_mode\": \"bn_stat_clean\", \"bn_momentum\": ${SHALLOW_MEAN_MOM}, \"bn_update_target\": \"shallow_mean_deep_both\", \"bn_shallow_mean_momentum\": ${SHALLOW_MEAN_MOM}, \"bn_deep_mean_momentum\": ${DM}, \"bn_deep_var_momentum\": ${DV}}"
        done
    done

    if [ "${STOP_EARLY}" -eq 1 ]; then
        break
    fi
done

echo ""
echo "================================================="
echo "SUMMARY — Hybrid dual-momentum sweep"
echo "================================================="

python3 - "${BASE_DIR}" <<'PYEOF'
import json
import pathlib
import statistics
import sys

base = pathlib.Path(sys.argv[1])
src_path = base / "source_only" / "final_acc_tcformer_otta.json"
if not src_path.exists():
    print("source_only result not found; cannot summarize.")
    raise SystemExit(0)

src = json.loads(src_path.read_text())
seed_dirs = sorted(p for p in base.glob("seed*") if p.is_dir())
print(f"Completed seed directories: {[p.name for p in seed_dirs]}")
print()

def summarize(name: str):
    rows = []
    for seed_dir in seed_dirs:
        path = seed_dir / name / "final_acc_tcformer_otta.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        deltas = [
            (data[f"Subject_{s}"] - src[f"Subject_{s}"]) * 100
            for s in range(1, 10)
        ]
        rows.append({
            "seed": seed_dir.name,
            "mean_delta": sum(deltas) / len(deltas),
            "wsd": min(deltas),
            "strict_ntr": sum(d < 0 for d in deltas),
            "material_ntr": sum(d < -0.5 for d in deltas),
        })
    return rows

condition_names = [
    "ref_hybrid_shared_mom0.01",
    "ref_mean_only_shared_mom0.0125",
]
for dm in ("0.0125", "0.015", "0.0175"):
    for dv in ("0.003", "0.005", "0.0075", "0.01"):
        condition_names.append(f"dual_dm{dm}_dv{dv}")

summary_rows = []
for name in condition_names:
    rows = summarize(name)
    if not rows:
        continue
    summary_rows.append({
        "name": name,
        "coverage": len(rows),
        "mean_of_means": statistics.mean(x["mean_delta"] for x in rows),
        "std_of_means": statistics.pstdev(x["mean_delta"] for x in rows) if len(rows) > 1 else 0.0,
        "worst_wsd": min(x["wsd"] for x in rows),
        "safe_count": sum(x["material_ntr"] == 0 for x in rows),
        "strict_safe_count": sum(x["strict_ntr"] == 0 for x in rows),
    })

print(f"{'condition':<28} {'cov':>5} {'meanΔ':>12} {'worstWSD':>10} {'safe@0.5':>10} {'safe<0':>8}")
print("-" * 92)
for row in sorted(summary_rows, key=lambda r: (-r["mean_of_means"], r["name"])):
    mean_s = f"{row['mean_of_means']:+.2f}±{row['std_of_means']:.2f}"
    print(
        f"{row['name']:<28} {row['coverage']:>5} {mean_s:>12} "
        f"{row['worst_wsd']:+.2f}% {row['safe_count']:>8}/{row['coverage']:<1} "
        f"{row['strict_safe_count']:>6}/{row['coverage']:<1}"
    )

safe_rows = [
    row for row in summary_rows
    if row["safe_count"] == row["coverage"] and row["worst_wsd"] > -0.5
]
print()
if safe_rows:
    best = max(safe_rows, key=lambda r: r["mean_of_means"])
    print(
        "Best condition under material-safety constraint: "
        f"{best['name']}  meanΔ={best['mean_of_means']:+.2f}±{best['std_of_means']:.2f}, "
        f"worstWSD={best['worst_wsd']:+.2f}%"
    )
else:
    print("No condition satisfied the material-safety constraint on all completed seeds.")
PYEOF

echo ""
echo "Results directory: ${BASE_DIR}"
echo "Stop early: ${STOP_EARLY}"
echo "End: $(date)"
