#!/bin/bash
# Focused multi-seed confirmation for the top dual-momentum candidates.

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

export MNE_DONTWRITE_HOME="${MNE_DONTWRITE_HOME:-true}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-intentflow}"
mkdir -p "${MPLCONFIGDIR}"

SOURCE_DIR="${1:-results/update_op_v2_20260401_125734/source_model}"
GPU_ID="${2:-0}"
SEEDS_CSV="${3:-0,1,2,3,4}"
HOURS="${4:-5}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="${5:-results/hybrid_dual_momentum_confirm_${TIMESTAMP}}"
CONFIG="configs/tcformer_otta/tcformer_otta_bs1.yaml"

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

run_condition() {
    local seed="$1"
    local seed_dir="$2"
    local name="$3"
    local kwargs="$4"

    echo ""
    echo "=== seed=${seed} | ${name} | remaining $(time_left_human) ==="
    run_logged "${name}" "${seed_dir}/${name}.log" \
        timeout 1800 python3 train_pipeline.py \
            --model tcformer_otta --dataset bcic2a \
            --seed "${seed}" \
            --gpu_id "${GPU_ID}" --config "${CONFIG}" \
            --results_dir "${seed_dir}/${name}" \
            --checkpoint_dir "${SOURCE_DIR}" \
            --model_kwargs "${kwargs}"
}

echo "================================================="
echo "Hybrid dual-momentum confirmation"
echo "Source: ${SOURCE_DIR}"
echo "Output: ${BASE_DIR}"
echo "GPU: ${GPU_ID}"
echo "Seeds: ${SEEDS_CSV}"
echo "Budget: ${HOURS}h (guard=${MIN_REMAINING_SEC}s)"
echo "Conditions:"
echo "  - ref_hybrid_shared_mom0.01"
echo "  - ref_mean_only_shared_mom0.0125"
echo "  - dual_dm0.0125_dv0.01   (strict-safe candidate)"
echo "  - dual_dm0.015_dv0.005   (mean-accuracy candidate)"
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

    for CONDITION in \
        "ref_hybrid_shared_mom0.01::{\"adapt_mode\": \"bn_stat_clean\", \"bn_momentum\": 0.01, \"bn_update_target\": \"shallow_mean_deep_both\"}" \
        "ref_mean_only_shared_mom0.0125::{\"adapt_mode\": \"bn_stat_clean\", \"bn_momentum\": 0.0125, \"bn_update_target\": \"shallow_mean_deep_mean_only\"}" \
        "dual_dm0.0125_dv0.01::{\"adapt_mode\": \"bn_stat_clean\", \"bn_momentum\": 0.01, \"bn_update_target\": \"shallow_mean_deep_both\", \"bn_shallow_mean_momentum\": 0.01, \"bn_deep_mean_momentum\": 0.0125, \"bn_deep_var_momentum\": 0.01}" \
        "dual_dm0.015_dv0.005::{\"adapt_mode\": \"bn_stat_clean\", \"bn_momentum\": 0.01, \"bn_update_target\": \"shallow_mean_deep_both\", \"bn_shallow_mean_momentum\": 0.01, \"bn_deep_mean_momentum\": 0.015, \"bn_deep_var_momentum\": 0.005}"
    do
        if ! has_time_left; then
            echo "[TIME] Remaining budget is below ${MIN_REMAINING_SEC}s. Stopping before launching a new run."
            STOP_EARLY=1
            break
        fi

        NAME="${CONDITION%%::*}"
        KWARGS="${CONDITION#*::}"
        run_condition "${SEED}" "${SEED_DIR}" "${NAME}" "${KWARGS}"
    done

    if [ "${STOP_EARLY}" -eq 1 ]; then
        break
    fi
done

echo ""
echo "================================================="
echo "SUMMARY — Hybrid dual-momentum confirmation"
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
subject_keys = [f"Subject_{s}" for s in range(1, 10)]
seed_dirs = sorted(p for p in base.glob("seed*") if p.is_dir())
print(f"Completed seed directories: {[p.name for p in seed_dirs]}")
print()

condition_names = [
    "ref_hybrid_shared_mom0.01",
    "ref_mean_only_shared_mom0.0125",
    "dual_dm0.0125_dv0.01",
    "dual_dm0.015_dv0.005",
]

rows = []
for name in condition_names:
    seed_rows = []
    for seed_dir in seed_dirs:
        path = seed_dir / name / "final_acc_tcformer_otta.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        deltas = [(data[k] - src[k]) * 100 for k in subject_keys]
        seed_rows.append({
            "seed": seed_dir.name,
            "mean_delta": sum(deltas) / len(deltas),
            "worst_wsd": min(deltas),
            "strict_ntr": sum(d < 0 for d in deltas),
            "material_ntr": sum(d < -0.5 for d in deltas),
        })
    if not seed_rows:
        continue
    rows.append({
        "name": name,
        "coverage": len(seed_rows),
        "mean_of_means": statistics.mean(x["mean_delta"] for x in seed_rows),
        "std_of_means": statistics.pstdev(x["mean_delta"] for x in seed_rows) if len(seed_rows) > 1 else 0.0,
        "worst_wsd": min(x["worst_wsd"] for x in seed_rows),
        "safe05": sum(x["material_ntr"] == 0 for x in seed_rows),
        "safe0": sum(x["strict_ntr"] == 0 for x in seed_rows),
    })

print(f"{'condition':<28} {'cov':>5} {'meanΔ':>12} {'worstWSD':>10} {'safe@0.5':>10} {'safe<0':>8}")
print("-" * 92)
for row in sorted(rows, key=lambda r: (-r["mean_of_means"], r["name"])):
    mean_s = f"{row['mean_of_means']:+.2f}±{row['std_of_means']:.2f}"
    safe05_s = f"{row['safe05']}/{row['coverage']}"
    safe0_s = f"{row['safe0']}/{row['coverage']}"
    print(
        f"{row['name']:<28} "
        f"{row['coverage']:>5} "
        f"{mean_s:>12} "
        f"{row['worst_wsd']:>10.2f} "
        f"{safe05_s:>10} "
        f"{safe0_s:>8}"
    )

print()
safe_rows = [r for r in rows if r["safe05"] == r["coverage"]]
if safe_rows:
    best = max(safe_rows, key=lambda r: r["mean_of_means"])
    print(
        "Best material-safe condition: "
        f"{best['name']} | meanΔ={best['mean_of_means']:+.2f}±{best['std_of_means']:.2f} | "
        f"worstWSD={best['worst_wsd']:.2f}"
    )
else:
    print("No condition achieved full material safety across completed seeds.")
PYEOF

echo ""
echo "Done: $(date)"
echo "Results saved to: ${BASE_DIR}"
