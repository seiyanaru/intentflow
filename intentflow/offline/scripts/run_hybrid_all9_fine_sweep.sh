#!/bin/bash
# =============================================================
# Next experiment: all-9 fine sweep around the hybrid sweet spot
# =============================================================
#
# Goal:
#   Probe the interval between mom=0.01 and mom=0.02 on all 9 subjects,
#   while comparing the two currently strongest designs:
#     1. shallow_mean_deep_both
#     2. shallow_mean_deep_mean_only
#
# Why this experiment:
#   - hybrid@0.01 is still the best all-9 point seen so far
#   - hybrid@0.02 improves S2/S7/S9 but regresses other subjects
#   - deep_mean_only@0.02 is almost tied with hybrid@0.01 and safer on all-9
#   - the true frontier is likely in the narrow band between 0.01 and 0.02
#
# Design:
#   targets  = {shallow_mean_deep_both, shallow_mean_deep_mean_only}
#   momenta  = {0.01, 0.0125, 0.015, 0.0175, 0.02}
#   subjects = all 9
#   seeds    = configurable, default 0..4 for overnight robustness
#
# Usage:
#   ./scripts/run_hybrid_all9_fine_sweep.sh \
#       [SOURCE_CKPT_DIR] [GPU_ID] [SEEDS_CSV] [HOURS] [OUTPUT_DIR]
#
# Example:
#   ./scripts/run_hybrid_all9_fine_sweep.sh \
#       results/update_op_v2_20260401_125734/source_model 0 0,1,2,3,4 5
# =============================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

SOURCE_DIR="${1:-results/update_op_v2_20260401_125734/source_model}"
GPU_ID="${2:-0}"
SEEDS_CSV="${3:-0,1,2,3,4}"
HOURS="${4:-5}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="${5:-results/hybrid_all9_fine_sweep_${TIMESTAMP}}"
CONFIG="configs/tcformer_otta/tcformer_otta_bs1.yaml"

# Priority order matters because the script may stop at the time budget.
# Run the currently safer target first, then probe the more aggressive variant.
TARGETS=(
    "shallow_mean_deep_mean_only"
    "shallow_mean_deep_both"
)
# Probe the middle of the suspected optimum band before the edges.
MOMS=(
    "0.015"
    "0.0125"
    "0.0175"
    "0.01"
    "0.02"
)

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
echo "Hybrid all-9 fine sweep"
echo "Source: ${SOURCE_DIR}"
echo "Output: ${BASE_DIR}"
echo "GPU: ${GPU_ID}"
echo "Seeds: ${SEEDS_CSV}"
echo "Targets: ${TARGETS[*]}"
echo "Momenta: ${MOMS[*]}"
echo "Budget: ${HOURS}h (guard=${MIN_REMAINING_SEC}s)"
echo "Start: $(date)"
echo "================================================="

CKPT_COUNT=$(ls "${SOURCE_DIR}/checkpoints/"*.ckpt 2>/dev/null | wc -l)
if [ "${CKPT_COUNT}" -lt 9 ]; then
    echo "[FATAL] Need 9 checkpoints in ${SOURCE_DIR}/checkpoints/, found ${CKPT_COUNT}"
    exit 1
fi

mkdir -p "${BASE_DIR}"

# source_only is seed-invariant here because OTTA is disabled and checkpoints are fixed
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

    for TARGET in "${TARGETS[@]}"; do
        for MOM in "${MOMS[@]}"; do
            if ! has_time_left; then
                echo ""
                echo "[TIME] Remaining budget is below ${MIN_REMAINING_SEC}s. Stopping before launching a new run."
                STOP_EARLY=1
                break 2
            fi

            TAG="${TARGET}_mom${MOM}"
            echo ""
            echo "=== seed=${SEED} | ${TAG} | remaining $(time_left_human) ==="
            run_logged "${TAG}" "${SEED_DIR}/${TAG}.log" \
                timeout 1800 python3 train_pipeline.py \
                    --model tcformer_otta --dataset bcic2a \
                    --seed "${SEED}" \
                    --gpu_id "${GPU_ID}" --config "${CONFIG}" \
                    --results_dir "${SEED_DIR}/${TAG}" \
                    --checkpoint_dir "${SOURCE_DIR}" \
                    --model_kwargs "{\"adapt_mode\": \"bn_stat_clean\", \"bn_momentum\": ${MOM}, \"bn_update_target\": \"${TARGET}\"}"
        done
    done

    if [ "${STOP_EARLY}" -eq 1 ]; then
        break
    fi
done

echo ""
echo "================================================="
echo "SUMMARY — Hybrid all-9 fine sweep"
echo "================================================="

python3 - "${BASE_DIR}" <<'PYEOF'
import json
import math
import pathlib
import statistics
import sys

base = pathlib.Path(sys.argv[1])
targets = [
    "shallow_mean_deep_both",
    "shallow_mean_deep_mean_only",
]
moms = ["0.01", "0.0125", "0.015", "0.0175", "0.02"]

src_path = base / "source_only" / "final_acc_tcformer_otta.json"
if not src_path.exists():
    print("source_only result not found; cannot summarize.")
    raise SystemExit(0)

src = json.loads(src_path.read_text())
seed_dirs = sorted(p for p in base.glob("seed*") if p.is_dir())
print(f"Completed seed directories: {[p.name for p in seed_dirs]}")
print()

def cond_name(target, mom):
    return f"{target}_mom{mom}"

rows = []
for target in targets:
    for mom in moms:
        name = cond_name(target, mom)
        seed_stats = []
        for seed_dir in seed_dirs:
            path = seed_dir / name / "final_acc_tcformer_otta.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            deltas = [
                (data[f"Subject_{s}"] - src[f"Subject_{s}"]) * 100
                for s in range(1, 10)
            ]
            seed_stats.append({
                "seed": seed_dir.name,
                "mean_delta": sum(deltas) / len(deltas),
                "wsd": min(deltas),
                "strict_ntr": sum(d < 0 for d in deltas),
                "material_ntr": sum(d < -0.5 for d in deltas),
            })
        if not seed_stats:
            continue
        mean_of_means = statistics.mean(x["mean_delta"] for x in seed_stats)
        std_of_means = statistics.pstdev(x["mean_delta"] for x in seed_stats) if len(seed_stats) > 1 else 0.0
        worst_wsd = min(x["wsd"] for x in seed_stats)
        safe_count = sum(x["material_ntr"] == 0 for x in seed_stats)
        strict_safe_count = sum(x["strict_ntr"] == 0 for x in seed_stats)
        rows.append({
            "target": target,
            "mom": mom,
            "coverage": len(seed_stats),
            "mean_of_means": mean_of_means,
            "std_of_means": std_of_means,
            "worst_wsd": worst_wsd,
            "safe_count": safe_count,
            "strict_safe_count": strict_safe_count,
        })

print(f"{'target':<30} {'mom':>7} {'cov':>5} {'meanΔ':>12} {'worstWSD':>10} {'safe@0.5':>10} {'safe<0':>8}")
print("-" * 92)
for row in sorted(rows, key=lambda r: (r["target"], float(r["mom"]))):
    mean_s = f"{row['mean_of_means']:+.2f}±{row['std_of_means']:.2f}"
    print(
        f"{row['target']:<30} {row['mom']:>7} {row['coverage']:>5} "
        f"{mean_s:>12} {row['worst_wsd']:+.2f}% {row['safe_count']:>8}/{row['coverage']:<1} "
        f"{row['strict_safe_count']:>6}/{row['coverage']:<1}"
    )

safe_rows = [
    row for row in rows
    if row["safe_count"] == row["coverage"] and row["worst_wsd"] > -0.5
]
print()
if safe_rows:
    best = max(safe_rows, key=lambda r: r["mean_of_means"])
    print(
        "Best condition under material-safety constraint: "
        f"{best['target']} @ mom={best['mom']}  "
        f"meanΔ={best['mean_of_means']:+.2f}±{best['std_of_means']:.2f}, "
        f"worstWSD={best['worst_wsd']:+.2f}%"
    )
else:
    print("No condition satisfied the material-safety constraint on all completed seeds.")
PYEOF

echo ""
echo "Results directory: ${BASE_DIR}"
echo "Stop early: ${STOP_EARLY}"
echo "End: $(date)"
