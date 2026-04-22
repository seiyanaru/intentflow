#!/bin/bash
# =============================================================
# Option 0: 2×2 interaction ablation — S2 × S7, shallow/deep × mean/var
# =============================================================
# Goal: Directly test whether the culprit is *shallow variance* specifically,
#       and whether S7's gain source is also variance-dependent.
#
# Design:
#   - Subjects: S2 (worst harm) and S7 (best gain) only
#   - bn_momentum: 0.01  (magnitude-controlled)
#   - adapt_mode: bn_stat_clean
#   - bs=1 (online simulation)
#   - 4 interaction conditions:
#       shallow_mean_only, shallow_var_only, deep_mean_only, deep_var_only
#   - Plus source_only and 'both' (full update) as anchors
#
# Interpretation matrix:
#   S2:
#     shallow_var_only ≈ both  → shallow variance = culprit          (confirms)
#     shallow_var_only ≈ 0     → variance not localized to shallow   (refutes)
#     deep_var_only    ≈ both  → culprit is in deep layers           (revises)
#   S7:
#     shallow_var_only ≈ both  → gain from shallow variance update
#     shallow_var_only ≈ 0     → gain from mean or deep layers
#   Cross:
#     S2: var harmful  + S7: var beneficial → subject-dependent rule needed
#     S2: var harmful  + S7: var neutral    → var-freeze safe globally
#
# Usage:
#   ./scripts/run_2x2_ablation.sh [SOURCE_CKPT_DIR] [GPU_ID]
# =============================================================

cd "$(dirname "$0")/.." || exit 1

SOURCE_DIR="${1:-results/update_op_v2_20260401_125734/source_model}"
GPU_ID="${2:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/2x2_ablation_${TIMESTAMP}"
CONFIG="configs/tcformer_otta/tcformer_otta_bs1_s27.yaml"
MOM=0.01

echo "================================================="
echo "Option 0: 2×2 ablation (S2/S7, shallow/deep × mean/var)"
echo "Source: ${SOURCE_DIR}"
echo "Output: ${BASE_DIR}"
echo "GPU: ${GPU_ID}"
echo "================================================="

CKPT_COUNT=$(ls "${SOURCE_DIR}/checkpoints/"*.ckpt 2>/dev/null | wc -l)
if [ "${CKPT_COUNT}" -lt 9 ]; then
    echo "[FATAL] Need 9 checkpoints in ${SOURCE_DIR}/checkpoints/, found ${CKPT_COUNT}"
    exit 1
fi

mkdir -p "${BASE_DIR}"

# ── source_only reference ──────────────────────────────────────────────────
echo ""
echo "--- source_only ---"
timeout 300 python3 train_pipeline.py \
    --model tcformer_otta --dataset bcic2a \
    --gpu_id "${GPU_ID}" --config "${CONFIG}" \
    --results_dir "${BASE_DIR}/source_only" \
    --checkpoint_dir "${SOURCE_DIR}" \
    --model_kwargs '{"adapt_mode": "source_only", "enable_otta": false}' \
    2>&1 | tee "${BASE_DIR}/source_only.log"
echo "=== source_only DONE at $(date) ==="

# ── Full update anchor ─────────────────────────────────────────────────────
echo ""
echo "--- both (full update anchor) ---"
timeout 300 python3 train_pipeline.py \
    --model tcformer_otta --dataset bcic2a \
    --gpu_id "${GPU_ID}" --config "${CONFIG}" \
    --results_dir "${BASE_DIR}/both" \
    --checkpoint_dir "${SOURCE_DIR}" \
    --model_kwargs "{\"adapt_mode\": \"bn_stat_clean\", \"bn_momentum\": ${MOM}, \"bn_update_target\": \"both\"}" \
    2>&1 | tee "${BASE_DIR}/both.log"
echo "=== both DONE at $(date) ==="

# ── 2×2 interaction conditions ─────────────────────────────────────────────
for TARGET in shallow_mean_only shallow_var_only deep_mean_only deep_var_only; do
    echo ""
    echo "================================================="
    echo "bn_update_target=${TARGET}, mom=${MOM}"
    echo "Start: $(date)"
    echo "================================================="

    timeout 300 python3 train_pipeline.py \
        --model tcformer_otta --dataset bcic2a \
        --gpu_id "${GPU_ID}" --config "${CONFIG}" \
        --results_dir "${BASE_DIR}/${TARGET}" \
        --checkpoint_dir "${SOURCE_DIR}" \
        --model_kwargs "{\"adapt_mode\": \"bn_stat_clean\", \"bn_momentum\": ${MOM}, \"bn_update_target\": \"${TARGET}\"}" \
        2>&1 | tee "${BASE_DIR}/${TARGET}.log"

    if [ $? -eq 0 ]; then
        echo "=== ${TARGET} COMPLETED at $(date) ==="
    else
        echo "=== ${TARGET} FAILED at $(date) ==="
    fi
done

echo ""
echo "================================================="
echo "SUMMARY — 2×2 ablation (S2 / S7)"
echo "================================================="

python3 - <<'PYEOF'
import re, os, sys

base = sys.argv[1] if len(sys.argv) > 1 else "."

def get_acc(path, subj):
    try:
        with open(path) as f:
            for line in f:
                m = re.search(rf'Subject\s+{subj}\s+=>\s+.*?Test Acc:\s+([\d.]+)', line)
                if m:
                    return float(m.group(1)) * 100
    except FileNotFoundError:
        pass
    return None

conditions = [
    ("source_only",       "source_only"),
    ("both",              "full update"),
    ("shallow_mean_only", "shallow × mean"),
    ("shallow_var_only",  "shallow × var"),
    ("deep_mean_only",    "deep    × mean"),
    ("deep_var_only",     "deep    × var"),
]

src_s2 = get_acc(f"{base}/source_only/results.txt", 2)
src_s7 = get_acc(f"{base}/source_only/results.txt", 7)

print(f"\n{'Condition':<20}  {'S2':>7}  {'S2 Δ':>7}  {'S7':>7}  {'S7 Δ':>7}")
print("-" * 58)
for tag, label in conditions:
    s2 = get_acc(f"{base}/{tag}/results.txt", 2)
    s7 = get_acc(f"{base}/{tag}/results.txt", 7)
    s2_str  = f"{s2:.2f}%" if s2 is not None else "  ---"
    s7_str  = f"{s7:.2f}%" if s7 is not None else "  ---"
    d2_str  = f"{s2 - src_s2:+.2f}%" if (s2 and src_s2 and tag != 'source_only') else "  ref "
    d7_str  = f"{s7 - src_s7:+.2f}%" if (s7 and src_s7 and tag != 'source_only') else "  ref "
    print(f"  {label:<20} {s2_str:>7}  {d2_str:>7}  {s7_str:>7}  {d7_str:>7}")

print()
print("Interaction interpretation:")

# Read key values
sv_s2 = get_acc(f"{base}/shallow_var_only/results.txt", 2)
sv_s7 = get_acc(f"{base}/shallow_var_only/results.txt", 7)
dv_s2 = get_acc(f"{base}/deep_var_only/results.txt",    2)
dv_s7 = get_acc(f"{base}/deep_var_only/results.txt",    7)
both_s2 = get_acc(f"{base}/both/results.txt", 2)
both_s7 = get_acc(f"{base}/both/results.txt", 7)

if all(v is not None for v in [sv_s2, sv_s7, dv_s2, dv_s7, both_s2, both_s7, src_s2, src_s7]):
    sv_s2_d = sv_s2 - src_s2
    sv_s7_d = sv_s7 - src_s7
    both_s2_d = both_s2 - src_s2
    both_s7_d = both_s7 - src_s7
    dv_s7_d = dv_s7 - src_s7

    # S2 culprit identification
    if abs(sv_s2_d - both_s2_d) < 1.5:
        print(f"  [S2] shallow_var_only ({sv_s2_d:+.2f}%) ≈ both ({both_s2_d:+.2f}%) → shallow variance = S2 culprit  CONFIRMED")
    else:
        print(f"  [S2] shallow_var_only ({sv_s2_d:+.2f}%) ≠ both ({both_s2_d:+.2f}%) → culprit NOT in shallow_var  REFUTED")

    # S7 gain source
    if sv_s7_d > 0.5:
        print(f"  [S7] shallow_var_only ({sv_s7_d:+.2f}%) > 0 → variance update contributes to S7 gain")
        print(f"       → var-freeze (mean_only) may reduce S7 gain  RISK EXISTS")
    else:
        print(f"  [S7] shallow_var_only ({sv_s7_d:+.2f}%) ≈ 0 → S7 gain does NOT require variance update")
        print(f"       → var-freeze safe for S7  RISK ABSENT")

    # Cross interpretation
    print()
    if sv_s2_d < -1.0 and sv_s7_d < 0.5:
        print("  => variance update harmful for S2, not needed for S7")
        print("  => var-freeze (mean_only) is plausibly safe as global design")
    elif sv_s2_d < -1.0 and sv_s7_d > 1.0:
        print("  => variance update: harmful for S2, beneficial for S7")
        print("  => subject-dependent adaptive rule likely needed")
    else:
        print("  => pattern unclear; inspect full table above")
PYEOF

echo ""
echo "All done: $(date)"
