#!/bin/bash
# =============================================================
# Experiment B: S2 direction ablation — what drives harmful drift?
# =============================================================
# Goal: Identify whether S2's harmful drift is driven by:
#   (1) mean update vs variance update (bn_update_target: mean_only vs var_only)
#   (2) shallow layers vs deep layers  (bn_update_target: shallow vs deep)
#
# Design:
#   - Subject: S2 only (most consistently harmed subject)
#   - bn_momentum: 0.01 (magnitude-controlled, isolates direction)
#   - adapt_mode: bn_stat_clean (no affine/dropout noise)
#   - bs=1 (online simulation)
#   - bn_update_target: {both, mean_only, var_only, shallow, deep}
#   - 5 conditions × 1 subject = 5 fast test runs
#   - Layer-wise drift norms saved to .npz for post-hoc analysis
#
# Interpretation:
#   var_only ≈ both  → variance update drives harm (distribution shape shift)
#   mean_only ≈ both → mean update drives harm    (distribution shift in location)
#   shallow ≈ both   → early feature layers carry the harmful drift
#   deep ≈ both      → late feature/TCN layers carry the harmful drift
#
# Usage:
#   ./scripts/run_direction_ablation.sh [SOURCE_CKPT_DIR] [GPU_ID]
# =============================================================

cd "$(dirname "$0")/.." || exit 1

SOURCE_DIR="${1:-results/update_op_v2_20260401_125734/source_model}"
GPU_ID="${2:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/direction_ablation_${TIMESTAMP}"
CONFIG="configs/tcformer_otta/tcformer_otta_bs1_s2only.yaml"
MOM=0.01

echo "================================================="
echo "Experiment B: S2 direction ablation (mom=${MOM})"
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

# source_only reference (no BN update)
echo ""
echo "--- source_only (reference, no BN update) ---"
timeout 300 python3 train_pipeline.py \
    --model tcformer_otta --dataset bcic2a \
    --gpu_id "${GPU_ID}" --config "${CONFIG}" \
    --results_dir "${BASE_DIR}/source_only" \
    --checkpoint_dir "${SOURCE_DIR}" \
    --model_kwargs '{"adapt_mode": "source_only", "enable_otta": false}' \
    2>&1 | tee "${BASE_DIR}/source_only.log"
echo "=== source_only DONE at $(date) ==="

# 5 bn_update_target conditions
for TARGET in both mean_only var_only shallow deep; do
    TAG="bn_clean_${TARGET}"
    echo ""
    echo "================================================="
    echo "bn_stat_clean, bn_update_target=${TARGET}, mom=${MOM}"
    echo "Start: $(date)"
    echo "================================================="

    timeout 300 python3 train_pipeline.py \
        --model tcformer_otta --dataset bcic2a \
        --gpu_id "${GPU_ID}" --config "${CONFIG}" \
        --results_dir "${BASE_DIR}/${TAG}" \
        --checkpoint_dir "${SOURCE_DIR}" \
        --model_kwargs "{\"adapt_mode\": \"bn_stat_clean\", \"bn_momentum\": ${MOM}, \"bn_update_target\": \"${TARGET}\"}" \
        2>&1 | tee "${BASE_DIR}/${TAG}.log"

    if [ $? -eq 0 ]; then
        echo "=== ${TAG} COMPLETED at $(date) ==="
    else
        echo "=== ${TAG} FAILED at $(date) ==="
    fi
done

echo ""
echo "================================================="
echo "SUMMARY — Experiment B (S2, direction ablation)"
echo "================================================="

echo ""
echo "--- S2 accuracy by bn_update_target ---"
python3 - <<'PYEOF'
import re, os, sys

base = sys.argv[1] if len(sys.argv) > 1 else "."

def get_s2_acc(path):
    try:
        with open(path) as f:
            for line in f:
                m = re.search(r'Subject\s+2\s+=>\s+([\d.]+)%', line)
                if m:
                    return float(m.group(1))
    except FileNotFoundError:
        pass
    return None

src_acc = get_s2_acc(f"{base}/source_only/results.txt")
print(f"  source_only  : {src_acc:.2f}%" if src_acc else "  source_only  : [not found]")

conditions = [
    ("both",      "bn_clean_both"),
    ("mean_only", "bn_clean_mean_only"),
    ("var_only",  "bn_clean_var_only"),
    ("shallow",   "bn_clean_shallow"),
    ("deep",      "bn_clean_deep"),
]

results = {}
for target, tag in conditions:
    acc = get_s2_acc(f"{base}/{tag}/results.txt")
    results[target] = acc
    if acc is not None and src_acc is not None:
        d = acc - src_acc
        sign = "+" if d >= 0 else ""
        flag = " *** HARM ***" if d < -1.0 else (" (gain)" if d > 0.5 else "")
        print(f"  {target:<12}: {acc:.2f}%  delta={sign}{d:.2f}%{flag}")
    else:
        print(f"  {target:<12}: [not found]")

print()
# Mechanistic interpretation
if all(results[t] is not None for t in ['both', 'mean_only', 'var_only', 'shallow', 'deep']) and src_acc:
    both_d   = results['both']      - src_acc
    mean_d   = results['mean_only'] - src_acc
    var_d    = results['var_only']  - src_acc
    shal_d   = results['shallow']   - src_acc
    deep_d   = results['deep']      - src_acc

    print("  Mechanism inference:")
    if abs(var_d - both_d) < abs(mean_d - both_d):
        print("  -> variance update dominates harm (var_d closer to both_d)")
    else:
        print("  -> mean update dominates harm (mean_d closer to both_d)")

    if abs(deep_d - both_d) < abs(shal_d - both_d):
        print("  -> deep layers dominate harm (deep_d closer to both_d)")
    else:
        print("  -> shallow layers dominate harm (shallow_d closer to both_d)")
PYEOF

echo ""
echo "--- Layer-wise drift norms (if .npz available) ---"
python3 - <<'PYEOF'
import os, glob
import numpy as np

base = sys.argv[1] if len(sys.argv) > 1 else "."

conditions = ["source_only", "bn_clean_both", "bn_clean_mean_only",
              "bn_clean_var_only", "bn_clean_shallow", "bn_clean_deep"]

import sys
base = sys.argv[1] if len(sys.argv) > 1 else "."

print(f"{'Condition':<20}  {'mean_drift':>10}  {'layer_0..5 (shallow)':>22}  {'layer_6..11 (deep)':>20}")
print("-" * 80)

for cond in conditions:
    npz_files = glob.glob(f"{base}/{cond}/**/*.npz", recursive=True)
    if not npz_files:
        print(f"  {cond:<20}  [no .npz]")
        continue
    # pick the subject-2 npz (or first available)
    npz_path = npz_files[0]
    for f in npz_files:
        if "subj2" in f or "subject_2" in f or "_s2" in f.lower():
            npz_path = f
            break
    try:
        data = np.load(npz_path, allow_pickle=True)
        if 'bn_drift_layers' in data:
            drift = data['bn_drift_layers']  # [n_steps, n_layers]
            mean_per_layer = drift.mean(axis=0)
            shallow = mean_per_layer[:6].sum()
            deep    = mean_per_layer[6:].sum()
            total   = mean_per_layer.sum()
            layer_str = " ".join(f"{v:.3f}" for v in mean_per_layer[:6])
            deep_str  = " ".join(f"{v:.3f}" for v in mean_per_layer[6:])
            print(f"  {cond:<20}  {total:>10.4f}  shallow=[{layer_str}]")
            print(f"  {'':20}  {'':10}  deep   =[{deep_str}]")
        else:
            print(f"  {cond:<20}  [bn_drift_layers not in npz: keys={list(data.keys())}]")
    except Exception as e:
        print(f"  {cond:<20}  [error: {e}]")
PYEOF

echo ""
echo "All done: $(date)"
