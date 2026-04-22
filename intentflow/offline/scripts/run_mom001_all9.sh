#!/bin/bash
# =============================================================
# Experiment A: mom=0.01, all 9 subjects, bs=1, bn_stat_clean
# =============================================================
# Goal: Verify that mom=0.01 reduces NTR-S and WSD across all 9 subjects,
#       not just the S2/S7/S9 subset.
#
# Design:
#   - All 9 subjects (BCIC-IV 2a)
#   - bn_momentum: 0.01  (best operating point from momentum sweep)
#   - adapt_mode: bn_stat_clean
#   - bs=1 (online simulation)
#   - Compare vs source_only baseline (same run) and prior bs=48 results
#
# Interpretation:
#   NTR-S ≤ 2/9 at mom=0.01 vs NTR-S=4/9 at mom=0.1 → momentum control works
#   WSD > -3% at mom=0.01                              → magnitude control sufficient
#   S2 harm persists at mom=0.01                       → direction is the causal axis
#
# Usage:
#   ./scripts/run_mom001_all9.sh [SOURCE_CKPT_DIR] [GPU_ID]
# =============================================================

cd "$(dirname "$0")/.." || exit 1

SOURCE_DIR="${1:-results/update_op_v2_20260401_125734/source_model}"
GPU_ID="${2:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/mom001_all9_${TIMESTAMP}"
CONFIG="configs/tcformer_otta/tcformer_otta_bs1.yaml"

echo "================================================="
echo "Experiment A: mom=0.01, all 9 subjects, bs=1"
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

# source_only reference
echo ""
echo "--- source_only (reference) ---"
timeout 1200 python3 train_pipeline.py \
    --model tcformer_otta --dataset bcic2a \
    --gpu_id "${GPU_ID}" --config "${CONFIG}" \
    --results_dir "${BASE_DIR}/source_only" \
    --checkpoint_dir "${SOURCE_DIR}" \
    --model_kwargs '{"adapt_mode": "source_only", "enable_otta": false}' \
    2>&1 | tee "${BASE_DIR}/source_only.log"

echo "=== source_only DONE at $(date) ==="

# bn_stat_clean with mom=0.01
echo ""
echo "================================================="
echo "bn_stat_clean, bn_momentum=0.01, all 9 subjects"
echo "Start: $(date)"
echo "================================================="

timeout 1800 python3 train_pipeline.py \
    --model tcformer_otta --dataset bcic2a \
    --gpu_id "${GPU_ID}" --config "${CONFIG}" \
    --results_dir "${BASE_DIR}/bn_clean_mom001" \
    --checkpoint_dir "${SOURCE_DIR}" \
    --model_kwargs '{"adapt_mode": "bn_stat_clean", "bn_momentum": 0.01}' \
    2>&1 | tee "${BASE_DIR}/bn_clean_mom001.log"

if [ $? -eq 0 ]; then
    echo "=== bn_clean_mom001 COMPLETED at $(date) ==="
else
    echo "=== bn_clean_mom001 FAILED at $(date) ==="
fi

echo ""
echo "================================================="
echo "SUMMARY — Experiment A (mom=0.01, all 9 subjects)"
echo "================================================="

echo ""
echo "--- source_only ---"
grep "Subject [0-9]\+ =>" "${BASE_DIR}/source_only/results.txt" 2>/dev/null \
    || grep -E "S[0-9]|acc" "${BASE_DIR}/source_only.log" | tail -20

echo ""
echo "--- bn_stat_clean, mom=0.01 ---"
grep "Subject [0-9]\+ =>" "${BASE_DIR}/bn_clean_mom001/results.txt" 2>/dev/null \
    || grep -E "S[0-9]|acc" "${BASE_DIR}/bn_clean_mom001.log" | tail -20

echo ""
echo "--- Delta (mom=0.01 - source) ---"
python3 - <<'PYEOF'
import re, glob, os, sys

base = sys.argv[1] if len(sys.argv) > 1 else "."

def parse_results(path):
    results = {}
    try:
        with open(path) as f:
            for line in f:
                m = re.search(r'Subject\s+(\d+)\s+=>\s+([\d.]+)%', line)
                if m:
                    results[int(m.group(1))] = float(m.group(2))
    except FileNotFoundError:
        pass
    return results

src = parse_results(f"{base}/source_only/results.txt")
ada = parse_results(f"{base}/bn_clean_mom001/results.txt")

if not src or not ada:
    print("  [results.txt not found — check .log files]")
    sys.exit(0)

deltas = []
harm_count = 0
for s in sorted(set(src) | set(ada)):
    if s in src and s in ada:
        d = ada[s] - src[s]
        deltas.append(d)
        sign = "+" if d >= 0 else ""
        flag = " *** HARM ***" if d < -1.0 else ""
        print(f"  S{s}: source={src[s]:.2f}%  adapted={ada[s]:.2f}%  delta={sign}{d:.2f}%{flag}")
        if d < 0:
            harm_count += 1

if deltas:
    ntr_s = harm_count / len(deltas)
    wsd = min(deltas)
    mean_delta = sum(deltas) / len(deltas)
    print(f"\n  NTR-S = {harm_count}/{len(deltas)} = {ntr_s:.1%}")
    print(f"  WSD   = {wsd:+.2f}%")
    print(f"  Mean  = {mean_delta:+.2f}%")
PYEOF

echo ""
echo "Reference (prior momentum sweep, bs=1):"
echo "  mom=0.1  S2/S7/S9: source=[67.71, 92.36, 88.19]  adapted=[59.72, 96.53, 89.24]"
echo "  mom=0.01 S2/S7/S9: source=[67.71, 92.36, 88.19]  adapted=[63.19, 93.75, 89.58]"
echo "  mom=0.001 S2/S7/S9: (near-zero drift, near-source accuracy)"
echo ""
echo "All done: $(date)"
