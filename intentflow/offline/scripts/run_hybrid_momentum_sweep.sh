#!/bin/bash
# =============================================================
# Proposal 1: Hybrid momentum sweep (S2/S3/S7/S9)
# =============================================================
#
# 目的:
#   mom=0.01 は `both` モード下で最適化された値であり、hybrid に最適とは限らない。
#   shallow var を凍結した今、momentum を上げても S2 を壊すリスクが低い。
#   higher mom → deep/mean update 強化 → S3/S7 gain が伸びる余地を探る。
#
# 設計:
#   bn_update_target: shallow_mean_deep_both (hybrid fixed)
#   bn_momentum: {0.005, 0.01, 0.02, 0.05}
#   subjects: S2/S3/S7/S9 (harm/gain/deep-var-gain/micro-harm の代表)
#   adapt_mode: bn_stat_clean, bs=1
#
# 判断基準:
#   最適化目標: mean Δ 最大
#   制約:       probe set (S2/S3/S7/S9) 上で NTR-S@0.5pp = 0/4 かつ WSD > -0.5%
#   → S2 が再び壊れるか、S3/S7 の gain がどう動くかを4被験者で絞る
#
# 反証条件:
#   mom を上げても S3/S7 の gain が伸びない → deep update の gain は mom 非依存
#   mom を上げると S2 が再び壊れる          → shallow var 凍結だけでは不十分
#   mom を下げると S9 の micro-harm が消える → S9 は momentum に敏感
#
# Usage:
#   ./scripts/run_hybrid_momentum_sweep.sh [SOURCE_CKPT_DIR] [GPU_ID] [OUTPUT_DIR]
# =============================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

SOURCE_DIR="${1:-results/update_op_v2_20260401_125734/source_model}"
GPU_ID="${2:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="${3:-results/hybrid_mom_sweep_${TIMESTAMP}}"
CONFIG="configs/tcformer_otta/tcformer_otta_bs1_s2379.yaml"

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

echo "================================================="
echo "Hybrid Momentum Sweep (S2/S3/S7/S9)"
echo "Source: ${SOURCE_DIR}"
echo "Output: ${BASE_DIR}"
echo "GPU: ${GPU_ID}"
echo "Start: $(date)"
echo "================================================="

CKPT_COUNT=$(ls "${SOURCE_DIR}/checkpoints/"*.ckpt 2>/dev/null | wc -l)
if [ "${CKPT_COUNT}" -lt 9 ]; then
    echo "[FATAL] Need 9 checkpoints in ${SOURCE_DIR}/checkpoints/, found ${CKPT_COUNT}"
    exit 1
fi

mkdir -p "${BASE_DIR}"

# ── source_only reference ──────────────────────────────────────────────────
echo ""
echo "=== [0/4] source_only ==="
run_logged "source_only" "${BASE_DIR}/source_only.log" \
    timeout 300 python3 train_pipeline.py \
        --model tcformer_otta --dataset bcic2a \
        --gpu_id "${GPU_ID}" --config "${CONFIG}" \
        --results_dir "${BASE_DIR}/source_only" \
        --checkpoint_dir "${SOURCE_DIR}" \
        --model_kwargs '{"adapt_mode": "source_only", "enable_otta": false}'

# ── 4 momentum conditions ──────────────────────────────────────────────────
for i in 1 2 3 4; do
    case $i in
        1) MOM=0.005 ;;
        2) MOM=0.01  ;;
        3) MOM=0.02  ;;
        4) MOM=0.05  ;;
    esac
    TAG="hybrid_mom${MOM}"

    echo ""
    echo "=== [${i}/4] hybrid, mom=${MOM} ==="
    run_logged "${TAG}" "${BASE_DIR}/${TAG}.log" \
        timeout 300 python3 train_pipeline.py \
            --model tcformer_otta --dataset bcic2a \
            --gpu_id "${GPU_ID}" --config "${CONFIG}" \
            --results_dir "${BASE_DIR}/${TAG}" \
            --checkpoint_dir "${SOURCE_DIR}" \
            --model_kwargs "{\"adapt_mode\": \"bn_stat_clean\", \"bn_momentum\": ${MOM}, \"bn_update_target\": \"shallow_mean_deep_both\"}"
done

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "================================================="
echo "SUMMARY — Hybrid Momentum Sweep"
echo "================================================="

python3 - "${BASE_DIR}" <<'PYEOF'
import re, sys, os
import numpy as np

base = sys.argv[1]

def parse(path, subjs=(2,3,7,9)):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                m = re.search(r'Subject\s+(\d+)\s+=>\s+.*?Test Acc:\s+([\d.]+)', line)
                if m and int(m.group(1)) in subjs:
                    out[int(m.group(1))] = float(m.group(2)) * 100
    except FileNotFoundError:
        pass
    return out

src = parse(f"{base}/source_only/results.txt")
moms = [0.005, 0.01, 0.02, 0.05]
tags = [f"hybrid_mom{m}" for m in moms]

print(f"\n{'mom':>6} | " + " | ".join(f"S{s} acc  Δ" for s in (2,3,7,9)))
print("-" * 75)

best_mom = None
best_mean = -999

for mom, tag in zip(moms, tags):
    row = parse(f"{base}/{tag}/results.txt")
    parts = [f"  {row[s]:.2f}%({row[s]-src[s]:+.2f})" if s in row else "  ---" for s in (2,3,7,9)]
    deltas = [row[s]-src[s] for s in (2,3,7,9) if s in row]
    mean_d = sum(deltas)/len(deltas) if deltas else None
    wsd    = min(deltas) if deltas else None
    ntr    = sum(1 for d in deltas if d < -0.5) if deltas else None
    mean_str = f"  mean={mean_d:+.2f}% WSD={wsd:+.2f}% NTR@0.5={ntr}/4" if mean_d is not None else ""
    print(f"  {mom:>5} | {'  |  '.join(parts)}{mean_str}")
    if mean_d is not None and ntr == 0 and wsd > -0.5 and mean_d > best_mean:
        best_mean = mean_d
        best_mom  = mom

print()
print(f"Reference (hybrid, mom=0.01, all-9 prior run):")
print(f"  S2=+0.00% S3=+0.70% S7=+1.04% S9=-0.34%  mean(4subj)=+0.35%")
print()
if best_mom:
    print(f"Best momentum on probe set (NTR@0.5=0, WSD>-0.5, max mean): mom={best_mom}  mean={best_mean:+.2f}%")
else:
    print(f"No momentum satisfies NTR@0.5=0 and WSD>-0.5 simultaneously.")
PYEOF

echo ""
echo "Results directory: ${BASE_DIR}"
echo "All done: $(date)"
