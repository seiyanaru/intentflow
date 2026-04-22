#!/bin/bash
# =============================================================
# Proposal 2: Hybrid deep-side decomposition (all 9 subjects)
# =============================================================
#
# 目的:
#   hybrid の gain が deep mean と deep var のどちらに乗っているかを確定する。
#   S3/S7 の gain source を特定することで、次の momentum/gate 設計の根拠を得る。
#
# 設計:
#   3 conditions × 9 subjects、bn_stat_clean、bs=1
#   全条件で shallow var は凍結（S2 保護を維持）
#
#   shallow_mean_deep_both      : deep = mean + var（現 hybrid、基準）
#   shallow_mean_deep_mean_only : deep = mean のみ（deep var を追加凍結）
#   shallow_mean_deep_var_only  : deep = var のみ（deep mean を追加凍結）
#
# 判断基準:
#   S3/S7 gain が deep_both ≈ deep_mean_only なら → deep var は不要
#   S3/S7 gain が deep_both ≈ deep_var_only  なら → deep var が主成分
#   両方に差があり deep_both が最良なら         → mean + var の相互作用が重要
#
# NOTE: momentum は前実験（run_hybrid_momentum_sweep.sh）の最良値を使うこと。
#       デフォルトは 0.01。ベスト値が出たら --model_kwargs で上書き可能。
#
# Usage:
#   ./scripts/run_hybrid_deep_decompose.sh [SOURCE_CKPT_DIR] [GPU_ID] [MOMENTUM] [OUTPUT_DIR]
# =============================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

SOURCE_DIR="${1:-results/update_op_v2_20260401_125734/source_model}"
GPU_ID="${2:-0}"
MOM="${3:-0.01}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="${4:-results/hybrid_deep_decompose_${TIMESTAMP}}"
CONFIG="configs/tcformer_otta/tcformer_otta_bs1.yaml"

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
echo "Proposal 2: Hybrid deep-side decomposition (all 9)"
echo "Source: ${SOURCE_DIR}"
echo "Output: ${BASE_DIR}"
echo "GPU: ${GPU_ID}"
echo "Momentum: ${MOM}"
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
echo "=== [0/3] source_only ==="
run_logged "source_only" "${BASE_DIR}/source_only.log" \
    timeout 1200 python3 train_pipeline.py \
        --model tcformer_otta --dataset bcic2a \
        --gpu_id "${GPU_ID}" --config "${CONFIG}" \
        --results_dir "${BASE_DIR}/source_only" \
        --checkpoint_dir "${SOURCE_DIR}" \
        --model_kwargs '{"adapt_mode": "source_only", "enable_otta": false}'

# ── 3 hybrid deep-side conditions ─────────────────────────────────────────
TARGETS="shallow_mean_deep_both shallow_mean_deep_mean_only shallow_mean_deep_var_only"
IDX=1
for TARGET in ${TARGETS}; do
    echo ""
    echo "=== [${IDX}/3] ${TARGET}, mom=${MOM} ==="
    run_logged "${TARGET}" "${BASE_DIR}/${TARGET}.log" \
        timeout 1800 python3 train_pipeline.py \
            --model tcformer_otta --dataset bcic2a \
            --gpu_id "${GPU_ID}" --config "${CONFIG}" \
            --results_dir "${BASE_DIR}/${TARGET}" \
            --checkpoint_dir "${SOURCE_DIR}" \
            --model_kwargs "{\"adapt_mode\": \"bn_stat_clean\", \"bn_momentum\": ${MOM}, \"bn_update_target\": \"${TARGET}\"}"
    IDX=$((IDX + 1))
done

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "================================================="
echo "SUMMARY — Deep-side Decomposition"
echo "================================================="

python3 - "${BASE_DIR}" <<'PYEOF'
import re, sys
import numpy as np

base = sys.argv[1]

def parse(path):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                m = re.search(r'Subject\s+(\d+)\s+=>\s+.*?Test Acc:\s+([\d.]+)', line)
                if m:
                    out[int(m.group(1))] = float(m.group(2)) * 100
    except FileNotFoundError:
        pass
    return out

src    = parse(f"{base}/source_only/results.txt")
d_both = parse(f"{base}/shallow_mean_deep_both/results.txt")
d_mno  = parse(f"{base}/shallow_mean_deep_mean_only/results.txt")
d_vno  = parse(f"{base}/shallow_mean_deep_var_only/results.txt")

subs = sorted(src.keys())

print(f"\n{'S':>2} | {'source':>7} | {'deep_both':>10} Δ | {'deep_mean':>10} Δ | {'deep_var':>9} Δ | {'best':>12}")
print("-" * 82)

for s in subs:
    sv = src.get(s)
    bv = d_both.get(s)
    mv = d_mno.get(s)
    vv = d_vno.get(s)
    if sv is None: continue

    db = f"{bv:.2f}({bv-sv:+.2f})" if bv is not None else "---"
    dm = f"{mv:.2f}({mv-sv:+.2f})" if mv is not None else "---"
    dv = f"{vv:.2f}({vv-sv:+.2f})" if vv is not None else "---"
    vals = {k: v for k, v in [('both',bv),('mean',mv),('var',vv)] if v is not None}
    best = max(vals, key=lambda k: vals[k]) if vals else "---"
    print(f"  {s:>1} | {sv:7.2f} | {db:>13} | {dm:>13} | {dv:>12} | {best:>12}")

def summary_stats(d, src):
    if not d: return None, None, None
    deltas = [d[s]-src[s] for s in src if s in d]
    return (
        sum(1 for x in deltas if x < -0.5),
        min(deltas),
        sum(deltas)/len(deltas)
    )

print()
print(f"  {'Condition':<28}  {'NTR@0.5':>8}  {'WSD':>8}  {'mean Δ':>8}")
print("  " + "-" * 58)
for label, d in [("shallow_mean_deep_both", d_both),
                  ("shallow_mean_deep_mean_only", d_mno),
                  ("shallow_mean_deep_var_only", d_vno)]:
    ntr, wsd, md = summary_stats(d, src)
    ntr_s = f"{ntr}/9" if ntr is not None else "---"
    wsd_s = f"{wsd:+.2f}%" if wsd is not None else "---"
    md_s  = f"{md:+.2f}%" if md is not None else "---"
    print(f"  {label:<28}  {ntr_s:>8}  {wsd_s:>8}  {md_s:>8}")

print()
print("  Deep-side gain decomposition (S3 and S7):")
for s in [3, 7]:
    sv = src.get(s, float('nan'))
    bv = d_both.get(s)
    mv = d_mno.get(s)
    vv = d_vno.get(s)
    if all(v is not None for v in (bv, mv, vv)):
        # deep var contribution = both - mean_only
        # deep mean contribution = both - var_only
        dv_contrib = bv - mv
        dm_contrib = bv - vv
        print(f"    S{s}: deep_var_contribution={dv_contrib:+.2f}%  deep_mean_contribution={dm_contrib:+.2f}%")
        if abs(dv_contrib) > abs(dm_contrib):
            print(f"         → deep VAR is the dominant gain source for S{s}")
        else:
            print(f"         → deep MEAN is the dominant gain source for S{s}")

PYEOF

echo ""
echo "Results directory: ${BASE_DIR}"
echo "All done: $(date)"
