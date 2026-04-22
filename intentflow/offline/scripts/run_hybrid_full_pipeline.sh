#!/bin/bash
# =============================================================
# Hybrid full pipeline: Proposal 1 → Proposal 2 (sequential)
# =============================================================
#
# 3 時間不在中に自動実行するためのラッパー。
# Proposal 1 (momentum sweep on S2/S3/S7/S9) が完了後、
# その最良 mom を Proposal 2 (deep decompose on all 9) に引き渡す。
#
# Usage:
#   ./scripts/run_hybrid_full_pipeline.sh [SOURCE_CKPT_DIR] [GPU_ID]
# =============================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

SOURCE_DIR="${1:-results/update_op_v2_20260401_125734/source_model}"
GPU_ID="${2:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="results/hybrid_pipeline_${TIMESTAMP}.log"
MOM_SWEEP_DIR="results/hybrid_mom_sweep_${TIMESTAMP}"
DECOMP_DIR="results/hybrid_deep_decompose_${TIMESTAMP}"

mkdir -p results
echo "Pipeline start: $(date)" | tee "${LOG_FILE}"
echo "Source: ${SOURCE_DIR}" | tee -a "${LOG_FILE}"
echo "GPU: ${GPU_ID}" | tee -a "${LOG_FILE}"

# ── Proposal 1: momentum sweep ─────────────────────────────────────────────
echo "" | tee -a "${LOG_FILE}"
echo "======================================" | tee -a "${LOG_FILE}"
echo "Phase 1: Hybrid momentum sweep" | tee -a "${LOG_FILE}"
echo "======================================" | tee -a "${LOG_FILE}"

if ./scripts/run_hybrid_momentum_sweep.sh "${SOURCE_DIR}" "${GPU_ID}" "${MOM_SWEEP_DIR}" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "Phase 1 completed successfully." | tee -a "${LOG_FILE}"
else
    status=${PIPESTATUS[0]:-1}
    echo "Phase 1 failed with status ${status}; aborting pipeline." | tee -a "${LOG_FILE}"
    exit "${status}"
fi

# Extract best momentum from results
BEST_MOM=$(python3 - "${MOM_SWEEP_DIR}" <<'PYEOF'
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
best_mom = "0.01"
best_mean = -999

for mom in ["0.005", "0.01", "0.02", "0.05"]:
    tag  = f"hybrid_mom{mom}"
    row  = parse(f"{base}/{tag}/results.txt")
    if not row: continue
    deltas = [row[s]-src[s] for s in (2,3,7,9) if s in row and s in src]
    if not deltas: continue
    ntr  = sum(1 for d in deltas if d < -0.5)
    wsd  = min(deltas)
    mean_d = sum(deltas)/len(deltas)
    if ntr == 0 and wsd > -0.5 and mean_d > best_mean:
        best_mean = mean_d
        best_mom  = mom

print(best_mom)
PYEOF
)

if [ -z "${BEST_MOM}" ]; then
    BEST_MOM="0.01"
    echo "Could not determine best momentum, falling back to 0.01" | tee -a "${LOG_FILE}"
fi

echo "" | tee -a "${LOG_FILE}"
echo "Best momentum from Phase 1 probe set: ${BEST_MOM}" | tee -a "${LOG_FILE}"

# ── Proposal 2: deep decomposition with best momentum ─────────────────────
echo "" | tee -a "${LOG_FILE}"
echo "======================================" | tee -a "${LOG_FILE}"
echo "Phase 2: Hybrid deep decomposition (mom=${BEST_MOM})" | tee -a "${LOG_FILE}"
echo "======================================" | tee -a "${LOG_FILE}"

if ./scripts/run_hybrid_deep_decompose.sh "${SOURCE_DIR}" "${GPU_ID}" "${BEST_MOM}" "${DECOMP_DIR}" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "Phase 2 completed successfully." | tee -a "${LOG_FILE}"
else
    status=${PIPESTATUS[0]:-1}
    echo "Phase 2 failed with status ${status}; aborting pipeline." | tee -a "${LOG_FILE}"
    exit "${status}"
fi

echo "" | tee -a "${LOG_FILE}"
echo "Pipeline complete: $(date)" | tee -a "${LOG_FILE}"
echo "Logs: ${LOG_FILE}"

# Final cross-experiment summary
echo ""
echo "================================================="
echo "CROSS-EXPERIMENT SUMMARY"
echo "================================================="
echo "Momentum sweep results: ${MOM_SWEEP_DIR}"
echo "Deep decompose results: ${DECOMP_DIR}"
echo ""
echo "Key questions answered:"
echo "  1. Is mom=0.01 optimal for hybrid on the probe set? → See Phase 1 summary"
echo "  2. Does deep var contribute to S3/S7 gain? → See Phase 2 summary"
