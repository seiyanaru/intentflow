#!/usr/bin/env bash
# =============================================================
# Proto-OTTA vs Policy-SafeCommit comparison on S2 (harm) and S7 (gain).
#
# Hypotheses:
#   H1: tcformer_proto_otta素体 outperforms tcformer_policy_safe_otta on S2
#       (S2 diagnostics 260429 already show proto素体=64.58 vs policy_safe=62.15)
#   H2: removing shallow_var_update from policy_safe_otta operator pool
#       recovers proto_otta素体 level on S2.
#   H3: On S7 (gain subject), the picture might flip — policy_safe_otta with
#       shallow_var allowed could exceed proto素体.
#
# Comparison axis (single factor per row):
#   - source_only            : enable_otta=false
#   - proto_otta_default     : tcformer_proto_otta素体 (alpha=0.3, energy on)
#   - policy_safe_default    : current policy_safe_otta (alpha=0.0, gates on)
#   - policy_safe_no_shallow : policy_safe_otta with shallow_var_update banned
#
# Each variant is run on S2 and S7 against a *single shared S7 source model*
# (newly trained here) and the existing S2 source models.
#
# Reuses existing S2 source checkpoints to preserve the 260429 baseline,
# trains S7 source once and reuses it for all four variants on S7.
#
# Usage:
#   ./scripts/run_proto_vs_policy_safe_s2_s7.sh [GPU_ID]
# =============================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

GPU_ID="${1:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/proto_vs_policy_safe_s2_s7_${TIMESTAMP}"
POLICY_CONFIG="configs/tcformer_policy_safe_otta/tcformer_policy_safe_otta_s2_smoke.yaml"
PROTO_CONFIG="configs/tcformer_proto_otta/tcformer_proto_otta_s2_smoke.yaml"
DATASET="bcic2a"
SEED=0

# Existing S2 source checkpoints (one per wrapper, already on disk).
S2_POLICY_CKPT="results/tcformer_policy_safe_otta_bcic2a_seed-0_aug-False_GPU0_20260429_2252"
S2_PROTO_CKPT="results/tcformer_proto_otta_bcic2a_seed-0_aug-False_GPU0_20260429_2157"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export MNE_DONTWRITE_HOME="${MNE_DONTWRITE_HOME:-true}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$BASE_DIR"

if [[ "${GPU_ID}" != "-1" ]]; then
  python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("[ProtoVsPolicy] ERROR: CUDA unavailable.", file=sys.stderr)
    sys.exit(1)
PY
fi

run_logged() {
    local label="$1"
    local log_path="$2"
    shift 2
    echo
    echo "=== ${label}: $(date) ==="
    if "$@" 2>&1 | tee "${log_path}"; then
        echo "=== ${label} DONE: $(date) ==="
    else
        local cmd_status=${PIPESTATUS[0]:-1}
        echo "=== ${label} FAILED (cmd=${cmd_status}): $(date) ==="
        return "${cmd_status}"
    fi
}

# -----------------------------------------------------------------
# Step 1: train a single S7 source model (used by every S7 variant).
# Trained with policy_safe_otta wrapper, enable_otta=false. The
# resulting checkpoint is plain TCFormerModule weights so both
# policy_safe and proto wrappers load it identically.
# -----------------------------------------------------------------
S7_SRC_DIR="${BASE_DIR}/s7_source_model"
run_logged "train_s7_source" "${BASE_DIR}/train_s7_source.log" \
    python train_pipeline.py \
        --model tcformer_policy_safe_otta \
        --dataset "${DATASET}" \
        --seed "${SEED}" \
        --gpu_id "${GPU_ID}" \
        --no_interaug \
        --config "${POLICY_CONFIG}" \
        --subject_ids 7 \
        --results_dir "${S7_SRC_DIR}" \
        --model_kwargs '{"enable_otta": false}'

# -----------------------------------------------------------------
# Step 2: evaluate four variants on S2 (existing checkpoints) and
# S7 (new shared checkpoint). Each run is test-only.
# -----------------------------------------------------------------
declare -a VARIANTS=(
  "source_only|tcformer_policy_safe_otta|${POLICY_CONFIG}|{\"enable_otta\": false}"
  "proto_otta_default|tcformer_proto_otta|${PROTO_CONFIG}|{\"enable_otta\": true, \"fusion_alpha\": 0.3, \"use_energy_gate\": true, \"proto_momentum\": 0.05}"
  "policy_safe_default|tcformer_policy_safe_otta|${POLICY_CONFIG}|{\"enable_otta\": true}"
  "policy_safe_no_shallow|tcformer_policy_safe_otta|${POLICY_CONFIG}|{\"enable_otta\": true, \"allowed_operators\": [\"prototype_update\", \"logit_bias_update\", \"deep_BN_update\", \"hybrid_BN_update\", \"no_update\", \"abstain\"]}"
)

run_subject() {
    local subject="$1"
    local ckpt_policy="$2"
    local ckpt_proto="$3"
    for entry in "${VARIANTS[@]}"; do
        IFS='|' read -r tag model cfg overrides <<<"${entry}"
        if [[ "${model}" == "tcformer_proto_otta" ]]; then
            ckpt="${ckpt_proto}"
        else
            ckpt="${ckpt_policy}"
        fi
        out_dir="${BASE_DIR}/s${subject}/${tag}"
        run_logged "s${subject}/${tag}" "${BASE_DIR}/s${subject}_${tag}.log" \
            python train_pipeline.py \
                --model "${model}" \
                --dataset "${DATASET}" \
                --seed "${SEED}" \
                --gpu_id "${GPU_ID}" \
                --no_interaug \
                --config "${cfg}" \
                --subject_ids "${subject}" \
                --checkpoint_dir "${ckpt}" \
                --results_dir "${out_dir}" \
                --model_kwargs "${overrides}"
    done
}

# S2: reuse existing checkpoints (per-wrapper).
run_subject 2 "${S2_POLICY_CKPT}" "${S2_PROTO_CKPT}"

# S7: shared new checkpoint (same dir works for both wrappers, weights compatible).
run_subject 7 "${S7_SRC_DIR}" "${S7_SRC_DIR}"

# -----------------------------------------------------------------
# Step 3: aggregate results.
# -----------------------------------------------------------------
python - "${BASE_DIR}" <<'PY'
from __future__ import annotations
import json, re, sys
from pathlib import Path

base = Path(sys.argv[1])
variants = ["source_only", "proto_otta_default", "policy_safe_default", "policy_safe_no_shallow"]
subjects = [2, 7]

def parse_acc(d: Path):
    f = d / "results.txt"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        m = re.search(r"Test Acc:\s+([\d.]+)", line)
        if m:
            return float(m.group(1)) * 100.0
    return None

rows = []
for subj in subjects:
    for v in variants:
        d = base / f"s{subj}" / v
        rows.append({"subject": subj, "variant": v, "acc": parse_acc(d)})

(base / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

md = ["| variant | S2 acc | S7 acc | mean | S7-S2 |", "|---|---:|---:|---:|---:|"]
for v in variants:
    s2 = next((r["acc"] for r in rows if r["variant"] == v and r["subject"] == 2), None)
    s7 = next((r["acc"] for r in rows if r["variant"] == v and r["subject"] == 7), None)
    if s2 is not None and s7 is not None:
        mean = (s2 + s7) / 2.0
        diff = s7 - s2
        md.append(f"| {v} | {s2:.2f} | {s7:.2f} | {mean:.2f} | {diff:+.2f} |")
    else:
        md.append(f"| {v} | {s2 if s2 is not None else '-'} | {s7 if s7 is not None else '-'} | - | - |")
(base / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
print(f"\nSaved to: {base}")
PY

echo
echo "Done: ${BASE_DIR}"
