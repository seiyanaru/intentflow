#!/usr/bin/env bash
# =============================================================
# A1: policy_safe_no_shallow の 9 subject sweep (1 seed, no_interaug)
#
# Goal:
#   (1) Verify "removing shallow_var_update from policy_safe_otta operator
#       pool improves both S2 and S7" generalises to all 9 subjects.
#   (2) Compare against source_only / proto_otta素体 / policy_safe_default
#       on the same per-subject source checkpoints (no checkpoint confound).
#
# Variants (single-factor changes vs source_only):
#   - source_only           : enable_otta=false
#   - proto_otta_default    : α=0.3, energy gate on
#   - policy_safe_default   : α=0.2 (from s2_smoke config)
#   - policy_safe_no_shallow: α=0.2, allowed_operators excludes shallow_var_update
#
# Source training:
#   - S2: reuse results/tcformer_policy_safe_otta_..._20260429_2252
#   - S7: reuse results/proto_vs_policy_safe_s2_s7_20260505_113330/s7_source_model
#   - S1,S3,S4,S5,S6,S8,S9: trained here, parallel across 2 free GPUs
#
# Usage:
#   ./scripts/run_a1_9subject_no_shallow.sh [GPU_TRAIN_A] [GPU_TRAIN_B]
# =============================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

GPU_A="${1:-0}"
GPU_B="${2:-2}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/a1_9subject_no_shallow_${TIMESTAMP}"
POLICY_CONFIG="configs/tcformer_policy_safe_otta/tcformer_policy_safe_otta_s2_smoke.yaml"
PROTO_CONFIG="configs/tcformer_proto_otta/tcformer_proto_otta_s2_smoke.yaml"
DATASET="bcic2a"
SEED=0

# Existing checkpoints to reuse (per-subject TCFormer source weights).
S2_CKPT="results/tcformer_policy_safe_otta_bcic2a_seed-0_aug-False_GPU0_20260429_2252"
S7_CKPT="results/proto_vs_policy_safe_s2_s7_20260505_113330/s7_source_model"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export MNE_DONTWRITE_HOME="${MNE_DONTWRITE_HOME:-true}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$BASE_DIR/sources"

echo "================================================="
echo "A1 9-subject sweep"
echo "Base dir : ${BASE_DIR}"
echo "Train GPUs: A=${GPU_A}, B=${GPU_B}"
echo "================================================="

# -----------------------------------------------------------------
# Step 1: train missing source models (S1, S3, S4, S5, S6, S8, S9).
# Distribute across two GPUs in parallel; each GPU runs its slice
# sequentially.
# -----------------------------------------------------------------
declare -a SLICE_A=(1 3 4 5)
declare -a SLICE_B=(6 8 9)

train_slice() {
    local gpu="$1"; shift
    local slice=("$@")
    for sid in "${slice[@]}"; do
        local out_dir="${BASE_DIR}/sources/s${sid}"
        local log_path="${BASE_DIR}/sources/s${sid}_train.log"
        echo "  [GPU ${gpu}] train s${sid} -> ${out_dir}"
        if ! python train_pipeline.py \
                --model tcformer_policy_safe_otta \
                --dataset "${DATASET}" \
                --seed "${SEED}" \
                --gpu_id "${gpu}" \
                --no_interaug \
                --config "${POLICY_CONFIG}" \
                --subject_ids "${sid}" \
                --results_dir "${out_dir}" \
                --model_kwargs '{"enable_otta": false}' \
                > "${log_path}" 2>&1; then
            echo "  [GPU ${gpu}] s${sid} FAILED (see ${log_path})"
            return 1
        fi
    done
}

echo
echo "=== Training 7 missing subjects (parallel) ==="
train_slice "${GPU_A}" "${SLICE_A[@]}" &
PID_A=$!
train_slice "${GPU_B}" "${SLICE_B[@]}" &
PID_B=$!
wait "${PID_A}" || { echo "Slice A failed"; exit 1; }
wait "${PID_B}" || { echo "Slice B failed"; exit 1; }
echo "=== Source training DONE ==="

# Resolve checkpoint per subject.
ckpt_for_subject() {
    case "$1" in
        2) echo "${S2_CKPT}";;
        7) echo "${S7_CKPT}";;
        *) echo "${BASE_DIR}/sources/s${1}";;
    esac
}

# -----------------------------------------------------------------
# Step 2: evaluate four variants on all 9 subjects (test-only).
# Use GPU_A for evaluation (single process; eval is fast).
# -----------------------------------------------------------------
declare -a VARIANTS=(
  "source_only|tcformer_policy_safe_otta|${POLICY_CONFIG}|{\"enable_otta\": false}"
  "proto_otta_default|tcformer_proto_otta|${PROTO_CONFIG}|{\"enable_otta\": true, \"fusion_alpha\": 0.3, \"use_energy_gate\": true, \"proto_momentum\": 0.05}"
  "policy_safe_default|tcformer_policy_safe_otta|${POLICY_CONFIG}|{\"enable_otta\": true}"
  "policy_safe_no_shallow|tcformer_policy_safe_otta|${POLICY_CONFIG}|{\"enable_otta\": true, \"allowed_operators\": [\"prototype_update\", \"logit_bias_update\", \"deep_BN_update\", \"hybrid_BN_update\", \"no_update\", \"abstain\"]}"
)

echo
echo "=== Evaluating 4 variants × 9 subjects ==="
for sid in 1 2 3 4 5 6 7 8 9; do
    ckpt=$(ckpt_for_subject "${sid}")
    for entry in "${VARIANTS[@]}"; do
        IFS='|' read -r tag model cfg overrides <<<"${entry}"
        out_dir="${BASE_DIR}/eval/s${sid}/${tag}"
        log_path="${BASE_DIR}/eval/s${sid}_${tag}.log"
        mkdir -p "${BASE_DIR}/eval"
        echo "  [eval] s${sid} / ${tag} (ckpt=${ckpt})"
        if ! python train_pipeline.py \
                --model "${model}" \
                --dataset "${DATASET}" \
                --seed "${SEED}" \
                --gpu_id "${GPU_A}" \
                --no_interaug \
                --config "${cfg}" \
                --subject_ids "${sid}" \
                --checkpoint_dir "${ckpt}" \
                --results_dir "${out_dir}" \
                --model_kwargs "${overrides}" \
                > "${log_path}" 2>&1; then
            echo "  [eval] s${sid} / ${tag} FAILED (see ${log_path})"
            return 1 2>/dev/null || exit 1
        fi
    done
done
echo "=== Eval DONE ==="

# -----------------------------------------------------------------
# Step 3: aggregate (mean, per-subject Δ, worst-subject Δ, NTR-S@0.5pp).
# -----------------------------------------------------------------
python - "${BASE_DIR}" <<'PY'
from __future__ import annotations
import json, re, sys
from pathlib import Path

base = Path(sys.argv[1])
variants = ["source_only", "proto_otta_default", "policy_safe_default", "policy_safe_no_shallow"]
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]

def parse_acc(d: Path):
    f = d / "results.txt"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        m = re.search(r"Test Acc:\s+([\d.]+)", line)
        if m:
            return float(m.group(1)) * 100.0
    return None

table = {v: {} for v in variants}
for v in variants:
    for s in subjects:
        table[v][s] = parse_acc(base / "eval" / f"s{s}" / v)

src = table["source_only"]

rows = []
for v in variants:
    accs = [table[v][s] for s in subjects if table[v][s] is not None]
    deltas = [
        table[v][s] - src[s]
        for s in subjects
        if table[v][s] is not None and src[s] is not None
    ]
    if not accs:
        continue
    n = len(accs)
    mean_acc = sum(accs) / n
    mean_delta = sum(deltas) / len(deltas) if deltas else None
    worst_delta = min(deltas) if deltas else None
    ntr_s = sum(1 for d in deltas if d < -0.5)  # negative transfer at 0.5pp
    rows.append({
        "variant": v,
        "n": n,
        "mean_acc": mean_acc,
        "mean_delta_vs_src": mean_delta,
        "worst_delta_vs_src": worst_delta,
        "ntr_s_at_0p5": ntr_s,
        "per_subject_acc": {s: table[v][s] for s in subjects},
    })

(base / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

md = ["# A1 9-subject sweep summary", ""]
md.append("| variant | n | mean_acc | Δmean | Δworst | NTR-S@0.5pp |")
md.append("|---|---:|---:|---:|---:|---:|")
for r in rows:
    md.append(
        "| {variant} | {n} | {mean_acc:.2f} | {dm} | {dw} | {ntr}/9 |".format(
            variant=r["variant"],
            n=r["n"],
            mean_acc=r["mean_acc"],
            dm="-" if r["mean_delta_vs_src"] is None else f"{r['mean_delta_vs_src']:+.2f}",
            dw="-" if r["worst_delta_vs_src"] is None else f"{r['worst_delta_vs_src']:+.2f}",
            ntr=r["ntr_s_at_0p5"],
        )
    )
md.append("")
md.append("## Per-subject accuracy")
md.append("")
header = "| variant | " + " | ".join(f"S{s}" for s in subjects) + " |"
sep = "|---|" + "|".join("---:" for _ in subjects) + "|"
md.append(header)
md.append(sep)
for r in rows:
    cells = []
    for s in subjects:
        a = r["per_subject_acc"][s]
        cells.append("-" if a is None else f"{a:.2f}")
    md.append(f"| {r['variant']} | " + " | ".join(cells) + " |")
(base / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
print(f"\nSaved to: {base}")
PY

echo
echo "All done: ${BASE_DIR}"
