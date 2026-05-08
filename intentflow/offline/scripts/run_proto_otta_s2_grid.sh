#!/usr/bin/env bash
# =============================================================
# Proto-OTTA S2 grid for proposal E'' first pass.
#
# Runs:
#   0. source_only reference (train once)
#   A. alpha=0.0, Energy on
#   B. alpha=0.1, Energy on
#   C. alpha=0.1, Energy off
#   D. alpha=0.3, Energy off
#
# The source model is trained only once, then all Proto-OTTA variants
# load the same checkpoint for a clean test-only comparison.
#
# Usage:
#   ./scripts/run_proto_otta_s2_grid.sh [GPU_ID] [SEED] [OUTPUT_DIR]
# =============================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

GPU_ID="${1:-0}"
SEED="${2:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="${3:-results/proto_otta_s2_grid_${TIMESTAMP}}"
CONFIG="configs/tcformer_proto_otta/tcformer_proto_otta_s2_smoke.yaml"
DATASET="bcic2a"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export MNE_DONTWRITE_HOME="${MNE_DONTWRITE_HOME:-true}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$BASE_DIR"

if [[ "${GPU_ID}" != "-1" ]]; then
  python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("[ProtoOTTA] ERROR: gpu_id>=0 requested but CUDA is unavailable.", file=sys.stderr)
    sys.exit(1)
PY
fi

run_logged() {
    local label="$1"
    local log_path="$2"
    shift 2

    echo ""
    echo "=== ${label}: $(date) ==="
    if "$@" 2>&1 | tee "${log_path}"; then
        echo "=== ${label} DONE: $(date) ==="
        return 0
    else
        local cmd_status=${PIPESTATUS[0]:-1}
        echo "=== ${label} FAILED (cmd=${cmd_status}): $(date) ==="
        return "${cmd_status}"
    fi
}

echo "================================================="
echo "Proto-OTTA S2 Grid"
echo "Config: ${CONFIG}"
echo "GPU: ${GPU_ID}"
echo "Seed: ${SEED}"
echo "Output: ${BASE_DIR}"
echo "================================================="

SOURCE_DIR="${BASE_DIR}/source_model"

SOURCE_OVERRIDES='{"enable_otta": false}'

run_logged "train_source_s2" "${BASE_DIR}/train_source_s2.log" \
    python train_pipeline.py \
        --model tcformer_proto_otta \
        --dataset "${DATASET}" \
        --seed "${SEED}" \
        --gpu_id "${GPU_ID}" \
        --no_interaug \
        --config "${CONFIG}" \
        --results_dir "${SOURCE_DIR}" \
        --model_kwargs "${SOURCE_OVERRIDES}"

declare -a TAGS=(
  "A_alpha0_energy_on"
  "B_alpha01_energy_on"
  "C_alpha01_energy_off"
  "D_alpha03_energy_off"
)

declare -a OVERRIDES=(
  '{"enable_otta": true, "fusion_alpha": 0.0, "use_energy_gate": true,  "proto_momentum": 0.05}'
  '{"enable_otta": true, "fusion_alpha": 0.1, "use_energy_gate": true,  "proto_momentum": 0.05}'
  '{"enable_otta": true, "fusion_alpha": 0.1, "use_energy_gate": false, "proto_momentum": 0.05}'
  '{"enable_otta": true, "fusion_alpha": 0.3, "use_energy_gate": false, "proto_momentum": 0.05}'
)

for idx in "${!TAGS[@]}"; do
    tag="${TAGS[$idx]}"
    overrides="${OVERRIDES[$idx]}"
    run_logged "${tag}" "${BASE_DIR}/${tag}.log" \
        python train_pipeline.py \
            --model tcformer_proto_otta \
            --dataset "${DATASET}" \
            --seed "${SEED}" \
            --gpu_id "${GPU_ID}" \
            --no_interaug \
            --config "${CONFIG}" \
            --checkpoint_dir "${SOURCE_DIR}" \
            --results_dir "${BASE_DIR}/${tag}" \
            --model_kwargs "${overrides}"
done

python - "${BASE_DIR}" "${SOURCE_DIR}" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

base = Path(sys.argv[1])
source = Path(sys.argv[2])
tags = [
    "A_alpha0_energy_on",
    "B_alpha01_energy_on",
    "C_alpha01_energy_off",
    "D_alpha03_energy_off",
]

def parse_acc(result_dir: Path) -> float | None:
    txt = result_dir / "results.txt"
    if not txt.exists():
        return None
    for line in txt.read_text().splitlines():
        m = re.search(r"Average Test Accuracy:\s+([\d.]+)", line)
        if m:
            return float(m.group(1))
    return None

def read_npz(result_dir: Path):
    files = list(result_dir.glob("proto_otta_stats_s2_*.npz"))
    if not files:
        return None
    return np.load(files[0])

summary = []
src_acc = parse_acc(source)
summary.append({"tag": "source_model_no_otta", "test_acc": src_acc})

for tag in tags:
    d = read_npz(base / tag)
    acc = parse_acc(base / tag)
    row = {"tag": tag, "test_acc": acc}
    if d is not None:
        y = d["label"]
        pred = d["pred"]
        orig = d["original_pred"]
        upd = d["adapted"].astype(bool)
        row.update(
            {
                "orig_acc_from_npz": float((orig == y).mean()),
                "final_acc_from_npz": float((pred == y).mean()),
                "changed": int((pred != orig).sum()),
                "changed_to_correct": int(((pred != orig) & (pred == y)).sum()),
                "changed_to_wrong": int(((pred != orig) & (orig == y) & (pred != y)).sum()),
                "updates": int(upd.sum()),
                "update_acc_orig": float((orig[upd] == y[upd]).mean()) if upd.any() else None,
                "pmax_pass_rate": float(d["gate_high_pmax"].mean()),
                "sal_pass_rate": float(d["gate_high_sal"].mean()),
                "energy_pass_rate": float(d["gate_safe_energy"].mean()),
                "mean_pmax": float(d["pmax"].mean()),
                "mean_fused_pmax": float(d["fused_pmax"].mean()),
                "mean_sal": float(d["sal"].mean()),
            }
        )
    summary.append(row)

out_json = base / "summary.json"
out_md = base / "summary.md"
out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

lines = [
    "# Proto-OTTA S2 Grid Summary",
    "",
    "| tag | test_acc | updates | update_acc | changed | +correct | +wrong | pmax_pass | sal_pass | energy_pass |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in summary:
    lines.append(
        "| {tag} | {test_acc} | {updates} | {update_acc} | {changed} | {changed_to_correct} | {changed_to_wrong} | {pmax_pass} | {sal_pass} | {energy_pass} |".format(
            tag=row["tag"],
            test_acc="-" if row.get("test_acc") is None else f"{row['test_acc']:.2f}",
            updates=row.get("updates", "-"),
            update_acc="-" if row.get("update_acc_orig") is None else f"{100*row['update_acc_orig']:.2f}",
            changed=row.get("changed", "-"),
            changed_to_correct=row.get("changed_to_correct", "-"),
            changed_to_wrong=row.get("changed_to_wrong", "-"),
            pmax_pass="-" if row.get("pmax_pass_rate") is None else f"{100*row['pmax_pass_rate']:.1f}",
            sal_pass="-" if row.get("sal_pass_rate") is None else f"{100*row['sal_pass_rate']:.1f}",
            energy_pass="-" if row.get("energy_pass_rate") is None else f"{100*row['energy_pass_rate']:.1f}",
        )
    )
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\n".join(lines))
print(f"\nWrote {out_json}")
print(f"Wrote {out_md}")
PY

echo ""
echo "All done: ${BASE_DIR}"
