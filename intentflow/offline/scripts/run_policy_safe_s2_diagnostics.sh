#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR="${1:-results/tcformer_policy_safe_otta_bcic2a_seed-0_aug-False_GPU0_20260429_2252}"
GPU_ID="${GPU_ID:-0}"
CONFIG="configs/tcformer_policy_safe_otta/tcformer_policy_safe_otta_s2_smoke.yaml"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="results/policy_safe_s2_diagnostics_${STAMP}"

mkdir -p "${OUT_ROOT}"

run_variant() {
  local name="$1"
  local kwargs="$2"
  local out_dir="${OUT_ROOT}/${name}"
  echo
  echo ">>> ${name}"
  python train_pipeline.py \
    --config "${CONFIG}" \
    --dataset bcic2a \
    --seed 0 \
    --gpu_id "${GPU_ID}" \
    --no_interaug \
    --checkpoint_dir "${CKPT_DIR}" \
    --results_dir "${out_dir}" \
    --model_kwargs "${kwargs}"
}

run_variant "source_only" \
  '{"enable_otta": false, "prototype_fusion_alpha": 0.0}'

run_variant "policy_soft_energy_alpha0" \
  '{"prototype_fusion_alpha": 0.0, "abstain_on_ood": true, "energy_blocks_update": true, "energy_abstain_margin": 2.0, "energy_abstain_z": 6.0}'

run_variant "policy_soft_energy_alpha02" \
  '{"prototype_fusion_alpha": 0.2, "abstain_on_ood": true, "energy_blocks_update": true, "energy_abstain_margin": 2.0, "energy_abstain_z": 6.0}'

run_variant "policy_no_energy_block_alpha02" \
  '{"prototype_fusion_alpha": 0.2, "abstain_on_ood": false, "energy_blocks_update": false}'

run_variant "policy_proto_alpha05" \
  '{"prototype_fusion_alpha": 0.5, "abstain_on_ood": true, "energy_blocks_update": true, "energy_abstain_margin": 2.0, "energy_abstain_z": 6.0}'

run_variant "policy_logit_bias_only" \
  '{"prototype_fusion_alpha": 0.0, "abstain_on_ood": true, "energy_blocks_update": true, "energy_abstain_margin": 2.0, "energy_abstain_z": 6.0, "allowed_operators": ["logit_bias_update", "no_update", "abstain"]}'

python - <<'PY' "${OUT_ROOT}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for result_file in sorted(root.glob("*/results.txt")):
    text = result_file.read_text()
    row = {"variant": result_file.parent.name}
    for line in text.splitlines():
        if "Average Test Accuracy:" in line:
            row["acc"] = line.split(":", 1)[1].strip()
        elif "Average Test Kappa:" in line:
            row["kappa"] = line.split(":", 1)[1].strip()
        elif "Average Test Loss:" in line:
            row["loss"] = line.split(":", 1)[1].strip()
    rows.append(row)

(root / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
md = ["| variant | acc | kappa | loss |", "|---|---:|---:|---:|"]
for row in rows:
    md.append(f"| {row.get('variant', '')} | {row.get('acc', '')} | {row.get('kappa', '')} | {row.get('loss', '')} |")
(root / "summary.md").write_text("\n".join(md) + "\n")
print(f"\nSaved summary: {root / 'summary.md'}")
PY
