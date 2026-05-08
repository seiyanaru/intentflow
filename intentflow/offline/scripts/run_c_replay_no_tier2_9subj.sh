#!/usr/bin/env bash
# Plan C ablation: replay buffer without the tier-2 sim_score gate.
#
# Uses the already-trained aug-True source checkpoints from Plan C and evaluates
# replay_uniform_no_tier2 on all 9 subjects. This isolates whether the replay
# simulated-reward gate itself is useful, versus merely having replay machinery.
#
# Usage:
#   PYTHON_BIN=/path/to/python ./scripts/run_c_replay_no_tier2_9subj.sh [BASE_DIR] [GPU_ID]
#
# Defaults:
#   BASE_DIR=results/c_aug_true_9subj_20260506_004923
#   GPU_ID=2
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

BASE_DIR="${1:-results/c_aug_true_9subj_20260506_004923}"
GPU_ID="${2:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

REPLAY_CFG="configs/tcformer_replay_safe_otta/tcformer_replay_safe_otta_s2_smoke.yaml"
VARIANT="replay_uniform_no_tier2"
OVERRIDES='{"enable_otta": true, "sim_score_tolerance": -999.0}'
SUBJECTS=(1 2 3 4 5 6 7 8 9)

echo "================================================="
echo "Plan C tier-2 ablation: replay_uniform_no_tier2"
echo "Base: ${BASE_DIR}"
echo "GPU: ${GPU_ID}"
echo "Python: ${PYTHON_BIN}"
echo "================================================="

eval_one() {
    local sid="$1"
    local ckpt="${BASE_DIR}/sources/s${sid}"
    local out="${BASE_DIR}/eval/s${sid}/${VARIANT}"
    local log="${BASE_DIR}/eval/s${sid}_${VARIANT}.log"

    if [[ ! -f "${ckpt}/checkpoints/subject_${sid}_model.ckpt" ]]; then
        echo "  [missing ckpt] s${sid}: ${ckpt}/checkpoints/subject_${sid}_model.ckpt" >&2
        return 1
    fi
    if [[ -f "${out}/results.txt" ]]; then
        echo "  [skip] s${sid} ${VARIANT} (results.txt exists)"
        return 0
    fi

    mkdir -p "${out}"
    echo "  [eval] s${sid} ${VARIANT}"
    "${PYTHON_BIN}" train_pipeline.py \
        --model tcformer_replay_safe_otta --dataset bcic2a \
        --seed 0 --gpu_id "${GPU_ID}" --interaug \
        --config "${REPLAY_CFG}" --subject_ids "${sid}" \
        --checkpoint_dir "${ckpt}" --results_dir "${out}" \
        --model_kwargs "${OVERRIDES}" \
        > "${log}" 2>&1
}

for sid in "${SUBJECTS[@]}"; do
    eval_one "${sid}"
done

echo
echo "=== Aggregate tier-2 ablation ==="
"${PYTHON_BIN}" - "${BASE_DIR}" <<'PY'
import json
import re
import sys
from pathlib import Path

base = Path(sys.argv[1])
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
variants = [
    "source_only",
    "policy_safe_no_shallow",
    "replay_safe_uniform",
    "replay_uniform_no_tier2",
    "replay_h6_weighted",
]


def parse_acc(path: Path):
    f = path / "results.txt"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        m = re.search(r"Test Acc:\s+([\d.]+)", line)
        if m:
            return float(m.group(1)) * 100.0
    return None


table = {
    variant: {
        subject: parse_acc(base / "eval" / f"s{subject}" / variant)
        for subject in subjects
    }
    for variant in variants
}
source = table["source_only"]
no_tier2 = table["replay_uniform_no_tier2"]

rows = []
for variant in variants:
    accs = [table[variant][s] for s in subjects if table[variant][s] is not None]
    deltas = [
        table[variant][s] - source[s]
        for s in subjects
        if table[variant][s] is not None and source[s] is not None
    ]
    tier2_deltas = []
    if variant != "replay_uniform_no_tier2":
        tier2_deltas = [
            table[variant][s] - no_tier2[s]
            for s in subjects
            if table[variant][s] is not None and no_tier2[s] is not None
        ]
    rows.append({
        "variant": variant,
        "n": len(accs),
        "mean_acc": sum(accs) / len(accs) if accs else None,
        "mean_delta_vs_source": sum(deltas) / len(deltas) if deltas else None,
        "worst_delta_vs_source": min(deltas) if deltas else None,
        "ntr_s_at_0p5": sum(1 for d in deltas if d < -0.5),
        "mean_delta_vs_no_tier2": sum(tier2_deltas) / len(tier2_deltas) if tier2_deltas else None,
        "worst_delta_vs_no_tier2": min(tier2_deltas) if tier2_deltas else None,
        "per_subj": table[variant],
    })

(base / "tier2_ablation_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

md = ["# Plan C: Replay Tier-2 Gate Ablation", ""]
md.append("| variant | mean | Δmean vs source | worst vs source | NTR-S | Δmean vs no_tier2 |")
md.append("|---|---:|---:|---:|---:|---:|")
for r in rows:
    def fmt(x, signed=False):
        if x is None:
            return "-"
        return f"{x:+.2f}" if signed else f"{x:.2f}"

    md.append(
        f"| {r['variant']} | {fmt(r['mean_acc'])} | "
        f"{fmt(r['mean_delta_vs_source'], signed=True)} | "
        f"{fmt(r['worst_delta_vs_source'], signed=True)} | "
        f"{r['ntr_s_at_0p5']}/9 | "
        f"{fmt(r['mean_delta_vs_no_tier2'], signed=True)} |"
    )

md.append("")
md.append("| variant | " + " | ".join(f"S{s}" for s in subjects) + " |")
md.append("|---|" + "|".join("---:" for _ in subjects) + "|")
for r in rows:
    cells = [
        "-" if r["per_subj"][s] is None else f"{r['per_subj'][s]:.2f}"
        for s in subjects
    ]
    md.append(f"| {r['variant']} | " + " | ".join(cells) + " |")

md.append("")
md.append("## Tier-2 Effect: replay_safe_uniform - replay_uniform_no_tier2")
md.append("")
md.append("| subject | replay_safe_uniform | no_tier2 | Δ |")
md.append("|---|---:|---:|---:|")
for s in subjects:
    a = table["replay_safe_uniform"][s]
    b = table["replay_uniform_no_tier2"][s]
    if a is None or b is None:
        md.append(f"| S{s} | - | - | - |")
    else:
        md.append(f"| S{s} | {a:.2f} | {b:.2f} | {a - b:+.2f} |")

(base / "tier2_ablation_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
print(f"\nSaved: {base / 'tier2_ablation_summary.md'}")
PY

echo
echo "=== P1 report with tier-2 ablation ==="
"${PYTHON_BIN}" scripts/analysis/analyze_replay_gate_p1.py \
    --result-dir "${BASE_DIR}" \
    --replay-variants replay_safe_uniform replay_uniform_no_tier2 \
    --out-prefix p1_tier2_ablation_9subj

echo
echo "Done: ${BASE_DIR}"
