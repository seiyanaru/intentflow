#!/usr/bin/env bash
# Plan C ablation: Best-of-K replay SafeCommit.
#
# Evaluates replay variants that score every candidate operator on the replay
# buffer and commit the passing operator with the largest sim_score. This tests
# whether replay simulated reward is useful for action selection, not just a
# binary safety gate.
#
# Usage:
#   PYTHON_BIN=/path/to/python ./scripts/run_c_replay_best_9subj.sh [BASE_DIR] [GPU_ID]
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
H6_CFG="configs/tcformer_replay_safe_otta/tcformer_replay_safe_otta_h6_weighted.yaml"
SUBJECTS=(1 2 3 4 5 6 7 8 9)

echo "================================================="
echo "Plan C: Best-of-K replay SafeCommit"
echo "Base: ${BASE_DIR}"
echo "GPU: ${GPU_ID}"
echo "Python: ${PYTHON_BIN}"
echo "================================================="

eval_one() {
    local sid="$1"
    local variant="$2"
    local cfg="$3"
    local overrides="$4"
    local ckpt="${BASE_DIR}/sources/s${sid}"
    local out="${BASE_DIR}/eval/s${sid}/${variant}"
    local log="${BASE_DIR}/eval/s${sid}_${variant}.log"

    if [[ ! -f "${ckpt}/checkpoints/subject_${sid}_model.ckpt" ]]; then
        echo "  [missing ckpt] s${sid}: ${ckpt}/checkpoints/subject_${sid}_model.ckpt" >&2
        return 1
    fi
    if [[ -f "${out}/results.txt" ]]; then
        echo "  [skip] s${sid} ${variant} (results.txt exists)"
        return 0
    fi

    mkdir -p "${out}"
    echo "  [eval] s${sid} ${variant}"
    "${PYTHON_BIN}" train_pipeline.py \
        --model tcformer_replay_safe_otta --dataset bcic2a \
        --seed 0 --gpu_id "${GPU_ID}" --interaug \
        --config "${cfg}" --subject_ids "${sid}" \
        --checkpoint_dir "${ckpt}" --results_dir "${out}" \
        --model_kwargs "${overrides}" \
        > "${log}" 2>&1
}

for sid in "${SUBJECTS[@]}"; do
    eval_one "${sid}" "replay_best_uniform" "${REPLAY_CFG}" \
        '{"enable_otta": true, "select_best_candidate": true}'
    eval_one "${sid}" "replay_best_h6_weighted" "${H6_CFG}" \
        '{"enable_otta": true, "select_best_candidate": true}'
done

echo
echo "=== Aggregate best-safe ablation ==="
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
    "replay_best_uniform",
    "replay_uniform_no_tier2",
    "replay_h6_weighted",
    "replay_best_h6_weighted",
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

rows = []
for variant in variants:
    accs = [table[variant][s] for s in subjects if table[variant][s] is not None]
    deltas = [
        table[variant][s] - source[s]
        for s in subjects
        if table[variant][s] is not None and source[s] is not None
    ]
    rows.append({
        "variant": variant,
        "n": len(accs),
        "mean_acc": sum(accs) / len(accs) if accs else None,
        "mean_delta_vs_source": sum(deltas) / len(deltas) if deltas else None,
        "worst_delta_vs_source": min(deltas) if deltas else None,
        "ntr_s_at_0p5": sum(1 for d in deltas if d < -0.5),
        "per_subj": table[variant],
    })

(base / "best_replay_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

def fmt(value, signed=False):
    if value is None:
        return "-"
    return f"{value:+.2f}" if signed else f"{value:.2f}"

md = ["# Plan C: Best-of-K Replay SafeCommit", ""]
md.append("| variant | mean | Δmean vs source | worst vs source | NTR-S |")
md.append("|---|---:|---:|---:|---:|")
for row in rows:
    md.append(
        f"| {row['variant']} | {fmt(row['mean_acc'])} | "
        f"{fmt(row['mean_delta_vs_source'], signed=True)} | "
        f"{fmt(row['worst_delta_vs_source'], signed=True)} | "
        f"{row['ntr_s_at_0p5']}/{row['n']} |"
    )

md.append("")
md.append("| variant | " + " | ".join(f"S{s}" for s in subjects) + " |")
md.append("|---|" + "|".join("---:" for _ in subjects) + "|")
for row in rows:
    cells = [
        "-" if row["per_subj"][s] is None else f"{row['per_subj'][s]:.2f}"
        for s in subjects
    ]
    md.append(f"| {row['variant']} | " + " | ".join(cells) + " |")

md.append("")
md.append("## Best-of-K Effect")
md.append("")
md.append("| subject | first_uniform | best_uniform | Δ | first_h6 | best_h6 | Δ |")
md.append("|---|---:|---:|---:|---:|---:|---:|")
for s in subjects:
    fu = table["replay_safe_uniform"][s]
    bu = table["replay_best_uniform"][s]
    fh = table["replay_h6_weighted"][s]
    bh = table["replay_best_h6_weighted"][s]
    md.append(
        f"| S{s} | {fmt(fu)} | {fmt(bu)} | "
        f"{fmt(None if fu is None or bu is None else bu - fu, signed=True)} | "
        f"{fmt(fh)} | {fmt(bh)} | "
        f"{fmt(None if fh is None or bh is None else bh - fh, signed=True)} |"
    )

(base / "best_replay_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
print(f"\nSaved: {base / 'best_replay_summary.md'}")
PY

echo
echo "=== P1 report with best-safe variants ==="
"${PYTHON_BIN}" scripts/analysis/analyze_replay_gate_p1.py \
    --result-dir "${BASE_DIR}" \
    --replay-variants replay_safe_uniform replay_best_uniform replay_h6_weighted replay_best_h6_weighted \
    --out-prefix p1_best_replay_9subj

echo
echo "Done: ${BASE_DIR}"
