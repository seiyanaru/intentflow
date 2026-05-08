#!/usr/bin/env bash
# Plan B ablation: replay SafeCommit without shallow_var_update.
#
# Uses existing Plan B checkpoints and evaluates S2/S4/S6/S7 over seeds 1-3.
#
# Usage:
#   PYTHON_BIN=/path/to/python ./scripts/run_b_replay_no_shallow_4subj_seeds.sh [BASE_DIR] [GPU_ID]
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

BASE_DIR="${1:-results/b_5seed_4subj_20260506_005153}"
GPU_ID="${2:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

REPLAY_CFG="configs/tcformer_replay_safe_otta/tcformer_replay_safe_otta_s2_smoke.yaml"
H6_CFG="configs/tcformer_replay_safe_otta/tcformer_replay_safe_otta_h6_weighted.yaml"
NO_SHALLOW='{"enable_otta": true, "allowed_operators": ["prototype_update","logit_bias_update","deep_BN_update","hybrid_BN_update","no_update","abstain"]}'
SUBJECTS=(2 4 6 7)
SEEDS=(1 2 3)

echo "================================================="
echo "Plan B: replay no-shallow ablation"
echo "Base: ${BASE_DIR}"
echo "GPU: ${GPU_ID}"
echo "Python: ${PYTHON_BIN}"
echo "================================================="

eval_one() {
    local sid="$1"
    local seed="$2"
    local variant="$3"
    local cfg="$4"
    local ckpt="${BASE_DIR}/sources/s${sid}_seed${seed}"
    local out="${BASE_DIR}/eval/s${sid}_seed${seed}/${variant}"
    local log="${BASE_DIR}/eval/s${sid}_seed${seed}_${variant}.log"

    if [[ ! -f "${ckpt}/checkpoints/subject_${sid}_model.ckpt" ]]; then
        echo "  [missing ckpt] s${sid} seed${seed}: ${ckpt}/checkpoints/subject_${sid}_model.ckpt" >&2
        return 1
    fi
    if [[ -f "${out}/results.txt" ]]; then
        echo "  [skip] s${sid} seed${seed} ${variant} (results.txt exists)"
        return 0
    fi

    mkdir -p "${out}"
    echo "  [eval] s${sid} seed${seed} ${variant}"
    "${PYTHON_BIN}" train_pipeline.py \
        --model tcformer_replay_safe_otta --dataset bcic2a \
        --seed "${seed}" --gpu_id "${GPU_ID}" --no_interaug \
        --config "${cfg}" --subject_ids "${sid}" \
        --checkpoint_dir "${ckpt}" --results_dir "${out}" \
        --model_kwargs "${NO_SHALLOW}" \
        > "${log}" 2>&1
}

for sid in "${SUBJECTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        eval_one "${sid}" "${seed}" "replay_uniform_no_shallow" "${REPLAY_CFG}"
        eval_one "${sid}" "${seed}" "replay_h6_no_shallow" "${H6_CFG}"
    done
done

echo
echo "=== Aggregate no-shallow seed-stability ablation ==="
"${PYTHON_BIN}" - "${BASE_DIR}" <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path

base = Path(sys.argv[1])
subjects = [2, 4, 6, 7]
seeds = [1, 2, 3]
variants = [
    "source_only",
    "policy_safe_no_shallow",
    "replay_safe_uniform",
    "replay_uniform_no_shallow",
    "replay_h6_weighted",
    "replay_h6_no_shallow",
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
    (variant, subject, seed): parse_acc(base / "eval" / f"s{subject}_seed{seed}" / variant)
    for variant in variants
    for subject in subjects
    for seed in seeds
}

rows = []
for variant in variants:
    all_accs = []
    all_source_deltas = []
    per_subj = {}
    for subject in subjects:
        accs = []
        source_deltas = []
        for seed in seeds:
            acc = table[(variant, subject, seed)]
            src = table[("source_only", subject, seed)]
            if acc is None:
                continue
            accs.append(acc)
            all_accs.append(acc)
            if src is not None:
                source_deltas.append(acc - src)
                all_source_deltas.append(acc - src)
        if accs:
            per_subj[subject] = {
                "mean": statistics.mean(accs),
                "std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                "vals": accs,
            }
    rows.append({
        "variant": variant,
        "n": len(all_accs),
        "mean_acc": statistics.mean(all_accs) if all_accs else None,
        "mean_delta_vs_source": statistics.mean(all_source_deltas) if all_source_deltas else None,
        "worst_delta_vs_source": min(all_source_deltas) if all_source_deltas else None,
        "ntr_s_at_0p5": sum(1 for d in all_source_deltas if d < -0.5),
        "per_subj": per_subj,
    })

(base / "no_shallow_replay_seed_stability_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

def fmt(value, signed=False):
    if value is None:
        return "-"
    return f"{value:+.2f}" if signed else f"{value:.2f}"

md = ["# Plan B: Replay No-Shallow Seed-Stability", ""]
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
    cells = []
    for subject in subjects:
        if subject not in row["per_subj"]:
            cells.append("-")
        else:
            stats = row["per_subj"][subject]
            cells.append(f"{stats['mean']:.2f} ± {stats['std']:.2f}")
    md.append(f"| {row['variant']} | " + " | ".join(cells) + " |")

(base / "no_shallow_replay_seed_stability_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
print(f"\nSaved: {base / 'no_shallow_replay_seed_stability_summary.md'}")
PY

echo
echo "Done: ${BASE_DIR}"
