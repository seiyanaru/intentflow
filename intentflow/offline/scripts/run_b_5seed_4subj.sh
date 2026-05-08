#!/usr/bin/env bash
# Plan B: seed-stability check on the 4 most informative subjects
# (S2 hardest harm, S4 worst-Δ, S6 hard, S7 gain). New seeds {1,2,3}
# (seed 0 already exists). Trains source per (subject, seed) with the
# policy_safe_otta wrapper enable_otta=false, then evaluates 4 variants.
#
# Variants: source_only, policy_safe_no_shallow, replay_safe_uniform,
# replay_h6_weighted. (proto/policy_safe_default omitted to save time;
# if needed they can be added later from same checkpoints.)
#
# Estimated wall time on a single GPU: 12 trains × ~20 min + 48 evals × ~40s
# ≈ 4.5 hrs.
#
# Usage: ./scripts/run_b_5seed_4subj.sh [GPU_ID]
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

GPU_ID="${1:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/b_5seed_4subj_${TIMESTAMP}"
mkdir -p "${BASE_DIR}/sources" "${BASE_DIR}/eval"

POLICY_CFG="configs/tcformer_policy_safe_otta/tcformer_policy_safe_otta_s2_smoke.yaml"
NO_SHALLOW_OPS='{"enable_otta": true, "allowed_operators": ["prototype_update","logit_bias_update","deep_BN_update","hybrid_BN_update","no_update","abstain"]}'
REPLAY_CFG="configs/tcformer_replay_safe_otta/tcformer_replay_safe_otta_s2_smoke.yaml"
H6_CFG="configs/tcformer_replay_safe_otta/tcformer_replay_safe_otta_h6_weighted.yaml"

SUBJECTS=(2 4 6 7)
SEEDS=(1 2 3)

echo "================================================="
echo "Plan B 5-seed 4-subject sweep"
echo "Output: ${BASE_DIR}, GPU=${GPU_ID}"
echo "Subjects: ${SUBJECTS[*]}, new seeds: ${SEEDS[*]}"
echo "================================================="

train_one() {
    local sid="$1"; local seed="$2"
    local out="${BASE_DIR}/sources/s${sid}_seed${seed}"
    local log="${BASE_DIR}/sources/s${sid}_seed${seed}.log"
    if [[ -f "${out}/checkpoints/subject_${sid}_model.ckpt" ]]; then
        echo "  [skip] s${sid} seed${seed} (ckpt exists)"; return 0
    fi
    echo "  [train] s${sid} seed${seed} -> ${out}"
    python train_pipeline.py \
        --model tcformer_policy_safe_otta --dataset bcic2a \
        --seed "${seed}" --gpu_id "${GPU_ID}" --no_interaug \
        --config "${POLICY_CFG}" --subject_ids "${sid}" \
        --results_dir "${out}" \
        --model_kwargs '{"enable_otta": false}' \
        > "${log}" 2>&1
}

eval_one() {
    local sid="$1"; local seed="$2"; local variant="$3"
    local model="$4"; local cfg="$5"; local overrides="$6"; local ckpt="$7"
    local out="${BASE_DIR}/eval/s${sid}_seed${seed}/${variant}"
    local log="${BASE_DIR}/eval/s${sid}_seed${seed}_${variant}.log"
    mkdir -p "$(dirname "${out}")"
    echo "  [eval] s${sid} seed${seed} ${variant}"
    python train_pipeline.py \
        --model "${model}" --dataset bcic2a \
        --seed "${seed}" --gpu_id "${GPU_ID}" --no_interaug \
        --config "${cfg}" --subject_ids "${sid}" \
        --checkpoint_dir "${ckpt}" --results_dir "${out}" \
        --model_kwargs "${overrides}" \
        > "${log}" 2>&1
}

echo
echo "=== Phase 1: train sources ==="
for sid in "${SUBJECTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        train_one "${sid}" "${seed}"
    done
done

echo
echo "=== Phase 2: evaluate 4 variants per (subject, seed) ==="
for sid in "${SUBJECTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ckpt="${BASE_DIR}/sources/s${sid}_seed${seed}"
        eval_one "${sid}" "${seed}" "source_only" \
            "tcformer_policy_safe_otta" "${POLICY_CFG}" \
            '{"enable_otta": false}' "${ckpt}"
        eval_one "${sid}" "${seed}" "policy_safe_no_shallow" \
            "tcformer_policy_safe_otta" "${POLICY_CFG}" \
            "${NO_SHALLOW_OPS}" "${ckpt}"
        eval_one "${sid}" "${seed}" "replay_safe_uniform" \
            "tcformer_replay_safe_otta" "${REPLAY_CFG}" \
            '{"enable_otta": true}' "${ckpt}"
        eval_one "${sid}" "${seed}" "replay_h6_weighted" \
            "tcformer_replay_safe_otta" "${H6_CFG}" \
            '{"enable_otta": true}' "${ckpt}"
    done
done

echo
echo "=== Phase 3: aggregate ==="
python - "${BASE_DIR}" <<'PY'
import json, re, sys
from pathlib import Path
import statistics
base = Path(sys.argv[1])
subjects = [2, 4, 6, 7]
seeds = [1, 2, 3]
variants = ["source_only","policy_safe_no_shallow","replay_safe_uniform","replay_h6_weighted"]
def parse_acc(d):
    f = d / "results.txt"
    if not f.exists(): return None
    for line in f.read_text().splitlines():
        m = re.search(r"Test Acc:\s+([\d.]+)", line)
        if m: return float(m.group(1)) * 100.0
    return None
table = {(v,s,seed): parse_acc(base / "eval" / f"s{s}_seed{seed}" / v) for v in variants for s in subjects for seed in seeds}
rows = []
for v in variants:
    per_subj = {}
    for s in subjects:
        accs = [table[(v,s,seed)] for seed in seeds if table[(v,s,seed)] is not None]
        if accs:
            per_subj[s] = {"mean": statistics.mean(accs), "std": statistics.pstdev(accs) if len(accs) > 1 else 0.0, "vals": accs}
    rows.append({"variant": v, "per_subj": per_subj})
(base / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str))
md = ["# Plan B: seed-stability on 4 subjects (new seeds 1-3)", ""]
md.append("| variant | " + " | ".join(f"S{s}" for s in subjects) + " |")
md.append("|---|" + "|".join("---:" for _ in subjects) + "|")
for r in rows:
    cells = []
    for s in subjects:
        if s in r["per_subj"]:
            m = r["per_subj"][s]["mean"]; sd = r["per_subj"][s]["std"]
            cells.append(f"{m:.2f} ± {sd:.2f}")
        else:
            cells.append("-")
    md.append(f"| {r['variant']} | " + " | ".join(cells) + " |")
(base / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
print(f"\nSaved: {base}")
PY

echo
echo "Done: ${BASE_DIR}"
