#!/usr/bin/env bash
# Plan C: aug-True (interaug enabled) regime, 9 subjects × seed 0.
# Trains source per subject and evaluates 4 variants. The aim is a direct
# comparison against the seminar-reported hybrid@0.01 mean=81.98.
#
# Estimated wall time: 9 trains × ~25 min + 36 evals × ~40s ≈ 4 hrs.
#
# Usage: ./scripts/run_c_aug_true_9subj.sh [GPU_ID]
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

GPU_ID="${1:-2}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/c_aug_true_9subj_${TIMESTAMP}"
mkdir -p "${BASE_DIR}/sources" "${BASE_DIR}/eval"

POLICY_CFG="configs/tcformer_policy_safe_otta/tcformer_policy_safe_otta_s2_smoke.yaml"
NO_SHALLOW_OPS='{"enable_otta": true, "allowed_operators": ["prototype_update","logit_bias_update","deep_BN_update","hybrid_BN_update","no_update","abstain"]}'
REPLAY_CFG="configs/tcformer_replay_safe_otta/tcformer_replay_safe_otta_s2_smoke.yaml"
H6_CFG="configs/tcformer_replay_safe_otta/tcformer_replay_safe_otta_h6_weighted.yaml"

SUBJECTS=(1 2 3 4 5 6 7 8 9)
SEED=0

echo "================================================="
echo "Plan C: aug-True 9-subject sweep"
echo "Output: ${BASE_DIR}, GPU=${GPU_ID}"
echo "================================================="

train_one() {
    local sid="$1"
    local out="${BASE_DIR}/sources/s${sid}"
    local log="${BASE_DIR}/sources/s${sid}.log"
    if [[ -f "${out}/checkpoints/subject_${sid}_model.ckpt" ]]; then
        echo "  [skip] s${sid} (ckpt exists)"; return 0
    fi
    echo "  [train aug-True] s${sid}"
    python train_pipeline.py \
        --model tcformer_policy_safe_otta --dataset bcic2a \
        --seed "${SEED}" --gpu_id "${GPU_ID}" --interaug \
        --config "${POLICY_CFG}" --subject_ids "${sid}" \
        --results_dir "${out}" \
        --model_kwargs '{"enable_otta": false}' \
        > "${log}" 2>&1
}

eval_one() {
    local sid="$1"; local variant="$2"; local model="$3"
    local cfg="$4"; local overrides="$5"; local ckpt="$6"
    local out="${BASE_DIR}/eval/s${sid}/${variant}"
    local log="${BASE_DIR}/eval/s${sid}_${variant}.log"
    mkdir -p "$(dirname "${out}")"
    echo "  [eval] s${sid} ${variant}"
    python train_pipeline.py \
        --model "${model}" --dataset bcic2a \
        --seed "${SEED}" --gpu_id "${GPU_ID}" --interaug \
        --config "${cfg}" --subject_ids "${sid}" \
        --checkpoint_dir "${ckpt}" --results_dir "${out}" \
        --model_kwargs "${overrides}" \
        > "${log}" 2>&1
}

echo
echo "=== Phase 1: train sources (aug-True) ==="
for sid in "${SUBJECTS[@]}"; do
    train_one "${sid}"
done

echo
echo "=== Phase 2: evaluate 4 variants per subject ==="
for sid in "${SUBJECTS[@]}"; do
    ckpt="${BASE_DIR}/sources/s${sid}"
    eval_one "${sid}" "source_only" \
        "tcformer_policy_safe_otta" "${POLICY_CFG}" \
        '{"enable_otta": false}' "${ckpt}"
    eval_one "${sid}" "policy_safe_no_shallow" \
        "tcformer_policy_safe_otta" "${POLICY_CFG}" \
        "${NO_SHALLOW_OPS}" "${ckpt}"
    eval_one "${sid}" "replay_safe_uniform" \
        "tcformer_replay_safe_otta" "${REPLAY_CFG}" \
        '{"enable_otta": true}' "${ckpt}"
    eval_one "${sid}" "replay_h6_weighted" \
        "tcformer_replay_safe_otta" "${H6_CFG}" \
        '{"enable_otta": true}' "${ckpt}"
done

echo
echo "=== Phase 3: aggregate ==="
python - "${BASE_DIR}" <<'PY'
import json, re, sys
from pathlib import Path
base = Path(sys.argv[1])
subjects = [1,2,3,4,5,6,7,8,9]
variants = ["source_only","policy_safe_no_shallow","replay_safe_uniform","replay_h6_weighted"]
def parse_acc(d):
    f = d / "results.txt"
    if not f.exists(): return None
    for line in f.read_text().splitlines():
        m = re.search(r"Test Acc:\s+([\d.]+)", line)
        if m: return float(m.group(1)) * 100.0
    return None
table = {v: {s: parse_acc(base / "eval" / f"s{s}" / v) for s in subjects} for v in variants}
src = table["source_only"]
rows = []
for v in variants:
    accs = [table[v][s] for s in subjects if table[v][s] is not None]
    deltas = [table[v][s] - src[s] for s in subjects if table[v][s] is not None and src[s] is not None]
    rows.append({"variant": v, "n": len(accs), "mean_acc": sum(accs)/len(accs) if accs else None,
                 "mean_delta": sum(deltas)/len(deltas) if deltas else None,
                 "worst_delta": min(deltas) if deltas else None,
                 "ntr_s": sum(1 for d in deltas if d < -0.5),
                 "per_subj": table[v]})
(base / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
md = ["# Plan C: aug-True 9-subject sweep", ""]
md.append("| variant | mean | Δmean | Δworst | NTR-S |")
md.append("|---|---:|---:|---:|---:|")
for r in rows:
    md.append("| {v} | {m:.2f} | {dm} | {dw} | {n}/9 |".format(
        v=r["variant"], m=r["mean_acc"] or 0,
        dm=("-" if r["mean_delta"] is None else f"{r['mean_delta']:+.2f}"),
        dw=("-" if r["worst_delta"] is None else f"{r['worst_delta']:+.2f}"),
        n=r["ntr_s"]))
md.append("")
md.append("| variant | " + " | ".join(f"S{s}" for s in subjects) + " |")
md.append("|---|" + "|".join("---:" for _ in subjects) + "|")
for r in rows:
    cells = ["-" if r["per_subj"][s] is None else f"{r['per_subj'][s]:.2f}" for s in subjects]
    md.append(f"| {r['variant']} | " + " | ".join(cells) + " |")
(base / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
print(f"\nSaved: {base}")
PY

echo
echo "Done: ${BASE_DIR}"
