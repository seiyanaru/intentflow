#!/usr/bin/env bash
# H6 (pmax × class-inv weighted sim_score) 9-subject sweep, reusing existing
# source checkpoints. Compares against the uniform-replay 9-subject result.
#
# Usage: ./scripts/run_replay_h6_9subject.sh [GPU_ID]
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

GPU_ID="${1:-0}"
A1_DIR="results/a1_9subject_no_shallow_20260505_182651"
S2_CKPT="results/tcformer_policy_safe_otta_bcic2a_seed-0_aug-False_GPU0_20260429_2252"
S7_CKPT="results/proto_vs_policy_safe_s2_s7_20260505_113330/s7_source_model"
UNIFORM_REPLAY_DIR="results/replay_safe_9subject_20260505_212942"
H6_CONFIG="configs/tcformer_replay_safe_otta/tcformer_replay_safe_otta_h6_weighted.yaml"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/replay_h6_9subject_${TIMESTAMP}"
mkdir -p "${BASE_DIR}/eval"

ckpt_for_subject() {
    case "$1" in
        2) echo "${S2_CKPT}";;
        7) echo "${S7_CKPT}";;
        *) echo "${A1_DIR}/sources/s${1}";;
    esac
}

echo "================================================="
echo "H6 (pmax_class_inv) 9-subject sweep"
echo "Output: ${BASE_DIR}"
echo "================================================="

for sid in 1 2 3 4 5 6 7 8 9; do
    ckpt=$(ckpt_for_subject "${sid}")
    out_dir="${BASE_DIR}/eval/s${sid}/replay_h6_weighted"
    log_path="${BASE_DIR}/eval/s${sid}_replay_h6_weighted.log"
    echo "  [eval] s${sid}: ckpt=${ckpt}"
    if ! python train_pipeline.py \
            --model tcformer_replay_safe_otta \
            --dataset bcic2a \
            --seed 0 \
            --gpu_id "${GPU_ID}" \
            --no_interaug \
            --config "${H6_CONFIG}" \
            --subject_ids "${sid}" \
            --checkpoint_dir "${ckpt}" \
            --results_dir "${out_dir}" \
            --model_kwargs '{"enable_otta": true}' \
            > "${log_path}" 2>&1; then
        echo "  s${sid} FAILED (see ${log_path})"
        exit 1
    fi
done

# Aggregate vs all variants from a1 + uniform replay.
python - "${BASE_DIR}" "${A1_DIR}" "${UNIFORM_REPLAY_DIR}" <<'PY'
from __future__ import annotations
import json, re, sys
from pathlib import Path

base = Path(sys.argv[1])
a1 = Path(sys.argv[2])
uniform_replay = Path(sys.argv[3])
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

variant_dirs = {
    "source_only": (a1, "source_only"),
    "proto_otta_default": (a1, "proto_otta_default"),
    "policy_safe_default": (a1, "policy_safe_default"),
    "policy_safe_no_shallow": (a1, "policy_safe_no_shallow"),
    "replay_safe_uniform": (uniform_replay, "replay_safe_default"),
    "replay_h6_weighted": (base, "replay_h6_weighted"),
}
table = {}
for v, (root, sub) in variant_dirs.items():
    table[v] = {s: parse_acc(root / "eval" / f"s{s}" / sub) for s in subjects}

src = table["source_only"]
rows = []
for v in variant_dirs.keys():
    accs = [table[v][s] for s in subjects if table[v][s] is not None]
    deltas = [table[v][s] - src[s] for s in subjects if table[v][s] is not None and src[s] is not None]
    rows.append({
        "variant": v,
        "n": len(accs),
        "mean_acc": (sum(accs) / len(accs)) if accs else None,
        "mean_delta_vs_src": (sum(deltas) / len(deltas)) if deltas else None,
        "worst_delta_vs_src": min(deltas) if deltas else None,
        "ntr_s_at_0p5": sum(1 for d in deltas if d < -0.5) if deltas else None,
        "per_subject_acc": {s: table[v][s] for s in subjects},
    })
(base / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

md = ["# H6 (pmax × class-inv weighted) 9-subject sweep — full comparison", ""]
md.append("| variant | n | mean_acc | Δmean | Δworst | NTR-S@0.5pp |")
md.append("|---|---:|---:|---:|---:|---:|")
for r in rows:
    md.append("| {variant} | {n} | {mean_acc:.2f} | {dm} | {dw} | {ntr}/9 |".format(
        variant=r["variant"],
        n=r["n"],
        mean_acc=r["mean_acc"] or 0.0,
        dm="-" if r["mean_delta_vs_src"] is None else f"{r['mean_delta_vs_src']:+.2f}",
        dw="-" if r["worst_delta_vs_src"] is None else f"{r['worst_delta_vs_src']:+.2f}",
        ntr=r["ntr_s_at_0p5"] if r["ntr_s_at_0p5"] is not None else "-",
    ))
md.append("")
md.append("## Per-subject accuracy")
md.append("")
md.append("| variant | " + " | ".join(f"S{s}" for s in subjects) + " |")
md.append("|---|" + "|".join("---:" for _ in subjects) + "|")
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
echo "Done: ${BASE_DIR}"
