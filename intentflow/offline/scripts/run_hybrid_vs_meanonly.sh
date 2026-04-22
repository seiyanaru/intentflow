#!/bin/bash
# =============================================================
# Hybrid vs Mean-only — causal minimum intervention, all 9 subjects
# =============================================================
#
# 実験設計：
#   3 conditions × 9 subjects, bn_stat_clean, mom=0.01, bs=1
#
#   1. source_only      : 適応なし（共通 reference）
#   2. hybrid           : shallow var 凍結 + shallow mean + deep both
#                         = culprit だけを止める最小介入
#   3. mean_only        : 全層 var 凍結（保守ベースライン）
#
# 見るべき指標：
#   - 各被験者 Δacc
#   - NTR-S, WSD, mean delta
#   - shallow / deep drift norm の差
#   - hybrid vs mean_only の trade-off（S7 型 gain の保持率）
#
# 反証条件（実験前に明示）：
#   hybrid が S2 を救えない → culprit は shallow var だけではない
#   hybrid と mean_only の S7 gain が同等 → deep/mean だけで十分
#   S1/S3/S6 型で hybrid が mean_only より大きく落ちる → hybrid が不安定
#
# Usage:
#   ./scripts/run_hybrid_vs_meanonly.sh [SOURCE_CKPT_DIR] [GPU_ID]
# =============================================================

cd "$(dirname "$0")/.." || exit 1

SOURCE_DIR="${1:-results/update_op_v2_20260401_125734/source_model}"
GPU_ID="${2:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/hybrid_vs_meanonly_${TIMESTAMP}"
CONFIG="configs/tcformer_otta/tcformer_otta_bs1.yaml"
MOM=0.01

echo "================================================="
echo "Hybrid vs Mean-only (all 9 subjects, mom=${MOM}, bs=1)"
echo "Source: ${SOURCE_DIR}"
echo "Output: ${BASE_DIR}"
echo "GPU: ${GPU_ID}"
echo "Start: $(date)"
echo "================================================="

CKPT_COUNT=$(ls "${SOURCE_DIR}/checkpoints/"*.ckpt 2>/dev/null | wc -l)
if [ "${CKPT_COUNT}" -lt 9 ]; then
    echo "[FATAL] Need 9 checkpoints in ${SOURCE_DIR}/checkpoints/, found ${CKPT_COUNT}"
    exit 1
fi

mkdir -p "${BASE_DIR}"

# ── 1. source_only ──────────────────────────────────────────────────────────
echo ""
echo "=== [1/3] source_only ==="
timeout 1200 python3 train_pipeline.py \
    --model tcformer_otta --dataset bcic2a \
    --gpu_id "${GPU_ID}" --config "${CONFIG}" \
    --results_dir "${BASE_DIR}/source_only" \
    --checkpoint_dir "${SOURCE_DIR}" \
    --model_kwargs '{"adapt_mode": "source_only", "enable_otta": false}' \
    2>&1 | tee "${BASE_DIR}/source_only.log"
echo "=== source_only DONE: $(date) ==="

# ── 2. hybrid: shallow_mean_deep_both ───────────────────────────────────────
echo ""
echo "=== [2/3] hybrid (shallow_mean_deep_both, mom=${MOM}) ==="
timeout 1800 python3 train_pipeline.py \
    --model tcformer_otta --dataset bcic2a \
    --gpu_id "${GPU_ID}" --config "${CONFIG}" \
    --results_dir "${BASE_DIR}/hybrid" \
    --checkpoint_dir "${SOURCE_DIR}" \
    --model_kwargs "{\"adapt_mode\": \"bn_stat_clean\", \"bn_momentum\": ${MOM}, \"bn_update_target\": \"shallow_mean_deep_both\"}" \
    2>&1 | tee "${BASE_DIR}/hybrid.log"

if [ $? -eq 0 ]; then
    echo "=== hybrid DONE: $(date) ==="
else
    echo "=== hybrid FAILED: $(date) ==="
fi

# ── 3. mean_only ────────────────────────────────────────────────────────────
echo ""
echo "=== [3/3] mean_only (all layers, mom=${MOM}) ==="
timeout 1800 python3 train_pipeline.py \
    --model tcformer_otta --dataset bcic2a \
    --gpu_id "${GPU_ID}" --config "${CONFIG}" \
    --results_dir "${BASE_DIR}/mean_only" \
    --checkpoint_dir "${SOURCE_DIR}" \
    --model_kwargs "{\"adapt_mode\": \"bn_stat_clean\", \"bn_momentum\": ${MOM}, \"bn_update_target\": \"mean_only\"}" \
    2>&1 | tee "${BASE_DIR}/mean_only.log"

if [ $? -eq 0 ]; then
    echo "=== mean_only DONE: $(date) ==="
else
    echo "=== mean_only FAILED: $(date) ==="
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "================================================="
echo "SUMMARY"
echo "================================================="

python3 - "${BASE_DIR}" <<'PYEOF'
import re, sys, glob
import numpy as np

base = sys.argv[1]

def parse_results(path):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                m = re.search(r'Subject\s+(\d+)\s+=>\s+.*?Test Acc:\s+([\d.]+)', line)
                if m:
                    out[int(m.group(1))] = float(m.group(2)) * 100
    except FileNotFoundError:
        pass
    return out

src  = parse_results(f"{base}/source_only/results.txt")
hyb  = parse_results(f"{base}/hybrid/results.txt")
mno  = parse_results(f"{base}/mean_only/results.txt")

subjects = sorted(src.keys())

print(f"\n{'S':>3} | {'source':>7} | {'hybrid':>8} | {'Δhyb':>7} | {'mean_only':>9} | {'Δmno':>7} | {'hyb>mno?':>9}")
print("-" * 72)

hyb_deltas, mno_deltas = [], []
hyb_harms,  mno_harms  = [], []

for s in subjects:
    sv   = src.get(s)
    hv   = hyb.get(s)
    mv   = mno.get(s)
    if sv is None:
        continue
    dh = (hv - sv) if hv else None
    dm = (mv - sv) if mv else None
    dhs = f"{dh:+.2f}%" if dh is not None else "  ---"
    dms = f"{dm:+.2f}%" if dm is not None else "  ---"
    hvs = f"{hv:.2f}%"  if hv is not None else "  ---"
    mvs = f"{mv:.2f}%"  if mv is not None else "  ---"
    cmp = ""
    if hv and mv:
        cmp = "hyb>" if hv > mv + 0.1 else ("mno>" if mv > hv + 0.1 else "~tie")
    harm_flag = ""
    if dh is not None and dh < -1.0: harm_flag = "  *** HARM (hyb)"
    if dm is not None and dm < -1.0: harm_flag += "  *** HARM (mno)"
    print(f"  {s:>1} | {sv:7.2f} | {hvs:>8} | {dhs:>7} | {mvs:>9} | {dms:>7} | {cmp:>9}{harm_flag}")
    if dh is not None:
        hyb_deltas.append(dh)
        if dh < -0.5: hyb_harms.append(s)
    if dm is not None:
        mno_deltas.append(dm)
        if dm < -0.5: mno_harms.append(s)

print()
if hyb_deltas:
    print(f"  Hybrid   : NTR-S={len(hyb_harms)}/9={len(hyb_harms)/9:.1%}  WSD={min(hyb_deltas):+.2f}%  mean={sum(hyb_deltas)/len(hyb_deltas):+.2f}%")
if mno_deltas:
    print(f"  Mean-only: NTR-S={len(mno_harms)}/9={len(mno_harms)/9:.1%}  WSD={min(mno_deltas):+.2f}%  mean={sum(mno_deltas)/len(mno_deltas):+.2f}%")

# Reference line
print()
print("  Reference (mom=0.01, both, all 9 — from prior run):")
print("    NTR-S=2/9=22.2%  WSD=-5.56%  mean=-0.08%")

# Specific subjects of interest
print()
print("  Key subjects:")
for s, label in [(2, "S2 worst-harm"), (7, "S7 best-gain")]:
    sv = src.get(s, float('nan'))
    hv = hyb.get(s)
    mv = mno.get(s)
    dh = f"{hv-sv:+.2f}%" if hv else "---"
    dm = f"{mv-sv:+.2f}%" if mv else "---"
    print(f"    {label}: source={sv:.2f}%  hybrid={hv:.2f}%(Δ={dh})  mean_only={mv:.2f}%(Δ={dm})" if hv and mv else f"    {label}: missing data")

# Hypothesis check
print()
print("  Hypothesis check:")
s2_hyb = hyb.get(2)
s2_src = src.get(2)
s7_hyb = hyb.get(7)
s7_mno = mno.get(7)
s7_src = src.get(7)
s7_mno_d = (s7_mno - s7_src) if s7_mno and s7_src else None
s7_hyb_d = (s7_hyb - s7_src) if s7_hyb and s7_src else None

if s2_hyb and s2_src:
    d = s2_hyb - s2_src
    verdict = "PASS: S2 rescued" if d > -1.5 else ("PARTIAL" if d > -3.0 else "FAIL: culprit not fully in shallow_var")
    print(f"    S2 hybrid Δ={d:+.2f}%  → {verdict}")

if s7_hyb_d is not None and s7_mno_d is not None:
    verdict = "PASS: hybrid retains gain" if s7_hyb_d > s7_mno_d + 0.3 else "INCONCLUSIVE: hybrid ~ mean_only for S7"
    print(f"    S7 hybrid Δ={s7_hyb_d:+.2f}% vs mean_only Δ={s7_mno_d:+.2f}%  → {verdict}")

PYEOF

# ── Drift norm comparison from .npz ─────────────────────────────────────────
echo ""
echo "--- Shallow vs Deep drift norm (from .npz) ---"
python3 - "${BASE_DIR}" <<'PYEOF'
import sys, glob
import numpy as np

base = sys.argv[1]

print(f"\n{'Condition':<12}  {'shallow_mean_drift':>18}  {'shallow_var_drift':>17}  {'deep_mean_drift':>15}  {'deep_var_drift':>14}")
print("-" * 85)

for cond in ["source_only", "hybrid", "mean_only"]:
    npz_files = glob.glob(f"{base}/{cond}/**/*.npz", recursive=True)
    if not npz_files:
        print(f"  {cond:<12}  [no .npz]")
        continue
    # aggregate across subjects
    all_layers = []
    for f in npz_files:
        try:
            d = np.load(f, allow_pickle=True)
            if 'bn_drift_layers' in d:
                all_layers.append(d['bn_drift_layers'])  # [steps, 12]
        except Exception:
            pass
    if not all_layers:
        print(f"  {cond:<12}  [bn_drift_layers not found]")
        continue
    arr = np.concatenate(all_layers, axis=0)  # [total_steps, 12]
    n_layers = arr.shape[1]
    half = n_layers // 2
    # Separate mean drift (odd index?) — we don't have mean/var separated in layers
    # bn_drift_layers = |Δmean|.norm + |Δvar|.norm combined per layer
    sh_total = arr[:, :half].mean()
    dp_total = arr[:, half:].mean()
    sh_max   = arr[:, :half].max()
    dp_max   = arr[:, half:].max()
    print(f"  {cond:<12}  sh_mean={sh_total:.5f} sh_max={sh_max:.4f}  dp_mean={dp_total:.5f} dp_max={dp_max:.4f}")

PYEOF

echo ""
echo "================================================="
echo "All done: $(date)"
echo "================================================="
