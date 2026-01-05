#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat << 'USAGE'
prune_paper_results.sh
  Paper experiment results をスリム化するための削除スクリプト（デフォルトは dry-run）。

Usage:
  ./intentflow/offline/scripts/prune_paper_results.sh --target <path> [--mode minimal|keep_top_figures] [--yes]

Examples:
  # dry-run（削除対象と削減見込みだけ表示）
  ./intentflow/offline/scripts/prune_paper_results.sh --target intentflow/offline/results/paper_experiments/bcic2a/20260104_171049

  # 実削除（--yes 必須）
  ./intentflow/offline/scripts/prune_paper_results.sh --target intentflow/offline/results/paper_experiments/bcic2a/20260104_171049 --mode minimal --yes

  # bcic2a配下の全experimentに適用（dry-run）
  ./intentflow/offline/scripts/prune_paper_results.sh --target intentflow/offline/results/paper_experiments/bcic2a

Modes:
  minimal:
    - 数値/設定/デバッグ（results.txt, final_acc_*.json, debug_*.json, config.yaml 等）以外は極力削除
    - checkpoints, curves, confmats, features/logits, runs/*/figures を削除
    - top-level figures/ は削除しない（必要なら手動で消してください）

  keep_top_figures:
    - minimal と同等。ただし runs/*/figures は削除、top-level figures/ は保持（=minimalと同じ挙動）

Notes:
  - 安全のため、target は paper_experiments 配下である必要があります。
USAGE
}

TARGET=""
MODE="minimal"
APPLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="${2:-}"; shift 2 ;;
    --mode)
      MODE="${2:-}"; shift 2 ;;
    --yes)
      APPLY=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "${TARGET}" ]]; then
  echo "--target is required" >&2
  usage
  exit 2
fi

# Normalize to absolute path
if [[ "${TARGET}" != /* ]]; then
  TARGET="$(cd "$(dirname "${TARGET}")" && pwd)/$(basename "${TARGET}")"
fi

if [[ "${TARGET}" != *"/results/paper_experiments/"* ]]; then
  echo "Refusing to operate outside paper_experiments: ${TARGET}" >&2
  exit 2
fi

if [[ "${MODE}" != "minimal" && "${MODE}" != "keep_top_figures" ]]; then
  echo "Unknown --mode: ${MODE}" >&2
  exit 2
fi

echo ">>> target: ${TARGET}"
echo ">>> mode:   ${MODE}"
if [[ "${APPLY}" -eq 1 ]]; then
  echo ">>> APPLY:  yes (will delete)"
else
  echo ">>> APPLY:  no (dry-run)"
fi

bytes_of() {
  local p="$1"
  if [[ ! -e "${p}" ]]; then
    echo 0
    return
  fi
  # Prefer GNU du -sb if available
  if du -sb "${p}" >/dev/null 2>&1; then
    du -sb "${p}" | awk '{print $1}'
  else
    # Fallback: KB * 1024 (approx)
    local kb
    kb="$(du -sk "${p}" | awk '{print $1}')"
    echo $((kb * 1024))
  fi
}

human() {
  python3 - << 'PY' "$1"
import sys
n=int(sys.argv[1])
for unit in ["B","KB","MB","GB","TB"]:
    if n < 1024 or unit == "TB":
        print(f"{n:.1f}{unit}")
        break
    n /= 1024
PY
}

collect_experiments() {
  local t="$1"
  if [[ -d "${t}/runs" || -d "${t}/data" || -f "${t}/sweep_summary.csv" ]]; then
    echo "${t}"
    return
  fi
  # treat as dataset dir: list timestamp dirs
  find "${t}" -maxdepth 1 -mindepth 1 -type d -name "20*" | sort
}

delete_targets_for_exp() {
  local exp="$1"
  # Remove heavy artifacts, keep small numeric outputs and configs.
  cat << EOF
${exp}/data/checkpoints
${exp}/data/confmats
${exp}/data/curves
${exp}/data/features_*.npz
${exp}/data/logits_*.npy
${exp}/runs/*/data/checkpoints
${exp}/runs/*/data/confmats
${exp}/runs/*/data/curves
${exp}/runs/*/data/features_*.npz
${exp}/runs/*/data/logits_*.npy
${exp}/runs/*/figures
EOF
}

do_one_exp() {
  local exp="$1"
  local total_bytes=0
  local count=0

  while IFS= read -r pattern; do
    [[ -z "${pattern}" ]] && continue
    # Expand globs safely
    shopt -s nullglob
    local matches=()
    # shellcheck disable=SC2206
    matches=( ${pattern} )
    shopt -u nullglob
    if [[ ${#matches[@]} -eq 0 ]]; then
      continue
    fi
    for m in "${matches[@]}"; do
      local b
      b="$(bytes_of "${m}")"
      total_bytes=$((total_bytes + b))
      count=$((count + 1))
      if [[ "${APPLY}" -eq 1 ]]; then
        rm -rf "${m}"
      else
        echo "[dry-run] rm -rf ${m}  ($(human "${b}"))"
      fi
    done
  done < <(delete_targets_for_exp "${exp}")

  if [[ "${APPLY}" -eq 1 ]]; then
    echo ">>> pruned: ${exp}  removed ~$(human "${total_bytes}")  targets=${count}"
  else
    echo ">>> would prune: ${exp}  remove ~$(human "${total_bytes}")  targets=${count}"
  fi
}

exps=()
while IFS= read -r e; do
  [[ -z "${e}" ]] && continue
  exps+=("${e}")
done < <(collect_experiments "${TARGET}")

if [[ ${#exps[@]} -eq 0 ]]; then
  echo "No experiment directories found under: ${TARGET}" >&2
  exit 1
fi

grand=0
for exp in "${exps[@]}"; do
  echo "=== ${exp} ==="
  before="$(bytes_of "${exp}")"
  do_one_exp "${exp}"
  after="$(bytes_of "${exp}")"
  if [[ "${APPLY}" -eq 1 ]]; then
    freed=$((before - after))
    echo ">>> before=$(human "${before}") after=$(human "${after}") freed~=$(human "${freed}")"
    grand=$((grand + freed))
  else
    # In dry-run, just show the current size as context
    echo ">>> current_size=$(human "${before}")"
  fi
  echo
done

if [[ "${APPLY}" -eq 1 ]]; then
  echo ">>> DONE. Total freed ~= $(human "${grand}")"
  echo ">>> Tip: PNG も消したければ、top-level figures/ や data/curves を追加で削除してください。"
else
  echo ">>> DONE (dry-run). Add --yes to apply."
fi


