---
name: research-direction-2605
description: DC-Replay検証後の研究方向の絞り込み(commitless主手法は筋が悪い/safety・理解主導へ)
metadata:
  type: project
---

2026-05下旬、DC-Replay(L1予測補正 / L2外部メモリ / L3 model-state commit)の内部検証と外部Deep Researchが一致した結論。

- **commitless memory/posterior correction を「次の提案手法」として磨くのは筋が悪い。** 理由: (1) 精度天井が低い([[dc-replay-empirical-ceilings]])、(2) アルゴリズム新規性が弱い(LAME / AdaNPC / BFT 2026 と被る)。
- **「大きく上げる ∧ 誰も壊さない」の同時強主張は、内部実測でも外部文献でも両立が薄い。** 片方を主目的、片方を制約にすべき(EEG-MI OTTAのgain相場は low single-digit pp)。
- **有力な核候補**: 手法新規性ではなく「理解(何が効き何が効かないか)+ safe-gain保証の評価設計」に置く。HSC / per-subject downside / L3 no-op / oracle天井 が手持ちの武器。
- **最重要外部競合**: BFT 2026 (arXiv:2601.07556, no-BP EEG-TTA)。原典PDF未確認、要精読。

**Why:** ユーザーは当初「全体平均+5%・汎用性」を目標にしたが、データ(oracle天井)と文献が否定。方向を safe-gain / 理解主導に再定義する根拠。
**How to apply:** 新手法を足す前に、この方向と矛盾しないか確認する。研究計画v0はこの前提で執筆中(safety主導、L3 no-opをmechanistic contributionに)。最終確定は計画レビュー後。
