# 研究計画 v0(叩き台) — Safe Online OTTA for EEG-MI

> 位置づけ: DC-Replay検証と oracle天井([260526_oracle_ceiling](260526_oracle_ceiling/260526_oracle_ceiling_analysis.md))、外部Deep Research([deep-research-report (1)](deep-research-report%20(1).md))を踏まえた研究計画の v0。**方向の最終確定はこの v0 を Deep Research にレビューさせた後**。手法新規性ではなく「理解 + safe-gain保証の評価」を軸に置く案。

---

## 0. 一行主張(候補)

> EEG-MI の batch=1 online OTTA で「特徴を変えない補正で安全に得られる gain の上限」を実証的に確定し、**per-subject harm を保証付きに抑える online adaptation policy** を Active BCI 制約下で定式化・評価する。

「大きく上げる」を主張に置かない。データ(oracle天井 ≲+3.4pp)と文献(low single-digit pp)がそれを支持しないため。

---

## 1. 背景・問題設定

- EEG-MI は cross-session / cross-subject で分布がずれ、source-only モデルが劣化する。
- OTTA は追従できるが、無教師更新で一部被験者を壊す(harmful adaptation)。
- Active BCI 制約: single-trial 即時予測、低レイテンシ、abstain-safe。

## 2. 確定済みの地図(本研究の出発点・根拠つき)

| 事実 | 根拠 |
|---|---|
| L3 model-state commit は精度に no-op(commit 0/111/150 で精度同一) | `regular_seminar_2605/tables/l3_diagnostics.csv` |
| 特徴を変えない補正の天井: L1 ≲ +3.4pp(3データセット)、top2 は source依存(2a+11.84/HGD+5.00) | `260526_oracle_ceiling/` |
| 静的グローバルprior補正は Δ0(全データセット) | 同上 |
| commitless correction の新規性は弱い(LAME/AdaNPC/BFT 2026) | Deep Research |

## 3. リサーチクエスチョン

- **RQ1(mechanistic)**: batch=1 OTTAで、適応の各責務(prior補正 / external memory / model-state commit / 特徴適応)は精度・安全性にどう寄与するか。どれが効き、どれが効かないか。
- **RQ2(safety)**: mean accuracy を維持・改善しつつ、per-subject harm(worst-drop, harmed-count)を制約下に抑える policy は構成可能か。
- **RQ3(generality)**: その知見・policy は 2a / 2b / HGD で一貫するか。

## 4. 仮説(1文、棄却可能)

> 「特徴を変えない補正(prior / memory / commit)は安全だが gain 天井が低く(L1 ≲ +3.4pp)、その範囲では model-state commit は不要(no-op)であり、gain の大半は L1/L2 の非破壊補正から出る。」

棄却条件: マルチseedで commit-on が commit-off に対し有意な mean gain を示す、または L1/L2 を超える gain が commitless で出る。

## 5. 貢献候補

- **C1 Mechanistic**: 「何が効き何が効かないか」の体系的分解(oracle天井 + 公平 ablation)。negative result(L3 no-op)を含む。
- **C2 Safety framework**: final / online / decision safety の定義と、harm制約付き adaptation policy(`mean↑ s.t. worst-drop ≤ τ, harmed ≤ k`)。
- **C3 Active BCI評価**: single-trial / latency / abstain / reset を含む MI-OTTA 評価プロトコル。abstain主評価は先行が薄く差別化点。

## 6. 実験計画

- **データセット**: BCIC2a(主)、BCIC2b、HGD。OpenBMI/Lee2019 は余力で追加。
- **split**: cross-session / cross-subject(Wimpff 2024 に準拠して比較可能性を確保)。
- **seed**: **複数seed必須**(single-seed gain はノイズと判別不能。2aで実証済み: +1.23→seed込み+0.26〜0.49)。
- **metrics**: mean acc, kappa, per-subject delta, worst-subject delta, HSC@0.5/1.0pp, online regret, max drawdown, recovery time, abstain rate, latency/trial。
- **baseline**: `source_only` / `replay_safe`(安全基準) / Wimpff-style OTTA / BFT-style no-BP(competitor) / DC各variant。
- **ablation軸(1因子ずつ)**: commit有無 / replay gate有無 / L1 correction有無 / L2 memory有無。→ 詳細設計時に `ablation-design` skill を使う。
- **達成基準**:
  - 成功: harm制約(HSC ≤ k, worst-drop ≤ τ)を満たしつつ mean gain が `replay_safe` を**有意に**上回る。
  - 棄却: マルチseedで安全policyが source_only に有意 gain 無し かつ harm制約も既存baselineを超えない → 手法貢献(C2)は棄却、C1+C3に絞る。

## 7. 競合と差別化(Deep Researchで詳細化)

| 競合 | 軸 | 差別化方針 |
|---|---|---|
| LAME / AdaNPC | parameter-free posterior/memory correction | 補正式でなく safety保証・評価で差別化 |
| **BFT 2026** (arXiv:2601.07556) | no-BP EEG-TTA(最重要競合) | **原典精読が最優先**。被り具合次第で主張を再調整 |
| Wimpff 2024 | MI-OTTA(alignment+BN+EM) | 必須比較対象。per-subject harm 主評価で差別化 |

## 8. リスク・未確定

- **BFT 2026 との被り**(要精読、新規性の生死)。
- 安全policyが trivial(no-update多用)に退化し「壊さないが上げない」になるリスク。
- gain がseedノイズ内に収まるリスク(→ 複数seed + CI で判定)。
- 「大きく上げる」を捨てきれない場合: 特徴適応(TTT/alignment)に振る別計画が必要(安全性トレードオフ・競合増)。

## 9. マイルストーン

1. ~~oracle天井~~(✓ 済)
2. マルチseed の source_only / replay_safe baseline を 2a/2b/HGD で固める
3. 公平 ablation(commit vs commitless, gate有無)→ `ablation-design` skill
4. safety指標の実装(online regret / max drawdown / harmful-correction 分解)
5. BFT 2026 精読 + 差別化表
6. この計画を Deep Research にレビューさせる
7. 方向の最終確定 → 手法 or 評価のどちらを主貢献にするか決定

---

## 次の一歩(最小)

マイルストーン2(マルチseed baseline)は、新規ランが要る(現状 2a は sweep 済みだが 2b/HGD は single-seed firstpass)。GPU実行。これが揃わないと RQ2/RQ3 の有意性が判定できない。**v0レビュー → baseline 拡充 → ablation の順**を推奨。
